###############################################################################
# Golden tests.
#
# Tests 2-4 are the real acceptance criteria for the vendoring: they compare this
# package against numbers produced by the original codebase, so a transcription
# mistake cannot pass silently.
#
# Tests needing the data are skipped (not failed) when it is not available, so
# `Pkg.test()` works on a checkout without it.
###############################################################################

using Test
using KrasPeptidePhageML
using DataFrames, Statistics, Random, Printf
using CairoMakie: Makie

const HAVE_MODEL = try
    isfile(data_path(:reference_model))
catch
    false
end

const HAVE_COUNTS = try
    isfile(data_path(:counts_csv)) && isfile(data_path(:samples_csv))
catch
    false
end

HAVE_MODEL || @warn "reference model not configured — golden tests 2-4 will be skipped"
HAVE_COUNTS || @warn "data CSVs not found — the import golden test will be skipped"

@testset "KrasPeptidePhageML" begin

    # ---------------------------------------------------------------- alphabet
    @testset "alphabet and encodings" begin
        @test length(AAs) == 20
        @test AA2INT['A'] == 1 && AA2INT['Y'] == 20
        @test seq2str(str2seq("ACDEFGH")) == "ACDEFGH"
        x = sample2hot([str2seq("ACDEFGH")])
        @test size(x) == (20, 7, 1)
        @test all(sum(x; dims = 1) .== 1)          # valid one-hot
        @test onehot2aa(x[:, :, 1]) == "ACDEFGH"
    end

    # ------------------------------------------------- the regularizer bug fix
    @testset "regularizer is actually differentiated" begin
        # The original passed the penalty as a zero-argument closure over the
        # OUTER model, which contributes exactly zero gradient under Flux's
        # explicit gradient mode. This pins the corrected behaviour.
        select = zeros(Bool, 5, 4); washed = zeros(Bool, 5, 4)
        select[1, :] .= true; washed[5, :] .= true
        model = build_model(select, washed; seed = 1)
        α = 0.05

        g_ok = KrasPeptidePhageML.Flux.gradient(m -> l2_nn(m, α), model)[1]
        W = model.states[1].m.layers[2].weight
        gW = g_ok.states[1].m.layers[2].weight
        @test gW !== nothing
        @test sum(abs, gW) > 0                                   # non-zero!
        @test isapprox(sum(abs, gW), 2α * sum(abs, W); rtol = 1e-5)

        # and the broken idiom really is zero, so the test means something
        g_bad = KrasPeptidePhageML.Flux.gradient(model) do m
            l2_nn(model, α) + 0 * sum(m.states[1].m.layers[2].weight)
        end[1]
        gbW = g_bad.states[1].m.layers[2].weight
        @test gbW === nothing || sum(abs, gbW) == 0
    end

    # ------------------------------------------------------- Monte-Carlo kernel
    @testset "MC kernel" begin
        select = zeros(Bool, 5, 4); washed = zeros(Bool, 5, 4)
        select[1, :] .= true; washed[5, :] .= true
        model = build_model(select, washed; seed = 3)
        s = zeros(20, 7); for i in 1:7; s[rand(1:20), i] = 1; end

        @test size(flip(s, 3, 2)) == size(s)
        @test sum(flip(s, 3, 2)[:, 2]) == 1 && flip(s, 3, 2)[3, 2] == 1
        @test size(Δenergies(s, 3, 2, model), 2) == 5

        before = copy(s)
        ToParetoFront!(s, model; modes = [1])
        @test all(sum(s; dims = 1) .== 1)              # still a valid one-hot
        @test size(s) == size(before)
        # a Pareto front means no single flip improves every selected mode
        @test isPareto(s, model; modes = [1])
    end

    # ---------------------------------------------------- GOLDEN TEST 2 + design
    if HAVE_MODEL
        @testset "GOLDEN: energy path reproduces published energies" begin
            ref = load_reference_model()
            @test ref isa Model
            @test length(ref.states) == 5            # the ZeroEnergy state survived import
            @test length(ref.ζ) == 32
            @test size(ref.μ) == (5, 32)

            df = designed_candidates()
            @test nrow(df) == 369
            E = panel_energies(ref, df.sequence)
            @test size(E) == (369, 5)
            @test all(E[:, MODES[:wash]] .== 0)      # wash is the zero-energy mode

            # The published columns ARE this model's per-mode energies. Weights are
            # Float32, so agreement is at Float32 precision (relative ~1.2e-7),
            # not bit-identical: summation order differs.
            for (col, mode) in (("min_positive_energy", :positive),
                                ("min_negative_energy", :negative),
                                ("min_mutation_energy", :mutation),
                                ("min_wt_energy", :wt))
                x = df[!, col]; y = E[:, MODES[mode]]
                @test cor(x, y) > 0.999999
                @test maximum(abs.(x .- y)) < 1e-5
            end
        end

        @testset "published specificity labels" begin
            ref = load_reference_model()
            df = designed_candidates()
            E = panel_energies(ref, df.sequence)
            labs = published_labels(E)
            @test length(labs) == 369
            @test all(l -> l in (:mut_specific, :wt_specific, :cross_specific, :none), labs)
            # the classes are mutually exclusive by construction but do not cover
            # the plane, so `:none` is expected to be present
            @test count(==(:mut_specific), labs) > 0
        end
    end

    # ------------------------------------------- GOLDEN TESTS 3 + 4 (need CSVs)
    if HAVE_MODEL && HAVE_COUNTS
        @testset "GOLDEN: import path and design" begin
            data, df_exp, seqs = load_lung_dataset(verbose = false)
            ref = load_reference_model()

            @test number_of_rounds(data) == 32
            @test number_of_sequences(data) == length(seqs)
            @test nrow(df_exp) == 32
            @test count(df_exp.latent) == 8
            # the latent round-2 nodes carry no reads
            @test all(iszero, data.counts[:, df_exp[df_exp.latent, :counts_column]])
            # the counts table round-trips its own summary columns
            @test [sum(view(data.counts, :, t)) for t in df_exp.counts_column] == df_exp.total_reads

            # GOLDEN TEST 3 — the reconstructed design IS the published one
            select, washed = build_select_washed(df_exp)
            @test select == ref.select
            @test washed == ref.washed

            # GOLDEN TEST 4 — one number validating the whole CSV -> Data chain
            ll = log_likelihood(ref, data)
            @test isapprox(ll, -392.4327; atol = 1e-3)
        end
    end

    # --------------------------------------------------------------- utilities
    @testset "cv splits and resampling" begin
        tr, te = cv_indices(100, 5, 2; seed = 1234)
        @test length(tr) + length(te) == 100
        @test isempty(intersect(tr, te))
        @test cv_indices(100, 5, 2; seed = 1234) == (tr, te)      # reproducible

        counts = Float64.(rand(MersenneTwister(1), 50:500, 200, 4))
        counts[:, 2] .= 0                                          # a latent column
        rs = resample_counts(counts, 1.0, MersenneTwister(7))
        for t in (1, 3, 4)
            @test isapprox(sum(rs[:, t]), sum(counts[:, t]); rtol = 1e-9)
        end
        @test all(rs[:, 2] .== 0)                                  # stays empty
        # `resample_counts` draws exactly `round(Int, frac * R)` reads, so the
        # total is the rounded half, not the half (here R is odd).
        half = resample_counts(counts, 0.5, MersenneTwister(7))
        @test sum(half[:, 1]) == round(Int, 0.5 * sum(counts[:, 1]))
    end

    # ------------------------------------------------------------------ figure
    # Cheap: the perturbation artifact is ~200 KB and no model is touched. This
    # pins the artifact's shape and that all 9 panels render, not their looks.
    @testset "combined figure" begin
        (; sequence, E_reference, runs) = load_perturbations()
        @test length(sequence) == size(E_reference, 1) == 369
        @test size(E_reference, 2) == length(MODES)
        @test all(size(r.E) == size(E_reference) for r in runs)
        @test count(r -> r.kind == "cv", runs) == 5
        @test count(r -> r.kind == "resample" && r.frac == 1.0, runs) == 10

        fig = figure_combined()
        axes = [c for c in fig.content if c isa Makie.Axis]
        @test length(axes) == 9

        out = joinpath(mktempdir(), "fig")
        figure_combined(outfile = out)
        for ext in ("svg", "png", "pdf")
            @test filesize("$out.$ext") > 10_000
        end
        # Vector text, but NOT editable text: Cairo's SVG surface subsets glyphs
        # into outline definitions referenced by <use>, so `<text>` never appears.
        # The original matplotlib figure used `svg.fonttype = "none"` and did
        # emit it. Pinned here so the difference is not rediscovered as a bug.
        svg = read("$out.svg", String)
        @test occursin("id=\"glyph", svg)
        @test !occursin("<text", svg)
    end
end
