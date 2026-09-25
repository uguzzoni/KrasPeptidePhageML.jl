###############################################################################
# 02_perturbations.jl — regenerate `results/perturbations.jld2`, the artifact
# the combined figure is drawn from.
#
#   julia --project=. scripts/02_perturbations.jl            # all 25 runs, ~3 h
#   julia --project=. scripts/02_perturbations.jl --schedule pilot --reps 1 --folds 2
#   julia --project=. scripts/02_perturbations.jl --only cv    # just the folds
#
# Two perturbations, both warm-started from the published model:
#
#   cv        5 folds over SEQUENCES — refit on 4/5 of the peptides
#   resample  reads resampled per column, multinomially, on the FIXED sequence
#             set: frac 1.0 is the bootstrap at the observed depth, 0.5 and 0.25
#             are rarefaction
#
# WARM START. Each refit begins at the published solution rather than from
# scratch, so what it measures is how far the perturbed data PULLS that solution
# — not how reproducible a fit is from a random init. This is deliberate and
# conservative: any movement reported is real, because the run had every chance
# to stay put. It also means the CV folds are NOT a held-out predictive test —
# the reference model was fitted on every sequence, so it has already seen every
# test fold. Do not read row 3 of the figure as generalisation error.
#
# Runs are appended to the output file as they finish, so an interrupted run
# loses at most one refit; `--resume` skips those already stored.
###############################################################################

using KrasPeptidePhageML
using JLD2, Random, Statistics, Printf
include(joinpath(@__DIR__, "common.jl"))

schedule = schedule_from_name(argval(ARGS, "--schedule", "warm"; as = String))
nfolds   = argval(ARGS, "--folds", 5)
nreps    = argval(ARGS, "--reps", 10)          # replicates at full depth
nreps_sub = argval(ARGS, "--reps-sub", 5)      # replicates per subsampled depth
fracs    = [parse(Float64, x) for x in split(argval(ARGS, "--fracs", "0.5,0.25"; as = String), ",")]
only     = argval(ARGS, "--only", "all"; as = String)   # all | cv | resample
seed     = argval(ARGS, "--seed", 1)
split_seed = argval(ARGS, "--split-seed", 1234)
resume   = argflag(ARGS, "--resume")
outfile  = argval(ARGS, "--out", joinpath(repo_root(), "results", "perturbations.jld2");
                  as = String)

@info "02_perturbations" schedule nfolds nreps fracs only outfile resume

data, df_exp, seqs = load_lung_dataset()
select, washed = build_select_washed(df_exp)
reference = load_reference_model()
S = number_of_sequences(data)

# The figure only ever looks at the 369 designed candidates, so that is all that
# is stored: 369x5 Float32 per run instead of the fitted model and a 20 438-row
# evaluation panel. 25 runs come to ~200 KB, which is committable.
cand = designed_candidates()
sequence = cand.sequence
E_reference = Float32.(panel_energies(reference, sequence))

# (kind, frac, rep) for every refit, in the order they are run
plan = Tuple{String,Float64,Int}[]
if only in ("all", "resample")
    append!(plan, [("resample", 1.0, r) for r in 1:nreps])
    for f in fracs, r in 1:nreps_sub
        push!(plan, ("resample", f, r))
    end
end
only in ("all", "cv") && append!(plan, [("cv", 1.0, f) for f in 1:nfolds])

runs = NamedTuple[]
if resume && isfile(outfile)
    runs = JLD2.load(outfile, "runs")
    done = Set((r.kind, r.frac, r.rep) for r in runs)
    filter!(p -> p ∉ done, plan)
    @info "resuming" n_done=length(runs) n_left=length(plan)
end

"Refit from the published solution on `d`, and return the candidates' energies."
function refit(d)
    model = build_warm_model(reference, select, washed)
    _, timings, _ = train_model!(model, d; schedule = schedule, verbose = false)
    return Float32.(panel_energies(model, sequence)), sum(t.seconds for t in timings)
end

t_start = time()
for (i, (kind, frac, rep)) in enumerate(plan)
    d = if kind == "cv"
        # The split is fixed by `split_seed`, independent of any model seed, so
        # every fold reconstructs the same partition on any machine.
        train_idx, _ = cv_indices(S, nfolds, rep; seed = split_seed)
        select_sequences(data, train_idx)
    else
        # The sequence set is NOT resampled — only the reads, per column — so the
        # energies stay comparable across replicates. The all-zero latent columns
        # stay empty because `resample_counts` skips columns with no reads.
        rng = MersenneTwister(hash((seed, rep, frac)) % typemax(Int32))
        Data(data.sequences, resample_counts(data.counts, frac, rng), data.ancestors)
    end

    E, secs = refit(d)
    push!(runs, (kind = kind, frac = frac, rep = rep, E = E))
    jldsave(outfile; sequence, E_reference, runs)      # checkpoint after each refit

    agree = 100 * mean(published_labels(E) .== published_labels(E_reference))
    @printf("[%2d/%2d] %-8s frac=%.2f rep=%d  %6.1f s  labels %5.1f %%  (elapsed %.1f min)\n",
            i, length(plan), kind, frac, rep, secs, agree, (time() - t_start) / 60)
end

@printf("\nwritten: %s  (%d runs, %.1f KB, %.1f min)\n",
        outfile, length(runs), filesize(outfile) / 1024, (time() - t_start) / 60)
