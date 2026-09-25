###############################################################################
# The published specificity classification, and how stable it is.
#
# Every model-derived conclusion in the paper flows through ONE object: the
# per-mode energy matrix, and specifically ABSOLUTE thresholds on `E[mutation]`
# and `E[wt]`. That is why the analyses here are about those labels.
#
# It is also why the labels are the fragile part: under read subsampling to 25 %
# the energy *rankings* survive (Spearman ~0.89) while the labels collapse to
# 30 %, because a modest systematic shift sweeps candidates across a fixed
# cutoff. Candidates whose label is unstable are, mechanically, the ones sitting
# ON a boundary.
###############################################################################

"""
Thresholds of the original analysis, on the `mutation` and `wt` energies.

    mut-specific    E[mutation] < -5    and E[wt] > 0
    wt-specific     E[mutation] > -3.5  and E[wt] < -0.5
    cross-specific  E[mutation] < -3.5  and E[wt] < 0
"""
const PUBLISHED_THRESHOLDS = (mut_hi = -5.0, wt_lo = 0.0,
                              wt_mut_hi = -3.5, wt_wt_hi = -0.5,
                              cross_mut = -3.5, cross_wt = 0.0)

"""
    published_labels(E; th=PUBLISHED_THRESHOLDS)

Apply the rule above to an energy matrix `E[s, mode]`, returning one of
`:mut_specific`, `:wt_specific`, `:cross_specific`, `:none` per row.

The three classes are mutually exclusive as written (mut needs `E[wt] > 0` while
cross needs `E[wt] < 0`; wt needs `E[mutation] > -3.5` while cross needs
`< -3.5`) but they do **not** cover the plane — candidates falling in no class
get `:none`, which is itself part of what gets measured.
"""
function published_labels(E::AbstractMatrix; th = PUBLISHED_THRESHOLDS)
    emut = @view E[:, MODES[:mutation]]
    ewt = @view E[:, MODES[:wt]]
    labels = fill(:none, size(E, 1))
    @inbounds for s in eachindex(labels)
        if emut[s] < th.mut_hi && ewt[s] > th.wt_lo
            labels[s] = :mut_specific
        elseif emut[s] > th.wt_mut_hi && ewt[s] < th.wt_wt_hi
            labels[s] = :wt_specific
        elseif emut[s] < th.cross_mut && ewt[s] < th.cross_wt
            labels[s] = :cross_specific
        end
    end
    return labels
end

"""
    designed_candidates()

The 369 published designed candidates: `sequence`, `published_class`, and the
four `min_*_energy` columns as published.

Shipped as `data/designed_candidates.csv` (exported from `sup_tables.xlsx`
Tables S2/S3) so the repo needs no XLSX dependency. Those energy columns are the
reference model's own per-mode energies, which makes them a golden test of the
whole vendored energy path — see `test/runtests.jl`.
"""
function designed_candidates(; path = bundled("designed_candidates.csv"))
    isfile(path) || error("designed candidates file not found: $path")
    tbl, header = readdlm(path, ',', String, '\n', header = true)
    cols = vec(header)
    text_cols = ("sequence", "published_class", "strategies")
    df = DataFrame()
    for (j, c) in enumerate(cols)
        v = tbl[:, j]
        df[!, Symbol(c)] = (c in text_cols) ? String.(strip.(v)) :
                           [isempty(strip(x)) ? NaN : parse(Float64, x) for x in v]
    end
    return df
end

"""
    affine_align(ref, y) -> (a, b)

Least-squares `y ≈ a * ref + b`.

Energies are identified only up to a per-mode additive constant (it is absorbed
into `μ`), and training can rescale them, so raw agreement with a reference model
understates how well the *ranking* is reproduced. Report both: if the slope and
intercept wander while Spearman holds, the ranking generalises and the absolute
cutoffs are a calibration of one particular fit.
"""
function affine_align(ref::AbstractVector, y::AbstractVector)
    keep = isfinite.(ref) .& isfinite.(y)
    x = ref[keep]; z = y[keep]
    length(x) > 2 || return (NaN, NaN)
    mx, mz = mean(x), mean(z)
    vx = sum(abs2, x .- mx)
    vx > 0 || return (NaN, NaN)
    a = sum((x .- mx) .* (z .- mz)) / vx
    return (a, mz - a * mx)
end

"""
    agreement_table(E_ref, Es, names; modes, subset)

Per mode and per run: Pearson, Spearman, the affine slope/intercept against the
reference, and the RMSE after alignment.
"""
function agreement_table(E_ref::AbstractMatrix, Es::Vector, names::Vector;
                         modes = [:positive, :negative, :mutation, :wt],
                         subset = nothing)
    rows = NamedTuple[]
    idx = subset === nothing ? Colon() : subset
    for m in modes
        i = MODES[m]
        r = Float64.(E_ref[idx, i])
        for (nm, E) in zip(names, Es)
            y = Float64.(E[idx, i])
            keep = isfinite.(r) .& isfinite.(y)
            a, b = affine_align(r[keep], y[keep])
            resid = y[keep] .- (a .* r[keep] .+ b)
            push!(rows, (mode = m, run = nm, n = sum(keep),
                         pearson = cor(r[keep], y[keep]),
                         spearman = corspearman(r[keep], y[keep]),
                         slope = a, intercept = b,
                         rmse_aligned = sqrt(mean(abs2, resid))))
        end
    end
    return DataFrame(rows)
end

"""
    label_agreement(E_ref, Es)

Per-candidate table: the reference label, and how many of `Es` reproduce it.

Read the result against the **same-data control**: refitting from the reference
on identical data already loses ~5 % of the labels to minibatch noise, so 95 %
is the ceiling, not 100 %. Only the shortfall below the control is a sampling
effect.
"""
function label_agreement(E_ref::AbstractMatrix, Es::Vector; sequences = nothing)
    ref = published_labels(E_ref)
    labs = [published_labels(E) for E in Es]
    df = DataFrame(published = ref)
    sequences === nothing || (df.sequence = collect(sequences))
    df.n_agree = [count(l -> l[k] == ref[k], labs) for k in eachindex(ref)]
    df.frac_agree = df.n_agree ./ max(1, length(Es))
    return df
end

jaccard(a::AbstractVector, b::AbstractVector) =
    (u = length(union(a, b)); u == 0 ? NaN : length(intersect(a, b)) / u)

"""
    topn_overlap(scores, n)

Pairwise Jaccard overlap of the top-`n` sets induced by each score vector
(higher is better). The robust counterpart to the label comparison: it survives
perturbations that break the absolute thresholds.
"""
function topn_overlap(scores::Vector, n::Int)
    tops = [partialsortperm(s, 1:min(n, length(s)); rev = true) for s in scores]
    k = length(scores)
    M = fill(NaN, k, k)
    for i in 1:k, j in 1:k
        M[i, j] = jaccard(tops[i], tops[j])
    end
    return M
end

"""
    cv_indices(S, nfolds, fold; seed=1234)

Indices of the `fold`-th train/test split of `S` sequences. The permutation is
fixed by `seed` (independent of any model seed) so every fold reconstructs the
same partition.
"""
function cv_indices(S::Int, nfolds::Int, fold::Int; seed::Int = 1234)
    @assert 1 ≤ fold ≤ nfolds
    p = randperm(MersenneTwister(seed), S)
    bounds = [(round(Int, (k - 1) * S / nfolds) + 1):round(Int, k * S / nfolds) for k in 1:nfolds]
    test = sort(p[bounds[fold]])
    train = sort(p[reduce(vcat, [collect(bounds[k]) for k in 1:nfolds if k != fold])])
    @assert length(train) + length(test) == S
    @assert isempty(intersect(train, test))
    return train, test
end
