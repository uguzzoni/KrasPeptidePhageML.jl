###############################################################################
# Evaluation metrics — and which of them actually mean anything.
#
# Three metrics are available and they are NOT equally informative. This was
# established empirically on the published model and data:
#
#  * ENRICHMENT correlation (`pooled_enrichment`) — predicted vs measured
#    log-enrichment over the nearest sequenced ancestor. THE metric to quote.
#    The per-sequence total `lRs` cancels in the ratio, so nothing about a
#    sample's own reads leaks into its prediction. Published model: r = 0.21 at
#    count threshold 10, 0.34 at 100, against ~0.01 for an untrained model.
#
#  * ABUNDANCE correlation (`pooled_abundance`) — DO NOT use as evidence of
#    predictive power. `log_abundances` adds `data.lRs`, each sequence's total
#    count across the tree, which the model receives as input. Measured on the
#    published dataset: fitted model 0.757, flat no-selection model 0.756,
#    UNTRAINED randomly initialised model 0.755. It measures the input data.
#
#  * per-read CROSS-ENTROPY (`predictive_logloss`) — the multinomial log-loss.
#    Comparable across models on the same dataset. Compare against `flat_model`,
#    not against an ancestor-frequency null: the latter has no access to `lRs`
#    and so flatters the model by ~1.2 nats/read.
#
# WHY POOLING. Enrichment is a ratio, so both the sample and its ancestor must be
# well measured. But the data are sparse — ~1.2 M reads over ~124 000 sequences is
# ~10 reads/sequence — so a high threshold leaves very few sequences per sample:
#
#     both counts > thr   |  thr=10  thr=30  thr=100  thr=1000
#     full data (median)  |    274      18        8         1
#     1/5 held-out fold   |     45       3        2         0
#
# Averaging PER-SAMPLE correlations above thr~30 therefore averages a handful of
# estimates each built on a handful of points; it is unstable and wanders
# non-monotonically in a way that looks like signal. These functions instead
# CENTRE each sample and POOL across samples, and report `n` with every value.
###############################################################################

_center(v) = v .- mean(v)

function _nearest_sequenced_ancestor(ancestors, dep, t::Int)
    a = ancestors[t]
    while a > 0
        dep[a] > 0 && return a
        a = ancestors[a]
    end
    return nothing
end

"""
    pooled_enrichment(model, data; thr=10, min_per_sample=10)

Predicted vs measured log-enrichment over the nearest *sequenced* ancestor
(walking past the latent round-2 nodes), pooled across samples after centring
each. A sequence contributes only if BOTH its count in the sample and in the
ancestor exceed `thr`.

Returns `(pearson, spearman, n, nsamples)` — always check `n`.

A model with constant selectivity predicts an enrichment that is mathematically
identical for every sequence (`lRs` and the normalisation cancel), so what
survives centring is floating-point noise. That is detected and reported as
`NaN` rather than as a spurious correlation — without the guard the flat null
produced an apparent r = -0.14.
"""
function pooled_enrichment(model::Model, data::Data; thr::Real = 10, min_per_sample::Int = 10)
    lN = log_abundances(model, data)
    C = data.counts
    dep = vec(sum(C; dims = 1))
    E = Float64[]; P = Float64[]; nsamp = 0
    for t in axes(C, 2)
        dep[t] > 0 || continue
        a = _nearest_sequenced_ancestor(data.ancestors, dep, t)
        a === nothing && continue
        idx = findall((view(C, :, t) .> thr) .& (view(C, :, a) .> thr))
        length(idx) >= min_per_sample || continue
        emp  = log.(C[idx, t] ./ dep[t]) .- log.(C[idx, a] ./ dep[a])
        pred = vec(lN[idx, t]) .- vec(lN[idx, a])
        k = isfinite.(emp) .& isfinite.(pred)
        sum(k) >= min_per_sample || continue
        append!(E, _center(emp[k])); append!(P, _center(pred[k]))
        nsamp += 1
    end
    length(E) < 20 && return (pearson = NaN, spearman = NaN, n = length(E), nsamples = nsamp)
    std(P) < 1e-8 * max(std(E), eps()) &&
        return (pearson = NaN, spearman = NaN, n = length(E), nsamples = nsamp)
    return (pearson = cor(E, P), spearman = corspearman(E, P), n = length(E), nsamples = nsamp)
end

"""
    pooled_abundance(model, data; thr=10, min_per_sample=10)

Predicted vs observed abundance (log scale), pooled the same way.

Reported as a control only — see the header. An untrained model scores ~0.755 on
the published data where the fitted model scores 0.757.
"""
function pooled_abundance(model::Model, data::Data; thr::Real = 10, min_per_sample::Int = 10)
    lN = log_abundances(model, data)
    C = data.counts
    f = normalize_counts(C)
    X = Float64[]; Y = Float64[]; nsamp = 0
    for t in axes(C, 2)
        idx = findall(view(C, :, t) .> thr)
        length(idx) >= min_per_sample || continue
        x = vec(f[idx, t]); y = exp.(vec(lN[idx, t]))
        k = (x .> 0) .& (y .> 0) .& isfinite.(y)
        sum(k) >= min_per_sample || continue
        append!(X, _center(log.(x[k]))); append!(Y, _center(log.(y[k])))
        nsamp += 1
    end
    length(X) < 20 && return (pearson = NaN, spearman = NaN, n = length(X), nsamples = nsamp)
    return (pearson = cor(X, Y), spearman = corspearman(X, Y), n = length(X), nsamples = nsamp)
end

"""
    predictive_logloss(model, data)

Multinomial cross-entropy of the predicted abundances, in **nats per read**, per
sample: `-sum_s counts[s,t] * logN[s,t] / R_t`.

Prefer this to differencing `log_likelihood` between a training and a held-out
subset: `log_likelihood` carries the data-only multinomial constant `data.lMt`,
which differs between subsets, so that difference is not a clean overfitting
measure. Comparing two *models* on the *same* data is fine either way, since
`lMt` cancels.
"""
function predictive_logloss(model::Model, data::Data)
    lN = log_abundances(model, data)
    C = data.counts
    ce = fill(NaN, size(C, 2))
    for t in axes(C, 2)
        R = sum(@view C[:, t])
        R > 0 || continue
        ce[t] = -sum(view(C, :, t) .* view(lN, :, t)) / R
    end
    return ce
end

"""
    resample_counts(counts, frac, rng)

Per-column multinomial resampling of reads:
`counts'[:, t] ~ Multinomial(round(frac * R_t), counts[:, t] / R_t)`,
independently for every column with `R_t > 0` — so the synthetic latent round-2
columns (all zero) stay zero.

`frac == 1` is the nonparametric bootstrap at the observed depth; `frac < 1` is
read subsampling. The sequence set is **not** resampled, so this captures
read-sampling noise only — not library-composition noise, and certainly not
biological replication.

Implemented by sequential binomial decomposition to avoid a Distributions
dependency.
"""
function resample_counts(counts::AbstractMatrix, frac::Real, rng::AbstractRNG)
    out = zeros(eltype(counts), size(counts))
    for t in axes(counts, 2)
        col = @view counts[:, t]
        R = sum(col)
        R > 0 || continue
        n = round(Int, frac * R)
        n > 0 || continue
        remaining_n = n
        remaining_p = float(R)
        @inbounds for s in axes(counts, 1)
            remaining_n > 0 || break
            c = col[s]
            c > 0 || continue
            p = min(1.0, c / remaining_p)
            draw = _rand_binomial(rng, remaining_n, p)
            out[s, t] = draw
            remaining_n -= draw
            remaining_p -= c
        end
    end
    return out
end

"Binomial sampler: inversion for small `n*p`, else a normal approximation with clamping."
function _rand_binomial(rng::AbstractRNG, n::Int, p::Float64)
    n <= 0 && return 0
    p <= 0 && return 0
    p >= 1 && return n
    if n * p < 30
        k = 0
        @inbounds for _ in 1:n
            rand(rng) < p && (k += 1)
        end
        return k
    end
    μ = n * p
    σ = sqrt(n * p * (1 - p))
    return clamp(round(Int, μ + σ * randn(rng)), 0, n)
end
