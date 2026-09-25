###############################################################################
# Constructing models.
###############################################################################

"""
    mode_nn(A, L)

The per-mode energy network of the published model: `flatten -> Dense(A*L, 2L,
relu) -> Dense(2L, 1)`. For A=20, L=7 that is 140 -> 14 -> 1, which matches the
published `model_nn_2l.jld2` layer shapes exactly.
"""
mode_nn(A::Int, L::Int) = Flux.Chain(Flux.flatten,
                                     Flux.Dense(A * L, 2 * L, Flux.relu),
                                     Flux.Dense(2 * L, 1, identity))

"""
    build_model(select, washed; seed=nothing, A=20, L=7)

A fresh 1-mut model: four `DeepEnergy` modes (`positive`, `negative`,
`mutation`, `wt`) plus a `ZeroEnergy` `wash` mode, with `μ` and `ζ` at zero.

Weights are `Float32` (Flux's default), as in the published model.
"""
function build_model(select::AbstractMatrix{Bool}, washed::AbstractMatrix{Bool};
                     seed::Union{Nothing,Int} = nothing,
                     A::Int = ALPHABET_SIZE, L::Int = PEPTIDE_LENGTH)
    seed === nothing || Random.seed!(seed)
    n_modes, n_rounds = size(select)
    @assert n_modes == length(MODES)
    return Model(
        (DeepEnergy(mode_nn(A, L)),   # 1 positive
         DeepEnergy(mode_nn(A, L)),   # 2 negative
         DeepEnergy(mode_nn(A, L)),   # 3 mutation
         DeepEnergy(mode_nn(A, L)),   # 4 wt
         ZeroEnergy()),               # 5 wash
        zeros(n_modes, n_rounds),
        zeros(n_rounds),
        select,
        washed,
    )
end

"""
    build_warm_model(reference, select, washed)

A copy of an already-fitted model to refit from — its energy networks, `μ` and
`ζ`. Requires the design to match the reference exactly, so the refit starts at
precisely the reference's optimum.

Use for **stability** analyses (refit from the published solution on perturbed
data and measure how far it moves): conservative by construction, since any
movement it reports is real.

Do **not** use it for held-out predictive claims. The published model was fitted
on every sequence, so a warm-started model has already seen the test fold.
"""
function build_warm_model(reference::Model, select::AbstractMatrix{Bool},
                          washed::AbstractMatrix{Bool})
    @assert size(reference.select) == size(select) "round structure differs from the reference model"
    @assert reference.select == select && reference.washed == washed (
        "design does not match the reference model — a warm start is only " *
        "meaningful when select/washed are identical")
    return Model(deepcopy(reference.states), copy(reference.μ), copy(reference.ζ),
                 select, washed)
end

"""
    flat_model(select, washed)

A model with all mode energies constant, i.e. **no sequence-dependent
selection**. This is the honest null for predictive metrics: it retains the same
`lRs` information the real model is handed, so whatever a fitted model beats it
by is genuine sequence signal.

It matters because the naive alternative flatters the model badly — see
`metrics.jl`: on the published data a flat model and even an *untrained* one
reproduce the abundance correlation as well as the fitted model.
"""
function flat_model(select::AbstractMatrix{Bool}, washed::AbstractMatrix{Bool})
    n_modes, n_rounds = size(select)
    return Model(ntuple(_ -> ZeroEnergy(), n_modes),
                 zeros(n_modes, n_rounds), zeros(n_rounds), select, washed)
end

"""
    load_model(path)

Load a fitted model from a `.jld2`, under the key `"model"`.

Handles both formats: one written by this package (loaded directly) and the
legacy format written by the original codebase — see `import_legacy_model`.
"""
function load_model(path::AbstractString)
    isfile(path) || error("model file not found: $path")
    raw = JLD2.load(path, "model")
    raw isa Model && return raw
    return import_legacy_model(raw)
end

"""
    import_legacy_model(raw)

Rebuild a native `Model` from a legacy-serialised one.

A model saved by the original codebase carries that codebase's type tags. With
those types unavailable here, JLD2 hands back `ReconstructedMutable` /
`ReconstructedStatic` placeholders and **silently drops every zero-field
component**:

  * the `ZeroEnergy` `wash` state vanishes from the `states` tuple — 4 states
    come back where the model has 5, and `μ` still has 5 rows, which is how the
    loss is detected;
  * `Flux.flatten` vanishes from each `Chain`, leaving 2 `Dense` layers;
  * each `Dense` keeps only `weight` and `bias`; its activation survives only
    inside the type *name*, e.g. `Dense{#relu,...}`.

So the chains are rebuilt from the weights, the activation is read back off the
type name, `flatten` is restored ahead of the first `Dense`, and the trailing
`ZeroEnergy` is put back. The result is verified by the golden energy test in
`test/runtests.jl`, which compares it against the published per-mode energies.
"""
function import_legacy_model(raw)
    for f in (:states, :μ, :ζ, :select, :washed)
        hasproperty(raw, f) || error("not a recognisable serialised model: missing field `$f`")
    end

    n_modes = size(raw.μ, 1)
    deep = [_rebuild_chain(s) for s in raw.states]
    n_missing = n_modes - length(deep)
    n_missing in (0, 1) ||
        error("cannot reconstruct: μ has $n_modes rows but $(length(deep)) energy states were recovered")

    # The dropped trailing state is the zero-field `wash` mode.
    states = n_missing == 1 ? (deep..., ZeroEnergy()) : Tuple(deep)

    return Model(states, Array(raw.μ), Array(raw.ζ),
                 Array{Bool}(raw.select), Array{Bool}(raw.washed))
end

function _rebuild_chain(state)
    hasproperty(state, :m) || error("energy state has no field `m`")
    layers = state.m.layers
    dense = [l for l in layers if hasproperty(l, :weight)]
    isempty(dense) && error("no Dense layers recovered from the serialised chain")
    built = [Flux.Dense(Array(l.weight), Array(l.bias), _activation_of(l)) for l in dense]
    # `Flux.flatten` is a plain function and is dropped by reconstruction; the
    # A x L x S input must be flattened before the first Dense, so put it back.
    return DeepEnergy(Flux.Chain(Flux.flatten, built...))
end

"Read a reconstructed `Dense`'s activation out of its type name (`Dense{#relu,...}`)."
function _activation_of(layer)
    name = string(typeof(layer))
    occursin("relu", name) && return Flux.relu
    occursin("tanh", name) && return tanh
    occursin("sigmoid", name) && return Flux.sigmoid
    return identity
end

"Load the published reference model (see `paths.jl` for configuration)."
load_reference_model(; override = nothing) = load_model(data_path(:reference_model; override = override))

"""
    panel_energies(model, sequences::AbstractVector{<:AbstractString})

`E[s, w]` for each peptide string, columns in mode-index order
(`positive, negative, mutation, wt, wash`).
"""
function panel_energies(model::Model, sequences::AbstractVector{<:AbstractString})
    seqs = sample2hot([str2seq(s) for s in sequences])
    return Float64.(energies(seqs, model))
end
