###############################################################################
# The selection model: per-mode energies, chemical potentials, and the
# likelihood of the observed read counts.
#
# VENDORED from the original implementation (`model.jl`, reachable subset:
#   tree/round semantics (:1-12), Model + trainable + @functor (:14-38),
#   energies(::Data,::Model) (:40), number_of_rounds/states (:47-49),
#   log_likelihood[_samples] (:91-131), log_selectivities (:133-153),
#   effective_energies (:164-166), potentials_select (:175-180),
#   log_abundances (:187-205), energies(::Sequences,::Model) (:211-219),
#   epochs2range (:279-280).
#
# `learn!` is NOT vendored from here — see train.jl. The original passes its
# regularizer as a zero-argument closure over the OUTER model, which contributes
# exactly zero gradient under Flux's explicit gradient mode; train.jl carries a
# corrected version taking `regf(m)`.
#
# Dropped: rare_binding_gauge[!], rare_binding_gauge_zeta[!] (:226-277).
###############################################################################

#=
A sample 't' refers to a phage population. It can either be an initial library or an
outcome of selections. Samples 't' are numbered from 1 to 'T', where 'T' is the number
of samples. They are arranged as nodes in a tree, with roots corresponding to initial
libraries.

On the tree, a round 'r' (of selection) corresponds to the unique edge connecting a sample to its ancestor.
Rounds are numbered from 1 to the number of edges in the tree. If there are 'T0' roots, then the round
connecting a non-root node 't' to its ancestor has index r = t - T0.

There are R = T - T0 rounds of selection.
=#

struct Model{St,M,Z,F}
    states::St
    μ::M # μ[w,r] chemical potential of state 'w' in round 'r'
    ζ::Z # exp(-ζ[r]) is the amplification factor at round 'r'
    select::F # select[w,r] is true if state 'w' is selected in round 'r'
    washed::F # washed[w,r] is true if state 'w' is washed out in round 'r'
    function Model(
        states,
        μ::AbstractMatrix,
        ζ::AbstractVector,
        select::AbstractMatrix,
        washed::AbstractMatrix,
    )
        @assert size(μ, 1) == number_of_states(states)
        @assert size(μ, 2) == length(ζ)
        @assert size(μ) == size(select) == size(washed)
        @assert all((select .== 0) .| (select .== 1))
        @assert all((washed .== 0) .| (washed .== 1))
        @assert iszero(select .& washed) # select and washed states are disjoint
        @assert all(any(select; dims = 1)) # at least one state selected in every round
        return new{typeof(states), typeof(μ), typeof(ζ), typeof(select)}(states, μ, ζ, select, washed)
    end
end
Flux.trainable(model::Model) = (states = model.states, μ = model.μ, ζ = model.ζ)
@functor Model #Functors.@functor

energies(data::Data, model::Model) = energies(data.sequences, model)

"""
    number_of_rounds(model)

Number of selection rounds modelled in `model`.
"""
number_of_rounds(model::Model) = length(model.ζ)
number_of_states(model::Model) = length(model.states)
number_of_states(states::Tuple) = length(states)

function log_likelihood_samples(
    lp::AbstractMatrix,
    ζ::AbstractVector,
    data::Data;
    batch::MiniBatch = MiniBatch(data), # current minibatch (optional)
)
    lN = log_abundances(lp, ζ, data; batch = batch)
    X = @. ifelse(iszero(batch.counts), zero(batch.counts * lN), batch.counts * lN)
    return vec(data.lMt) / number_of_sequences(data) .+ mean_(X; dims = 1)
end

"""
    log_likelihood_samples(model, data)

Log-likelihood of each experiment, computed as sum(R[s,t] * log(N[s,t]) over s),
for each sample 't'.
"""
function log_likelihood_samples(
    model::Model,
    data::Data;
    rare_binding::Bool = false,
    batch::MiniBatch = MiniBatch(data) # current minibatch (optional)
)
    lp = log_selectivities(model, batch; rare_binding = rare_binding)
    return log_likelihood_samples(lp, model.ζ, data; batch = batch)
end

function log_likelihood(model::Model, data::Data; kwargs...)
    ls = log_likelihood_samples(model, data; kwargs...)
    return sum(ls)
end

function log_likelihood(
    lp::AbstractMatrix,
    ζ::AbstractVector,
    data::Data;
    batch::MiniBatch = MiniBatch(data)
)
    ls = log_likelihood_samples(lp, ζ, data; batch = batch)
    return sum(ls)
end

function log_selectivities(model::Model, data::Union{Data, MiniBatch}; kwargs...)
    return log_selectivities(model, data.sequences; kwargs...)
end

function log_selectivities(
    model::Model,
    sequences::Sequences;
    rare_binding::Bool = false
)
    E = energies(sequences, model)
    μ_num = potentials_select(model.μ; select = model.select)
    if rare_binding
        μ_den = potentials_select(model.μ; select = model.washed)
    else
        μ_den = potentials_select(model.μ; select = model.washed .| model.select)
    end
    # Eeff[s,w,r]
    Eeff_num = effective_energies(E, μ_num)
    Eeff_den = effective_energies(E, μ_den)
    return logsumexp_(-Eeff_num; dims = 2) - logsumexp_(-Eeff_den; dims = 2) # lp[s,r]
end

"""
    effective_energies(E, μ)

Tensor of effective energies, Eeff[s,w,r] = E[s,w] - μ[w,r].
"""
function effective_energies(E::AbstractMatrix, μ::AbstractMatrix)
    return unsqueeze_right(E) .- unsqueeze_left(μ)
end

"""
    potentials_select(μ; select)

Assigns `-Inf` chemical potential to non-selected states. In other words, replaces
entries in `μ` with `Inf` if `select` is false for the corresponding state,
or 0 otherwise.
"""
function potentials_select(μ::AbstractArray; select::AbstractArray)
    # select is a keyword arg so it is not differentiated
    @assert size(μ) == size(select)
    mask = @ignore_derivatives select_mask((!).(select))
    return μ .- mask
end

"""
    log_abundances(model, data; rare_binding = false, batch = MiniBatch(data))

Estimates of abundances, log(N[s,t]), for sequences in 'batch'.
"""
function log_abundances(
    model::Model,
    data::Data;
    rare_binding::Bool = false,
    batch::MiniBatch = MiniBatch(data)
)
    lp = log_selectivities(model, batch; rare_binding = rare_binding) # lp[s,r]
    return log_abundances(lp, model.ζ, data; batch = batch)
end

function log_abundances(
    lp::AbstractMatrix,
    ζ::AbstractVector,
    data::Data;
    batch::MiniBatch = MiniBatch(data)
)
    lPZ = node_costs(lp .- reshape(ζ, 1, :), data.ancestors)
    return batch.lRs .+ lPZ .- tree_logsumexp(lPZ .+ data.lRt, data.ancestors; dim = 2)
end

# energies(sequences::Sequences, model::Model) = mapreduce(hcat, model.states) do state
#     energies(sequences, state)
# end

function energies(sequences::Sequences, model::Model)
    # Create an array to store the energy vectors
    energy_vectors = map(model.states) do state
        energies(sequences, state)
    end
    
    # Use the generic cat function which works on both CPU and GPU arrays
    return cat(energy_vectors..., dims=2)
end


epochs2range(epochs::UnitRange) = epochs
epochs2range(epochs::Integer) = 1:epochs
