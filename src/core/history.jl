###############################################################################
# Minimal training history.
#
# Replaces `ValueHistories.MVHistory`, which the original used only as the default
# `history` keyword of `learn!` and only ever through `push!(h, key, value)` and
# `get(h[key])`. Reimplementing it here drops the ValueHistories dependency.
###############################################################################

"""
    TrainHistory()

Append-only record of scalars per key, plus the iteration at which each was
recorded. Mirrors the small part of the `MVHistory` API that training used:

    push!(h, :loglikelihood, value)
    iters, values = get(h, :loglikelihood)
    h[:loglikelihood]                     # the values alone
"""
struct TrainHistory
    iters::Dict{Symbol,Vector{Int}}
    values::Dict{Symbol,Vector{Float64}}
    counter::Base.RefValue{Int}
end

TrainHistory() = TrainHistory(Dict{Symbol,Vector{Int}}(), Dict{Symbol,Vector{Float64}}(), Ref(0))

function Base.push!(h::TrainHistory, key::Symbol, value::Real)
    h.counter[] += 1
    push!(get!(h.iters, key, Int[]), h.counter[])
    push!(get!(h.values, key, Float64[]), Float64(value))
    return h
end

Base.getindex(h::TrainHistory, key::Symbol) = get(h.values, key, Float64[])
Base.haskey(h::TrainHistory, key::Symbol) = haskey(h.values, key)
Base.keys(h::TrainHistory) = keys(h.values)
Base.get(h::TrainHistory, key::Symbol) = (get(h.iters, key, Int[]), get(h.values, key, Float64[]))
Base.length(h::TrainHistory) = sum(length, values(h.values); init = 0)

function Base.show(io::IO, h::TrainHistory)
    print(io, "TrainHistory(")
    print(io, join(("$k: $(length(v)) pts" for (k, v) in h.values), ", "))
    print(io, ")")
end
