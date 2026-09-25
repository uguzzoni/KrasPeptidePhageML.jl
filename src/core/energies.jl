###############################################################################
# Energy states.
#
# VENDORED VERBATIM from the original implementation (`energies.jl`, reachable subset only:
#   ConstEnergy struct + energies + ConstEnergy(::ZeroEnergy)  (:40-46, :60)
#   ZeroEnergy + energies + rrule                             (:52-59)
#   DeepEnergy + energies                                     (:66-70)
#
# Dropped: IndepSite, Epistasis, GlobBias, SimpleAR, AndEnergy — the 1-mut model
# uses only DeepEnergy (4 modes) + ZeroEnergy (the `wash` mode). Dropping them
# also removes the only users of tensordot/NumArray/logsoftmax/OneHot.
#
# `ConstEnergy` is kept because `energies(seq, ::ZeroEnergy)` DELEGATES to it.
# The ZeroEnergy rrule is load-bearing: without it Zygote tries to differentiate
# `repeat` over a 0-dimensional Bool array.
###############################################################################

#
# inespecific constant energy model
#
struct ConstEnergy{E<:AbstractArray{<:Real,0}}
    e::E
end
@functor ConstEnergy
ConstEnergy(x::Real) = ConstEnergy(fill(x))
ConstEnergy(::Type{T} = Float64) where {T} = ConstEnergy(fill(zero(T)))
energies(sequences::Sequences, state::ConstEnergy) = repeat(state.e, size(sequences, 3))


#
# inespecific zero energy model
#
struct ZeroEnergy{E<:AbstractArray{<:Real,0}} end
ZeroEnergy(T::Type{<:Real} = Bool) = ZeroEnergy{Array{T,0}}()
@functor ZeroEnergy
energies(sequences::Sequences, state::ZeroEnergy) = energies(sequences, ConstEnergy(state))
function ChainRulesCore.rrule(::typeof(energies), sequences::Sequences, state::ZeroEnergy)
    zero_energies_pullback(_) = (NoTangent(), NoTangent(), NoTangent())
    return energies(sequences, state), zero_energies_pullback
end
ConstEnergy(::ZeroEnergy{E}) where {E} = ConstEnergy{E}(fill(false))


#
# deep neural network energy model
#
struct DeepEnergy{T}
    m::T
end
energies(sequences::Sequences, state::DeepEnergy) = vec(state.m(sequences))
@functor DeepEnergy
