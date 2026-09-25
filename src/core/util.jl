###############################################################################
# Numeric helpers.
#
# VENDORED VERBATIM from the original implementation (`util.jl`), reachable subset only:
#   logsumexp_ (:2), mean_ (:4), select_mask (:94),
#   unsqueeze_left (:96), unsqueeze_right (:97), log_multinomial (:104-106).
#
# Everything else in that file is unreachable from the model / training / MC API
# (it served IndepSite, Epistasis, SimpleAR, the RBM states and simulate).
#
# Note these reduce a dimension and DROP it — call sites depend on that shape.
###############################################################################

# these functions reduce a dimension and drop it
logsumexp_(A::AbstractArray; dims = :) = dropdims(logsumexp(A; dims = dims); dims = dims)
mean_(A::AbstractArray; dims = :) = dropdims(mean(A; dims = dims); dims = dims)

"""
Given an array `select`, returns another array with `Inf` where `select` is > 0,
and zeros elsewhere.
"""
select_mask(select::AbstractArray) = (select .> 0) .* Inf

unsqueeze_left(A::AbstractArray) = reshape(A, 1, size(A)...)
unsqueeze_right(A::AbstractArray) = reshape(A, size(A)..., 1)

"""
    log_multinomial(N; dims = :)

Log of multinomial coefficients, reduced across the given dimension of `N`.
"""
function log_multinomial(N::AbstractArray; dims = :)
    return loggamma.(sum(N; dims) .+ 1) .- sum(loggamma.(N .+ 1); dims)
end
