###############################################################################
# Tree algebra over the sample tree (`ancestors`).
#
# VENDORED VERBATIM from the original implementation (`ancestors.jl`, reachable subset:
#   semantics (:1-12), valid_ancestors (:14-19), number_of_* (:21-23),
#   node_costs + rrule (:43-88), subtree_sum (:109-123), tree_sum + rrule
#   (:125-152), tree_logsumexp + rrule (:154-176), subtree_maximum (:178-191),
#   tree_maximum (:193-206).
#
# Dropped: isroot/isleaf (:24-25), find_root (:27-41), edge_costs (:95-107).
#
# The three rrules are LOAD-BEARING: node_costs and tree_logsumexp mutate or
# reduce in ways Zygote cannot differentiate on its own. Do not "simplify" them.
###############################################################################

#=
We can represent a rooted tree as a list of ancestors.
So that ancestors[i] gives the ancestor of node 'i'.
If 'i' is a root node, ancestors[i] == 0.
We assume that 'ancestors' always visits parents before children.

Edges are identified by the index of the descendant node.

We can also represent forests of rooted trees.
=#

# TODO: make 'ancestors' a keyword arg, since it is never differentiated

function valid_ancestors(ancestors)
    for (i, a) in enumerate(ancestors)
        0 ≤ a < i || return false # children are traversed before parents?
    end
    return true
end

number_of_nodes(ancestors) = length(ancestors)
number_of_roots(ancestors) = count(iszero, ancestors)
number_of_edges(ancestors) = number_of_nodes(ancestors) - number_of_roots(ancestors)

"""
    node_costs(X, ancestors)

Given edge costs `X`, returns costs associated to traversals from roots to nodes.
"""
function node_costs(X::AbstractArray, ancestors; dim::Int = ndims(X))
    @assert size(X, dim) == number_of_edges(ancestors)
    Y_sz = ntuple(d -> d == dim ? number_of_nodes(ancestors) : size(X, d), Val(ndims(X)))
    Y = fill!(similar(X, typeof(sum(X)), Y_sz), 0)
    edge_index = 0
    for (i, a) in enumerate(ancestors)
        if a > 0
            edge_cost = selectdim(X, dim, edge_index += 1)
            selectdim(Y, dim, i) .= selectdim(Y, dim, a) .+ edge_cost
        end
    end
    return Y
end

# since this involves mutation and loops, we define the rrule manually
function ChainRulesCore.rrule(
    ::typeof(node_costs),
    X::AbstractArray,
    ancestors;
    dim::Int = ndims(X)
)
    Y = node_costs(X, ancestors; dim = dim)
    function node_costs_pullback(dY)
        @assert size(dY) == size(Y)
        dX = zero(X)
        n_roots = cumsum(iszero.(ancestors))
        @assert issorted([i - n_roots[i] for i in eachindex(ancestors)])
        for (i, a) in Iterators.reverse(enumerate(ancestors))
            if a > 0
                i_ = i - n_roots[i]
                a_ = a - n_roots[a]
                selectdim(dX, dim, i_) .+= selectdim(dY, dim, i)
                if ancestors[a] > 0
                    selectdim(dX, dim, a_) .+= selectdim(dX, dim, i_)
                end
            end
        end
        return NoTangent(), dX, NoTangent()
    end
    return Y, node_costs_pullback
end

"""
    subtree_sum(X, ancestors)

Given node costs `X`, computes the total cost of the subtree subyacent at each node.
"""
function subtree_sum(X::AbstractArray, ancestors; dim::Int = ndims(X))
    @assert size(X, dim) == number_of_nodes(ancestors)
    Y = copy(X)
    for (i, a) in Iterators.reverse(enumerate(ancestors))
        if a > 0
            selectdim(Y, dim, a) .+= selectdim(Y, dim, i)
        end
    end
    return Y
end

"""
    tree_sum(X, ancestors)

Given node costs `X`, assigs to each node the total cost of the tree that contains it.
"""
function tree_sum(X::AbstractArray, ancestors; dim::Int = ndims(X))
    Y = subtree_sum(X, ancestors; dim = dim)
    for (i, a) in enumerate(ancestors)
        if a > 0
            selectdim(Y, dim, i) .= selectdim(Y, dim, a)
        end
    end
    return Y
end

function ChainRulesCore.rrule(
    ::typeof(tree_sum),
    X::AbstractArray,
    ancestors;
    dim::Int = ndims(X)
)
    Y = tree_sum(X, ancestors; dim = dim)
    function tree_sum_pullback(dY)
        dX = tree_sum(dY, ancestors; dim = dim)
        return NoTangent(), dX, NoTangent()
    end
    return Y, tree_sum_pullback
end

"""
    tree_logsumexp(X, ancestors; dim = ndims(X))

Assigns the value of log(sum(exp.(X) for each tree)) to each node.
"""
function tree_logsumexp(X::AbstractArray, ancestors; dim::Int = ndims(X))
    X_max = tree_maximum(X, ancestors; dim = dim)
    return log.(tree_sum(exp.(X .- X_max), ancestors; dim = dim)) .+ X_max
end

function ChainRulesCore.rrule(
    ::typeof(tree_logsumexp),
    X::AbstractArray,
    ancestors;
    dim::Int = ndims(X)
)
    Y = tree_logsumexp(X, ancestors; dim = dim)
    function tree_logsumexp_pullback(dY)
        dX = tree_sum(dY, ancestors; dim = dim) .* exp.(X .- Y)
        return NoTangent(), dX, NoTangent()
    end
    return Y, tree_logsumexp_pullback
end

"""
    subtree_maximum(X, ancestors; dim = ndims(X))

Assigns to each node the maximum cost of the nodes at the subyacent subtree.
"""
function subtree_maximum(X::AbstractArray, ancestors; dim::Int = ndims(X))
    Y = copy(X)
    for (i, a) in Iterators.reverse(enumerate(ancestors))
        if a > 0
            selectdim(Y, dim, a) .= max.(selectdim(Y, dim, i), selectdim(Y, dim, a))
        end
    end
    return Y
end

"""
    tree_maximum(X, ancestors; dim = ndims(X))

Assigns to each node the maximum cost of all nodes at the tree that contains it.
"""
function tree_maximum(X::AbstractArray, ancestors; dim::Int = ndims(X))
    Y = subtree_maximum(X, ancestors; dim = dim)
    for (i, a) in enumerate(ancestors)
        if a > 0
            selectdim(Y, dim, i) .= selectdim(Y, dim, a)
        end
    end
    return Y
end
