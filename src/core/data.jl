###############################################################################
# Data container: one-hot sequences + read counts over a tree of samples.
#
# VENDORED VERBATIM from the original implementation (`data.jl`, reachable subset:
#   Sequences alias (:1), Data + both ctors + @functor (:3-33),
#   number_of_samples/rounds/sequences (:36-39),
#   MiniBatch (:43-64), Data(::MiniBatch, anc) + select_sequences (:66-67),
#   minibatches/minibatch_count (:85-132), normalize_counts (:156-159).
#
# Dropped: number_of_libraries, alphabet_size, sequence_length, data_split,
# selectivities(::Data), log_selectivities(::Data) — none are reachable from the
# training / evaluation / MC API.
#
# `lRs` (log total counts of a sequence across the tree) is what makes the raw
# abundance correlation a poor model metric: it is handed to the model as input.
###############################################################################

const Sequences{V<:Real} = AbstractArray{V,3}

struct Data{Seq, Cnt, Anc, lCn}
    # sequences[a,i,s] == 1 if letter 'i' of sequence 's' is an 'a' (one-hot)
    sequences::Seq
    # counts[s,t] = reads of sequence 's' in sample 't'
    counts::Cnt
    # ancestors[t] = ancestor of sample 't'; for roots ancestors[t] == 0
    ancestors::Anc
    # summary Statistics
    lRt::lCn # total counts in each sample
    lRs::lCn # total counts of a sequence in each tree
    lMt::lCn # log of multinomial factors
    function Data(sequences::Sequences, counts::AbstractMatrix, ancestors)
        lRt = log.(sum(counts; dims = 1))
        lRs = log.(tree_sum(counts, ancestors; dim = 2))
        lMt = log_multinomial(counts; dims = 1)
        @assert size(counts, 1) == size(sequences, 3) # number of sequences
        @assert size(counts, 2) == number_of_nodes(ancestors)
        @assert all(sum(sequences; dims = 1) .== 1) # valid one-hot representation
        @assert all(0 .≤ counts .< Inf)
        @assert valid_ancestors(ancestors)
        return new{typeof(sequences), typeof(counts), typeof(ancestors), typeof(lMt)}(
            sequences, counts, ancestors, lRt, lRs, lMt
        )
    end
    function Data(sequences::Sequences, counts::AbstractMatrix, ancestors, lRt, lRs, lMt)
        return new{typeof(sequences), typeof(counts), typeof(ancestors), typeof(lRt)}(
            sequences, counts, ancestors, lRt, lRs, lMt
        )
    end
end
@functor Data #Functors.@functor


number_of_samples(data::Data) = number_of_nodes(data.ancestors)
number_of_libraries(data::Data) = number_of_roots(data.ancestors)
number_of_rounds(data::Data) = number_of_edges(data.ancestors)
number_of_sequences(data::Data) = size(data.sequences, 3)
alphabet_size(data::Data) = size(data.sequences, 1)
sequence_length(data::Data) = size(data.sequences, 2)

struct MiniBatch{Seq, Cnt, lCnt}
    sequences::Seq
    counts::Cnt
    lRs::lCnt
    """
        MiniBatch(data, select = :)

    Returns a subset of `data` containing only the selection `select`
    of sequences (either a list of indices or a bool vector of flags).
    """
    function MiniBatch(data::Data, select = :)
        sequences = data.sequences[:,:,select]
        counts = data.counts[select,:]
        lRs = data.lRs[select,:]
        @assert size(counts, 1) == size(sequences, 3) # number of sequences
        @assert size(counts, 2) == size(data.counts, 2) # number of samples
        return new{typeof(sequences), typeof(counts), typeof(lRs)}(sequences, counts, lRs)
    end
end

number_of_sequences(batch::MiniBatch) = size(batch.counts, 1)
number_of_samples(batch::MiniBatch) = size(batch.counts, 2)

Data(batch::MiniBatch, ancestors) = Data(batch.sequences, batch.counts, ancestors)
select_sequences(data::Data, select) = Data(MiniBatch(data, select), data.ancestors)

function minibatches(data::Data, n::Int; fullonly=false)
    S = number_of_sequences(data)
    p = randperm(S)
    slices = minibatches(S, n; fullonly = fullonly)
    batches = [MiniBatch(data, p[b]) for b in slices]
    return batches
end

"""
    minibatch_count(data, n; fullonly=false)

Number of minibatches.
"""
function minibatch_count(data::Data, n::Int; fullonly=false)
    S = size(data.sequences, 3)
    return minibatch_count(S, n; fullonly=fullonly)
end

"""
    minibatches(samples, n, fullonly=false)

Partition `samples` into minibatches of length `n`.
If `fullonly` is `true`, all slices are of length `n`, otherwise
the last slice can be smaller.
"""
function minibatches(samples::Int, n::Int; fullonly=false)
    @assert samples ≥ 0 && n ≥ 0
    if fullonly
        endidx = fld(samples, n) * n
    else
        endidx = min(cld(samples, n), samples) * n
    end
    return [b:min(b + n - 1, samples) for b = 1:n:endidx]
end

"""
    minibatch_count(samples, n; fullonly=false)

Number of minibatches.
"""
function minibatch_count(samples::Int, n::Int; fullonly=false)
    if fullonly
        endidx = fld(samples, n) * n
    else
        endidx = min(cld(samples, n), samples) * n
    end
    return length(1:n:endidx)
end

"""
    normalize_counts(data)

Normalizes counts to sum 1 in every round.
"""
function normalize_counts(data::Data)
    return Data(data.sequences, normalize_counts(data.counts), data.ancestors)
end
normalize_counts(counts::AbstractArray; dims = 1) = counts ./ sum(counts; dims = dims)
