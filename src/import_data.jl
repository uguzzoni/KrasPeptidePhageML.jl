###############################################################################
# Data import: CSV -> (Data, df_exp, sequences).
#
#
#   data/lung_counts.csv    sequence + one integer column per tree node.
#                           Column 1 after `sequence` is the root (the initial
#                           library); the remaining 32 are the rounds, in the
#                           row order of the sample table.
#   data/lung_samples.csv   one row per round, the `df_exp` table.
#
# The two join on `counts_column`, the 1-based index of that round's column in
# the counts matrix (= `column` + 1, the root occupying column 1). The 8 latent
# round-2 nodes are present in both tables: their count columns are all-zero and
# they carry `latent = true`.
#
###############################################################################

# ---------------------------------------------------------------------------
# Amino-acid alphabet and sequence encodings.
# ---------------------------------------------------------------------------

const AA2INT = Dict('A' => 1, 'C' => 2, 'D' => 3, 'E' => 4, 'F' => 5,
                    'G' => 6, 'H' => 7, 'I' => 8, 'K' => 9, 'L' => 10,
                    'M' => 11, 'N' => 12, 'P' => 13, 'Q' => 14, 'R' => 15,
                    'S' => 16, 'T' => 17, 'V' => 18, 'W' => 19, 'Y' => 20,
                    '-' => 21)

const INT2AA = Dict(k => aa for (aa, k) in AA2INT)

const AAs = "ACDEFGHIKLMNPQRSTVWY"

const ALPHABET_SIZE = 20
const PEPTIDE_LENGTH = 7

"Integer-code an amino-acid string; anything outside the alphabet becomes `q_other`."
function str2seq(str::AbstractString; dic2int::Dict = AA2INT, q_other::Int = 21)
    return Tuple(get(dic2int, a, q_other) for a in str)
end

"Inverse of [`str2seq`](@ref)."
seq2str(seq; dic2str::Dict = INT2AA) = join(dic2str[a] for a in seq)

"""
    sample2hot(sample; A=20)

One-hot encode a vector of integer-coded sequences into an `A x L x M` array.
"""
function sample2hot(sample; A::Int = ALPHABET_SIZE)
    M = length(sample)
    L = length(sample[1])
    x = zeros(Int8, A, L, M)
    for (k, s) in enumerate(sample), (j, a) in enumerate(s)
        x[a, j, k] = 1
    end
    return x
end

"""
    onehot2aa(seq)

Decode a one-hot array back to strings. Gaps (all-zero columns) decode to `'A'`
— an upstream quirk kept because the model never emits them.
"""
function onehot2aa(seq::AbstractArray{<:Real})
    A = reduce(string, [AAs[Tuple(a)[1]] for a in argmax(seq; dims = 1)]; dims = 2, init = "")
    return ndims(seq) == 2 ? only(A) : reshape(A, size(seq)[3:end])
end

aa2onehot(str) = sample2hot([str2seq(s) for s in str])

# ---------------------------------------------------------------------------
# The two CSV tables.
# ---------------------------------------------------------------------------

"Column types of `lung_samples.csv`; anything not listed is read as `String`."
const SAMPLE_COLUMN_TYPES = Dict(
    "column" => Int, "counts_column" => Int, "replica" => Int, "round" => Int,
    "output" => Bool, "latent" => Bool, "n_sample" => Int,
    "original_column" => Int, "original_position" => Int,
    "ancestor_column" => Int, "ancestor_counts_column" => Int,
    "total_reads" => Int, "n_sequences_observed" => Int,
)

_parse_field(::Type{String}, s::AbstractString) = String(s)
_parse_field(::Type{Bool}, s::AbstractString) = (s == "true")
_parse_field(::Type{T}, s::AbstractString) where {T<:Number} = parse(T, s)

"""
    read_samples_csv(file) -> DataFrame

Read the sample table. One row per round, ordered by `column` (1..n_rounds);
`counts_column` indexes the counts matrix, where column 1 is the root.
"""
function read_samples_csv(file::AbstractString)
    isfile(file) || error("samples file not found: $file")
    raw, header = readdlm(file, ',', String, '\n'; header = true)
    colnames = strip.(vec(header))
    df = DataFrame()
    for (j, name) in enumerate(colnames)
        T = get(SAMPLE_COLUMN_TYPES, name, String)
        df[!, name] = [_parse_field(T, raw[i, j]) for i in axes(raw, 1)]
    end
    @assert df.column == 1:nrow(df) "sample table must be ordered by `column`, 1..n"
    @assert df.counts_column == df.column .+ 1 "`counts_column` must be `column + 1`"
    return df
end

"""
    read_counts_csv(file) -> (sequences, counts, colnames)

Read the counts table: `sequences` is the peptide of each row as a string,
`counts` an `n_sequences x n_nodes` `Matrix{Float64}` whose first column is the
root library, and `colnames` the name of each of those columns (`"root"` first).
"""
function read_counts_csv(file::AbstractString)
    isfile(file) || error("counts file not found: $file")
    raw, header = readdlm(file, ',', String, '\n'; header = true)
    sequences = String.(vec(raw[:, 1]))
    counts = parse.(Float64, raw[:, 2:end])
    colnames = strip.(vec(header))[2:end]          # drop the `sequence` column
    @assert length(unique(length.(sequences))) == 1 "peptides have inhomogeneous length"
    @assert first(colnames) == "root" "first count column must be the root library"
    return sequences, counts, colnames
end

"""
    load_lung_dataset(; counts_csv=nothing, samples_csv=nothing, verbose=true)

Rebuild `(data, df_exp, sequences)` from the two CSVs.

  * `data`      — `Data` holding one-hot sequences, the counts matrix (root in
                  column 1) and the ancestor tree;
  * `df_exp`    — the sample table, one row per round;
  * `sequences` — the peptides as strings, in the row order of `data`.

Paths default to `config/paths.toml` / the environment — see `paths.jl`.
"""
function load_lung_dataset(; counts_csv = nothing, samples_csv = nothing, verbose::Bool = true)
    file_counts  = data_path(:counts_csv;  override = counts_csv)
    file_samples = data_path(:samples_csv; override = samples_csv)

    df_exp = read_samples_csv(file_samples)
    sequences, counts, colnames = read_counts_csv(file_counts)

    @assert size(counts, 2) == nrow(df_exp) + 1 "counts has $(size(counts,2)) columns for $(nrow(df_exp)) rounds + root"
    @assert colnames[2:end] == df_exp.column_name "counts columns and sample rows disagree"

    # `ancestor_counts_column` is already an index into the counts matrix; the
    # root has no ancestor.
    ancestors = tuple(0, df_exp.ancestor_counts_column...)

    data = Data(sample2hot([str2seq(s) for s in sequences]), counts, ancestors)

    verbose && @info "load_lung_dataset" S=number_of_sequences(data) n_samples=number_of_samples(data) n_rounds=number_of_rounds(data) n_latent=count(df_exp.latent)
    return data, df_exp, sequences
end

# ---------------------------------------------------------------------------
# Experimental design: which modes are selected / washed in each round.
# ---------------------------------------------------------------------------

"Mode -> state index of the model."
const MODES = Dict(:positive => 1, :negative => 2, :mutation => 3, :wt => 4, :wash => 5)
const MODE_NAMES = [:positive, :negative, :mutation, :wt, :wash]

"""
    build_select_washed(df_exp)

  * `negative` — selected on the latent round-2 nodes (the washed-away population)
  * `positive` — selected on every sequenced round
  * `wt`       — `((mutation in {WT,EV}) & output) | (round != 2 & !output)`
  * `mutation` — `(mutation not in {WT,EV}) & output`
  * washing    — **only `wash`**, in every round
"""
function build_select_washed(df_exp::DataFrame)
    n_rounds = nrow(df_exp)
    select = zeros(Bool, length(MODES), n_rounds)
    washed = zeros(Bool, length(MODES), n_rounds)

    is_wt = (df_exp.mutation .== "WT") .| (df_exp.mutation .== "EV")

    select[MODES[:negative], df_exp[df_exp.round .== 2, :column]] .= true
    select[MODES[:positive], df_exp[df_exp.round .!= 2, :column]] .= true
    select[MODES[:wt], df_exp[(is_wt .& df_exp.output) .|
                              ((df_exp.round .!= 2) .& .!df_exp.output), :column]] .= true
    select[MODES[:mutation], df_exp[(.!is_wt) .& df_exp.output, :column]] .= true
    washed[MODES[:wash], :] .= true

    @assert iszero(select .& washed) "select and washed must be disjoint"
    @assert all(any(select; dims = 1)) "every round needs at least one selected state"
    return select, washed
end
