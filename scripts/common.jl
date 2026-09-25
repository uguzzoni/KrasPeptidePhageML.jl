###############################################################################
# Argument parsing shared by the scripts. Included, not a module — these are
# command-line conveniences, not package API.
###############################################################################

using Printf

"""
    argval(args, flag, default; as = typeof(default))

`--flag value` from `args`, parsed as `as`, or `default` when absent.
"""
function argval(args, flag::AbstractString, default; as = typeof(default))
    i = findfirst(==(flag), args)
    i === nothing && return default
    i < length(args) || error("$flag needs a value")
    v = args[i + 1]
    return as === String ? v : parse(as, v)
end

"`--flag` present?"
argflag(args, flag::AbstractString) = findfirst(==(flag), args) !== nothing

"`\"default\"`/`\"warm\"`/`\"pilot\"` -> the matching schedule constant."
function schedule_from_name(name::AbstractString)
    name == "default" && return DEFAULT_SCHEDULE
    name == "warm" && return WARM_SCHEDULE
    name == "pilot" && return PILOT_SCHEDULE
    error("unknown schedule $name (default | warm | pilot)")
end

finite_mean(v) = (f = filter(isfinite, collect(v)); isempty(f) ? NaN : mean(f))
