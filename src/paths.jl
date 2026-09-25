###############################################################################
# Where the data lives.
#
# The dataset is two CSV tables in `data/` (`lung_counts.csv` ~9 MB,
# `lung_samples.csv` a few KB) plus the fitted reference model. Paths come from
# `config/paths.toml`, which is git-ignored; `config/paths.toml.example` is
# committed as a template.
#
# Design note: resolution happens INSIDE these functions, never at module load.
# the original did the opposite — it read ENV["DATASET_KRAS"] at load time and
# threw a bare `ErrorException("")` if unset, which made importing it
# fail outright, and additionally required several unrelated count files to exist
# before a lung run could start. This package always loads; it only complains
# when you actually ask for a file, and then it names the file.
#
# Precedence:  explicit argument  >  ENV  >  config/paths.toml
###############################################################################

const _PATH_ENV = Dict(
    :counts_csv      => "KRAS_COUNTS_CSV",
    :samples_csv     => "KRAS_SAMPLES_CSV",
    :reference_model => "KRAS_REFERENCE_MODEL",
)

"Defaults for files that ship in `data/`."
const _PATH_BUNDLED = Dict(
    :counts_csv      => "lung_counts.csv",
    :samples_csv     => "lung_samples.csv",
    :reference_model => "model_nn_2l.jld2",
)

repo_root() = normpath(joinpath(@__DIR__, ".."))
config_file() = joinpath(repo_root(), "config", "paths.toml")

"Data files shipped with the repo (small enough to version)."
bundled(parts...) = joinpath(repo_root(), "data", parts...)

_config() = isfile(config_file()) ? TOML.parsefile(config_file()) : Dict{String,Any}()

"""
    data_path(key; override=nothing)

Resolve one data path. `key` is one of `:counts_csv`, `:samples_csv`,
`:reference_model`.

Raises a message that names the key, the config file and the environment
variable when it cannot resolve — rather than failing at import time.
"""
function data_path(key::Symbol; override = nothing)
    override === nothing || return String(override)

    envvar = get(_PATH_ENV, key, nothing)
    if envvar !== nothing && haskey(ENV, envvar)
        return ENV[envvar]
    end

    cfg = _config()
    if haskey(cfg, String(key))
        p = cfg[String(key)]
        return isabspath(p) ? p : normpath(joinpath(repo_root(), p))
    end

    # everything ships in `data/`, so fall back to the bundled copy
    if haskey(_PATH_BUNDLED, key)
        p = bundled(_PATH_BUNDLED[key])
        isfile(p) && return p
    end

    error("""
          could not resolve data path for `:$key`.
          Set it in one of (highest precedence first):
            1. the `override` argument,
            2. the environment variable $(something(envvar, "—")),
            3. `$(config_file())` as   $(key) = "/abs/path/to/file"
          Copy `config/paths.toml.example` to `config/paths.toml` to get started;
          `data/README.md` explains which file each key expects.
          """)
end

"""
    check_data_paths(; keys=...)

Report which configured paths exist. Useful as the first cell of a notebook.
"""
function check_data_paths(; keys = (:counts_csv, :samples_csv, :reference_model))
    rows = NamedTuple[]
    for k in keys
        p, ok, err = try
            q = data_path(k); (q, isfile(q), "")
        catch e
            ("", false, sprint(showerror, e) |> x -> first(split(x, '\n')))
        end
        push!(rows, (key = k, path = p, exists = ok, problem = err))
        @printf("  %-17s %s  %s\n", k, ok ? "OK  " : "MISSING", isempty(p) ? err : p)
    end
    return rows
end
