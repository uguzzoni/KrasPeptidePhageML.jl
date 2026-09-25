"""
    KrasPeptidePhageML

An implementation of the phage-display selection model used for
KRAS-mutant-specific peptide discovery as described in Uguzzoni et al. (2026): 
import counts from CSV, fit the model, generate candidate peptides, and evaluate stability.


Quick start:

```julia
using KrasPeptidePhageML
check_data_paths()                                    # is the data configured?
data, df_exp, seqs = load_lung_dataset()
select, washed = build_select_washed(df_exp)
model = build_model(select, washed; seed = 1)
train_model!(model, data; schedule = PILOT_SCHEDULE)
```
"""
module KrasPeptidePhageML

using Random, Statistics, Printf, TOML
using StatsBase
using StatsBase: corspearman
using DataFrames
import DelimitedFiles: readdlm
using JLD2
using SpecialFunctions: loggamma
using LogExpFunctions: logsumexp
using ChainRulesCore: rrule, NoTangent, @ignore_derivatives
using Functors: @functor
import ChainRulesCore
import Flux
using CairoMakie

# ---- configuration ---------------------------------------------------------
export data_path, check_data_paths, repo_root, bundled

# ---- model core (vendored) --------------------------------------------------
export Data, MiniBatch, Model, DeepEnergy, ZeroEnergy, ConstEnergy
export energies, log_selectivities, log_abundances, log_likelihood, log_likelihood_samples
export select_sequences, normalize_counts, minibatches, minibatch_count
export number_of_sequences, number_of_samples, number_of_rounds, number_of_states
export TrainHistory

# ---- Monte-Carlo sequence design (vendored) ---------------------------------
export Δenergies, flip, flip!, MCParetoTrial!, MCPareto!, ToParetoFront!
export ParetoAnneal!, isPareto          # were unexported upstream

# ---- data import ------------------------------------------------------------
export AA2INT, INT2AA, AAs, str2seq, seq2str, sample2hot, onehot2aa, aa2onehot
export read_counts_csv, read_samples_csv
export load_lung_dataset, build_select_washed, MODES, MODE_NAMES

# ---- model construction, training, generation ------------------------------
export mode_nn, build_model, build_warm_model, flat_model
export load_model, load_reference_model, panel_energies
export l2_nn, learn!, train_model!
export DEFAULT_SCHEDULE, WARM_SCHEDULE, PILOT_SCHEDULE
export generate_candidates, GenerationSpec, default_specs, energy_thresholds
export run_chains, select_candidates

# ---- evaluation and figures ------------------------------------------------
export pooled_enrichment, pooled_abundance, predictive_logloss, resample_counts
export published_labels, PUBLISHED_THRESHOLDS, label_agreement, agreement_table, topn_overlap
export designed_candidates, cv_indices
export figure_combined, load_perturbations

include("paths.jl")

# Model core. Order matters only for type aliases used in signatures:
# `Sequences` is defined in core/data.jl and used by core/energies.jl.
include("core/data.jl")
include("core/energies.jl")
include("core/model.jl")
include("core/util.jl")
include("core/ancestors.jl")
include("core/history.jl")
include("core/mc.jl")

# Data import.
include("import_data.jl")

# Modelling workflow.
include("build.jl")
include("train.jl")
#include("generate.jl")
include("metrics.jl")
include("labels.jl")
include("figures.jl")

end # module
