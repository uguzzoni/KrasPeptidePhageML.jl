###############################################################################
# 01_train.jl — fit the lung model from scratch on the full published dataset.
#
#   julia --project=. scripts/01_train.jl [--seed 1] [--schedule default]
#                                         [--out results/model_seed1.jld2]
#
# `--schedule default` is 400 epochs and takes hours; `pilot` is two epochs and
# exists to check that the pipeline runs. The fitted model is written to
# `results/` and is NOT committed (see .gitignore) — the repository ships the
# published `data/model_nn_2l.jld2` as its reference model instead.
#
# This will not land on the published optimum. The original used a longer
# multi-phase schedule that was not recorded in full, and `learn!` here carries
# the corrected regulariser (see NOTES_provenance.md), so the trajectory differs
# from the start. What the run is for: an independent fit to compare against.
###############################################################################

using KrasPeptidePhageML
using JLD2, Statistics, Printf
include(joinpath(@__DIR__, "common.jl"))

seed     = argval(ARGS, "--seed", 1)
schedule = schedule_from_name(argval(ARGS, "--schedule", "default"; as = String))
holdout  = argval(ARGS, "--holdout", 0.0)        # fraction held out for the curve
outfile  = argval(ARGS, "--out",
                  joinpath(repo_root(), "results", @sprintf("model_seed%d.jld2", seed));
                  as = String)

@info "01_train" seed schedule holdout outfile

data, df_exp, seqs = load_lung_dataset()
select, washed = build_select_washed(df_exp)

# An optional held-out split, only so `train_model!` can record a test curve.
# Leave it at 0 for a production fit: the published reference was fitted on
# everything, and a model trained on a subset is not comparable to it.
data_train, data_test = if holdout > 0
    S = number_of_sequences(data)
    nfolds = max(2, round(Int, 1 / holdout))
    tr, te = cv_indices(S, nfolds, 1)
    @info "holdout" n_train=length(tr) n_test=length(te)
    select_sequences(data, tr), select_sequences(data, te)
else
    data, nothing
end

model = build_model(select, washed; seed = seed)
@printf("initial log-likelihood %.4f\n", log_likelihood(model, data_train))

t0 = time()
history, timings, curve = train_model!(model, data_train;
                                       schedule = schedule, eval_data = data_test)
total = time() - t0

# Per-read cross-entropy against the flat no-selection null, which retains the
# same `lRs` information the fitted model is handed — see metrics.jl for why the
# abundance correlation is NOT evidence of anything.
null = flat_model(select, washed)
ce = finite_mean(predictive_logloss(model, data_train))
ce_null = finite_mean(predictive_logloss(null, data_train))
enr = pooled_enrichment(model, data_train)

@printf("\ntrained in %.1f min\n", total / 60)
@printf("  log-likelihood        %.4f\n", log_likelihood(model, data_train))
@printf("  cross-entropy         %.4f nats/read  (flat null %.4f)\n", ce, ce_null)
@printf("  enrichment pearson    %.3f  (spearman %.3f, n = %d over %d samples)\n",
        enr.pearson, enr.spearman, enr.n, enr.nsamples)

# Energies of the 369 designed candidates, so the fit can be compared with the
# reference model on the object every published conclusion flows through.
cand = designed_candidates()
E = panel_energies(model, cand.sequence)
E_ref = panel_energies(load_reference_model(), cand.sequence)
tab = agreement_table(E_ref, [E], ["fit"])
println()
show(stdout, tab; allrows = true)
println("\n\nlabel agreement with the reference: ",
        @sprintf("%.1f %%", 100 * mean(published_labels(E) .== published_labels(E_ref))))

mkpath(dirname(outfile))
jldsave(outfile; model, history, timings, curve, df_exp,
        E_panel = Float32.(E), E_reference = Float32.(E_ref),
        sequence = cand.sequence, total_seconds = total,
        config = Dict("seed" => seed, "schedule" => schedule, "holdout" => holdout,
                      "S" => number_of_sequences(data_train)))
@printf("\nwritten: %s  (%.1f MB)\n", outfile, filesize(outfile) / 1024^2)
