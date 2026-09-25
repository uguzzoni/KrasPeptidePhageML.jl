# Workplan — `kras-peptide-phage-ml`, a standalone repository

## Context

The KRAS phage-display work currently spans an analysis project plus two personal packages
(a model library and an import library) whose loaded copies differ from their git HEADs, and whose
notebooks reference paths that no longer exist. Reproducing anything requires the whole
constellation. Worse, the import library **fails to load at all** unless `ENV["DATASET_KRAS"]` is set
*and* several unrelated count files are present — even for a lung-only run.

Goal: a self-contained repository at **`~/Projects/julia_packages/kras-peptide-phage-ml`** that reads
the dataset from CSV, trains the 1-mut model, generates peptide candidates, and runs test notebooks
producing real results (including `figure_combined`) — depending on **no personal packages**. Only
the reachable parts of the *model* library get vendored; the import library is replaced outright by
two CSV tables (§B), so the load-time failure above is moot.

Decisions taken with the user: **CairoMakie** for figures (pure Julia); **the dataset ships inside
the repo** as two CSVs (9.4 MB total), still behind configurable paths but with bundled defaults, so
a checkout is reproducible with nothing to fetch; notebooks = **fast smoke test + figure rebuilt
from compact committed artifacts**; generation reimplemented **clean and seeded**, with the
original's bugs documented rather than reproduced.

---

## What gets vendored — verified dependency closures

### A. Model core — from the model library (~600 of 1688 lines)

Vendor from the copy that is actually loaded at runtime (verified byte-identical to the
checked-in source tree). Only these files are reachable:

| file | keep | drop |
|---|---|---|
| `data.jl` | 1–67, 85–132, 156–159 | `data_split`, `selectivities`, `alphabet_size` (optional) |
| `energies.jl` | `DeepEnergy` 66–70, `ZeroEnergy` 52–59, `ConstEnergy` 40–42/46/60 | `IndepSite`, `Epistasis`, `GlobBias`, `SimpleAR`, `AndEnergy` |
| `model.jl` | 14–40, 47–49, 56–153, 164–219, 279–280 | `rare_binding_gauge*` (optional) |
| `util.jl` | `logsumexp_`, `mean_`, `select_mask`, `unsqueeze_left/right`, `log_multinomial` | everything else |
| `ancestors.jl` | `valid_ancestors`, `number_of_{nodes,roots,edges}`, `node_costs`+rrule, `subtree_sum`, `tree_sum`+rrule, `tree_logsumexp`+rrule, `subtree_maximum`, `tree_maximum` | `isroot/isleaf`, `find_root`, `edge_costs` |
| the MC module | all 131 lines | — |

**Entirely droppable** (no reachable helper lives in them): `rbm.jl`, `indep_model.jl`,
`optimize_depletion.jl`, `simulate.jl`, `node.jl`, `analysis.jl`, `learn_mu.jl`, and `io.jl` +
`path_selectivity.jl` (which the module never even `include`s — `path_selectivity.jl` would not
compile).

**Must not be lost** (silent breakage otherwise):
- `@functor Model`, `@functor DeepEnergy`, and `Flux.trainable(::Model) = (states, μ, ζ)` — without
  these the optimiser never updates the networks, or tries to update `select`/`washed`.
- Three custom `rrule`s, all load-bearing for AD: `energies(::Sequences, ::ZeroEnergy)`
  (`energies.jl:56`), `node_costs` (`ancestors.jl:63`), `tree_logsumexp` (`ancestors.jl:164`).
- `ConstEnergy` is needed *indirectly* — `energies(seq, ::ZeroEnergy)` delegates to it. Either keep
  it or inline `repeat(fill(false), size(seq,3))` and keep the rrule.

**Rename gotcha:** the MC module hard-codes module-qualified type names in three signatures
(lines 56, 85, 104) and in one call (line 123). These must become bare names in the new module.

**Dependencies this removes:** `Optim`, `OneHot`, `AbstractTrees`, `Distributions`,
`LinearAlgebra`, `SparseArrays`, `StatsFuns` all become unnecessary.

### B. Data import — two CSV tables, nothing vendored

**Superseded, 2026-09-23.** The import library is not vendored at all any more. The dataset is no
longer rebuilt from the raw experiment: it ships as two aligned CSVs, exported from the original
project by `notebooks/answer_reviewers/src/export_published_csv.jl`, which dumps
`load_published_data()` — i.e. the exact training set of `model_nn_2l.jld2`.

| file | contents |
|---|---|
| `data/lung_counts.csv` | `sequence` + one integer column per tree node: `root` (the initial library) then the 32 rounds, in the row order of the sample table. 123 777 rows, 9.4 MB. |
| `data/lung_samples.csv` | the `df_exp` table, one row per round (32 rows, 3.4 KB). |

The two join on `counts_column` = `column` + 1, the root occupying counts column 1. The 8 latent
round-2 nodes appear in both tables: their count columns are all-zero and they carry `latent = true`.
Ancestors are read straight out of `ancestor_counts_column`, so no tree has to be reconstructed.

Everything the old import did to *produce* that dataset — sample-name parsing, merging the initial
library in, the stop-codon/contamination filter, the count and frequency thresholds, the round 2→3
reframing, inserting the latent round, building the ancestor tuple — is baked into the CSVs and is
**not** repeated here. That also retires the quirks that used to have to be preserved verbatim; see
§Fixes.

What survives, in the single file `src/import_data.jl` (~200 lines, replacing the four files of the
old `src/io/`):

- alphabet and encodings, kept from `utils.jl`: `AA2INT`, `INT2AA`, `AAs`, `str2seq`, `seq2str`,
  `sample2hot`, `onehot2aa`, `aa2onehot` — `dist` went with the contamination filter;
- `read_counts_csv` / `read_samples_csv` — typed readers that **assert** the join
  (`counts_column == column + 1`, counts header == `column_name`, root first) instead of trusting it;
- `load_lung_dataset` — the two readers → `Data`;
- `build_select_washed` — the 1-mut design rules, unchanged.

**Dependencies this removes:** `SequenceLogos`, `PyPlot`, and `StatsBase` as an import dependency
(it stays, used by `metrics.jl`). `DelimitedFiles` is the only reader needed.

### C. Analysis, metrics, figures — from `notebooks/answer_reviewers/src`

Carry over, reorganised: `build_model`/`build_select_washed`/`l2_nn`/`learn_reg!`/`train_model!`
(`load_lung_data`'s reframing and frequency filter are now pre-applied in the CSVs — §B),
`resample_counts`,
`published_labels`/`agreement_table`/`topn_overlap`/`label_agreement`, the **corrected** pooled
estimators from `eval_metrics.jl`, and the 9 panels of `make_figure_combined.jl` (ported to
CairoMakie).

**Drop:** `scripts_training/utils.jl` — verified **0 call sites** in the current code
(`selectivities_safe2`, `compute_selectivity_correlation`, `fitler_samples`, `filter_sequences`,
`adaptive_training_phase` are all dead).
**Drop from the evaluation panel:** the XLSX read of `sup_tables.xlsx`, the `enriched` group
(`peptides_enrich_selected.jld2`) and the 20 000-sequence `background` group — the combined figure
uses **only the `designed` group**. Ship `data/designed_candidates.csv` instead (369 sequences +
published class + the four published `min_*_energy` columns). This removes `XLSX`.

---

## Deliberate fixes, and quirks preserved on purpose

**Fixed** (each is a real defect):
1. **The regularizer.** The original `learn!` passes `reg` as a zero-argument closure over the *outer*
   model; under Flux's explicit gradient mode the penalty contributes **exactly zero gradient**
   (measured `0.0` vs `1.95e1` for the correct form). The vendored `learn!` takes `regf(m)`, a
   function *of* the differentiated model. A regression test pins this.
2. `datafiles.jl` replaced by explicit path configuration with error messages that name the missing
   file, and with bundled defaults so the keys are optional.
3. Generation: see below.

**Preserved verbatim** — the experimental design, which `build_select_washed` must still reproduce
exactly from `df_exp`:
- 4 cell lines × 2 replicas × (RI, RII−, RII+) + latent round 2 → 32 rounds; `select`/`washed` per
  the 1-mut rules with **wash-only washing**. Pinned by Verification 3.

**No longer this repo's problem.** Three quirks used to have to be carried verbatim because they
decided which sequences entered training: the contamination filter (drop everything within Hamming
distance 1 of `"WSLGYTG"`), `filter_experiment`'s fraction over a **per-sequence row** sum rather
than a per-sample column sum, and `parser_sample_names_lung`'s uppercase mismatch. They are now
frozen inside `data/lung_counts.csv` and the code implementing them is deleted. Regenerating the
CSVs means going back to `export_published_csv.jl` in the original project — that script is the only
remaining link to it.

---

## Repository layout

```
kras-peptide-phage-ml/
  Project.toml                 Flux, Functors, ChainRulesCore, SpecialFunctions,
                               LogExpFunctions, DataFrames, DelimitedFiles, JLD2,
                               StatsBase, Statistics, Random, Printf, TOML, CairoMakie
  README.md  WORKPLAN.md  NOTES_provenance.md
  src/
    KrasPeptidePhageML.jl       module + includes
    core/  data.jl energies.jl model.jl util.jl ancestors.jl mc.jl history.jl
    import_data.jl              alphabet + the two CSV tables -> Data, df_exp
    build.jl train.jl generate.jl metrics.jl labels.jl figures.jl paths.jl
  data/  lung_counts.csv  lung_samples.csv  designed_candidates.csv  README.md
  config/paths.toml.example
  results/                     compact per-run artifacts (committed, ~10 KB each)
  scripts/ 01_train.jl 02_generate.jl 03_perturbations.jl 04_figure.jl
  notebooks/ quickstart.ipynb reproduce_figure.ipynb
  test/runtests.jl
```

`history.jl` is a ~15-line `TrainHistory` replacing `MVHistory`, which drops `ValueHistories`.

**Paths.** `src/paths.jl` reads `config/paths.toml` (git-ignored; an `.example` is committed),
overridable per call and by `ENV`. Keys: `counts_csv`, `samples_csv`, `reference_model` — all three
fall back to the bundled copy in `data/`, so the config file is **optional** and only needed to
point at data kept elsewhere. Resolution happens **inside functions**, never at module load.

**Compact results.** Per perturbation run store only what the figure needs — `E_panel` (369×5
Float32), published-label agreement, per-mode correlations, timings, config — ~10 KB instead of
7 MB. 15 runs ≈ 150 KB, committable. Full models are optional and git-ignored.

---

## Generation (`src/generate.jl`)

Recipe recovered from the original project's `lung/generation/generation_model.ipynb`
— verified as the source of Table S2: the union of its three output files is exactly the 369
sequences and the `strategies` column reproduces with **0 mismatches**.

Hyperparameters to keep exactly: `β = vcat(fill(1e-2,10), fill(1.0,10), fill(10.0,10), fill(100.0,10))`;
1000 independent chains; uniform random one-hot starts (no seed library);
`ParetoAnneal!` then `ToParetoFront!`; thresholds `E_thr = mean(E) .- std(E)` over 10 000 random
training sequences. Modes per strategy: **Energy optimization** `modes=[m]`, m ∈ 1:4;
**Pareto 2-state** `modes=[1,m]`, m ∈ 2:4; **Pareto 3-state** `modes=[1,m]`, m ∈ 3:4 — the MC still
optimises two modes; "3-state" comes from the *selection filter* adding the negative-energy
criterion.

Fixed relative to the original (documented in `NOTES_provenance.md`): selection indices are applied
to the array they were computed from (the original mixed Pareto-refined energies with
annealed-only arrays); the wt block uses the wt array (the original used positive+mutation); no
accidental truncation (the original's 86/27/28 were `BoundsError` side effects); and the RNG is
seeded. Exact reproduction of the 369 is impossible regardless — the original seeded nothing.

Validation instead of exact match: generated candidates must land in the expected quadrants of the
`E[mutation]`/`E[wt]` plane and overlap substantially with the 369.

---

## Implementation steps

1. **Skeleton + paths** — `Project.toml`, module, `paths.jl`, `data/README.md`. `Pkg.instantiate()`
   must succeed with no personal packages in the manifest.
2. **Vendor the core** — the six files per §A, unqualifying the module-qualified references, with
   the corrected `learn!(…; regf)` and `TrainHistory`. Gate: `Pkg.precompile()` clean.
3. **Import path** — §B: `import_data.jl`, the two CSV readers plus `load_lung_dataset` and
   `build_select_washed`. No vendoring; the dataset arrives pre-built.
4. **Golden tests** (the gate that proves the vendoring is faithful) — see Verification 2–4.
5. **Training** — `train.jl` + `scripts/01_train.jl`.
6. **Generation** — `generate.jl` + `scripts/02_generate.jl`.
7. **Metrics and labels** — `metrics.jl` (pooled estimators, log-loss, flat-model null),
   `labels.jl`.
8. **Perturbations** — `scripts/03_perturbations.jl` (warm CV folds + read bootstrap/subsampling),
   writing compact artifacts.
9. **Figures** — port the 9 panels to CairoMakie; `scripts/04_figure.jl`.
10. **Notebooks** — `quickstart.ipynb` (minutes) and `reproduce_figure.ipynb` (from committed
    artifacts).
11. **Docs** — `README.md` (quickstart, data setup) and `NOTES_provenance.md` (vendored-from map
    with line ranges, preserved quirks, fixed bugs, generation deviations).

---

## Verification

1. `julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'` — no personal packages in the
   manifest (no personal package appears in it).
2. **Energy path golden test.** `energies` for the 369 designed candidates must match the
   `min_{positive,negative,mutation,wt}_energy` columns of `data/designed_candidates.csv` to
   **< 1e-5**. This exercises the whole vendored energy stack (`DeepEnergy`, `ZeroEnergy`, `cat`,
   `sample2hot`, alphabet order) in one assertion. It holds at `max|diff| ≈ 1e-6`, which is Float32
   round-off (relative 1.2e-7), not an export-precision artifact — hence 1e-5 and not 1e-6.
3. **Design golden test.** `build_select_washed` on the `df_exp` read back from
   `lung_samples.csv` equals the reference model's stored `select`/`washed` — 160/160 entries each.
   ✅ holds.
4. **Import golden test.** `log_likelihood(reference_model, load_lung_dataset())` = **−392.4327**
   (4 dp). One number validating the whole CSV→`Data` chain: column/row alignment, the root in
   column 1, the all-zero latent columns, the ancestor tuple, one-hot encoding. ✅ holds at
   −392.43271686. Since the CSVs *are* the published training set, this is now an exact-match test
   rather than a re-derivation — the structural assertions in `read_*_csv` and the round-trip of
   `total_reads` cover what the old pipeline steps used to.
5. **Regularizer regression test.** Gradient of the penalty w.r.t. the differentiated model equals
   `2α·Σ|W|` and is **non-zero** — pins the bug fixed in §Fixes.
6. **Training smoke test.** A few epochs on a subsample strictly decrease the objective.
7. **Generation smoke test.** 20 chains yield valid 7-mers over the 20-letter alphabet, in the
   expected energy quadrant for their strategy.
8. **Figure test.** `reproduce_figure.ipynb` regenerates `figure_combined.svg` from the committed
   artifacts; assert 9 axes present and the SVG contains `<text>` elements with zero outlined glyph
   paths (editable text).

Tests 2–4 are the real acceptance criteria: they compare the vendored code against numbers produced
by the original stack, so a vendoring mistake cannot pass silently.

---

## Out of scope

The 3-mut model; import paths for the other tissues; the DRB and FACS analyses; the full
robustness suite as a test (the scripts are there, the notebooks use precomputed artifacts); and
retraining to match the published optimum — reaching it needs the original longer multi-phase
schedule, which is noted but not replicated.
