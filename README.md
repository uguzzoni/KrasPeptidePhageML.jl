# KrasPeptidePhageML.jl

Julia implementation of the machine-learning model used in:

> **Integrated whole-cell phage display and machine learning identify candidate
> targeting ligands in KRAS-defined lung cancer contexts**
> Guido Uguzzoni, Dimas Carolina Belisario, Andrea Pagnani, Serena Marchiò
> *Journal* (year). DOI: *to be added*

The package is the computational companion to that study. It contains the
published phage-display sequencing data, the fitted model behind the analysis,
and the code needed to re-read the data, re-fit the model, score peptides, and
regenerate the robustness figure.

The wet-lab parts of the study — biopanning, peptide synthesis, flow-cytometry
validation — are described in the paper and are not part of this repository.

---

## What the model does

Whole-cell phage display was performed on four lung cancer cell lines spanning
KRAS wild-type and mutant backgrounds, over three rounds of selection and two
biological replicates. The resulting NGS read counts are modelled as the
combined contribution of several **latent selection modes**: instead of scoring
each experiment separately, a single probabilistic model explains all 32
sequenced rounds at once, and each peptide receives one *energy* per mode.

An energy is a learned sequence score inferred from read counts — lower energy
means stronger predicted association with that selection mode. It is a
statistical quantity, not a measured binding affinity.

The five modes, as they are wired to the experiments in
[`build_select_washed`](src/import_data.jl):

| Mode | Attached to | Interpretation |
| --- | --- | --- |
| `mutation` | outputs of the KRAS-mutant lines (G12C / G12D / G12V) | mutant-associated selection |
| `wt` | outputs of the KRAS-WT line, plus the non-round-2 inputs | WT-associated selection |
| `positive` | every sequenced round | KRAS-unspecific selection |
| `negative` | the round-2 latent nodes (washed-away population) | control-associated selection |
| `wash` | all rounds | non-selective losses |

The first four correspond to the four latent selection components described in
the Methods; `wash` carries the round-to-round depletion that is not selection.

Each mode maps sequence → energy through its own feed-forward network: one-hot
encoded heptapeptide in, two hidden layers of 20 and 5 SELU units, one scalar
out. Network weights and the experiment-specific mixing coefficients are fitted
jointly by maximising a multinomial likelihood of the observed read counts,
with L2 regularisation and mini-batch stochastic optimisation (batch size 128).
The formulation follows the phage-display selection model of Uguzzoni et al.,
adapted here to a whole-cell, multi-cell-line design.

### Specificity classes

Every model-derived conclusion in the paper passes through absolute thresholds
on two energies ([`src/labels.jl`](src/labels.jl)):

```
mut-specific     E[mutation] < -5      and  E[wt] >  0
wt-specific      E[mutation] > -3.5    and  E[wt] < -0.5
cross-specific   E[mutation] < -3.5    and  E[wt] <  0
```

Candidates in none of the three regions are labelled `none`.

---

## Installation

Requires Julia ≥ 1.10.

```bash
git clone <repository-url>
cd KrasPeptidePhageML.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Then check that the data resolves and the package loads:

```julia
julia> using KrasPeptidePhageML
julia> check_data_paths()
  counts_csv        OK    .../data/lung_counts.csv
  samples_csv       OK    .../data/lung_samples.csv
  reference_model   OK    .../data/model_nn_2l.jld2
```

Run the test suite with `julia --project=. -e 'using Pkg; Pkg.test()'`. Tests
that need the data are skipped, not failed, if it is not configured.

---

## Quick start

```julia
using KrasPeptidePhageML

data, df_exp, seqs = load_lung_dataset()   # 123 777 peptides × (root + 32 rounds)
model = load_reference_model()             # the published fit

cand = designed_candidates()               # the 369 model-designed peptides
E    = panel_energies(model, cand.sequence)   # 369 × 5 energy matrix
published_labels(E)                        # :mut_specific, :wt_specific, ...
```

Fitting a model of your own:

```julia
select, washed = build_select_washed(df_exp)
model = build_model(select, washed; seed = 1)
train_model!(model, data; schedule = PILOT_SCHEDULE)   # 2 epochs, a smoke test
```

Three training schedules are provided: `DEFAULT_SCHEDULE` (400 epochs, hours —
what a production fit uses), `WARM_SCHEDULE` (120 epochs, for refits starting
from an already-fitted model), and `PILOT_SCHEDULE` (2 epochs, to verify that
the pipeline runs).

Two notebooks walk through the whole thing:

- [notebooks/quickstart.ipynb](notebooks/quickstart.ipynb) — data, reference
  model, labels, the designed candidates, a pilot fit
- [notebooks/reproduce_figure.ipynb](notebooks/reproduce_figure.ipynb) — the
  robustness figure and the numbers behind it

---

## Data

Everything ships in [`data/`](data/); nothing is downloaded.

| File | Contents |
| --- | --- |
| `lung_counts.csv` | 123 777 heptapeptides × 33 count columns (initial library + 32 sequenced rounds) |
| `lung_samples.csv` | one row per round: cell line, KRAS status, replicate, round, input/output, position in the selection tree |
| `model_nn_2l.jld2` | the fitted reference model used for the published analysis |
| `designed_candidates.csv` | the 369 model-designed peptides with their published per-mode energies, design strategy, and specificity class |

The peptide design step itself — Monte-Carlo sampling in the fitted energy
landscape — is not part of this package. Its output ships instead, as
`designed_candidates.csv`, and is what every downstream result is computed
from; the procedure is described in the paper.

The four cell lines are A427 (G12D), CORL23 (G12V), H23 (G12C) and H1993
(KRAS WT). Rounds marked `latent` carry no reads: they are the washed-away
populations, present in the selection tree but never sequenced.

From these 369 designed candidates, 98 non-redundant heptapeptides were
synthesised and 88 were successfully solubilised and tested by flow cytometry,
as reported in the paper.

To point the package at data kept elsewhere, copy
[`config/paths.toml.example`](config/paths.toml.example) to `config/paths.toml`
and edit it. Each path can also be set through an environment variable
(`KRAS_COUNTS_CSV`, `KRAS_SAMPLES_CSV`, `KRAS_REFERENCE_MODEL`) or passed
directly as a function argument; see [`src/paths.jl`](src/paths.jl) for the
precedence rules.

Loading the reference model and evaluating it on the imported counts gives
`log_likelihood = -392.4327`. That single number validates the entire
CSV → `Data` chain, and the golden tests assert it.

---

## Reproducing the analysis

```bash
# 1. fit a model from scratch on the full dataset  (hours)
julia --project=. scripts/01_train.jl --seed 1 --schedule default

# 2. the robustness refits: 5 CV folds + 20 read resamplings  (~3 h)
julia --project=. scripts/02_perturbations.jl

# 3. redraw the combined figure from the stored refits  (seconds)
julia --project=. scripts/03_figure.jl
```

Step 3 alone is enough to regenerate [the figure](figures/figure_combined.svg):
it reads only `results/perturbations.jld2`, a ~200 KB artifact committed to the
repository that holds the 369 × 5 energy matrix of every refit. Steps 1 and 2
accept `--schedule pilot` for a fast end-to-end check.

A from-scratch fit will not land exactly on the published optimum. The original
model was trained with a longer multi-phase schedule that was not recorded in
full, and the regulariser has since been corrected, so the trajectory differs
from the first epoch. Step 1 is an *independent* fit to compare against the
reference, not a bit-for-bit reconstruction of it; the script prints the
comparison it supports (energy correlations and label agreement on the 369
candidates).

### Robustness analysis

Two perturbations quantify how much of the published analysis survives
resampling, both warm-started from the published solution so that any movement
reported is real:

- **Read-level resampling** — counts resampled multinomially at 100 %, 50 % and
  25 % of the original depth, with the sequence set held fixed (10 replicates
  at full depth, 5 at each reduced depth). This tests sensitivity to
  read-sampling noise only, not to library composition or biological
  replication.
- **Sequence-level cross-validation** — 5 folds over peptides, refitting on
  four fifths of the sequences.


---

## Repository layout

```
src/
  core/          the selection model: data structures, energies, likelihood,
                 Monte-Carlo sequence design  (vendored from the original code)
  paths.jl       data-path resolution
  import_data.jl CSV → Data, the alphabet, the mode/experiment wiring
  build.jl       model construction, per-mode networks, energy panels
  train.jl       the training loop, regulariser, schedules
  metrics.jl     enrichment, predictive log-loss, count resampling
  labels.jl      published thresholds, label agreement, CV splits
  figures.jl     the combined 3×3 robustness figure
scripts/         the three command-line stages above
notebooks/       quickstart and figure reproduction
data/            the published dataset, reference model, designed candidates
results/         the perturbation artifact the figure is drawn from
LICENSE          MIT, for the code
data/LICENSE     CC BY 4.0, for the data
```

---

## Licence

Code is released under the [MIT licence](LICENSE). The datasets in
[`data/`](data/) and the derived results in [`results/`](results/) are released
under [CC BY 4.0](data/LICENSE). Reuse of either requires attribution to the
paper.

## Citing

If you use this package or the data it ships, please cite the paper:

```bibtex
@article{uguzzoni_kras_peptides,
  title   = {Integrated whole-cell phage display and machine learning identify
             candidate targeting ligands in KRAS-defined lung cancer contexts},
  author  = {Uguzzoni, Guido and Belisario, Dimas Carolina and
             Pagnani, Andrea and Marchi{\`o}, Serena},
  journal = {},
  year    = {},
  doi     = {}
}
```

## Contact

Questions about the model and this package: Guido Uguzzoni
(<guido.uguzzoni@cea.fr>). Questions about the experimental work: Serena
Marchiò.
