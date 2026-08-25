# An open dataset of soil compaction tests

**2,854 Proctor compaction tests** assembled from six public sources, each
record complete in the index properties required for modelling and each
carrying the compactive energy its source stated — spanning four Proctor
energy levels from 355 to 2,693 kJ/m³ and both sides of the coarse/fine
boundary.

A wider pool of **7,273 quality-controlled records** and **90 digitised
compaction curves (1,048 measured points)** are released alongside it.

Every record carries its originating source, so any evaluation can respect
provenance rather than treating records from one laboratory as independent.

**Trained weights ship with the data.** `predict.py` predicts maximum dry
density and optimum moisture content for a soil you describe, from boosters
fitted on all 2,854 records — see [Predict a soil](#predict-a-soil).

---

## Why this exists

Correlations predicting maximum dry density (MDD) and optimum moisture content
(OMC) from index properties have been published for four decades. They rest on
datasets of one to four hundred specimens, usually from a single laboratory,
almost always at a single compactive energy, and almost never released. Four
decades of correlations have therefore accumulated that cannot be compared with
one another, because no two rest on the same soils.

This dataset is an attempt to fix the input side of that problem.

## What is here

![Data acquisition and screening, from initial query to the modelling dataset](figures/data_flow.png)

*How the two released datasets were assembled. The principal flow runs down the
left; exclusions branch to the right with the count and the reason. 11,119
candidate URLs and 618 candidate datasets reduce to four harmonised streams of
8,457 records, then to the 7,273 that pass the zero-air-voids admissibility
check — released as `compaction_parameters_full.csv` — and finally to the 2,854
complete in every input the model needs, released as
`compaction_parameters.csv`. This is Figure 1 of the accompanying paper.*

| File | Records | Content |
|---|---|---|
| `data/compaction_parameters.csv` | 2,854 | **The modelling dataset.** Complete in PI, fines, energy and Gs; no imputation of soil properties. This is the file the accompanying paper reports on. |
| `data/compaction_parameters_full.csv` | 7,273 | The wider pool: every record passing the admissibility filter, including those with incomplete property sets. Impose your own completeness requirement rather than inheriting ours. |
| `data/curve_points.csv` | 1,048 | Digitised (water content, dry density) points forming 90 complete compaction curves |
| `data/curve_parameters.csv` | 90 | Four-parameter fits of those curves, with identifiability flags |
| `data/sources.csv` | 6 | Provenance, licence and record counts per source |
| `data/references.csv` | 378 | Per-record citation |
| `models/model_mdd.json` | — | **Trained weights for maximum dry density.** Gradient-boosted, fitted on all 2,854 records |
| `models/model_omc.json` | — | **Trained weights for optimum moisture content.** Same records, same configuration |
| `models/model_meta.json` | — | Input order, hyperparameters and the out-of-fold accuracy of both, so the weights never travel without their honest performance |
| `models/source_pfn.csv` | 2,854 | **The TabPFN artefact.** The dataset prepared as the context the model predicts from: the seven inputs in order with energy already logged, plus the two targets, and no identifier column |
| `models/tabpfn_meta.json` | — | The record for that model, which has no weights by construction — the context path, its digest, the input order and the out-of-fold accuracy |
| `predict.py` | — | **The command-line predictor.** Loads the weights and applies them to soils you supply |
| `examples/example_soils.csv` | 6 | Worked input for the batch mode of `predict.py` |

Field definitions, units and derivations: **[`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md)**.

### Columns of the modelling dataset

| Column | Meaning | Unit |
|---|---|---|
| `LL`, `PL`, `PI` | liquid limit, plastic limit, plasticity index | % |
| `fines_pct`, `sand_pct` | fraction passing 0.075 mm; sand fraction | % |
| `energy_kJm3` | compactive energy as stated by the source | kJ/m³ |
| `Gs` | specific gravity of solids | — |
| `MDD_Mgm3` | maximum dry density | Mg/m³ |
| `OMC_frac` | optimum moisture content | **decimal fraction** |
| `group` | the unit a cross-validation split should respect | — |
| `record_id`, `source`, `test_standard` | provenance | — |

Two columns carry **deliberate gaps**, and they should be left as they are:

- `sand_pct` is absent on 169 records. Every record in the corpus reporting a
  non-standard compactive effort omits the sand fraction, so requiring it would
  delete all four-energy coverage. Gradient-boosted trees and several other
  learners accept the gaps directly.
- `LL` and `PL` are absent on 134 records. These are non-plastic soils, for
  which the Atterberg tests do not apply; `PI = 0` records the fact that
  matters. The published values for these soils are placeholders — 113 report a
  plastic limit above the liquid limit, which cannot be true — so they were
  removed rather than propagated.

## Predict a soil

The weights in `models/` are the artefact of the accompanying paper: two
gradient-boosted models, one per target, fitted on every record here with
hyperparameters fixed in advance rather than searched. `predict.py` applies
them.

```bash
pip install -r requirements.txt
```

**One soil.** Give the index properties you have and the Proctor standard:

```bash
python predict.py --ll 38 --pl 20 --fines 72 --sand 25 --gs 2.70 --effort SP
```

```
soil 1
  MDD  1.775 Mg/m3    +/- 0.068 out-of-fold MAE
  OMC  16.90 %          +/- 1.89 out-of-fold MAE
  degree of saturation at optimum 0.876
```

**Many soils.** A CSV carrying the input columns, with `test_standard` or
`energy_kJm3` for the effort:

```bash
python predict.py --csv examples/example_soils.csv --out predictions.csv
```

**Machine-readable.** `--json` writes the model name, the predictions and the
accuracy figures to stdout in either mode.

### Which model

| | `--model xgboost` (default) | `--model tabpfn` |
|---|---|---|
| MDD R², out-of-fold | 0.818 at 0.068 Mg/m³ | **0.823 at 0.066** |
| OMC R², out-of-fold | 0.781 at 1.89 % | **0.784 at 1.87** |
| Impossible pairs, of 2,854 | 11 | **2** |
| Time per call | a fraction of a second | **~5 minutes on CPU** |
| Install | `requirements.txt`, three packages | plus `requirements-tabpfn.txt` — torch, and a checkpoint downloaded on first use |

TabPFN is the paper's leading model and the one Sections 5.2–5.5 are reported
with, so it is offered here. It ships **no weights**, and cannot: it predicts by
in-context learning, taking no gradient step on this dataset at all, so the
2,854 records are supplied as context and the query read off in a single forward
pass.

**`models/source_pfn.csv` is what it loads**, and stands in the same relation to
TabPFN as `model_mdd.json` does to the boosters — change it and you have changed
the model. It is the dataset prepared: the seven inputs in the order the model
expects them, compactive energy already converted to its natural logarithm, the
two targets, the documented gaps carried through as blanks, and deliberately no
`record_id`, `source` or `group`, so that no identifier can reach the model as a
feature. `predict.py` checks it against the digest in `models/tabpfn_meta.json`
before running, and refuses rather than quoting an accuracy that would describe
a different model. Rebuild it from the dataset at any time:

```bash
python scripts/build_source_pfn.py
```

The default is the gradient-boosted model because the two are not distinguishable
in accuracy on these data — 0.005 in R², against a between-fold dispersion of
0.012 to 0.020 — while they differ by three orders of magnitude in cost. On a
lean clay the two return 1.775 against 1.788 Mg/m³ and 16.90 against 16.66 %,
differences an order of magnitude inside their own error bars. Where the choice
does matter is physical admissibility: TabPFN implies an impossible degree of
saturation five times less often, which is why the paper carries it through.
Reach for it when a prediction sits near the zero-air-voids line, or when a
handful of soils justifies the wait.

### What the inputs are

| Flag | Column | Unit | |
|---|---|---|---|
| `--ll` | `LL` | % | liquid limit; omit for a non-plastic soil |
| `--pl` | `PL` | % | plastic limit; omit for a non-plastic soil |
| `--pi` | `PI` | % | plasticity index; **derived as LL − PL** when both are given, so pass it only for a non-plastic soil, as `0` |
| `--fines` | `fines_pct` | % | passing 0.075 mm; **required** |
| `--sand` | `sand_pct` | % | sand fraction; may be omitted |
| `--gs` | `Gs` | — | specific gravity; defaults to 2.68, the corpus median |
| `--effort` | `test_standard` | — | `SP`, `MP`, `RSP` or `RMP` |
| `--energy` | `energy_kJm3` | kJ/m³ | an alternative to `--effort`, for an effort not on the list |

Energy enters the model as its natural logarithm; `predict.py` takes kJ/m³ and
converts, so never pass a logarithm. `LL`, `PL` and `sand_pct` may be left
blank: the boosters were fitted with those gaps present, for the reasons above,
and route a blank down a learned default branch. Everything else is required.

### What comes back, and what to trust

Each prediction carries the degree of saturation it implies at the peak, and a
flag when that exceeds unity — the pair then plots above the zero-air-voids line
and describes a soil that cannot exist. On the released data this happens for
11 of 2,854 records; it happens more often when the inputs are far from the
corpus, and inputs outside the range the corpus spans are flagged as
extrapolation.

**Quote the out-of-fold accuracy, not the fit.** Under random five-fold
cross-validation the released hyperparameters give R² 0.818 for density at
0.068 Mg/m³ and 0.781 for the optimum at 1.89 % water content, and TabPFN 0.823
and 0.784. Those figures are recorded in `models/model_meta.json` and
`models/tabpfn_meta.json`, and reported alongside every prediction. The weights shipped here were then refitted on all 2,854 records
with nothing held back, so what they score on their own training data is not an
estimate of anything. Two further cautions: a grouped split, which respects
provenance, is roughly 0.10 in R² harder than the random split reported, so
expect less on a source the model has not seen; and the prediction is of the
peak alone, not of the curve either side of it.

Regenerate either artefact from the released data at any time:

```bash
python scripts/train_model.py        # models/model_mdd.json, model_omc.json
python scripts/build_source_pfn.py   # models/source_pfn.csv
```

### What drives the prediction

![Shapley attributions for both released models](figures/shap_attribution.png)

*What each input contributes, for both models this repository ships. (a, d)
XGBoost, exact TreeSHAP over all 2,854 records; (b, e) TabPFN, permutation
estimate over 400; (c, f) mean |SHAP| as a share of the total. In the beeswarm
panels each point is one record, placed by its SHAP value and coloured by the
feature's own value, low to high. This is Figure 5 of the accompanying paper.*

Fines content and the liquid limit carry roughly two-thirds of the attribution
on both targets, and the two models agree on the ordering at a Spearman
coefficient of 0.96 — which is the useful thing to know before trusting a
prediction: **a soil whose fines content or liquid limit you are unsure of is a
soil the prediction is unsure of.** Specific gravity and the plasticity index
contribute little, so the 2.68 default for `--gs` costs less than it might
appear.

`ln(energy)` looks negligible in the attribution share, and is not. Only 101
records of 2,854, 3.5 %, sit at a non-standard compactive effort, so a mean
absolute Shapley value over the whole dataset is dominated by the 96.5 % where
the input does not vary. Panels (a) and (d) show what it does when it does
vary — the yellow points, high energy, sit clearly apart from the rest. Compare
the two efforts on one soil and the difference is 0.17 Mg/m³ and 3.5 % water
content, which is not a small effect.

## Fit your own model

```python
import pandas as pd, numpy as np

d = pd.read_csv("data/compaction_parameters.csv")
d["log_energy"] = np.log(d.energy_kJm3)

X = d[["LL", "PL", "PI", "fines_pct", "sand_pct", "log_energy", "Gs"]]
y_density  = d.MDD_Mgm3      # Mg/m3
y_moisture = d.OMC_frac      # fraction, multiply by 100 for percent
```

**Group your folds.** LTPP contributes 1,976 records naming 905 distinct test
sections, which are not independent publications. A random split places
near-duplicate soils from one laboratory on both sides of the partition. Use
`GroupKFold` on `group` if you want an estimate of performance on a source the
model has not seen; a random split answers a different and easier question, and
in our hands the two differ by roughly 0.10 in R².

**What to beat.** Under the random five-fold split, the accompanying paper
reports R² 0.823 for density from TabPFN, 0.818 from the gradient-boosted model
released here and 0.797 from a multilayer perceptron — three model classes
within 0.03 of one another, on identical folds. A nested search over nine
hyperparameters of the ensemble recovered nothing over the values fixed a
priori. The ceiling looks like a property of the data rather than of the
learner, so the return on a further algorithm is probably small; the return on
more records, or on the curve either side of the peak, is not.

## Quality control

Physical admissibility is assessed through the degree of saturation implied at
the compaction peak. With ρ_w = 1 Mg/m³,

```
e     = Gs / MDD − 1
S_opt = OMC · Gs / e
```

A record with `S_opt > 1` plots above the zero-air-voids line and is impossible
as published. Records are retained where `0.20 ≤ S_opt ≤ 1.0`, MDD lies in
0.8–2.6 Mg/m³ and OMC in 1–70 %.

**Of 8,242 harmonised records, 969 (11.8 %) failed** — 914 above the
zero-air-voids line, 55 below the lower bound. Roughly one published compaction
record in nine, taken at face value, describes a soil that cannot exist. The
rate is far higher among records extracted from the literature than among
deposited datasets, which is the argument for applying the check as a filter
rather than reporting it as a diagnostic.

Two deposited datasets that met every inclusion criterion were **excluded for
integrity**: in each, every apparently genuine specimen is followed by roughly
nine near-copies perturbed in the fourth to sixth decimal place, and 194 of 199
records in one of them plot above the zero-air-voids line.

Verify the release:

```bash
shasum -a 256 -c CHECKSUMS.sha256
python scripts/validate_corpus.py
```

## Sources

| # | Source | Region | Licence |
|---|---|---|---|
| 1 | [LTPP Standard Data Release 39](https://infopave.fhwa.dot.gov/Data/StandardDataRelease), Material Test volume | United States | US federal public domain |
| 2 | [Soranzo (2024)](https://doi.org/10.5281/zenodo.14251190), geotechnical database | Austria | CC BY 4.0 |
| 3 | [Geotechnical Properties of Soils: A Dataset from Literature (2025)](https://doi.org/10.6084/m9.figshare.28681187) | multi-source | CC BY 4.0 |
| 4 | [Geotechnical Properties and CBR Values, Türkiye (2026)](https://doi.org/10.5281/zenodo.20737270) | Türkiye | CC BY 4.0 |
| 5 | [Compaction Characteristics Data (2026)](https://doi.org/10.6084/m9.figshare.32955851) — the only source spanning more than one compactive energy | multi-source | CC BY 4.0 |
| 6 | [Biopolymer-stabilised soils (2026)](https://doi.org/10.5281/zenodo.19242689), untreated parent soils only | multi-source | CC BY 4.0 |

If you use this dataset, **cite the original sources as well as this
compilation**. Per-record citations are in `data/references.csv`.

## What is not here

**Complete compaction curves, at scale.** No public repository holds them. The
four to six measured points are discarded at publication and only the peak
reported; this is explicitly true of LTPP, whose documentation states that the
points other than the peak are not loaded into the database. The 90 curves in
`data/curve_points.csv` were recovered by digitising figures, and are the
exception rather than the rule.

**Laboratory data from national databases.** The British Geological Survey
National Geotechnical Properties Database defines AGS4 groups for compaction
points and parameters and exposes 72,849 boreholes through an open API, but a
random sample of 600 boreholes returned 57 project files containing no
compaction or Atterberg group whatever. The public export carries borehole logs
only; laboratory data requires a licence.

## Limitations

- **Energy coverage is thin and confounded.** Four levels are present, but only
  101 records (3.5 %) are at non-standard effort and all derive from a single
  curated source.
- **One programme supplies 69 % of records.** LTPP is laboratory data from one
  programme under one protocol.
- **Only the peak is modelled** in the main dataset. The curve either side of
  optimum — which determines the acceptable water-content window on site — is
  present for 90 soils only.
- **Gs is imputed on most records** at 2.68, the corpus median of measured
  values, and propagates into the derived `S_opt`.
- **Geographic imbalance.** The United States, Türkiye and Austria predominate;
  tropical residual, volcanic and temperate glacial soils are sparse.

## Licence

[CC BY 4.0](LICENSE). Individual sources carry their own terms, recorded in
`data/sources.csv`; all are CC BY 4.0 or public domain.
