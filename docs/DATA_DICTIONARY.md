# Data dictionary

Units are stated for every field. Where a quantity is derived rather than
measured, the derivation is given.

---

## `data/compaction_parameters.csv` — 2,854 records

The modelling dataset. Every record reports plasticity index, fines content,
compactive energy and specific gravity, so a model can be fitted without
imputing any soil property.

| Field | Unit | Description |
|---|---|---|
| `record_id` | — | Identifies the **test**, not the row. 2,762 distinct values over 2,854 rows: 170 rows share an identifier with another, 166 of them LTPP repeat tests on one pavement layer and 4 a collision in identifier construction between two different figshare soils. **Do not join on it** — a merge multiplies rows and silently corrupts any metric computed afterwards. Use the row order, which is stable across the release |
| `source` | — | Contributing source; join to `sources.csv` |
| `group` | — | **The unit a cross-validation split should respect.** LTPP records name 905 distinct test sections belonging to one testing programme, so they carry their state code (`LTPP-4`, `LTPP-35`, …); every other record carries its reference or DOI. 162 groups: 54 LTPP states and 108 other sources |
| `test_standard` | — | `SP` standard, `MP` modified, `RSP` reduced standard, `RMP` reduced modified Proctor |
| `LL` | % | Liquid limit. Absent on the 134 non-plastic records |
| `PL` | % | Plastic limit. Absent on the same 134 records |
| `PI` | % | Plasticity index, `LL − PL`. Present on every record; `0` denotes non-plastic |
| `fines_pct` | % | Fraction passing the 0.075 mm sieve |
| `sand_pct` | % | Sand fraction. Absent on 169 records, all of which report a non-standard compactive effort |
| `energy_kJm3` | kJ/m³ | Compactive energy **as stated by the source**, never inferred. A record whose source did not identify the standard followed was excluded rather than assigned one |
| `Gs` | — | Specific gravity of solids. Imputed at 2.68 where unreported |
| `MDD_Mgm3` | Mg/m³ | Maximum dry density |
| `OMC_frac` | fraction | Optimum moisture content **as a decimal fraction**. Multiply by 100 for percent |

### Compactive energy levels

| `test_standard` | Energy (kJ/m³) | Records |
|---|---|---|
| `RSP` reduced standard | 355.2 | 28 |
| `SP` standard | 592.5 | 2,753 |
| `RMP` reduced modified | 1,346.6 | 7 |
| `MP` modified | 2,693.3 | 66 |

All 101 non-standard records derive from a single curated source
(`figshare:10.6084/m9.figshare.32955851`) spanning 51 distinct references, so
compactive energy is partly confounded with provenance. Any pooled metric on
this file is predominantly a standard-Proctor metric.

### Composition

| Source | Records |
|---|---|
| `ltpp-sdr39` | 1,976 |
| `figshare:10.6084/m9.figshare.28681187` | 395 |
| `zenodo:10.5281/zenodo.20737270` | 269 |
| `figshare:10.6084/m9.figshare.32955851` | 169 |
| `zenodo:10.5281/zenodo.14251190` | 38 |
| `zenodo:10.5281/zenodo.19242689` | 7 |

---

## `data/compaction_parameters_full.csv` — 7,273 records

Every record passing the admissibility filter, including those with incomplete
property sets. Use this file to impose a completeness requirement of your own.
It carries the fields above, without `group`, plus:

| Field | Unit | Description |
|---|---|---|
| `reference_or_doi` | — | Citation or DOI of the originating publication |
| `gravel_pct`, `silt_pct`, `clay_pct` | % | Further gradation fractions where reported |
| `D10`, `D30`, `D50`, `D60` | mm | Particle diameters at the stated passing percentages |
| `Cu`, `Cc` | — | Uniformity and curvature coefficients |
| `uscs` | — | Unified Soil Classification System symbol where reported |
| `Gs_imputed` | bool | `True` where `Gs` was set to the corpus median rather than measured |
| `S_opt` | fraction | Degree of saturation at the compaction peak, derived below |

Coverage is uneven by construction: the fields beyond the modelling set are
reported by some sources and not others.

---

## Derived quantities

With ρ_w = 1 Mg/m³, the void ratio and degree of saturation at the compaction
peak follow from the record itself:

```
e     = Gs / MDD − 1
S_opt = OMC · Gs / e
```

`S_opt` is derived, not measured, and inherits any error in `Gs` — which is
imputed on most records. It is the basis of the admissibility filter: a record
with `S_opt > 1` plots above the zero-air-voids line and cannot exist as
published. Records are retained where `0.20 ≤ S_opt ≤ 1.0`.

Across this corpus `S_opt` has a mean of 0.815 and a coefficient of variation
of 11 %, so soils reach their compaction peak at approximately 82 % saturation
irrespective of plasticity, gradation or compactive effort.

---

## `data/curve_points.csv` — 1,048 points

Digitised compaction curves, one row per measured point. This is the part of
the release describing curve *shape* rather than only its peak.

| Field | Unit | Description |
|---|---|---|
| `curve_id` | — | Join to `curve_parameters.csv` |
| `source_doi` | — | DOI of the article the figure came from |
| `w_pct` | % | Water content |
| `rho_d_Mgm3` | Mg/m³ | Dry density at that water content |

Recovered by digitising vector figures in open-access articles. Point
coordinates carry digitisation error in addition to the original measurement
error, and should not be treated as equivalent in precision to tabulated
values.

---

## `data/curve_parameters.csv` — 90 curves

Four-parameter fits of the curves above, parameterised in saturation space so
that a fitted curve cannot cross the zero-air-voids line:

```
S(w) = S_opt · (w / w_opt) ^ k_d     for w <= w_opt
S(w) = S_opt · (w / w_opt) ^ k_w     for w >  w_opt
```

With `k_d > 1` and `k_w < 1` the curve rises monotonically dry of optimum,
falls monotonically wet of it, peaks exactly at `w_opt`, and satisfies `S ≤ 1`
throughout. The compaction parameters follow as

```
OMC = w_opt
MDD = Gs · rho_w · S_opt / (S_opt + w_opt · Gs)
```

Identifiability flags record where a parameter is poorly constrained by the
available points — `k_w` in particular, because a saturated wet branch follows
the zero-air-voids line and the loss surface becomes flat in that direction.

---

## `data/sources.csv`

| Field | Description |
|---|---|
| `source_key` | Join to the `source` column of the data files |
| `name`, `region`, `doi_or_note`, `licence` | Provenance and terms |
| `records_raw`, `records_retained` | Counts before and after quality control |

## `data/references.csv`

Per-record citation, so that any observation can be traced to the publication
that reported it.

---

## `models/` — the released weights

Two gradient-boosted models, one per target, fitted on all 2,854 records of
`data/compaction_parameters.csv` and saved in XGBoost's portable JSON format.
`predict.py` loads them; `scripts/train_model.py` regenerates them.

| File | Content |
|---|---|
| `model_mdd.json` | Weights for `MDD_Mgm3`, Mg/m³ |
| `model_omc.json` | Weights for `OMC_frac`, decimal fraction — multiply by 100 for percent |
| `model_meta.json` | Input order, hyperparameters, the documented gaps, and the out-of-fold accuracy of both |
| `source_pfn.csv` | **The TabPFN artefact**, 2,854 rows. The dataset prepared as the context that model predicts from — see below |
| `tabpfn_meta.json` | The record for `predict.py --model tabpfn`. **No weights**: TabPFN takes no gradient step on this dataset, so `source_pfn.csv` is supplied as context at inference and is itself the artefact. The file records the context path, its digest, the expected record count, the input order and the out-of-fold accuracy |

### Inputs, in the order the weights expect them

| Position | Field | Unit | Note |
|---|---|---|---|
| 1 | `PL` | % | May be blank; 134 training records are |
| 2 | `PI` | % | Required. `0` denotes non-plastic |
| 3 | `fines_pct` | % | Required |
| 4 | `sand_pct` | % | May be blank; 169 training records are |
| 5 | `log_energy` | ln(kJ/m³) | **The natural logarithm of `energy_kJm3`**, not the energy itself |
| 6 | `Gs` | — | Required |

`LL` is **not** a model input. Only two of the three consistency limits are
algebraically independent, and carrying the third costs 0.041 in R² for density
on a source the model has not seen, so the liquid limit is released with the
dataset and held out of the mapping. `predict.py` still accepts it, and uses it
only to derive `PI` as `LL - PL` and to reject `PL` above `LL`.

The order matters: the boosters carry no column names, so a permuted input
produces a confident and wrong answer rather than an error. `predict.py` builds
the matrix from `model_meta.json`, and `scripts/validate_corpus.py` checks that
the recorded order is the one above.

### Fields of `model_meta.json`

| Field | Description |
|---|---|
| `n_records` | Records the weights were fitted on; 2,854, with nothing held back |
| `features` | The six inputs, in the order above |
| `params` | The XGBoost hyperparameters, fixed a priori rather than searched |
| `missing_by_design` | The documented gaps present during fitting, by column |
| `cross_validated` | R², MAE and RMSE per target from random five-fold cross-validation of this configuration, each computed over the pooled out-of-fold predictions of all 2,854 records. Not from the weights shipped here, which saw every record. **This is an interpolation figure**: under a random split 96.6 % of records share a provenance group with their training fold |
| `transfer` | **The figure to quote for a new soil.** The same configuration scored with folds blocked on the 162 provenance groups, and with each of the six sources held out in turn |
| `zav_violations` | Predicted pairs implying a degree of saturation above the zero-air-voids line, out of 2,854, under each design. MDD and OMC are predicted independently, so nothing forbids this |
| `models` | Target to weight file |

`cross_validated` and `transfer` report `OMC_frac` in the unit of the column, a
decimal fraction: MAE 0.0188 is 1.88 % water content.

`tabpfn_meta.json` carries the same `features`, `missing_by_design`,
`cross_validated`, `transfer` and `zav_violations` fields, in the same units,
plus `context` and `context_records`. `predict.py` refuses to run the TabPFN backend if the context
file no longer holds the recorded number of records, since the accuracy
alongside the prediction would then describe a different dataset.

### `models/source_pfn.csv` — the TabPFN context

Eight columns, 2,854 rows, built by `scripts/build_source_pfn.py` from
`data/compaction_parameters.csv`.

| Position | Field | Unit | Note |
|---|---|---|---|
| 1–6 | `PL`, `PI`, `fines_pct`, `sand_pct`, `log_energy`, `Gs` | as above | The model inputs, in the order the model expects. `log_energy` is **already** ln(kJ/m³) — do not log it again |
| 7 | `MDD_Mgm3` | Mg/m³ | Context target |
| 8 | `OMC_frac` | fraction | Context target |

The 134 blank `PL` and 169 blank `sand_pct` are carried through as blanks:
TabPFN ingests them directly, and imputing here would silently change the model
the paper reports.

No `LL`, `record_id`, `source`, `group` or `test_standard`, by design — an identifier
column reaching the model as a feature is the failure this omission prevents.
Rows follow the order of `data/compaction_parameters.csv`, which is how the two
are matched; `record_id` cannot serve, being shared by 170 rows.

`models/tabpfn_meta.json` records `context_sha256` over this file. `predict.py`
verifies it before loading and `scripts/validate_corpus.py` additionally checks
the file reproduces the dataset row for row, so drift is caught rather than
quietly changing what `--model tabpfn` means.
