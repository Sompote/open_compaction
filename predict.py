"""Predict maximum dry density and optimum moisture content from index properties.

Loads the gradient-boosted weights in `models/` — fitted on all 2,854 records of
`data/compaction_parameters.csv` — and applies them to soils you supply. The
paper's leading model, TabPFN, is available as `--model tabpfn`; see below.

One soil, from the command line:

    python predict.py --ll 38 --pl 20 --fines 72 --sand 25 --gs 2.70 --effort SP

Many soils, from a CSV carrying the input columns:

    python predict.py --csv examples/example_soils.csv --out predictions.csv

Machine-readable output, for either mode:

    python predict.py --ll 38 --pl 20 --fines 72 --gs 2.70 --effort SP --json

The model reads the six inputs of Section 4.1 of the paper: PL, PI, fines, sand,
compactive energy and Gs. Energy enters as its natural logarithm; this script
takes it in kJ/m3 and does the conversion, so never pass a logarithm.

The liquid limit is accepted but is not a model input. Only two of the three
consistency limits are algebraically independent, and the paper measures the
third to cost accuracy on a source the model has not seen, so LL is released
with the dataset and held out of the mapping. It is used here for two things
only: PI is derived as LL - PL when both are given, and PL above LL is rejected
as impossible. Passing LL alone therefore no longer informs the prediction.

PL and sand may be omitted: the boosters were fitted with those gaps present and
route a blank down a learned default branch, which is what the 134 non-plastic
and 169 no-gradation records of the corpus taught them to do. Every other input
is required.

Two models are available. `--model xgboost`, the default, uses the boosters in
`models/`. `--model tabpfn` instead reproduces the paper's leading model, which
has no weights to load by construction: TabPFN predicts by in-context learning,
so the 2,854 records of `models/source_pfn.csv` are supplied as context and the
query read off in a single forward pass. That file is the TabPFN artefact, and
plays the part the weight files play for the boosters; it is checked against a
recorded digest before use. It is more accurate on paper
and produces fewer physically impossible pairs, but costs roughly five minutes
per call on CPU against a fraction of a second, and needs `pip install -r
requirements-tabpfn.txt`. On the soils tried here the two agree far inside their
own error bars, which is the paper's point: the 0.005 in R2 between them is not
a difference this dataset can resolve.

The reported accuracy is always out-of-fold, from the paper, and is printed
beside every prediction: R2 0.819 for density at 0.068 Mg/m3 and 0.783 for the
optimum at 1.88 % water content for the boosters, 0.824 at 0.066 and 0.784 at
1.87 for TabPFN. The in-sample fit of weights trained on every record is not an
estimate of anything.

Those are random-fold figures, and under a random split 96.6 % of records share
a provenance group with their training fold. For a soil from a laboratory or a
study the corpus does not contain, quote the `transfer` block of the metadata
instead: 0.722 and 0.681 for the boosters with folds blocked on the 162
provenance groups, 0.392 and 0.225 with a whole source held out, against 0.727
and 0.696, and 0.520 and 0.614, for TabPFN.
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")

# the four Proctor efforts present in the corpus, kJ/m3
EFFORTS = {"RSP": 355.2, "SP": 592.5, "RMP": 1346.6, "MP": 2693.3}
# columns a --csv is allowed to carry, beyond any it keeps for its own use
INPUTS = ["LL", "PL", "PI", "fines_pct", "sand_pct", "energy_kJm3", "Gs"]
OPTIONAL = {"LL", "PL", "sand_pct"}
# the range each input spans in the training corpus; outside it the prediction
# is an extrapolation and is flagged as one rather than refused
RANGE = {"LL": (13.0, 152.8), "PL": (1.0, 49.2), "PI": (0.0, 126.2),
         "fines_pct": (1.5, 100.0), "sand_pct": (0.0, 98.5),
         "energy_kJm3": (355.2, 2693.3), "Gs": (2.30, 2.94)}


def load_xgboost():
    """The two boosters and the metadata recording how they were fitted."""
    meta_path = os.path.join(MODELS, "model_meta.json")
    if not os.path.exists(meta_path):
        sys.exit(f"no weights at {MODELS}; run `python scripts/train_model.py` first")
    with open(meta_path) as f:
        meta = json.load(f)
    import xgboost as xgb

    predictors = {}
    for target, name in meta["models"].items():
        m = xgb.XGBRegressor()
        m.load_model(os.path.join(MODELS, name))
        predictors[target] = m.predict
    return predictors, meta


def load_tabpfn():
    """TabPFN over the released corpus as context; no weights exist to load.

    The context is the dataset itself, so this reconstructs the model rather
    than restoring it. Expect minutes per call on CPU.
    """
    meta_path = os.path.join(MODELS, "tabpfn_meta.json")
    if not os.path.exists(meta_path):
        sys.exit(f"no {meta_path}; the TabPFN backend needs it for the context path")
    with open(meta_path) as f:
        meta = json.load(f)

    # the context first, so that a bad one fails at once rather than after the
    # seconds torch takes to import
    context = os.path.join(HERE, *meta["context"].split("/"))
    if not os.path.exists(context):
        sys.exit(f"no {meta['context']}; run `python scripts/build_source_pfn.py`")

    # the context is what the boosters' weight files are: change it and the model
    # changes, so it is checked against the digest recorded when it was built
    digest = hashlib.sha256(open(context, "rb").read()).hexdigest()
    if digest != meta.get("context_sha256"):
        sys.exit(f"{meta['context']} does not match the digest in tabpfn_meta.json; "
                 "the recorded accuracy would describe a different model. Rebuild "
                 "it with `python scripts/build_source_pfn.py`")

    d = pd.read_csv(context)
    if list(d.columns) != meta["context_columns"]:
        sys.exit(f"{meta['context']} carries {list(d.columns)}, expected "
                 f"{meta['context_columns']}")
    if len(d) != meta["context_records"]:
        sys.exit(f"{meta['context']} holds {len(d)} records, expected "
                 f"{meta['context_records']}; the recorded accuracy would not apply")
    # already prepared: log_energy is computed and the columns are in order
    X = d[meta["features"]].astype(float)

    # single-threaded BLAS and the large-dataset opt-in, both before torch is
    # imported: TabPFN segfaults on the larger folds otherwise
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
    try:
        from tabpfn import TabPFNRegressor
    except ImportError:
        sys.exit("the TabPFN backend needs `pip install -r requirements-tabpfn.txt`")

    def fit_and_predict(target):
        def go(Q):
            m = TabPFNRegressor(**meta["params"]).fit(X, d[target].astype(float))
            return m.predict(Q)
        return go

    print(f"TabPFN over {len(d)} context records from {meta['context']}; "
          "this takes minutes on CPU", file=sys.stderr)
    return {t: fit_and_predict(t) for t in ("MDD_Mgm3", "OMC_frac")}, meta


def load_models(kind):
    return load_tabpfn() if kind == "tabpfn" else load_xgboost()


def prepare(d, feats):
    """Input frame to model matrix: derive PI, take ln(energy), order the columns."""
    d = d.copy()
    for c in INPUTS:
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # PI is LL - PL on every record of the corpus, so derive it where it is absent
    derived = d.PI.isna() & d.LL.notna() & d.PL.notna()
    d.loc[derived, "PI"] = d.loc[derived, "LL"] - d.loc[derived, "PL"]

    missing = [c for c in INPUTS if c not in OPTIONAL and d[c].isna().any()]
    if missing:
        rows = sorted(set(np.where(d[missing].isna().any(axis=1))[0] + 1))
        sys.exit(f"rows {rows}: {', '.join(missing)} are required and cannot be blank")
    bad = d.PL > d.LL
    if bad.any():
        sys.exit(f"rows {sorted(np.where(bad)[0] + 1)}: PL above LL is not possible")

    d["log_energy"] = np.log(d.energy_kJm3)
    return d, d[feats].astype(float)


def flags(row):
    """Which inputs fall outside the range the corpus covers."""
    out = []
    for c, (lo, hi) in RANGE.items():
        v = row.get(c)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            if v < lo or v > hi:
                out.append(f"{c} outside {lo}-{hi}")
    return out


def predict(d, models, meta):
    """Predictions, the implied degree of saturation, and the admissibility check."""
    d, X = prepare(d, meta["features"])
    # to float64, so that rounding for display is exact rather than showing the
    # float32 the booster returns
    mdd = np.asarray(models["MDD_Mgm3"](X), dtype=float)
    omc = np.asarray(models["OMC_frac"](X), dtype=float)

    # degree of saturation implied at the peak, with rho_w = 1 Mg/m3; a pair
    # giving S_opt > 1 plots above the zero-air-voids line and cannot exist
    e = d.Gs.to_numpy() / mdd - 1.0
    s_opt = omc * d.Gs.to_numpy() / e

    out = pd.DataFrame({
        "MDD_Mgm3": np.round(mdd, 3),
        "OMC_pct": np.round(100.0 * omc, 2),
        "S_opt": np.round(s_opt, 3),
        "admissible": s_opt <= 1.0,
        "extrapolation": [", ".join(flags(r)) for _, r in d.iterrows()],
    })
    return d, out


def report(d, out, meta):
    """Human-readable output, one block per soil."""
    cv = meta["cross_validated"]
    grouped = meta.get("transfer", {}).get("grouped_5fold")
    for i, r in out.iterrows():
        label = d.get("name", pd.Series(dtype=object)).get(i) or f"soil {i + 1}"
        print(f"\n{label}")
        print(f"  MDD  {r.MDD_Mgm3:.3f} Mg/m3    "
              f"+/- {cv['MDD_Mgm3']['mae']:.3f} out-of-fold MAE")
        print(f"  OMC  {r.OMC_pct:.2f} %          "
              f"+/- {100 * cv['OMC_frac']['mae']:.2f} out-of-fold MAE")
        print(f"  degree of saturation at optimum {r.S_opt:.3f}"
              + ("" if r.admissible else "   INADMISSIBLE: above the zero-air-voids line"))
        if r.extrapolation:
            print(f"  extrapolation: {r.extrapolation}")
    if grouped:
        print(f"\nThe MAE above is the random-fold figure. For a soil from a source "
              f"the model\nhas not seen, expect {grouped['MDD_Mgm3']['mae']:.3f} Mg/m3 "
              f"and {100 * grouped['OMC_frac']['mae']:.2f} % instead "
              f"(models/*_meta.json, transfer).")


def main():
    p = argparse.ArgumentParser(
        description="Predict MDD and OMC with the released weights.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Efforts: " + ", ".join(f"{k} = {v} kJ/m3" for k, v in EFFORTS.items()))
    p.add_argument("--model", choices=["xgboost", "tabpfn"], default="xgboost",
                   help="xgboost (default) loads the released weights and returns "
                        "at once; tabpfn reproduces the paper's leading model "
                        "from the corpus as context, at minutes per call on CPU")
    p.add_argument("--csv", help="input file carrying the columns of INPUTS")
    p.add_argument("--out", help="write predictions here instead of to the screen")
    p.add_argument("--json", action="store_true", help="emit JSON on stdout")
    p.add_argument("--ll", type=float, help="liquid limit, %%; not a model input, used only to derive PI")
    p.add_argument("--pl", type=float, help="plastic limit, %%")
    p.add_argument("--pi", type=float, help="plasticity index, %%; derived from LL - PL if omitted")
    p.add_argument("--fines", type=float, help="fraction passing 0.075 mm, %%")
    p.add_argument("--sand", type=float, help="sand fraction, %%")
    p.add_argument("--gs", type=float, default=2.68,
                   help="specific gravity of solids; default 2.68, the corpus median")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--effort", choices=sorted(EFFORTS), help="Proctor standard")
    g.add_argument("--energy", type=float, help="compactive energy, kJ/m3")
    a = p.parse_args()

    models, meta = load_models(a.model)

    if a.csv:
        d = pd.read_csv(a.csv)
        if "energy_kJm3" not in d.columns and "test_standard" in d.columns:
            d["energy_kJm3"] = d.test_standard.map(EFFORTS)
    else:
        if a.fines is None:
            p.error("give --csv, or --fines together with an effort and the "
                    "properties you have")
        energy = a.energy if a.energy is not None else EFFORTS.get(a.effort)
        if energy is None:
            p.error("give --effort or --energy")
        d = pd.DataFrame([{"LL": a.ll, "PL": a.pl, "PI": a.pi, "fines_pct": a.fines,
                           "sand_pct": a.sand, "energy_kJm3": energy, "Gs": a.gs}])

    d, out = predict(d, models, meta)

    keep = [c for c in ("record_id", "name") if c in d.columns]
    result = pd.concat([d[keep + INPUTS].reset_index(drop=True), out], axis=1)

    if a.out:
        result.to_csv(a.out, index=False)
        n = int((~out.admissible).sum())
        print(f"wrote {a.out}: {len(result)} predictions"
              + (f", {n} above the zero-air-voids line" if n else ""))
    elif a.json:
        # via to_json so that a blank LL, PL or sand survives as null
        records = result.to_json(orient="records") or "[]"
        json.dump({"model": a.model,
                   "predictions": json.loads(records),
                   "accuracy_out_of_fold": meta["cross_validated"]},
                  sys.stdout, indent=2)
        print()
    else:
        report(d, out, meta)
        print(f"\nOut-of-fold accuracy of {a.model}, from the paper: "
              f"R2 {meta['cross_validated']['MDD_Mgm3']['r2']:.3f} for density, "
              f"{meta['cross_validated']['OMC_frac']['r2']:.3f} for the optimum.")


if __name__ == "__main__":
    main()
