"""Build `models/source_pfn.csv`, the context TabPFN predicts from.

XGBoost ships fitted weights. TabPFN cannot: it takes no gradient step on this
dataset, and predicts by supplying the training records as context and reading
the query off in a single forward pass. The context is therefore the artefact,
and this script builds it — the same role `train_model.py` plays for the
boosters.

    python scripts/build_source_pfn.py

The file holds exactly what the model is given: the six inputs in the order
`models/tabpfn_meta.json` records, with compactive energy already converted to
its natural logarithm, and the two targets. Nothing else, deliberately —
`record_id`, `source` and `group` are omitted so that no identifier can be fed
to the model as a feature by accident, and the liquid limit is omitted because
the paper's input set carries the plastic limit and the plasticity index only.

Row order follows `data/compaction_parameters.csv` exactly, which is how the two
are joined: `record_id` names a test rather than a row and is shared by 170 of
them, so it cannot be used as a key.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "compaction_parameters.csv")
MODELS = os.path.join(HERE, "models")
OUT = os.path.join(MODELS, "source_pfn.csv")

FEATURES = ["PL", "PI", "fines_pct", "sand_pct", "log_energy", "Gs"]
TARGETS = ["MDD_Mgm3", "OMC_frac"]
# out-of-fold performance of TabPFN on this context under random five-fold
# cross-validation, from the paper
CV = {"MDD_Mgm3": {"r2": 0.824, "mae": 0.0664, "rmse": 0.0913, "unit": "Mg/m3"},
      "OMC_frac": {"r2": 0.784, "mae": 0.0187, "rmse": 0.0268, "unit": "fraction"}}
# the same context scored under the two transfer-facing designs of the paper.
# CV above is an interpolation figure: under the random split 96.6 % of records
# share a provenance group with their training fold
TRANSFER = {
    "grouped_5fold": {
        "note": "folds blocked on the 162 provenance groups, same training-set size",
        "MDD_Mgm3": {"r2": 0.727, "mae": 0.0858, "rmse": 0.1136, "unit": "Mg/m3"},
        "OMC_frac": {"r2": 0.696, "mae": 0.0228, "rmse": 0.0318, "unit": "fraction"}},
    "leave_one_source_out": {
        "note": "each of the six sources predicted by a model trained on the other five",
        "MDD_Mgm3": {"r2": 0.520, "mae": 0.1162, "rmse": 0.1506, "unit": "Mg/m3"},
        "OMC_frac": {"r2": 0.614, "mae": 0.0276, "rmse": 0.0358, "unit": "fraction"}},
}
# predicted pairs implying a degree of saturation above the zero-air-voids line,
# out of 2,854, under each design
ZAV_VIOLATIONS = {"random_5fold": 1, "grouped_5fold": 0, "leave_one_source_out": 41}


def main():
    os.makedirs(MODELS, exist_ok=True)
    d = pd.read_csv(DATA)
    d["log_energy"] = np.log(d.energy_kJm3)

    context = d[FEATURES + TARGETS].astype(float)
    # the two documented gaps survive as blanks: TabPFN ingests them directly,
    # and imputing here would silently change the model the paper reports
    context.to_csv(OUT, index=False)

    digest = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
    meta_path = os.path.join(MODELS, "tabpfn_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["context"] = "models/source_pfn.csv"
    meta["context_records"] = len(context)
    meta["context_columns"] = FEATURES + TARGETS
    meta["context_sha256"] = digest
    # the context is the artefact, so its feature list, its documented gaps and
    # its honest accuracy are written here rather than maintained by hand
    meta["features"] = FEATURES
    meta["missing_by_design"] = {c: int(context[c].isna().sum())
                                 for c in FEATURES if context[c].isna().any()}
    meta["cross_validated"] = CV
    meta["transfer"] = TRANSFER
    meta["zav_violations"] = ZAV_VIOLATIONS
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    gaps = {c: int(context[c].isna().sum()) for c in FEATURES if context[c].isna().any()}
    print(f"models/source_pfn.csv   {len(context)} records, "
          f"{len(FEATURES)} inputs and {len(TARGETS)} targets")
    print(f"                        documented gaps carried through: {gaps}")
    print(f"                        sha256 {digest[:16]}... recorded in tabpfn_meta.json")


if __name__ == "__main__":
    main()
