"""Check the released dataset and weights against the invariants in the README.

Run from the repository root:

    python scripts/validate_corpus.py

Exits non-zero if any invariant fails, so it can be used in continuous
integration. The weights section is skipped, not failed, where xgboost is
absent, so the dataset can be validated without it.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = ["LL", "PL", "PI", "fines_pct", "sand_pct", "energy_kJm3", "Gs"]
# what the boosters were fitted on: energy as its natural logarithm
MODEL_FEATURES = ["LL", "PL", "PI", "fines_pct", "sand_pct", "log_energy", "Gs"]
fails = []


def check(condition, message):
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        fails.append(message)


def main():
    d = pd.read_csv(os.path.join(HERE, "data", "compaction_parameters.csv"))
    print(f"modelling dataset: {len(d)} records, {d.shape[1]} columns\n")

    check(len(d) == 2854, "2,854 records")
    check(list(d.columns)[:4] == ["record_id", "source", "group", "test_standard"],
          "provenance columns lead the schema")
    check(d.MDD_Mgm3.notna().all() and d.OMC_frac.notna().all(),
          "no missing target values")
    check(d.record_id.notna().all(), "every record carries an identifier")
    # record_id names a test, not a row: 78 ids are shared by 170 rows, almost
    # all LTPP repeat tests on one pavement layer. Joining on it multiplies
    # rows, so the invariant asserted here is the true one
    shared = int(d.record_id.duplicated(keep=False).sum())
    check(shared == 170, f"170 rows share an identifier with another: {shared}")
    check(d.record_id.nunique() == 2762, "2,762 distinct identifiers")

    # the two documented gaps, and no others
    gaps = {c: int(d[c].isna().sum()) for c in FEATURES if d[c].isna().any()}
    check(gaps == {"LL": 134, "PL": 134, "sand_pct": 169},
          f"only the documented gaps are present: {gaps}")

    # non-plastic soils carry no limits, and nothing else does
    npl = d.PI == 0
    check(int(npl.sum()) == 134, "134 non-plastic records")
    check(d.loc[npl, ["LL", "PL"]].isna().all().all(),
          "non-plastic records carry no consistency limits")
    check(d.loc[~npl, ["LL", "PL"]].notna().all().all(),
          "every plastic record carries both limits")
    check(int((d.PL > d.LL).sum()) == 0, "no record reports PL above LL")

    # physical admissibility
    e = d.Gs / d.MDD_Mgm3 - 1.0
    s_opt = d.OMC_frac * d.Gs / e
    check(bool(((s_opt >= 0.20) & (s_opt <= 1.0)).all()),
          "every record satisfies 0.20 <= S_opt <= 1.0")
    check(bool(d.MDD_Mgm3.between(0.8, 2.6).all()), "MDD within 0.8-2.6 Mg/m3")
    check(bool(d.OMC_frac.between(0.01, 0.7).all()), "OMC within 1-70 %")

    # compactive energy
    levels = sorted(d.energy_kJm3.unique())
    check(levels == [355.2, 592.5, 1346.6, 2693.3],
          f"four Proctor energy levels present: {levels}")
    check(int((d.energy_kJm3 != 592.5).sum()) == 101,
          "101 records at non-standard effort")

    # provenance
    check(d.group.nunique() == 162, "162 provenance groups")
    check(d.source.nunique() == 6, "six contributing sources")

    weights(d)

    print()
    if fails:
        print(f"{len(fails)} check(s) failed")
        sys.exit(1)
    print("all checks passed")


def tabpfn_context(d, models):
    """`models/source_pfn.csv` must be the dataset, prepared and nothing more.

    TabPFN has no weights, so this file is its artefact: if it drifts from the
    dataset, the accuracy recorded beside it describes a different model.
    """
    tab_path = os.path.join(models, "tabpfn_meta.json")
    if not os.path.exists(tab_path):
        return
    with open(tab_path) as f:
        tab = json.load(f)
    check(tab["features"] == MODEL_FEATURES,
          "the TabPFN backend expects the same inputs in the same order")
    check(tab["context_records"] == len(d),
          "the TabPFN context is every record of the modelling dataset")

    path = os.path.join(HERE, *tab["context"].split("/"))
    if not os.path.exists(path):
        check(False, f"the TabPFN context file {tab['context']} is present")
        return
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    check(digest == tab.get("context_sha256"),
          f"{tab['context']} matches the digest recorded in tabpfn_meta.json")

    c = pd.read_csv(path)
    check(list(c.columns) == tab["context_columns"],
          f"{tab['context']} carries the seven inputs and two targets, in order")
    check(len(c) == len(d), f"{tab['context']} holds {len(d)} records")
    # no identifier may reach the model as a feature
    check(not ({"record_id", "source", "group", "test_standard"} & set(c.columns)),
          f"{tab['context']} carries no identifier column")

    if len(c) == len(d) and list(c.columns) == tab["context_columns"]:
        want = d.copy()
        want["log_energy"] = np.log(want.energy_kJm3)
        same = all(np.allclose(c[col].to_numpy(dtype=float),
                               want[col].to_numpy(dtype=float),
                               rtol=0, atol=1e-12, equal_nan=True)
                   for col in tab["context_columns"])
        check(same, f"{tab['context']} reproduces the dataset row for row")


def weights(d):
    """The released boosters load, expect the documented inputs, and predict."""
    print("\nreleased weights")
    models = os.path.join(HERE, "models")
    meta_path = os.path.join(models, "model_meta.json")
    if not os.path.exists(meta_path):
        print("  skip  no models/ directory; run scripts/train_model.py")
        return
    try:
        import xgboost as xgb
    except ImportError:
        print("  skip  xgboost not installed")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    tabpfn_context(d, models)

    check(meta["features"] == MODEL_FEATURES,
          f"weights expect the documented inputs in order: {meta['features']}")
    check(meta["n_records"] == len(d), "weights were fitted on every record")
    check(set(meta["models"]) == {"MDD_Mgm3", "OMC_frac"},
          "one set of weights per target")

    d = d.copy()
    d["log_energy"] = np.log(d.energy_kJm3)
    X = d[MODEL_FEATURES].astype(float)
    for target, name in meta["models"].items():
        path = os.path.join(models, name)
        if not os.path.exists(path):
            check(False, f"models/{name} is present")
            continue
        m = xgb.XGBRegressor()
        m.load_model(path)
        p = np.asarray(m.predict(X), dtype=float)
        y = d[target].astype(float).to_numpy()
        r2 = 1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        # in-sample, and therefore not an accuracy: the threshold is set to
        # catch weights wired to the wrong columns, which score 0.68 or worse
        # under any permutation of the seven, not to judge the model
        check(r2 > 0.85, f"models/{name} reproduces its training target, R2 {r2:.3f}")
        check(meta["cross_validated"][target]["r2"] < r2,
              f"the recorded out-of-fold R2 for {target} is below the in-sample fit")


if __name__ == "__main__":
    main()
