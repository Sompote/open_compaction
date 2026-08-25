"""Refit the released weights from the released data.

The two boosters in `models/` are produced here: one per target, fitted on every
record of `data/compaction_parameters.csv` with the hyperparameters fixed a
priori in the paper, and saved in XGBoost's portable JSON format.

    python scripts/train_model.py

Nothing is held back, so the fit these weights achieve on the training records
is not an estimate of their accuracy. The honest figures are the out-of-fold
ones recorded in `models/model_meta.json`, and they come from five-fold
cross-validation reported in the paper, not from this script.
"""
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "compaction_parameters.csv")
MODELS = os.path.join(HERE, "models")

PARAMS = dict(n_estimators=600, learning_rate=0.04, max_depth=5, subsample=0.85,
              colsample_bytree=0.85, min_child_weight=5, reg_lambda=1.0,
              objective="reg:squarederror", random_state=0, n_jobs=8, verbosity=0)
FEATS = ["LL", "PL", "PI", "fines_pct", "sand_pct", "log_energy", "Gs"]
# out-of-fold performance of this configuration under random five-fold
# cross-validation, from the paper -- recorded in the metadata so the artefact
# never travels without its honest accuracy
CV = {"MDD_Mgm3": {"r2": 0.818, "mae": 0.0681, "rmse": 0.0929, "unit": "Mg/m3"},
      "OMC_frac": {"r2": 0.781, "mae": 0.0189, "rmse": 0.0270, "unit": "fraction"}}


def main():
    os.makedirs(MODELS, exist_ok=True)
    d = pd.read_csv(DATA)
    d["log_energy"] = np.log(d.energy_kJm3)
    X = d[FEATS].astype(float)

    meta = {"n_records": len(d), "features": FEATS, "params": PARAMS,
            "missing_by_design": {c: int(d[c].isna().sum())
                                  for c in FEATS if d[c].isna().any()},
            "cross_validated": CV, "models": {}}
    for target in ["MDD_Mgm3", "OMC_frac"]:
        m = xgb.XGBRegressor(**PARAMS).fit(X, d[target].astype(float))
        name = "model_mdd.json" if target == "MDD_Mgm3" else "model_omc.json"
        m.save_model(os.path.join(MODELS, name))
        meta["models"][target] = name
        print(f"models/{name:16} fitted on {len(d)} records, target {target}")

    with open(os.path.join(MODELS, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("models/model_meta.json  features, hyperparameters and out-of-fold scores")


if __name__ == "__main__":
    main()
