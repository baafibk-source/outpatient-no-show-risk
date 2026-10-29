"""
Fairness check for the no-show model.

Two questions:
  1. Does the model behave similarly across groups (accuracy, calibration, share flagged High)?
  2. Do we lose anything by dropping the patient-characteristic features
     (welfare_programme, is_male, has_handicap, n_conditions)?

Run after prep_noshow.py:   python fairness_check.py
Writes outputs/fairness_by_group.csv and outputs/fairness_ablation.csv
"""
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from prep_noshow import FEATURES, TARGET, time_split
from model_noshow import SENSITIVE_FEATURES

DATA_PATH, OUT_DIR, SEED = "noshow_features.parquet", "outputs", 42
LOW_CUT, HIGH_CUT = 0.20, 0.35          # same risk bands as the dashboard
GROUPS = ["sex", "age_band", "welfare_programme", "has_handicap"]


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_and_score(features, train, valid, test):
    model = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.03, max_depth=5, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, eval_metric="aucpr",
        early_stopping_rounds=100, enable_categorical=True, tree_method="hist", random_state=SEED)
    model.fit(train[features], train[TARGET], eval_set=[(valid[features], valid[TARGET])], verbose=False)
    raw_valid = model.predict_proba(valid[features])[:, 1]
    raw_test = model.predict_proba(test[features])[:, 1]
    # Platt scaling fitted on the validation period, applied to test (same idea as calibrate_noshow.py)
    platt = LogisticRegression(C=1e6).fit(_logit(raw_valid).reshape(-1, 1), valid[TARGET])
    calibrated = platt.predict_proba(_logit(raw_test).reshape(-1, 1))[:, 1]
    return raw_test, calibrated


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_parquet(DATA_PATH)
    train_full, test = time_split(df, 0.2)
    train, valid = time_split(train_full, 0.15)

    # ---- ablation: what do the sensitive features buy us?
    variants = {
        "all features": FEATURES,
        "drop welfare_programme": [f for f in FEATURES if f != "welfare_programme"],
        "drop welfare + sex + disability + n_conditions": [f for f in FEATURES if f not in SENSITIVE_FEATURES],
    }
    welfare = test["welfare_programme"].to_numpy() == 1
    rows, scored = [], {}
    for name, feats in variants.items():
        raw, cal = fit_and_score(feats, train, valid, test)
        scored[name] = (raw, cal)
        high = cal >= HIGH_CUT - 0.005          # whole-percent rounding, as in the dashboard
        rows.append({
            "variant": name,
            "roc_auc": round(roc_auc_score(test[TARGET], raw), 4),
            "high_share_welfare": round(high[welfare].mean(), 3),
            "high_share_non_welfare": round(high[~welfare].mean(), 3),
            "observed_rate_in_high_band": round(test[TARGET].to_numpy()[high].mean(), 3),
        })
    ablation = pd.DataFrame(rows)
    ablation.to_csv(f"{OUT_DIR}/fairness_ablation.csv", index=False)
    print("\n== Ablation ==\n", ablation.to_string(index=False))

    # ---- group breakdown for the full model
    raw, cal = scored["all features"]
    t = test.assign(raw=raw, risk=cal)
    t["band"] = pd.cut(np.floor(t["risk"] * 100 + 0.5), [-1, LOW_CUT * 100, HIGH_CUT * 100, 101],
                       right=False, labels=["Low", "Medium", "High"])
    t["sex"] = np.where(t["is_male"] == 1, "Male", "Female")
    t["age_band"] = pd.cut(t["age"], [-1, 12, 17, 29, 44, 64, 120],
                           labels=["0-12", "13-17", "18-29", "30-44", "45-64", "65+"])
    out = []
    for col in GROUPS:
        for key, g in t.groupby(col, observed=True):
            if len(g) < 100:
                continue
            out.append({
                "attribute": col, "group": str(key), "n": len(g),
                "observed_rate": round(g[TARGET].mean(), 3),
                "mean_predicted": round(float(g["risk"].mean()), 3),
                "roc_auc": round(roc_auc_score(g[TARGET], g["raw"]), 3) if g[TARGET].nunique() > 1 else np.nan,
                "share_high": round(float((g["band"] == "High").mean()), 3),
            })
    by_group = pd.DataFrame(out)
    by_group.to_csv(f"{OUT_DIR}/fairness_by_group.csv", index=False)
    print("\n== By group (full model) ==\n", by_group.to_string(index=False))


if __name__ == "__main__":
    main()
