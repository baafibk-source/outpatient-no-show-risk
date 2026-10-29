"""
Outpatient No-Show (DNA) Prediction - Probability calibration

Run AFTER model_noshow.py:
    python calibrate_noshow.py

The raw XGBoost scores over-predict at the top end (riskiest tenth of the test
week: ~48% predicted vs ~38% observed). This fits Platt scaling on the
validation week that sits between training and the held-out test week, so the
test week is never used for fitting or for choosing the method.

Method choice: Platt vs isotonic was compared with leave-one-day-out Brier score
INSIDE the validation week (printed when you run this; Platt was marginally best). Platt
also preserves the ranking exactly and, being linear in log-odds, lets SHAP
contributions be rescaled exactly (calibrated contribution = slope * SHAP).

Caveat: the validation week was also used for XGBoost early stopping, so it is
not completely untouched - but it is disjoint from the held-out test week.

Output:
    outputs/calibration.json  - Platt slope/intercept + before/after metrics
"""

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from prep_noshow import FEATURES, TARGET, time_split

DATA_PATH = "noshow_features.parquet"
OUT_DIR = "outputs"

df = pd.read_parquet(DATA_PATH)
train_full, test = time_split(df, test_frac=0.2)
train, valid = time_split(train_full, test_frac=0.15)  # same split as model_noshow.py

with open(f"{OUT_DIR}/model_features.json") as f:
    categories = json.load(f)["neighbourhood_categories"]
model = xgb.XGBClassifier()
model.load_model(f"{OUT_DIR}/xgb_noshow.json")


def raw_scores(d):
    X = d[FEATURES].copy()
    X["neighbourhood"] = pd.Categorical(X["neighbourhood"].astype(str), categories=categories)
    return model.predict_proba(X)[:, 1]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ece(y, p, bins=10):
    """Expected calibration error over equal-count risk bins."""
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q")
    return float((g.size() / len(p) * (g["p"].mean() - g["y"].mean()).abs()).sum())


def top_decile(y, p):
    top = np.argsort(-p)[: len(p) // 10]
    return float(p[top].mean()), float(np.asarray(y)[top].mean())


p_valid = raw_scores(valid)
y_valid = valid[TARGET].values


def compare_methods():
    """Leave-one-day-out Brier score inside the validation week (test week untouched)."""
    days = valid["appointment_date"].values
    scores = {"platt": [], "isotonic": [], "raw": []}
    for d in np.unique(days):
        held, fit = days == d, days != d
        pl = LogisticRegression(C=1e6).fit(logit(p_valid[fit]).reshape(-1, 1), y_valid[fit])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_valid[fit], y_valid[fit])
        scores["platt"].append(brier_score_loss(y_valid[held], pl.predict_proba(logit(p_valid[held]).reshape(-1, 1))[:, 1]))
        scores["isotonic"].append(brier_score_loss(y_valid[held], iso.predict(p_valid[held])))
        scores["raw"].append(brier_score_loss(y_valid[held], p_valid[held]))
    return {k: round(float(np.mean(v)), 5) for k, v in scores.items()}


method_scores = compare_methods()
print("\nLeave-one-day-out Brier inside validation:", method_scores)

# Fit Platt scaling on the whole validation week
platt = LogisticRegression(C=1e6).fit(logit(p_valid).reshape(-1, 1), valid[TARGET])
slope, intercept = float(platt.coef_[0, 0]), float(platt.intercept_[0])


def calibrate(p):
    return 1 / (1 + np.exp(-(slope * logit(p) + intercept)))


def report(name, y, p_raw):
    out = {}
    for label, p in [("before", p_raw), ("after", calibrate(p_raw))]:
        pred, obs = top_decile(y, p)
        out[label] = {
            "brier": round(brier_score_loss(y, p), 4),
            "ece": round(ece(y, p), 4),
            "mean_predicted": round(float(p.mean()), 4),
            "top_decile_predicted": round(pred, 4),
            "top_decile_observed": round(obs, 4),
            "roc_auc": round(roc_auc_score(y, p), 4),
        }
    out["observed_rate"] = round(float(np.mean(y)), 4)
    print(f"\n{name}: observed rate {out['observed_rate']:.3f}")
    print(pd.DataFrame(out["before"], index=["before"]).T.join(pd.DataFrame(out["after"], index=["after"]).T))
    return out


result = {
    "method": "platt",
    "slope": round(slope, 6),
    "intercept": round(intercept, 6),
    "fitted_on": {
        "start": str(valid["appointment_date"].min().date()),
        "end": str(valid["appointment_date"].max().date()),
        "rows": int(len(valid)),
    },
    "method_choice_brier_leave_one_day_out": method_scores,
    "validation": report("Validation week (in-sample for the calibrator)", valid[TARGET].values, p_valid),
    "test": report("Held-out test week", test[TARGET].values, raw_scores(test)),
}
print(f"\nPlatt: calibrated log-odds = {slope:.4f} * raw log-odds + {intercept:.4f}")

with open(f"{OUT_DIR}/calibration.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved {OUT_DIR}/calibration.json")
