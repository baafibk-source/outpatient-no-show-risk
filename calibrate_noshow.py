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

ECE (expected calibration error): scores are split into 10 equal-COUNT bins (deciles of the
predicted probability, via pd.qcut), not equal-width bins, so every bin has about the same
number of appointments even though most scores sit between 5% and 40%. ECE is the
size-weighted average of |mean predicted - observed rate| over those bins. Ten bins on ~26k
test rows is ~2,600 per bin, enough that sampling noise in each bin is about +/-1 point; the
absolute value of ECE depends on the binning, so compare it only against figures that use
the same 10 equal-count bins. The reliability diagram uses the same bins.

Output:
    outputs/calibration.json       - Platt slope/intercept + before/after metrics
    outputs/calibration_curve.png  - before/after reliability diagram on the held-out test week
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from monitoring.stats import ECE_BINS, ece
from prep_noshow import TARGET, time_split

DATA_PATH = "noshow_features.parquet"
OUT_DIR = "outputs"

df = pd.read_parquet(DATA_PATH)
train_full, test = time_split(df, test_frac=0.2)
train, valid = time_split(train_full, test_frac=0.15)  # same split as model_noshow.py

with open(f"{OUT_DIR}/model_features.json") as f:
    _meta = json.load(f)
categories = _meta["neighbourhood_categories"]
FEATURES = _meta["features"]  # whatever the model was trained on (e.g. without --drop-sensitive features)
model = xgb.XGBClassifier()
model.load_model(f"{OUT_DIR}/xgb_noshow.json")


def raw_scores(d):
    X = d[FEATURES].copy()
    X["neighbourhood"] = pd.Categorical(X["neighbourhood"].astype(str), categories=categories)
    return model.predict_proba(X)[:, 1]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


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


def reliability_diagram(y, p_raw, path):
    """Predicted vs observed no-show rate per equal-count bin, before and after Platt scaling.
    Platt is monotonic, so the bins hold the same appointments before and after."""
    y = np.asarray(y)
    q = pd.qcut(p_raw, ECE_BINS, labels=False, duplicates="drop")
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    top = 0
    for label, p, style in [("Before (raw XGBoost)", p_raw, dict(marker="o", color="#c2410c")),
                            ("After (Platt scaling)", calibrate(p_raw), dict(marker="s", color="#0f766e"))]:
        g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q").mean()
        ax.plot(g["p"], g["y"], lw=1.8, label=f"{label}, ECE {ece(y, p):.3f}", **style)
        top = max(top, g["p"].max(), g["y"].max())
    lim = float(np.ceil(top * 20) / 20)
    ax.plot([0, lim], [0, lim], ls="--", c="grey", label="Perfect calibration")
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Mean predicted no-show risk (per bin)",
           ylabel="Observed no-show rate (per bin)",
           title=f"Reliability on the held-out test week\n({ECE_BINS} equal-count bins, n={len(y):,})")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


result = {
    "method": "platt",
    "slope": round(slope, 6),
    "intercept": round(intercept, 6),
    "fitted_on": {
        "start": str(valid["appointment_date"].min().date()),
        "end": str(valid["appointment_date"].max().date()),
        "rows": int(len(valid)),
    },
    "ece_binning": {"bins": ECE_BINS, "strategy": "equal-count (quantile) bins of predicted risk"},
    "method_choice_brier_leave_one_day_out": method_scores,
    "validation": report("Validation week (in-sample for the calibrator)", valid[TARGET].values, p_valid),
    "test": report("Held-out test week", test[TARGET].values, raw_scores(test)),
}
print(f"\nPlatt: calibrated log-odds = {slope:.4f} * raw log-odds + {intercept:.4f}")

with open(f"{OUT_DIR}/calibration.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved {OUT_DIR}/calibration.json")

reliability_diagram(test[TARGET].values, raw_scores(test), f"{OUT_DIR}/calibration_curve.png")
print(f"Saved {OUT_DIR}/calibration_curve.png")
