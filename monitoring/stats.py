"""Drift and performance statistics. Pure functions over arrays and DataFrames, no app imports."""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

MONITORED_FEATURES = ["lead_time_days", "age", "booked_hour", "neighbourhood"]
CATEGORICAL = {"neighbourhood"}
PSI_BINS = 10
EPS = 1e-4          # floor for an empty bin's share, so the log in PSI stays finite
ECE_BINS = 10       # equal-count bins, same definition as calibrate_noshow.py


def psi(expected, actual):
    """Population Stability Index between two sets of bin shares:
    sum((actual - expected) * ln(actual / expected)). Common reading: <0.1 stable, 0.1-0.2 moderate
    shift, >0.2 significant shift."""
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


# --- numeric features: reference deciles ---------------------------------------------------------
def numeric_reference(values, bins=PSI_BINS):
    """Bin edges at the reference quantiles (duplicates dropped, e.g. lead time has many zeros) and
    the share of reference rows in each bin."""
    v = np.asarray(values, dtype=float)
    edges = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1]))
    return {"type": "numeric", "edges": [float(x) for x in edges], "proportions": _shares(v, edges).tolist()}


def _shares(values, edges):
    idx = np.searchsorted(np.asarray(edges, dtype=float), np.asarray(values, dtype=float), side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1)
    return counts / max(counts.sum(), 1)


# --- categorical features: share per training category, plus a bucket for anything unseen --------
def categorical_reference(values):
    s = pd.Series(values).astype(str)
    shares = s.value_counts(normalize=True)
    return {"type": "categorical", "categories": list(shares.index), "proportions": shares.tolist()}


def _cat_shares(values, categories):
    s = pd.Series(values).astype(str)
    known = s.value_counts().reindex(categories).fillna(0).to_numpy()
    unseen = len(s) - known.sum()
    return np.append(known, unseen) / max(len(s), 1)


def feature_psi(ref, values):
    if ref["type"] == "categorical":
        # The reference has no unseen bucket, so a new category shows up as drift (floored to EPS)
        return psi(np.append(ref["proportions"], 0.0), _cat_shares(values, ref["categories"]))
    return psi(ref["proportions"], _shares(values, ref["edges"]))


def build_reference(train: pd.DataFrame, valid_scores, base_rates: dict, model_features: list) -> dict:
    """Snapshot saved at training time (outputs/reference_stats.json).
    Feature shares come from the TRAINING rows. Score shares come from raw XGBoost scores on the
    validation period: scores on the rows the model was fitted on would be too optimistic a baseline."""
    feats = {}
    for f in MONITORED_FEATURES:
        feats[f] = categorical_reference(train[f]) if f in CATEGORICAL else numeric_reference(train[f])
    return {
        "rows": int(len(train)),
        "psi_bins": PSI_BINS,
        "features": feats,
        "score": {**numeric_reference(valid_scores), "basis": "raw XGBoost scores on the validation period"},
        "base_rate": base_rates,
        "model_features": model_features,
    }


# --- performance -------------------------------------------------------------------------------
def ece(y, p, bins=ECE_BINS):
    """Expected calibration error over equal-count risk bins: the size-weighted mean of
    |mean predicted - observed rate|."""
    y, p = np.asarray(y), np.asarray(p)
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q")
    return float((g.size() / len(p) * (g["p"].mean() - g["y"].mean()).abs()).sum())


def window_performance(y, p_calibrated, p_raw):
    """AUC on the raw score (Platt scaling is monotonic, so AUC is identical) and calibration on the
    displayed, calibrated probability."""
    y = np.asarray(y)
    both = y.min() != y.max()
    return {
        "rows": int(len(y)),
        "observed_rate": round(float(y.mean()), 4),
        "mean_predicted": round(float(np.mean(p_calibrated)), 4),
        "roc_auc": round(float(roc_auc_score(y, p_raw)), 4) if both else None,
        "ece": round(ece(y, p_calibrated), 4) if len(y) >= ECE_BINS * 20 else None,
        "brier": round(float(brier_score_loss(y, p_calibrated)), 4),
    }
