"""
Outpatient No-Show (DNA) Prediction - Modelling, Evaluation & Explainability

Run AFTER prep_noshow.py:
    pip install -r requirements.txt
    python model_noshow.py                    # train + evaluate + explain
    python model_noshow.py --drop-sensitive   # retrain without welfare, sex, disability and condition-count features
    python model_noshow.py --skip-shap        # skip the (slow) SHAP charts, e.g. in CI

Outputs (in ./outputs):
    model_comparison.csv     - metrics for both models
    metrics.json             - single source of truth for every number quoted in the app and README
    pr_curve.png             - precision-recall curves
    capture_by_decile.png    - "if we contact the top X%, how many no-shows do we reach?"
    shap_importance.png      - global feature importance (SHAP)            [unless --skip-shap]
    shap_beeswarm.png        - direction & spread of each feature's effect [unless --skip-shap]
    xgb_noshow.json          - saved XGBoost model used by the dashboard
    model_features.json      - feature list and neighbourhood categories used by the dashboard
    reference_stats.json     - training-time snapshot of monitored features and scores (for drift checks)

Calibration (Platt scaling) is fitted afterwards by calibrate_noshow.py.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

from monitoring.stats import build_reference
from prep_noshow import FEATURES, TARGET, time_split

DATA_PATH = "noshow_features.parquet"
SEED = 42

# Features that describe who the patient is rather than how the appointment was booked.
# n_conditions is included because it is a sum that contains the disability flag, so
# dropping has_handicap alone would leave that information in the model.
# See fairness_check.py for the accuracy cost and the effect on who gets flagged High.
SENSITIVE_FEATURES = ["welfare_programme", "is_male", "has_handicap", "n_conditions"]


# -----------------------------------------------------------------------------
# EVALUATION HELPERS
# -----------------------------------------------------------------------------
def capture_at_top(y_true, scores, top_frac):
    """Share of all no-shows found if we act on the top X% highest-risk appointments."""
    y = np.asarray(y_true)
    n_top = int(np.ceil(len(scores) * top_frac))
    top_idx = np.argsort(-np.asarray(scores))[:n_top]
    return y[top_idx].sum() / y.sum()


def evaluate(name, y_true, scores):
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "pr_auc_lift_vs_random": average_precision_score(y_true, scores) / np.mean(y_true),
        "brier": brier_score_loss(y_true, scores),
        "noshows_captured_top10pct": capture_at_top(y_true, scores, 0.10),
        "noshows_captured_top20pct": capture_at_top(y_true, scores, 0.20),
    }


def bootstrap_ci(y_true, scores, patient_ids, n_boot=300, seed=SEED):
    """95% intervals for ROC AUC and top-20% capture, resampling PATIENTS (not rows)
    because one patient can have several appointments."""
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y_true), np.asarray(scores)
    groups = pd.Series(np.arange(len(y))).groupby(np.asarray(patient_ids)).apply(lambda g: g.to_numpy()).to_list()
    aucs, caps = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
        caps.append(capture_at_top(y[idx], s[idx], 0.20))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    clo, chi = np.percentile(caps, [2.5, 97.5])
    return {"roc_auc_ci95": [round(lo, 3), round(hi, 3)], "capture_top20_ci95": [round(clo, 3), round(chi, 3)]}


def main(drop_sensitive=False, skip_shap=False, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    features = [f for f in FEATURES if not (drop_sensitive and f in SENSITIVE_FEATURES)]
    numeric_cols = [c for c in features if c != "neighbourhood"]

    # -------------------------------------------------------------------------
    # 1. DATA
    # -------------------------------------------------------------------------
    df = pd.read_parquet(DATA_PATH)
    print("== Train vs test ==")
    train_full, test = time_split(df, test_frac=0.2)

    # Carve a time-based validation set out of the END of train for early stopping
    # (and, later, for fitting the calibration). The test set is only touched once.
    print("\n== Train vs validation (inside train) ==")
    train, valid = time_split(train_full, test_frac=0.15)

    X_train, y_train = train[features], train[TARGET]
    X_valid, y_valid = valid[features], valid[TARGET]
    X_test, y_test = test[features], test[TARGET]
    prevalence = y_test.mean()
    print(f"\nTest no-show rate (PR-AUC of a random model): {prevalence:.3f}\n")

    # -------------------------------------------------------------------------
    # 2. BASELINE: LOGISTIC REGRESSION
    # -------------------------------------------------------------------------
    logit = Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), ["neighbourhood"]),
        ])),
        ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])
    logit.fit(X_train, y_train)
    logit_scores = logit.predict_proba(X_test)[:, 1]

    # -------------------------------------------------------------------------
    # 3. XGBOOST
    # -------------------------------------------------------------------------
    # No scale_pos_weight: re-weighting classes distorts predicted probabilities, and for
    # operational use (e.g. "35% risk") we want probabilities that mean something.
    xgb_model = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.03, max_depth=5, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        eval_metric="aucpr", early_stopping_rounds=100,
        enable_categorical=True, tree_method="hist", random_state=SEED,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    print(f"XGBoost best iteration: {xgb_model.best_iteration}")
    xgb_scores = xgb_model.predict_proba(X_test)[:, 1]

    # -------------------------------------------------------------------------
    # 4. COMPARE + SAVE METRICS
    # -------------------------------------------------------------------------
    results = pd.DataFrame([
        evaluate("Logistic Regression", y_test, logit_scores),
        evaluate("XGBoost", y_test, xgb_scores),
    ]).round(3)
    print("\n", results.to_string(index=False), "\n")
    results.to_csv(f"{out_dir}/model_comparison.csv", index=False)

    # How much does each block of features actually add? (honest context for the headline number)
    def quick_auc(cols):
        m = xgb.XGBClassifier(
            n_estimators=2000, learning_rate=0.03, max_depth=5, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            eval_metric="aucpr", early_stopping_rounds=100,
            enable_categorical=True, tree_method="hist", random_state=SEED,
        )
        m.fit(train[cols], y_train, eval_set=[(valid[cols], y_valid)], verbose=False)
        return round(float(roc_auc_score(y_test, m.predict_proba(test[cols])[:, 1])), 4)

    history_cols = ["prior_appts", "prior_no_shows", "prior_no_show_rate",
                    "is_first_appt", "days_since_last_appt", "multiple_appts_same_day"]
    ablations = {
        "xgboost_lead_time_only_roc_auc": quick_auc(["lead_time_days"]),
        "xgboost_without_history_features_roc_auc": quick_auc([c for c in features if c not in history_cols]),
        "xgboost_all_selected_features_roc_auc": round(float(roc_auc_score(y_test, xgb_scores)), 4),
    }
    print("Feature-block ablation:", ablations)

    metrics = {
        "data": {
            "rows_after_cleaning": int(len(df)),
            "patients": int(df["patient_id"].nunique()),
            "overall_no_show_rate": round(float(df[TARGET].mean()), 4),
            "date_range": [str(df["appointment_date"].min().date()), str(df["appointment_date"].max().date())],
        },
        "split": {
            "train_rows": int(len(train)), "valid_rows": int(len(valid)), "test_rows": int(len(test)),
            "train_end_exclusive": str(valid["appointment_date"].min().date()),
            "test_start": str(test["appointment_date"].min().date()),
            "test_end": str(test["appointment_date"].max().date()),
            "train_no_show_rate": round(float(train[TARGET].mean()), 4),
            "valid_no_show_rate": round(float(valid[TARGET].mean()), 4),
            "test_no_show_rate": round(float(prevalence), 4),
        },
        "features": features,
        "dropped_sensitive_features": SENSITIVE_FEATURES if drop_sensitive else [],
        "xgboost_best_iteration": int(xgb_model.best_iteration),
        "test": {r["model"]: {k: v for k, v in r.items() if k != "model"} for r in results.to_dict("records")},
        "xgboost_test_ci": bootstrap_ci(y_test, xgb_scores, test["patient_id"]),
        "ablations": ablations,
    }
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Precision-recall curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, s in [("Logistic Regression", logit_scores), ("XGBoost", xgb_scores)]:
        p, r, _ = precision_recall_curve(y_test, s)
        ax.plot(r, p, label=f"{name} (PR-AUC {average_precision_score(y_test, s):.3f})")
    ax.axhline(prevalence, ls="--", c="grey", label=f"Random ({prevalence:.3f})")
    ax.set(xlabel="Recall (share of no-shows found)", ylabel="Precision",
           title="Precision-recall on held-out later dates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/pr_curve.png", dpi=150)
    plt.close(fig)

    # Capture curve - the chart an operational audience will understand
    fracs = np.arange(0.1, 1.01, 0.1)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, s in [("Logistic Regression", logit_scores), ("XGBoost", xgb_scores)]:
        ax.plot(fracs * 100, [capture_at_top(y_test, s, f) * 100 for f in fracs], marker="o", label=name)
    ax.plot(fracs * 100, fracs * 100, ls="--", c="grey", label="Random targeting")
    ax.set(xlabel="% of appointments contacted (highest risk first)",
           ylabel="% of all no-shows reached",
           title="Who to contact: no-shows reached by targeting")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/capture_by_decile.png", dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 5. EXPLAINABILITY (SHAP)
    # -------------------------------------------------------------------------
    if not skip_shap:
        import shap  # imported here so the rest of the script (and CI) works without it

        shap_sample = X_test.sample(min(3000, len(X_test)), random_state=SEED)
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer(shap_sample)

        plt.figure()
        shap.plots.bar(shap_values, max_display=15, show=False)
        plt.title("What drives no-show risk (mean |SHAP|)")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/shap_importance.png", dpi=150)
        plt.close()

        plt.figure()
        shap.plots.beeswarm(shap_values, max_display=15, show=False)
        plt.title("How each feature pushes risk up or down")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/shap_beeswarm.png", dpi=150)
        plt.close()

        top_i = int(np.argmax(xgb_model.predict_proba(shap_sample)[:, 1]))
        row = shap_values[top_i]
        contrib = pd.Series(row.values, index=shap_sample.columns).sort_values(key=abs, ascending=False)
        print("Example: highest-risk appointment in sample")
        print(f"  Predicted no-show probability: {xgb_model.predict_proba(shap_sample.iloc[[top_i]])[0, 1]:.1%}")
        for feat, val in contrib.head(5).items():
            print(f"  {feat} = {shap_sample.iloc[top_i][feat]} {'raises' if val > 0 else 'lowers'} risk")

    # -------------------------------------------------------------------------
    # 6. SAVE
    # -------------------------------------------------------------------------
    xgb_model.save_model(f"{out_dir}/xgb_noshow.json")
    with open(f"{out_dir}/model_features.json", "w") as f:
        json.dump({"features": features,
                   "neighbourhood_categories": list(df["neighbourhood"].cat.categories)}, f)
    # Reference snapshot for drift monitoring: training-row feature shares and validation-period scores
    reference = build_reference(
        train, xgb_model.predict_proba(X_valid)[:, 1],
        base_rates={"train": metrics["split"]["train_no_show_rate"], "valid": metrics["split"]["valid_no_show_rate"],
                    "test": metrics["split"]["test_no_show_rate"]},
        model_features=features)
    with open(f"{out_dir}/reference_stats.json", "w") as f:
        json.dump(reference, f)
    print(f"\nSaved model, metrics, reference stats and charts to ./{out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train and evaluate the no-show model.")
    ap.add_argument("--drop-sensitive", action="store_true",
                    help="exclude welfare_programme, is_male, has_handicap and n_conditions from the model")
    ap.add_argument("--skip-shap", action="store_true", help="skip the SHAP charts")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()
    main(drop_sensitive=args.drop_sensitive, skip_shap=args.skip_shap, out_dir=args.out_dir)
