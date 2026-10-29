"""
Outpatient No-Show (DNA) Prediction - Modelling, Evaluation & Explainability

Run AFTER prep_noshow.py:
    pip install xgboost shap scikit-learn matplotlib
    python model_noshow.py

Outputs (in ./outputs):
    model_comparison.csv     - metrics for both models
    pr_curve.png             - precision-recall curves
    capture_by_decile.png    - "if we contact the top X%, how many no-shows do we reach?"
    shap_importance.png      - global feature importance (SHAP)
    shap_beeswarm.png        - direction & spread of each feature's effect
    xgb_noshow.json          - saved XGBoost model for the Gradio app
"""

import os
import json

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
import shap

from prep_noshow import FEATURES, TARGET, time_split

DATA_PATH = "noshow_features.pkl"
OUT_DIR = "outputs"
SEED = 42
os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. DATA
# -----------------------------------------------------------------------------
df = pd.read_pickle(DATA_PATH)
print("== Train vs test ==")
train_full, test = time_split(df, test_frac=0.2)

# Carve a time-based validation set out of the END of train for early stopping.
# The test set is only touched once, at the very end.
print("\n== Train vs validation (inside train) ==")
train, valid = time_split(train_full, test_frac=0.15)

X_train, y_train = train[FEATURES], train[TARGET]
X_valid, y_valid = valid[FEATURES], valid[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

numeric_cols = [c for c in FEATURES if c != "neighbourhood"]
prevalence = y_test.mean()
print(f"\nTest no-show rate (PR-AUC of a random model): {prevalence:.3f}\n")


# -----------------------------------------------------------------------------
# 2. EVALUATION HELPERS
# -----------------------------------------------------------------------------
def capture_at_top(y_true, scores, top_frac):
    """Share of all no-shows found if we act on the top X% highest-risk appointments."""
    n_top = int(np.ceil(len(scores) * top_frac))
    top_idx = np.argsort(-scores)[:n_top]
    return y_true.iloc[top_idx].sum() / y_true.sum()


def evaluate(name, y_true, scores):
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "pr_auc_lift_vs_random": average_precision_score(y_true, scores) / y_true.mean(),
        "brier": brier_score_loss(y_true, scores),
        "noshows_captured_top10pct": capture_at_top(y_true, scores, 0.10),
        "noshows_captured_top20pct": capture_at_top(y_true, scores, 0.20),
    }


# -----------------------------------------------------------------------------
# 3. BASELINE: LOGISTIC REGRESSION
# -----------------------------------------------------------------------------
logit = Pipeline([
    ("prep", ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), ["neighbourhood"]),
    ])),
    ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
])
logit.fit(X_train, y_train)
logit_scores = logit.predict_proba(X_test)[:, 1]


# -----------------------------------------------------------------------------
# 4. XGBOOST
# -----------------------------------------------------------------------------
# Note: no scale_pos_weight. Re-weighting classes distorts predicted probabilities,
# and for operational use (e.g. "70% risk") we want probabilities that mean something.
# Ranking metrics (ROC/PR-AUC) are unaffected by this choice.
xgb_model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="aucpr",
    early_stopping_rounds=100,
    enable_categorical=True,
    tree_method="hist",
    random_state=SEED,
)
xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
print(f"XGBoost best iteration: {xgb_model.best_iteration}")
xgb_scores = xgb_model.predict_proba(X_test)[:, 1]


# -----------------------------------------------------------------------------
# 5. COMPARE
# -----------------------------------------------------------------------------
results = pd.DataFrame([
    evaluate("Logistic Regression", y_test, logit_scores),
    evaluate("XGBoost", y_test, xgb_scores),
]).round(3)
print("\n", results.to_string(index=False), "\n")
results.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)

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
fig.savefig(f"{OUT_DIR}/pr_curve.png", dpi=150)
plt.close(fig)

# Capture curve - the chart an operational NHS audience will understand
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
fig.savefig(f"{OUT_DIR}/capture_by_decile.png", dpi=150)
plt.close(fig)


# -----------------------------------------------------------------------------
# 6. EXPLAINABILITY (SHAP)
# -----------------------------------------------------------------------------
# Sample the test set to keep SHAP fast; results are stable at this size.
shap_sample = X_test.sample(min(3000, len(X_test)), random_state=SEED)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(shap_sample)

plt.figure()
shap.plots.bar(shap_values, max_display=15, show=False)
plt.title("What drives no-show risk (mean |SHAP|)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/shap_importance.png", dpi=150)
plt.close()

plt.figure()
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.title("How each feature pushes risk up or down")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/shap_beeswarm.png", dpi=150)
plt.close()

# Plain-English explanation for one high-risk appointment (useful for the demo / call)
top_i = int(np.argmax(xgb_model.predict_proba(shap_sample)[:, 1]))
row = shap_values[top_i]
contrib = pd.Series(row.values, index=shap_sample.columns).sort_values(key=abs, ascending=False)
print("Example: highest-risk appointment in sample")
print(f"  Predicted no-show probability: {xgb_model.predict_proba(shap_sample.iloc[[top_i]])[0, 1]:.1%}")
for feat, val in contrib.head(5).items():
    direction = "raises" if val > 0 else "lowers"
    print(f"  {feat} = {shap_sample.iloc[top_i][feat]} {direction} risk")


# -----------------------------------------------------------------------------
# 7. SAVE
# -----------------------------------------------------------------------------
xgb_model.save_model(f"{OUT_DIR}/xgb_noshow.json")
with open(f"{OUT_DIR}/model_features.json", "w") as f:
    json.dump({"features": FEATURES,
               "neighbourhood_categories": list(df["neighbourhood"].cat.categories)}, f)
print(f"\nSaved model, metrics and charts to ./{OUT_DIR}/")
