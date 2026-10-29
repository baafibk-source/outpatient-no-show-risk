"""Shared fixtures: a tiny synthetic dataset and a tiny XGBoost model, so no test needs the Kaggle
data or the trained artefacts in outputs/."""
import json
import logging

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from fastapi.testclient import TestClient

from dashboard.llm import LLM, build_model_card
from dashboard.model import Scorer
from dashboard.data import load_test_week
from dashboard.server import create_app
from dashboard.state import AppState
from monitoring.stats import build_reference

FEATURES = ["lead_time_days", "same_day_booking", "appt_weekday", "booked_hour", "sms_received", "age",
            "hypertension", "diabetes", "alcoholism", "prior_appts", "prior_no_shows", "prior_no_show_rate",
            "is_first_appt", "days_since_last_appt", "multiple_appts_same_day", "neighbourhood"]
HOODS = ["Centro", "Jardim Camburi", "Other"]
TEST_START = "2016-06-01"


def synthetic_frame(n_per_day=250, seed=0):
    rng = np.random.default_rng(seed)
    days = [d for d in pd.date_range("2016-05-09", "2016-06-08") if d.day_of_week < 5]
    rows = []
    for d in days:
        lead = rng.choice([0, 1, 3, 7, 14, 30], n_per_day, p=[.3, .1, .1, .2, .2, .1])
        age = rng.integers(0, 90, n_per_day)
        p = 0.03 + 0.012 * np.minimum(lead, 25)           # no-shows rise with lead time
        rows.append(pd.DataFrame({
            "appointment_date": d, "lead_time_days": lead, "age": age,
            "no_show": (rng.random(n_per_day) < p).astype(int),
            "booked_hour": rng.integers(7, 18, n_per_day), "sms_received": rng.integers(0, 2, n_per_day),
            "hypertension": rng.integers(0, 2, n_per_day), "diabetes": rng.integers(0, 2, n_per_day),
            "alcoholism": 0, "prior_appts": 0, "prior_no_shows": 0, "prior_no_show_rate": -1.0,
            "is_first_appt": 1, "days_since_last_appt": -1, "multiple_appts_same_day": 0,
            "neighbourhood": rng.choice(HOODS, n_per_day),
        }))
    df = pd.concat(rows, ignore_index=True)
    df["same_day_booking"] = (df["lead_time_days"] == 0).astype(int)
    df["appt_weekday"] = df["appointment_date"].dt.day_of_week
    df["appointment_id"] = np.arange(1000, 1000 + len(df))
    df["patient_id"] = np.arange(len(df)) // 2          # two appointments per patient, some same day
    df["lead_time_band"] = pd.cut(df["lead_time_days"], [-1, 0, 7, 14, 30, 60, np.inf],
                                  labels=["same_day", "1-7d", "8-14d", "15-30d", "31-60d", "60d+"])
    df["neighbourhood"] = df["neighbourhood"].astype(pd.CategoricalDtype(HOODS))
    return df


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    df = synthetic_frame()
    train = df[df["appointment_date"] < "2016-05-25"]
    model = xgb.XGBClassifier(n_estimators=15, max_depth=3, enable_categorical=True, tree_method="hist", random_state=0)
    model.fit(train[FEATURES], train["no_show"])
    calibration = {"slope": 0.95, "intercept": -0.2, "fitted_on": {"start": "2016-05-25", "end": "2016-05-31"},
                   "test": {"after": {"top_decile_predicted": 0.4, "top_decile_observed": 0.37}}}
    scorer = Scorer(model, FEATURES, HOODS, calibration, avg_rate=float(df["no_show"].mean()))
    path = tmp_path_factory.mktemp("data") / "features.parquet"
    df.to_parquet(path, index=False)
    week = load_test_week(path, scorer, TEST_START)
    metrics = {
        "data": {"rows_after_cleaning": len(df), "overall_no_show_rate": float(df["no_show"].mean())},
        "split": {"train_rows": len(train), "test_rows": len(week.test), "test_start": TEST_START,
                  "test_end": "2016-06-08", "train_no_show_rate": 0.2, "valid_no_show_rate": 0.19, "test_no_show_rate": 0.18},
        "test": {"XGBoost": {"roc_auc": 0.7, "noshows_captured_top20pct": 0.4}},
    }
    valid = df[(df["appointment_date"] >= "2016-05-25") & (df["appointment_date"] < TEST_START)]
    reference = build_reference(train, model.predict_proba(valid[FEATURES])[:, 1],
                                {"train": 0.2, "valid": 0.19, "test": 0.18}, FEATURES)
    return {"df": df, "scorer": scorer, "week": week, "metrics": metrics, "calibration": calibration,
            "reference": reference}


def make_state(tiny, **over):
    args = dict(scorer=tiny["scorer"], metrics=tiny["metrics"], calibration=tiny["calibration"], week=tiny["week"],
                llm=LLM(), comparison=[{"model": "XGBoost", "roc_auc": 0.7, "pr_auc": 0.3,
                                         "noshows_captured_top20pct": 0.4}],
                model_card=build_model_card(tiny["metrics"], tiny["calibration"], None),
                reference=tiny["reference"], model_version="test-model")
    args.update(over)
    return AppState(**args)


@pytest.fixture()
def client(tiny):
    with TestClient(create_app(state=make_state(tiny))) as c:
        yield c


@pytest.fixture()
def client_factory(tiny):
    """Client with a customised state, e.g. a fake LLM or no worklist."""
    clients = []

    def build(**over):
        c = TestClient(create_app(state=make_state(tiny, **over)))
        c.__enter__()
        clients.append(c)
        return c
    yield build
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture()
def prediction_lines():
    """Capture the JSON prediction log lines (the logger does not propagate, so caplog can't see it)."""
    lines = []

    class H(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())
    h = H()
    logger = logging.getLogger("dna.predictions")
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
    yield lines
    logger.removeHandler(h)
