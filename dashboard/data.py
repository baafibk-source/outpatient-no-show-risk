"""The held-out test week: scored once at startup, powers the overview and worklist."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import HIGH_PCT, LABELS, LOW_PCT
from .model import Scorer


@dataclass
class TestWeek:
    full: pd.DataFrame   # every cleaned appointment (for lead-time rates)
    test: pd.DataFrame   # held-out rows with risk, band, date and repeat-booking columns

    __test__ = False  # not a pytest class


def load_test_week(path: Path, scorer: Scorer, test_start: str) -> TestWeek:
    """`test_start` comes from outputs/metrics.json, so the cut-off is the one training used."""
    df = pd.read_parquet(path)
    test = df[df["appointment_date"] >= pd.Timestamp(test_start)].copy()
    test["raw_risk"] = scorer.raw_proba(scorer.frame(test))
    test["risk"] = np.round(scorer.calibrate(test["raw_risk"]), 4)
    pct = np.floor(test["risk"].to_numpy() * 100 + 0.5)  # whole percent as displayed
    test["band"] = pd.cut(pct, [-1, LOW_PCT, HIGH_PCT, 101], right=False,
                          labels=["Low", "Medium", "High"]).astype(str)
    test["date"] = test["appointment_date"].dt.strftime("%Y-%m-%d")
    test["neighbourhood"] = test["neighbourhood"].astype(str)
    # Repeat bookings: several appointments for one patient on one day need one contact, not several
    grp = test.groupby(["patient_id", "date"])
    test["appts_that_day"] = grp["appointment_id"].transform("count")
    test = test.sort_values(["risk", "appointment_id"], ascending=[False, True])
    test["is_patient_top_row"] = ~test.duplicated(["patient_id", "date"])
    test = test.set_index("appointment_id", drop=False)
    return TestWeek(full=df, test=test)


def repeat_summary(test: pd.DataFrame):
    """How much the top of each day's list is taken up by repeat bookings of the same patient."""
    days = []
    for date, g in test.groupby("date"):
        top = g.head(20)
        days.append({"date": date, "top20_patients": int(top["patient_id"].nunique()),
                     "max_rows_one_patient": int(top["patient_id"].value_counts().max())})
    return {
        "patients_with_repeat_bookings": int(test.loc[test["appts_that_day"] > 1, "patient_id"].nunique()),
        "repeat_rows": int((~test["is_patient_top_row"]).sum()),
        "top20_by_day": days,
    }


def capture_curve(y, s, steps=20):
    order = np.argsort(-s)
    y_sorted = np.asarray(y)[order]
    total = y_sorted.sum()
    out = []
    for k in range(steps + 1):
        frac = k / steps
        n = int(np.ceil(len(s) * frac))
        out.append({"contacted": round(frac, 3), "reached": round(float(y_sorted[:n].sum() / total), 4)})
    return out


def global_importance(scorer: Scorer, test: pd.DataFrame):
    sample = test.sample(min(3000, len(test)), random_state=42)
    vals = np.abs(scorer.explainer(scorer.frame(sample)).values * scorer.slope).mean(axis=0)
    order = np.argsort(-vals)
    return [{"feature": scorer.features[i], "label": LABELS[scorer.features[i]],
             "mean_abs_shap": round(float(vals[i]), 4)} for i in order]


def lead_time_rates(full: pd.DataFrame):
    return (full.groupby("lead_time_band", observed=True)["no_show"].agg(["mean", "size"]).reset_index())
