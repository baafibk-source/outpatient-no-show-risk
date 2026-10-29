"""Scoring: calibrated probability, risk band, suggested action and SHAP explanation."""

from typing import Sequence

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from prep_noshow import to_other

from .config import ACTIONS, BINARY, HIGH_PCT, LABELS, LOW_PCT, WEEKDAYS
from .schemas import AppointmentIn


def pct_int(p):
    """Whole percent exactly as the UI displays it (round half up), so a band never contradicts the number shown."""
    return int(float(p) * 100 + 0.5)


def band(p):
    pct = pct_int(p)
    if pct >= HIGH_PCT:
        return "High"
    if pct >= LOW_PCT:
        return "Medium"
    return "Low"


def action_for(level, lead_time_days):
    """Suggested contact. Short-notice bookings leave no time for a call 'days before'."""
    lead = int(lead_time_days)
    if level == "Low":
        return ACTIONS["Low"]
    if lead == 0:
        return ("Booked for today: call the patient now to confirm they can still come, and offer a later slot if not."
                if level == "High" else
                "Booked for today: send an SMS now confirming the time and how to cancel if plans have changed.")
    if lead == 1:
        return ("Booked for tomorrow: call the patient today to confirm, and offer to rebook if the time no longer suits."
                if level == "High" else
                "Booked for tomorrow: send an SMS today confirming the time and how to cancel.")
    if lead <= 3:
        return ("Short notice: phone the patient the day before and offer to rebook if the time no longer suits."
                if level == "High" else
                "Short notice: send an SMS the day before the appointment.")
    return ACTIONS[level]


def fmt_value(feat, val):
    if feat == "neighbourhood":
        return str(val)
    val = float(val)
    if feat == "appt_weekday":
        return WEEKDAYS[int(val)] if 0 <= int(val) < len(WEEKDAYS) else "Sunday"
    if feat == "prior_no_show_rate":
        return "n/a" if val < 0 else f"{val:.0%}"
    if feat == "days_since_last_appt" and val < 0:
        return "n/a"
    if feat in BINARY:
        return "Yes" if int(val) == 1 else "No"
    if feat == "lead_time_days":
        return f"{int(val)} days"
    return str(int(val)) if val == int(val) else f"{val:.2f}"


def to_frame(rows: pd.DataFrame, features: Sequence[str], categories: Sequence[str]) -> pd.DataFrame:
    X = rows[list(features)].copy()
    X["neighbourhood"] = pd.Categorical(X["neighbourhood"].astype(str), categories=list(categories))
    return X


def build_row(a: AppointmentIn, features: Sequence[str], categories: Sequence[str]) -> pd.DataFrame:
    """Model input for one made-up appointment. Unknown or rare neighbourhoods become 'Other'."""
    prior_no_shows = min(a.prior_no_shows, a.prior_appts)
    has = lambda c: int(c in a.conditions)
    row = {
        "lead_time_days": a.lead_time_days,
        "same_day_booking": int(a.lead_time_days == 0),
        "appt_weekday": WEEKDAYS.index(a.weekday) if a.weekday in WEEKDAYS else 1,
        "booked_hour": a.booked_hour,
        "sms_received": int(a.sms_received),
        "age": a.age,
        "hypertension": has("Hypertension"),
        "diabetes": has("Diabetes"),
        "alcoholism": has("Alcoholism"),
        "prior_appts": a.prior_appts,
        "prior_no_shows": prior_no_shows,
        "is_first_appt": int(a.prior_appts == 0),
        "prior_no_show_rate": prior_no_shows / a.prior_appts if a.prior_appts > 0 else -1.0,
        "days_since_last_appt": a.days_since_last_appt if a.prior_appts > 0 else -1,
        "multiple_appts_same_day": int(a.multiple_appts_same_day),
        "neighbourhood": to_other(a.neighbourhood, categories),
    }
    return to_frame(pd.DataFrame([row]), features, categories)


class Scorer:
    """The XGBoost model plus its Platt calibration. The calibration is linear in log-odds, so SHAP
    contributions scale exactly by the slope."""

    def __init__(self, model: xgb.XGBClassifier, features, categories, calibration=None, avg_rate=0.2):
        self.model = model
        self.features = list(features)
        self.categories = list(categories)
        self.calibration = calibration
        self.slope = calibration["slope"] if calibration else 1.0
        self.intercept = calibration["intercept"] if calibration else 0.0
        self.avg_rate = avg_rate
        self.explainer = shap.TreeExplainer(model)
        self.default_neighbourhood = "Jardim Camburi" if "Jardim Camburi" in self.categories else self.categories[0]

    def frame(self, rows: pd.DataFrame) -> pd.DataFrame:
        return to_frame(rows, self.features, self.categories)

    def row(self, a: AppointmentIn) -> pd.DataFrame:
        return build_row(a, self.features, self.categories)

    def raw_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def calibrate(self, p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return 1 / (1 + np.exp(-(self.slope * np.log(p / (1 - p)) + self.intercept)))

    def explain(self, X: pd.DataFrame):
        """Calibrated probability plus every feature's SHAP contribution for one row."""
        raw = float(self.raw_proba(X)[0])
        p = round(float(self.calibrate(raw)), 4)  # same precision the UI receives, so band and displayed % agree
        values = self.explainer(X)[0].values * self.slope  # contributions in calibrated log-odds
        order = np.argsort(-np.abs(values))
        factors = [{
            "feature": self.features[i],
            "label": LABELS[self.features[i]],
            "value": fmt_value(self.features[i], X.iloc[0][self.features[i]]),
            "shap": round(float(values[i]), 4),
        } for i in order]
        level = band(p)
        return {"risk": round(p, 4), "raw_risk": round(raw, 4), "band": level,
                "action": action_for(level, X.iloc[0]["lead_time_days"]),
                "ratio_to_average": round(p / self.avg_rate, 2), "factors": factors}
