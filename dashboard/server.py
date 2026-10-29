"""
DNA (Did Not Attend) Risk Dashboard - API + static frontend
Proof of concept trained on the public Kaggle "Medical Appointment No Shows"
dataset (Brazil, 2016). Not NHS data and not for clinical or operational use.

Run from the project root:
    pip install -r dashboard/requirements.txt
    uvicorn dashboard.server:app --reload
Then open http://127.0.0.1:8000

Files needed (looked up in ./, ./outputs, ../outputs):
    xgb_noshow.json, model_features.json, model_comparison.csv
    calibration.json     (from calibrate_noshow.py - without it raw scores are shown)
    noshow_features.parquet  (optional - powers the worklist and overview)
"""

import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from prep_noshow import to_other

log = logging.getLogger("dna")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def find(fname):
    # Fixed locations relative to this file only; never user input or the working directory
    for folder in [ROOT, ROOT / "outputs", HERE]:
        path = folder / fname
        if path.exists():
            return path
    return None


def require(fname):
    path = find(fname)
    if path is None:
        raise FileNotFoundError(f"Could not find {fname} in the project root or ./outputs")
    return path


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
with open(require("model_features.json")) as f:
    meta = json.load(f)
FEATURES = meta["features"]
NEIGHBOURHOODS = meta["neighbourhood_categories"]

model = xgb.XGBClassifier()
model.load_model(require("xgb_noshow.json"))
explainer = shap.TreeExplainer(model)

# Platt calibration (fitted on the pre-test validation week by calibrate_noshow.py).
# It is linear in log-odds, so SHAP contributions scale exactly by the slope.
_cal_path = find("calibration.json")
CALIBRATION = json.loads(_cal_path.read_text()) if _cal_path else None
CAL_SLOPE = CALIBRATION["slope"] if CALIBRATION else 1.0
CAL_INTERCEPT = CALIBRATION["intercept"] if CALIBRATION else 0.0


def raw_proba(X):
    return model.predict_proba(X)[:, 1]


def calibrate(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(CAL_SLOPE * np.log(p / (1 - p)) + CAL_INTERCEPT)))

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
AVG_RATE = 0.20
LOW_CUT, HIGH_CUT = 0.20, 0.35
LOW_PCT, HIGH_PCT = 20, 35  # the same cut-offs in whole percent, as the UI shows them


def pct_int(p):
    """Whole percent exactly as the UI displays it (round half up), so a band never contradicts the number shown."""
    return int(float(p) * 100 + 0.5)

LABELS = {
    "lead_time_days": "Days between booking and appointment",
    "same_day_booking": "Booked same day",
    "appt_weekday": "Appointment day",
    "booked_hour": "Hour booking was made",
    "sms_received": "SMS reminder sent",
    "age": "Age",
    "hypertension": "Hypertension",
    "diabetes": "Diabetes",
    "alcoholism": "Alcoholism",
    "prior_appts": "Previous appointments",
    "prior_no_shows": "Previous no-shows",
    "prior_no_show_rate": "Previous no-show rate",
    "is_first_appt": "First appointment",
    "days_since_last_appt": "Days since last appointment",
    "multiple_appts_same_day": "Several appointments that day",
    "neighbourhood": "Clinic neighbourhood",
}
BINARY = {"same_day_booking", "sms_received", "hypertension", "diabetes", "alcoholism",
          "is_first_appt", "multiple_appts_same_day"}

ACTIONS = {
    "High": "Phone the patient a few days before and offer to rebook if the time no longer suits.",
    "Medium": "Send an SMS or automated reminder a few days before the appointment.",
    "Low": "Standard process. No extra contact needed.",
}


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


def band(p):
    pct = pct_int(p)
    if pct >= HIGH_PCT:
        return "High"
    if pct >= LOW_PCT:
        return "Medium"
    return "Low"


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


def to_frame(rows: pd.DataFrame) -> pd.DataFrame:
    X = rows[FEATURES].copy()
    X["neighbourhood"] = pd.Categorical(X["neighbourhood"].astype(str), categories=NEIGHBOURHOODS)
    return X


def explain(X: pd.DataFrame):
    """Calibrated probability plus every feature's SHAP contribution for one row."""
    raw = float(raw_proba(X)[0])
    p = round(float(calibrate(raw)), 4)  # same precision the UI receives, so band and displayed % agree
    values = explainer(X)[0].values * CAL_SLOPE  # contributions in calibrated log-odds
    order = np.argsort(-np.abs(values))
    factors = [{
        "feature": FEATURES[i],
        "label": LABELS[FEATURES[i]],
        "value": fmt_value(FEATURES[i], X.iloc[0][FEATURES[i]]),
        "shap": round(float(values[i]), 4),
    } for i in order]
    level = band(p)
    return {"risk": round(p, 4), "raw_risk": round(raw, 4), "band": level,
            "action": action_for(level, X.iloc[0]["lead_time_days"]),
            "ratio_to_average": round(p / AVG_RATE, 2), "factors": factors}


# -----------------------------------------------------------------------------
# Held-out final week: scored once at startup, powers overview + worklist
# -----------------------------------------------------------------------------
def load_test_week():
    path = find("noshow_features.parquet")
    if path is None:
        return None, None
    df = pd.read_parquet(path)
    dates = np.sort(df["appointment_date"].unique())
    cutoff = dates[int(len(dates) * 0.8)]  # same split as prep_noshow.time_split
    test = df[df["appointment_date"] >= cutoff].copy()
    test["raw_risk"] = raw_proba(to_frame(test))
    test["risk"] = np.round(calibrate(test["raw_risk"]), 4)
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
    return df, test


FULL, TEST = load_test_week()


def repeat_summary():
    """How much the top of each day's list is taken up by repeat bookings of the same patient."""
    days = []
    for date, g in TEST.groupby("date"):
        top = g.head(20)
        days.append({"date": date, "top20_patients": int(top["patient_id"].nunique()),
                     "max_rows_one_patient": int(top["patient_id"].value_counts().max())})
    return {
        "patients_with_repeat_bookings": int(TEST.loc[TEST["appts_that_day"] > 1, "patient_id"].nunique()),
        "repeat_rows": int((~TEST["is_patient_top_row"]).sum()),
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


@lru_cache(maxsize=1)
def global_importance():
    sample = TEST.sample(min(3000, len(TEST)), random_state=42)
    vals = np.abs(explainer(to_frame(sample)).values * CAL_SLOPE).mean(axis=0)
    order = np.argsort(-vals)
    return [{"feature": FEATURES[i], "label": LABELS[FEATURES[i]], "mean_abs_shap": round(float(vals[i]), 4)}
            for i in order]


def comparison_table():
    path = find("model_comparison.csv")
    return pd.read_csv(path).to_dict(orient="records") if path else []


# -----------------------------------------------------------------------------
# LLM (Hugging Face Inference Providers). Falls back to templates without a token.
# -----------------------------------------------------------------------------
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
HF_TOKEN = os.getenv("HF_TOKEN")
llm_client = None
if HF_TOKEN:
    from huggingface_hub import InferenceClient
    llm_client = InferenceClient(token=HF_TOKEN, timeout=60)

MODEL_CARD = """About the underlying prediction model:
- XGBoost classifier trained on 110,527 public outpatient appointments from Brazil (2016, Kaggle).
- Not NHS data. A proof of concept, not a clinical or operational tool.
- Held-out results: ROC AUC 0.73; contacting the top 20% highest-risk appointments reaches about 38% of no-shows (about 1.9x random).
- Scores are calibrated with Platt scaling fitted on the week before the test week, so they read as probabilities; the highest scores still run a few points above observed rates.
- Average no-show rate is about 20%. Risk bands: Low under 20%, Medium 20-35%, High 35% and above.
- Strongest driver overall is booking lead time (same-day bookings 4.6% no-show vs 24-34% when booked one or more days ahead).
- Limitations: only about six weeks of data so attendance history is short; the SMS reminder effect is confounded with lead time; neighbourhood likely reflects deprivation or travel distance.
- Scores should be used to offer patients extra support, never to penalise them."""

GUARDRAILS = """Rules:
- Use ONLY the facts provided. Never invent numbers, factors or patient details.
- The risk score comes from the prediction model. Do not change or re-estimate it.
- Do not give clinical or medical advice.
- Keep language plain, respectful and non-judgemental about patients.
- If asked something the facts cannot answer, say so briefly."""


# Abuse limits for the paid LLM calls. In memory, so they reset when the Space restarts.
LLM_PER_IP_HOUR = int(os.getenv("LLM_PER_IP_HOUR", "10"))
LLM_GLOBAL_DAY = int(os.getenv("LLM_GLOBAL_DAY", "300"))
# Number of reverse proxies in front of the app that append to X-Forwarded-For
PROXY_HOPS = int(os.getenv("PROXY_HOPS", "1"))
_llm_lock = threading.Lock()
_llm_by_ip = defaultdict(deque)
_llm_day = {"date": None, "count": 0}
LIMIT_NOTE = "The AI assistant has reached its usage limit, so this is template text. Try again later."


def client_ip(request: Request):
    # The rightmost entries are added by the proxies, so the client can't spoof them
    hops = [h.strip() for h in request.headers.get("x-forwarded-for", "").split(",") if h.strip()]
    if len(hops) >= PROXY_HOPS > 0:
        return hops[-PROXY_HOPS]
    return request.client.host if request.client else "unknown"


def llm_allowed(request: Request):
    """Take one LLM call from the per-IP hourly and global daily budgets, or return False."""
    now, today = time.time(), datetime.now(timezone.utc).date()
    ip = client_ip(request)
    with _llm_lock:
        if _llm_day["date"] != today:
            _llm_day.update(date=today, count=0)
            _llm_by_ip.clear()
        hits = _llm_by_ip[ip]
        while hits and hits[0] <= now - 3600:
            hits.popleft()
        if len(hits) >= LLM_PER_IP_HOUR or _llm_day["count"] >= LLM_GLOBAL_DAY:
            return False
        hits.append(now)
        _llm_day["count"] += 1
        return True


def call_llm(messages, max_tokens=350):
    # Extra token budget: reasoning models (e.g. gpt-oss) think before answering
    resp = llm_client.chat_completion(messages=messages, model=LLM_MODEL,
                                      max_tokens=max_tokens + 1200, temperature=0.3)
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("empty response")
    return text


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


Band = Literal["Low", "Medium", "High"]


class Factor(Strict):
    label: str = Field(max_length=80)
    value: str = Field(max_length=80)
    shap: float = Field(ge=-50, le=50)


class Context(Strict):
    risk: float = Field(ge=0, le=1)
    band: Band
    action: Optional[str] = Field(None, max_length=300)
    factors: List[Factor] = Field(max_length=20)


def ctx_action(ctx: "Context"):
    return ctx.action or ACTIONS.get(ctx.band, "")


def ctx_text(ctx: Context):
    lines = [f"Predicted no-show risk: {ctx.risk:.0%} ({ctx.band} risk).",
             f"Suggested action (from the risk band and booking lead time): {ctx_action(ctx)}",
             "Main factors (from SHAP, most important first):"]
    lines += [f"- {f.label} = {f.value} ({'raises' if f.shap > 0 else 'lowers'} risk)" for f in ctx.factors[:6]]
    return "\n".join(lines)


def fallback_summary(ctx: Context):
    ups = [f"{f.label.lower()} ({f.value})" for f in ctx.factors[:6] if f.shap > 0][:3]
    downs = [f"{f.label.lower()} ({f.value})" for f in ctx.factors[:6] if f.shap <= 0][:2]
    text = f"This appointment has a {ctx.risk:.0%} predicted chance of a no-show ({ctx.band} risk). "
    if ups:
        text += "Risk is pushed up mainly by " + ", ".join(ups) + ". "
    if downs:
        text += "It is pulled down by " + ", ".join(downs) + ". "
    return text + f"\n\nSuggested action: {ctx_action(ctx)}"


def fallback_reminder(channel):
    if channel == "sms":
        return ("Hi [Name], this is a reminder of your appointment at [Clinic] on [Date] at [Time]. "
                "If you can no longer make it, please call [Phone] so we can offer the slot to someone else. Thank you.")
    return ("Hello, may I speak to [Name]? I'm calling from [Clinic] to remind you about your appointment on [Date] "
            "at [Time]. Is that time still convenient for you? If not, I can help you rebook now. "
            "Is there anything that might make it difficult to attend, such as travel? Thank you, we look forward to seeing you.")


# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------
# API docs are off unless ENABLE_DOCS=1 (e.g. for local development)
_docs = os.getenv("ENABLE_DOCS") == "1"
app = FastAPI(title="DNA Risk Dashboard API", debug=False,
              docs_url="/docs" if _docs else None, redoc_url="/redoc" if _docs else None,
              openapi_url="/openapi.json" if _docs else None)


@app.exception_handler(RequestValidationError)
async def invalid_input(request: Request, exc: RequestValidationError):
    # Don't echo the submitted values back (they can be huge, or NaN which JSON can't encode)
    errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg", "")} for e in exc.errors()[:20]]
    return JSONResponse({"detail": errors}, status_code=422)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    # Log the type only (messages can carry request details); never send a traceback to the client
    log.error("Unhandled %s on %s", type(exc).__name__, request.url.path)
    return JSONResponse({"detail": "Internal server error"}, status_code=500)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    # No X-Frame-Options / frame-ancestors: Hugging Face embeds the Space in an iframe
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    resp.headers.setdefault("Content-Security-Policy",
                            "script-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'")
    return resp


class AppointmentIn(Strict):
    lead_time_days: int = Field(14, ge=0, le=365)
    age: int = Field(35, ge=0, le=110)
    weekday: Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"] = "Tuesday"
    booked_hour: int = Field(10, ge=0, le=23)
    sms_received: bool = True
    conditions: List[Literal["Hypertension", "Diabetes", "Alcoholism"]] = Field([], max_length=3)
    prior_appts: int = Field(0, ge=0, le=500)
    prior_no_shows: int = Field(0, ge=0, le=500)
    days_since_last_appt: int = Field(0, ge=0, le=3650)
    multiple_appts_same_day: bool = False
    neighbourhood: Optional[str] = Field(None, max_length=100)


def build_row(a: AppointmentIn) -> pd.DataFrame:
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
        "neighbourhood": to_other(a.neighbourhood, NEIGHBOURHOODS),
    }
    return to_frame(pd.DataFrame([row]))


DEFAULT_HOOD = "Jardim Camburi" if "Jardim Camburi" in NEIGHBOURHOODS else NEIGHBOURHOODS[0]


def need_test_week():
    if TEST is None:
        raise HTTPException(503, "noshow_features.parquet not found. Run prep_noshow.py to enable the worklist.")


@app.get("/api/meta")
def get_meta():
    return {
        "weekdays": WEEKDAYS,
        "neighbourhoods": NEIGHBOURHOODS,
        "default_neighbourhood": DEFAULT_HOOD,
        "bands": {"low_below": LOW_CUT, "high_from": HIGH_CUT},
        "average_rate": AVG_RATE,
        "actions": ACTIONS,
        "llm_enabled": llm_client is not None,
        "llm_model": LLM_MODEL if llm_client else None,
        "has_worklist": TEST is not None,
        "calibrated": CALIBRATION is not None,
    }


@app.get("/api/overview")
def get_overview():
    need_test_week()
    t = TEST
    per_day = []
    for date, g in t.groupby("date"):
        counts = g["band"].value_counts()
        per_day.append({
            "date": date,
            "weekday": pd.Timestamp(date).day_name(),
            "appointments": int(len(g)),
            "expected_dna": round(float(g["risk"].sum()), 1),
            "actual_dna": int(g["no_show"].sum()),
            "high": int(counts.get("High", 0)),
            "medium": int(counts.get("Medium", 0)),
            "low": int(counts.get("Low", 0)),
        })

    lead = (FULL.groupby("lead_time_band", observed=True)["no_show"]
            .agg(["mean", "size"]).reset_index())
    lead_labels = {"same_day": "Same day", "1-7d": "1–7 days", "8-14d": "8–14 days",
                   "15-30d": "15–30 days", "31-60d": "31–60 days", "60d+": "60+ days"}

    # Calibration: predicted (before and after) vs observed rate by risk decile.
    # Platt is monotonic, so the deciles are the same appointments either way.
    deciles = pd.qcut(t["raw_risk"], 10, labels=False, duplicates="drop")
    calib = (t.groupby(deciles).agg(raw=("raw_risk", "mean"), predicted=("risk", "mean"),
                                    observed=("no_show", "mean"), n=("risk", "size")).reset_index(drop=True))

    return {
        "window": {"start": t["date"].min(), "end": t["date"].max()},
        "totals": {
            "appointments": int(len(t)),
            "expected_dna": round(float(t["risk"].sum())),
            "actual_dna": int(t["no_show"].sum()),
            "actual_rate": round(float(t["no_show"].mean()), 4),
            "high": int((t["band"] == "High").sum()),
            "medium": int((t["band"] == "Medium").sum()),
            "low": int((t["band"] == "Low").sum()),
            "high_band_rate": round(float(t.loc[t["band"] == "High", "no_show"].mean()), 4),
            "low_band_rate": round(float(t.loc[t["band"] == "Low", "no_show"].mean()), 4),
        },
        "per_day": per_day,
        "capture": capture_curve(t["no_show"].values, t["risk"].values),
        "lead_time": [{"band": lead_labels.get(r.lead_time_band, str(r.lead_time_band)),
                       "rate": round(float(r["mean"]), 4), "n": int(r["size"])}
                      for _, r in lead.iterrows()],
        "calibration": [{"raw": round(float(r.raw), 4), "predicted": round(float(r.predicted), 4),
                         "observed": round(float(r.observed), 4), "n": int(r.n)} for r in calib.itertuples()],
        "calibration_summary": CALIBRATION,
        "repeats": repeat_summary(),
        "importance": global_importance()[:10],
        "models": comparison_table(),
    }


@app.get("/api/appointments")
def list_appointments(date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
                      neighbourhood: Optional[str] = Query(None, max_length=100),
                      band_filter: Optional[Band] = None, group_patients: bool = True,
                      limit: int = Query(200, ge=1, le=500), offset: int = Query(0, ge=0, le=1_000_000)):
    need_test_week()
    t = TEST
    if date:
        t = t[t["date"] == date]
    if neighbourhood:
        t = t[t["neighbourhood"] == neighbourhood]
    if band_filter:
        t = t[t["band"] == band_filter]
    n_appointments, expected = int(len(t)), float(t["risk"].sum())
    counts = t["band"].value_counts()
    strip = t["risk"].values
    if group_patients:
        # One row per patient per day, at their highest-risk booking
        t = t[t["is_patient_top_row"]]
    page = t.iloc[offset: offset + limit]
    rows = [{
        "id": int(r.appointment_id),
        "date": r.date,
        "neighbourhood": r.neighbourhood,
        "age": int(r.age),
        "lead_time_days": int(r.lead_time_days),
        "sms_received": bool(r.sms_received),
        "prior_appts": int(r.prior_appts),
        "prior_no_shows": int(r.prior_no_shows),
        "risk": round(float(r.risk), 4),
        "band": r.band,
        "attended": not bool(r.no_show),
        "appts_that_day": int(r.appts_that_day),
    } for r in page.itertuples()]
    return {
        "total": int(len(t)),
        "appointments": n_appointments,
        "grouped": group_patients,
        "counts": {b: int(counts.get(b, 0)) for b in ["High", "Medium", "Low"]},
        "expected_dna": round(expected, 1),
        "risk_strip": [round(float(x), 3) for x in strip[:: max(1, len(strip) // 400)]],
        "rows": rows,
    }


@app.get("/api/appointments/{appointment_id}")
def get_appointment(appointment_id: int):
    need_test_week()
    if appointment_id not in TEST.index:
        raise HTTPException(404, "Appointment is not in the held-out week")
    r = TEST.loc[[appointment_id]]
    out = explain(to_frame(r))
    row = r.iloc[0]
    siblings = TEST[(TEST["patient_id"] == row["patient_id"]) & (TEST["date"] == row["date"])]
    out.update({"id": appointment_id, "date": row["date"], "neighbourhood": row["neighbourhood"],
                "attended": not bool(row["no_show"]), "appts_that_day": int(len(siblings)),
                "missed_that_day": int(siblings["no_show"].sum())})
    return out


@app.post("/api/predict")
def predict(a: AppointmentIn):
    return explain(build_row(a))


@app.post("/api/summary")
def summary(ctx: Context, request: Request):
    if llm_client is None:
        return {"text": fallback_summary(ctx), "source": "template"}
    if not llm_allowed(request):
        return {"text": fallback_summary(ctx), "source": "template", "error": "rate_limited"}
    msgs = [
        {"role": "system", "content": "You help outpatient clinic admin staff understand a no-show risk prediction.\n"
                                      + GUARDRAILS + "\n\n" + MODEL_CARD},
        {"role": "user", "content": ctx_text(ctx) + "\n\nWrite a summary for clinic staff in 3-4 short sentences: "
                                    "the risk level, the two or three main reasons in plain English, "
                                    "then a final line starting 'Suggested action:'. No headings, no bullet points."},
    ]
    try:
        return {"text": call_llm(msgs, max_tokens=250), "source": "llm"}
    except Exception as e:
        log.warning("LLM summary failed: %s", type(e).__name__)
        return {"text": fallback_summary(ctx), "source": "template", "error": "unavailable"}


class ReminderIn(Strict):
    context: Context
    channel: Literal["sms", "call"] = "sms"


@app.post("/api/reminder")
def reminder(body: ReminderIn, request: Request):
    if llm_client is None:
        return {"text": fallback_reminder(body.channel), "source": "template"}
    if not llm_allowed(request):
        return {"text": fallback_reminder(body.channel), "source": "template", "error": "rate_limited"}
    fmt = ("an SMS under 300 characters" if body.channel == "sms"
           else "a short, friendly phone call script for a receptionist (under 120 words)")
    tone = ("Because this appointment has a higher chance of being missed, gently make it easy to rebook "
            "and ask if anything might make attending difficult."
            if body.context.band in ("High", "Medium") else "Keep it a simple, standard reminder.")
    msgs = [
        {"role": "system", "content": "You write appointment reminders on behalf of an outpatient clinic.\n"
         "Strict privacy rules: NEVER mention the risk score, predictions, past missed appointments, age, "
         "health conditions or any personal details. Patients must never feel judged.\n"
         "Use placeholders exactly like [Name], [Clinic], [Date], [Time], [Phone]. Warm, clear British English."},
        {"role": "user", "content": f"Write {fmt}. {tone} Return only the message text."},
    ]
    try:
        return {"text": call_llm(msgs, max_tokens=250).strip('"'), "source": "llm"}
    except Exception as e:
        log.warning("LLM reminder failed: %s", type(e).__name__)
        return {"text": fallback_reminder(body.channel), "source": "template", "error": "unavailable"}


class ChatMessage(Strict):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatIn(Strict):
    context: Context
    messages: List[ChatMessage] = Field(min_length=1, max_length=8)

    # Staff questions are capped at 500 characters; assistant replies may be longer
    @model_validator(mode="after")
    def check_question(self):
        if any(m.role == "user" and len(m.content) > 500 for m in self.messages) or self.messages[-1].role != "user":
            raise ValueError("questions must be at most 500 characters and the last message must be a question")
        return self


@app.post("/api/chat")
def chat(body: ChatIn, request: Request):
    if llm_client is None:
        return {"text": "The AI assistant isn't enabled on this server. Set an HF_TOKEN environment variable "
                        "to turn it on.", "source": "template"}
    if not llm_allowed(request):
        return {"text": LIMIT_NOTE + "\n\n" + fallback_summary(body.context), "source": "template",
                "error": "rate_limited"}
    msgs = [{"role": "system", "content":
             "You are an assistant inside a no-show risk dashboard. Answer questions about the current "
             "prediction and how the model works, in 2-5 sentences.\n" + GUARDRAILS + "\n\n" + MODEL_CARD
             + "\n\nCurrent prediction:\n" + ctx_text(body.context)}]
    msgs += [{"role": m.role, "content": m.content} for m in body.messages]
    try:
        return {"text": call_llm(msgs, max_tokens=300), "source": "llm"}
    except Exception as e:
        log.warning("LLM chat failed: %s", type(e).__name__)
        return {"text": "The AI model isn't responding right now. Try again shortly.",
                "source": "template", "error": "unavailable"}


# -----------------------------------------------------------------------------
# Frontend
# -----------------------------------------------------------------------------
STATIC = HERE / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
