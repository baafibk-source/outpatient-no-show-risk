"""HTTP API. Paths and response shapes are unchanged from the single-file server."""

import logging
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

from .config import ACTIONS, HIGH_CUT, LOW_CUT, WEEKDAYS
from .data import capture_curve, lead_time_rates, repeat_summary
from .llm import LIMIT_NOTE, chat_messages, fallback_reminder, fallback_summary, \
    reminder_messages, summary_messages
from .schemas import AppointmentIn, Band, ChatIn, ContextIn, ReminderIn
from .state import AppState

log = logging.getLogger("dna")
router = APIRouter()


def S(request: Request) -> AppState:
    return request.app.state.dna


@router.get("/api/meta")
def get_meta(request: Request):
    s = S(request)
    return {
        "weekdays": WEEKDAYS,
        "neighbourhoods": s.scorer.categories,
        "default_neighbourhood": s.scorer.default_neighbourhood,
        "bands": {"low_below": LOW_CUT, "high_from": HIGH_CUT},
        "average_rate": s.scorer.avg_rate,
        "actions": ACTIONS,
        "llm_enabled": s.llm.enabled,
        "llm_model": s.llm.model if s.llm.enabled else None,
        "has_worklist": s.week is not None,
        "calibrated": s.calibration is not None,
    }


@router.get("/api/overview")
def get_overview(request: Request):
    s = S(request)
    week = s.need_week()
    t = week.test
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

    lead = lead_time_rates(week.full)
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
        "calibration_summary": s.calibration,
        "repeats": repeat_summary(t),
        "importance": s.importance[:10],
        "models": s.comparison,
    }


@router.get("/api/appointments")
def list_appointments(request: Request,
                      date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
                      neighbourhood: Optional[str] = Query(None, max_length=100),
                      band_filter: Optional[Band] = None, group_patients: bool = True,
                      limit: int = Query(200, ge=1, le=500), offset: int = Query(0, ge=0, le=1_000_000)):
    t = S(request).need_week().test
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


@router.get("/api/appointments/{appointment_id}")
def get_appointment(appointment_id: int, request: Request):
    s = S(request)
    test = s.need_week().test
    if appointment_id not in test.index:
        raise HTTPException(404, "Appointment is not in the held-out week")
    r = test.loc[[appointment_id]]
    out = s.scorer.explain(s.scorer.frame(r))
    row = r.iloc[0]
    siblings = test[(test["patient_id"] == row["patient_id"]) & (test["date"] == row["date"])]
    out.update({"id": appointment_id, "date": row["date"], "neighbourhood": row["neighbourhood"],
                "attended": not bool(row["no_show"]), "appts_that_day": int(len(siblings)),
                "missed_that_day": int(siblings["no_show"].sum())})
    return out


@router.post("/api/predict")
def predict(a: AppointmentIn, request: Request):
    s = S(request)
    return s.scorer.explain(s.scorer.row(a))


# --- LLM endpoints: the context is rebuilt on the server; template fallbacks need no token --------
@router.post("/api/summary")
def summary(body: ContextIn, request: Request):
    s = S(request)
    ctx = s.resolve_context(body)
    if not s.llm.enabled:
        return {"text": fallback_summary(ctx), "source": "template"}
    if not s.llm.allowed(request):
        return {"text": fallback_summary(ctx), "source": "template", "error": "rate_limited"}
    try:
        return {"text": s.llm.call(summary_messages(s.model_card, ctx), max_tokens=250), "source": "llm"}
    except Exception as e:
        log.warning("LLM summary failed: %s", type(e).__name__)
        return {"text": fallback_summary(ctx), "source": "template", "error": "unavailable"}


@router.post("/api/reminder")
def reminder(body: ReminderIn, request: Request):
    s = S(request)
    ctx = s.resolve_context(body.context)
    if not s.llm.enabled:
        return {"text": fallback_reminder(body.channel), "source": "template"}
    if not s.llm.allowed(request):
        return {"text": fallback_reminder(body.channel), "source": "template", "error": "rate_limited"}
    try:
        text = s.llm.call(reminder_messages(body.channel, ctx["band"]), max_tokens=250)
        return {"text": text.strip('"'), "source": "llm"}
    except Exception as e:
        log.warning("LLM reminder failed: %s", type(e).__name__)
        return {"text": fallback_reminder(body.channel), "source": "template", "error": "unavailable"}


@router.post("/api/chat")
def chat(body: ChatIn, request: Request):
    s = S(request)
    if not s.llm.enabled:
        return {"text": "The AI assistant isn't enabled on this server. Set an HF_TOKEN environment variable "
                        "to turn it on.", "source": "template"}
    ctx = s.resolve_context(body.context)
    if not s.llm.allowed(request):
        return {"text": LIMIT_NOTE + "\n\n" + fallback_summary(ctx), "source": "template", "error": "rate_limited"}
    try:
        return {"text": s.llm.call(chat_messages(s.model_card, ctx, body.messages), max_tokens=300), "source": "llm"}
    except Exception as e:
        log.warning("LLM chat failed: %s", type(e).__name__)
        return {"text": "The AI model isn't responding right now. Try again shortly.",
                "source": "template", "error": "unavailable"}
