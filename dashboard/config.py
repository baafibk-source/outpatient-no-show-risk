"""Paths, constants and environment settings for the dashboard."""

import os
from pathlib import Path
from typing import Optional, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STATIC = HERE / "static"

# Fixed locations relative to this package only; never user input or the working directory
SEARCH_DIRS = [ROOT, ROOT / "outputs", HERE]


def find(fname: str, dirs: Optional[Sequence[Path]] = None) -> Optional[Path]:
    for folder in (dirs if dirs is not None else SEARCH_DIRS):
        path = Path(folder) / fname
        if path.exists():
            return path
    return None


def require(fname: str, dirs: Optional[Sequence[Path]] = None) -> Path:
    path = find(fname, dirs)
    if path is None:
        raise FileNotFoundError(f"Could not find {fname} in the project root or ./outputs")
    return path


WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

# Risk bands are operational policy choices, not something learned from the data, so they are fixed
# here. fairness_check.py uses the same cut-offs. Compared in whole percent, as the UI displays them.
LOW_PCT, HIGH_PCT = 20, 35
LOW_CUT, HIGH_CUT = LOW_PCT / 100, HIGH_PCT / 100

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
    # Only means "no earlier visit inside the six-week window of the data"
    "is_first_appt": "No earlier visit in data",
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

# API docs are off unless ENABLE_DOCS=1 (e.g. for local development)
ENABLE_DOCS = os.getenv("ENABLE_DOCS") == "1"

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_PER_IP_HOUR = int(os.getenv("LLM_PER_IP_HOUR", "10"))
LLM_GLOBAL_DAY = int(os.getenv("LLM_GLOBAL_DAY", "300"))

# Number of reverse proxies in front of the app that each append one entry to X-Forwarded-For.
# The per-IP LLM limit reads the entry that many places from the RIGHT, because the proxies add
# those entries and the client cannot spoof them. NOT verified for Hugging Face Spaces: its docs do
# not say how many hops there are, so 1 is an assumption. Check by logging the header on your Space,
# and set PROXY_HOPS accordingly. 0 means "no proxy": use the connection address and ignore the
# header. If the value is wrong the per-IP limit becomes too strict or too loose, but the global
# daily cap (LLM_GLOBAL_DAY) still bounds total cost.
PROXY_HOPS = int(os.getenv("PROXY_HOPS", "1"))
