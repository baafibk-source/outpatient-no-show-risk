"""One structured JSON log line per prediction, with NO personal details: only a timestamp, the
calibrated score, the risk band, the model version and which screen asked. No appointment id, patient
id, inputs, neighbourhood or free text is ever logged."""

import json
import logging
import sys
import threading
from collections import Counter
from datetime import datetime, timezone

log = logging.getLogger("dna.predictions")


def setup():
    """JSON lines to stdout (what Hugging Face Spaces and most platforms collect). Idempotent."""
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


class LiveCounter:
    """In-memory tally of predictions since the app started (resets on restart)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.since = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.total = 0
        self.by_band = Counter()

    def add(self, band):
        with self._lock:
            self.total += 1
            self.by_band[band] += 1

    def snapshot(self):
        with self._lock:
            return {"since": self.since, "predictions": self.total,
                    "by_band": {b: self.by_band.get(b, 0) for b in ("High", "Medium", "Low")}}


def log_prediction(counter: LiveCounter, score: float, band: str, model_version: str, source: str):
    counter.add(band)
    log.info(json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": "prediction", "source": source,
        "score": round(float(score), 4), "band": band, "model_version": model_version,
    }, separators=(",", ":")))
