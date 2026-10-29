"""Everything the app loads at startup, kept on app.state instead of module globals so the app can be
built (and tested) with a different set of artefacts."""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional, Sequence
from pathlib import Path

import pandas as pd
import xgboost as xgb
from fastapi import HTTPException

from monitoring.prediction_log import LiveCounter
from monitoring.report import build_report

from .config import LABELS, SEARCH_DIRS, find, require
from .data import TestWeek, global_importance, lead_time_rates, load_test_week
from .llm import LLM, build_model_card
from .model import Scorer
from .schemas import ContextIn

log = logging.getLogger("dna")


@dataclass
class AppState:
    scorer: Scorer
    metrics: dict
    calibration: Optional[dict]
    week: Optional[TestWeek]
    llm: LLM
    comparison: list
    model_card: str
    reference: Optional[dict] = None
    model_version: str = "unversioned"
    live: LiveCounter = field(default_factory=LiveCounter)

    @cached_property
    def monitoring(self):
        """Drift/performance report from replaying the held-out days (cached: it never changes)."""
        if self.week is None or self.reference is None:
            raise HTTPException(503, "Monitoring needs noshow_features.parquet and outputs/reference_stats.json "
                                     "(run prep_noshow.py and model_noshow.py).")
        return build_report(self.reference, self.week.test, LABELS)

    @cached_property
    def importance(self):
        return global_importance(self.scorer, self.week.test)

    def need_week(self):
        if self.week is None:
            raise HTTPException(503, "noshow_features.parquet not found. Run prep_noshow.py to enable the worklist.")
        return self.week

    def resolve_context(self, ctx: ContextIn) -> dict:
        """Build the prediction an LLM prompt is about, on the server."""
        if ctx.appointment_id is not None:
            test = self.need_week().test
            if ctx.appointment_id not in test.index:
                raise HTTPException(404, "Appointment is not in the held-out week")
            return self.scorer.explain(self.scorer.frame(test.loc[[ctx.appointment_id]]))
        return self.scorer.explain(self.scorer.row(ctx.appointment))


def load_state(dirs: Optional[Sequence[Path]] = None, token: Optional[str] = None) -> AppState:
    dirs = dirs if dirs is not None else SEARCH_DIRS
    meta = json.loads(require("model_features.json", dirs).read_text())
    metrics = json.loads(require("metrics.json", dirs).read_text())
    cal_path = find("calibration.json", dirs)
    calibration = json.loads(cal_path.read_text()) if cal_path else None

    model_path = require("xgb_noshow.json", dirs)
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    ref_path = find("reference_stats.json", dirs)
    # Average no-show rate over all cleaned appointments, from the training script's own output
    scorer = Scorer(model, meta["features"], meta["neighbourhood_categories"], calibration,
                    avg_rate=metrics["data"]["overall_no_show_rate"])

    data_path = find("noshow_features.parquet", dirs)
    week = load_test_week(data_path, scorer, metrics["split"]["test_start"]) if data_path else None
    lead = None
    if week is not None:
        lead = [{"band": r.lead_time_band, "mean": float(r["mean"])} for _, r in lead_time_rates(week.full).iterrows()]

    comp_path = find("model_comparison.csv", dirs)
    return AppState(
        scorer=scorer, metrics=metrics, calibration=calibration, week=week,
        llm=LLM(token=token if token is not None else os.getenv("HF_TOKEN")),
        comparison=pd.read_csv(comp_path).to_dict(orient="records") if comp_path else [],
        model_card=build_model_card(metrics, calibration, lead),
        reference=json.loads(ref_path.read_text()) if ref_path else None,
        # Short content hash of the model file: changes whenever the model is retrained
        model_version="xgb-" + hashlib.sha256(model_path.read_bytes()).hexdigest()[:8],
    )
