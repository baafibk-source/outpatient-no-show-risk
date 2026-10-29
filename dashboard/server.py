"""
DNA (Did Not Attend) Risk Dashboard - API + static frontend
Proof of concept trained on the public Kaggle "Medical Appointment No Shows"
dataset (Brazil, 2016). Not NHS data and not for clinical or operational use.

Run from the project root:
    pip install -r requirements.txt
    uvicorn dashboard.server:app --reload
Then open http://127.0.0.1:8000

Files needed (looked up in ./, ./outputs, ./dashboard):
    xgb_noshow.json, model_features.json, metrics.json   (from model_noshow.py)
    calibration.json         (from calibrate_noshow.py - without it raw scores are shown)
    model_comparison.csv     (from model_noshow.py)
    noshow_features.parquet  (optional - powers the worklist and overview)

Modules: config (paths, constants, env), schemas (request bodies), model (scoring), data (test week),
llm (assistant), state (startup loading), routes (API).
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Sequence

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from monitoring.prediction_log import setup as setup_prediction_log

from .config import ENABLE_DOCS, STATIC
from .routes import router
from .state import AppState, load_state

log = logging.getLogger("dna")


def create_app(dirs: Optional[Sequence[Path]] = None, state: Optional[AppState] = None) -> FastAPI:
    """Build the app. Artefacts load in the lifespan (not at import), from `dirs`, unless a ready-made
    `state` is passed in, which is how tests supply a tiny model."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_prediction_log()
        app.state.dna = state if state is not None else load_state(dirs)
        yield

    app = FastAPI(title="DNA Risk Dashboard API", debug=False, lifespan=lifespan,
                  docs_url="/docs" if ENABLE_DOCS else None, redoc_url="/redoc" if ENABLE_DOCS else None,
                  openapi_url="/openapi.json" if ENABLE_DOCS else None)

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

    app.include_router(router)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app


app = create_app()
