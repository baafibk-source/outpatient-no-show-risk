# DNA Risk Dashboard

A web frontend for the outpatient no-show (Did Not Attend) model. It has three views:

- **Worklist**: the held-out final week (1–8 June 2016), scored by the XGBoost model. Pick a day and a neighbourhood to get appointments ranked by risk. Select one to see its SHAP explanation, the suggested contact, and a draft SMS or call script. Tick "Show what happened" to compare scores with what actually happened. Repeat bookings for the same patient on the same day are grouped into one row by default. The suggested contact depends on both the risk band and how far ahead the appointment was booked.
- **Risk check**: score a single made-up appointment, with live updates as you change any field.
- **Model**: capture curve, lead-time effect, calibration, feature importance, model comparison and limitations.

Proof of concept on public Kaggle data (Brazil, 2016). Not NHS data and not for clinical or operational use.

## Run

From the project root (the folder containing `outputs/`, `prep_noshow.py` and `noshow_features.parquet`):

```bash
pip install -r requirements.txt
uvicorn dashboard.server:app --reload
```

Open http://127.0.0.1:8000.

- `outputs/xgb_noshow.json`, `outputs/model_features.json` and `outputs/metrics.json` are required.
- `outputs/calibration.json` makes the scores read as probabilities. Create it with `python calibrate_noshow.py`, which fits Platt scaling on the week before the held-out week. Without it, the dashboard shows raw model scores.
- `noshow_features.parquet` powers the Worklist and Model views. Create it with `python prep_noshow.py`. Without it, only Risk check works.
- To turn on the AI assistant (staff summaries, reminders, Q&A), set `HF_TOKEN` (and optionally `LLM_MODEL`) before starting. Without a token, the dashboard falls back to template text.

## Structure

- `server.py`: `create_app()` builds the FastAPI app; artefacts load in a lifespan function (not at import), so tests can pass their own.
- `config.py` (paths, constants, env settings), `schemas.py` (request bodies and limits), `model.py` (scoring, bands, SHAP), `data.py` (held-out week), `llm.py` (assistant, templates, rate limits, model card), `state.py` (startup loading), `routes.py` (API).
- `static/`: plain HTML, CSS and JS. No build step.

Numbers shown in the app and quoted to the assistant come from `outputs/metrics.json` and `outputs/calibration.json`; the held-out start date is `split.test_start` in `metrics.json`. Risk bands (20% and 35%) are fixed operational thresholds in `config.py`.

| Endpoint | Purpose |
|---|---|
| `GET /api/meta` | Dropdown options, band cut-offs, LLM status |
| `GET /api/overview` | Week totals, per-day bands, capture curve, calibration, importance, lead-time rates |
| `GET /api/appointments?date=&neighbourhood=&band_filter=&group_patients=&limit=&offset=` | Ranked worklist (one row per patient per day by default) |
| `GET /api/appointments/{id}` | Risk and SHAP factors for one appointment |
| `POST /api/predict` | Score a custom appointment |
| `POST /api/summary`, `/api/reminder`, `/api/chat` | LLM explanations, with guardrails. Body: `{"appointment_id": 123}` or `{"appointment": {...}}` (inside `context` for reminder and chat). The server rebuilds the prediction itself; client-supplied factor text is not accepted |

Settings (environment): `HF_TOKEN`, `LLM_MODEL`, `LLM_PER_IP_HOUR`, `LLM_GLOBAL_DAY`, `ENABLE_DOCS`, and `PROXY_HOPS` (number of reverse proxies appending to `X-Forwarded-For`; used for the per-IP LLM limit. The right value for Hugging Face Spaces is not documented, so 1 is an assumption: check the header on your Space. `0` ignores the header).
