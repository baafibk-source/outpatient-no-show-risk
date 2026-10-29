# DNA Risk Dashboard

A web frontend for the outpatient no-show (Did Not Attend) model. It has three views:

- **Worklist**: the held-out final week (1–8 June 2016), scored by the XGBoost model. Pick a day and a neighbourhood to get appointments ranked by risk. Select one to see its SHAP explanation, the suggested contact, and a draft SMS or call script. Tick "Show what happened" to compare scores with what actually happened. Repeat bookings for the same patient on the same day are grouped into one row by default. The suggested contact depends on both the risk band and how far ahead the appointment was booked.
- **Risk check**: score a single made-up appointment, with live updates as you change any field.
- **Model**: capture curve, lead-time effect, calibration, feature importance, model comparison and limitations.

Proof of concept on public Kaggle data (Brazil, 2016). Not NHS data and not for clinical or operational use.

## Run

From the project root (the folder containing `outputs/` and `noshow_features.parquet`):

```bash
pip install -r dashboard/requirements.txt
uvicorn dashboard.server:app --reload
```

Open http://127.0.0.1:8000.

- `outputs/xgb_noshow.json` and `outputs/model_features.json` are required.
- `outputs/calibration.json` makes the scores read as probabilities. Create it with `python calibrate_noshow.py`, which fits Platt scaling on the week before the held-out week. Without it, the dashboard shows raw model scores.
- `noshow_features.parquet` powers the Worklist and Model views. Create it with `python prep_noshow.py`. Without it, only Risk check works.
- To turn on the AI assistant (staff summaries, reminders, Q&A), set `HF_TOKEN` (and optionally `LLM_MODEL`) before starting. Without a token, the dashboard falls back to template text.

## Structure

- `server.py`: FastAPI app. It loads the model, scores the test week once at startup, and serves the JSON API and static files.
- `static/`: plain HTML, CSS and JS. No build step.

| Endpoint | Purpose |
|---|---|
| `GET /api/meta` | Dropdown options, band cut-offs, LLM status |
| `GET /api/overview` | Week totals, per-day bands, capture curve, calibration, importance, lead-time rates |
| `GET /api/appointments?date=&neighbourhood=&band_filter=&group_patients=&limit=&offset=` | Ranked worklist (one row per patient per day by default) |
| `GET /api/appointments/{id}` | Risk and SHAP factors for one appointment |
| `POST /api/predict` | Score a custom appointment |
| `POST /api/summary`, `/api/reminder`, `/api/chat` | LLM explanations, with the same guardrails as the Gradio app |
