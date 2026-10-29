"""LLM explanations (Hugging Face Inference Providers), with template fallbacks and abuse limits.

Prompts only ever contain facts the SERVER computed (see state.resolve_context): the client sends an
appointment id or validated inputs, never text that ends up in a prompt except the staff question."""

import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Optional

from fastapi import Request

from .config import ACTIONS, HIGH_PCT, LLM_GLOBAL_DAY, LLM_MODEL, LLM_PER_IP_HOUR, LOW_PCT, PROXY_HOPS

log = logging.getLogger("dna")

LIMIT_NOTE = "The AI assistant has reached its usage limit, so this is template text. Try again later."

GUARDRAILS = """Rules:
- Use ONLY the facts provided. Never invent numbers, factors or patient details.
- The risk score comes from the prediction model. Do not change or re-estimate it.
- Do not give clinical or medical advice.
- Keep language plain, respectful and non-judgemental about patients.
- If asked something the facts cannot answer, say so briefly."""


class LLM:
    def __init__(self, token: Optional[str] = None, model: str = LLM_MODEL, per_ip_hour: int = LLM_PER_IP_HOUR,
                 global_day: int = LLM_GLOBAL_DAY, proxy_hops: int = PROXY_HOPS, client=None):
        self.model = model
        self.per_ip_hour, self.global_day, self.proxy_hops = per_ip_hour, global_day, proxy_hops
        self.client = client
        if client is None and token:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(token=token, timeout=60)
        self._lock = threading.Lock()
        self._by_ip = defaultdict(deque)
        self._day = {"date": None, "count": 0}

    @property
    def enabled(self):
        return self.client is not None

    def client_ip(self, request: Request):
        # The rightmost entries are added by the proxies, so the client can't spoof them
        hops = [h.strip() for h in request.headers.get("x-forwarded-for", "").split(",") if h.strip()]
        if len(hops) >= self.proxy_hops > 0:
            return hops[-self.proxy_hops]
        return request.client.host if request.client else "unknown"

    def allowed(self, request: Request):
        """Take one LLM call from the per-IP hourly and global daily budgets, or return False."""
        now, today = time.time(), datetime.now(timezone.utc).date()
        ip = self.client_ip(request)
        with self._lock:
            if self._day["date"] != today:
                self._day.update(date=today, count=0)
                self._by_ip.clear()
            hits = self._by_ip[ip]
            while hits and hits[0] <= now - 3600:
                hits.popleft()
            if len(hits) >= self.per_ip_hour or self._day["count"] >= self.global_day:
                return False
            hits.append(now)
            self._day["count"] += 1
            return True

    def call(self, messages, max_tokens=350):
        # Extra token budget: reasoning models (e.g. gpt-oss) think before answering
        resp = self.client.chat_completion(messages=messages, model=self.model,
                                           max_tokens=max_tokens + 1200, temperature=0.3)
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise ValueError("empty response")
        return text


def build_model_card(metrics: dict, calibration: Optional[dict], lead_rates: Optional[list]) -> str:
    """The facts the assistant may quote about the model, built from outputs/metrics.json and
    outputs/calibration.json so they can never drift from the numbers the training scripts produced."""
    d, sp = metrics["data"], metrics["split"]
    xgb = metrics["test"]["XGBoost"]
    capture = xgb["noshows_captured_top20pct"]
    lines = [
        "About the underlying prediction model:",
        f"- XGBoost classifier trained on {sp['train_rows']:,} of {d['rows_after_cleaning']:,} cleaned public outpatient "
        f"appointments from Brazil (2016, Kaggle); {sp['test_rows']:,} later appointments "
        f"({sp['test_start']} to {sp['test_end']}) were held out for testing.",
        "- Not NHS data. A proof of concept, not a clinical or operational tool.",
        f"- Held-out results: ROC AUC {xgb['roc_auc']:.2f}; contacting the top 20% highest-risk appointments reaches "
        f"about {capture:.0%} of no-shows (about {capture / 0.2:.1f}x random).",
    ]
    if calibration:
        t, f = calibration["test"]["after"], calibration["fitted_on"]
        lines.append(f"- Scores are calibrated with Platt scaling fitted on the validation period ({f['start']} to {f['end']}), "
                     f"so they read as probabilities; the highest-scoring tenth is still predicted at "
                     f"{t['top_decile_predicted']:.0%} against {t['top_decile_observed']:.0%} observed.")
    lines.append(f"- Average no-show rate is about {d['overall_no_show_rate']:.0%} (it fell from {sp['train_no_show_rate']:.1%} "
                 f"in training to {sp['test_no_show_rate']:.1%} in the test period). "
                 f"Risk bands: Low under {LOW_PCT}%, Medium {LOW_PCT}-{HIGH_PCT}%, High {HIGH_PCT}% and above.")
    if lead_rates:
        same = next((r["mean"] for r in lead_rates if r["band"] == "same_day"), None)
        other = [r["mean"] for r in lead_rates if r["band"] not in ("same_day", "60d+")]
        if same is not None and other:
            lines.append(f"- Strongest driver overall is booking lead time (same-day bookings {same:.1%} no-show vs "
                         f"{min(other):.0%}-{max(other):.0%} when booked one or more days ahead).")
    lines += [
        "- Limitations: only about six weeks of data so attendance history is short; the SMS reminder effect is "
        "confounded with lead time; neighbourhood likely reflects deprivation or travel distance.",
        "- Scores should be used to offer patients extra support, never to penalise them.",
    ]
    return "\n".join(lines)


# --- Prompt text. `ctx` is the server-built explanation dict from Scorer.explain ---------------
def ctx_action(ctx):
    return ctx.get("action") or ACTIONS.get(ctx["band"], "")


def ctx_text(ctx):
    lines = [f"Predicted no-show risk: {ctx['risk']:.0%} ({ctx['band']} risk).",
             f"Suggested action (from the risk band and booking lead time): {ctx_action(ctx)}",
             "Main factors (from SHAP, most important first):"]
    lines += [f"- {f['label']} = {f['value']} ({'raises' if f['shap'] > 0 else 'lowers'} risk)" for f in ctx["factors"][:6]]
    return "\n".join(lines)


def fallback_summary(ctx):
    top = ctx["factors"][:6]
    ups = [f"{f['label'].lower()} ({f['value']})" for f in top if f["shap"] > 0][:3]
    downs = [f"{f['label'].lower()} ({f['value']})" for f in top if f["shap"] <= 0][:2]
    text = f"This appointment has a {ctx['risk']:.0%} predicted chance of a no-show ({ctx['band']} risk). "
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


def summary_messages(model_card, ctx):
    return [
        {"role": "system", "content": "You help outpatient clinic admin staff understand a no-show risk prediction.\n"
                                      + GUARDRAILS + "\n\n" + model_card},
        {"role": "user", "content": ctx_text(ctx) + "\n\nWrite a summary for clinic staff in 3-4 short sentences: "
                                    "the risk level, the two or three main reasons in plain English, "
                                    "then a final line starting 'Suggested action:'. No headings, no bullet points."},
    ]


def reminder_messages(channel, band):
    fmt = ("an SMS under 300 characters" if channel == "sms"
           else "a short, friendly phone call script for a receptionist (under 120 words)")
    tone = ("Because this appointment has a higher chance of being missed, gently make it easy to rebook "
            "and ask if anything might make attending difficult."
            if band in ("High", "Medium") else "Keep it a simple, standard reminder.")
    return [
        {"role": "system", "content": "You write appointment reminders on behalf of an outpatient clinic.\n"
         "Strict privacy rules: NEVER mention the risk score, predictions, past missed appointments, age, "
         "health conditions or any personal details. Patients must never feel judged.\n"
         "Use placeholders exactly like [Name], [Clinic], [Date], [Time], [Phone]. Warm, clear British English."},
        {"role": "user", "content": f"Write {fmt}. {tone} Return only the message text."},
    ]


def chat_messages(model_card, ctx, history):
    msgs = [{"role": "system", "content":
             "You are an assistant inside a no-show risk dashboard. Answer questions about the current "
             "prediction and how the model works, in 2-5 sentences.\n" + GUARDRAILS + "\n\n" + model_card
             + "\n\nCurrent prediction:\n" + ctx_text(ctx)}]
    return msgs + [{"role": m.role, "content": m.content} for m in history]
