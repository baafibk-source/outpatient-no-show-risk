"""Build the monitoring report by REPLAYING the held-out days (a simulation, not live traffic)."""

import pandas as pd

from .alerts import THRESHOLDS, evaluate, psi_status
from .stats import MONITORED_FEATURES, feature_psi, psi, window_performance, _shares

SIMULATION_NOTE = ("This replays the held-out days (1-8 June 2016) one at a time against the "
                   "training-time reference. There is no live traffic behind it. It shows what the checks "
                   "would report, not how the deployed model is doing today.")


def build_report(reference: dict, test: pd.DataFrame, labels: dict, status_of=psi_status, thresholds=THRESHOLDS) -> dict:
    """`test` needs: appointment_date, no_show, raw_risk, risk and the monitored feature columns."""
    days = sorted(test["appointment_date"].dt.strftime("%Y-%m-%d").unique())
    day_of = test["appointment_date"].dt.strftime("%Y-%m-%d")

    def cumulative(fn):
        """PSI of everything replayed so far: day 1, days 1-2, ... Stable once a few days have accrued."""
        return [{"date": d, "days": i + 1, "psi": round(fn(test[day_of <= d]), 4)} for i, d in enumerate(days)]

    def judged(rows):
        """Status from the cumulative windows long enough to judge (see min_days_for_drift)."""
        vals = [r["psi"] for r in rows if r["days"] >= thresholds["min_days_for_drift"]]
        return status_of(max(vals), thresholds) if vals else "ok"

    features = []
    for f in MONITORED_FEATURES:
        ref = reference["features"][f]
        cum = cumulative(lambda g: feature_psi(ref, g[f]))
        by_day = [{"date": d, "psi": round(feature_psi(ref, test.loc[day_of == d, f]), 4)} for d in days]
        features.append({"feature": f, "label": labels.get(f, f), "psi": round(feature_psi(ref, test[f]), 4),
                         "status": judged(cum), "cumulative": cum, "by_day": by_day})

    sref = reference["score"]
    score = {
        "psi": round(psi(sref["proportions"], _shares(test["raw_risk"], sref["edges"])), 4),
        "basis": sref["basis"],
        "by_day": [{"date": d, "psi": round(psi(sref["proportions"], _shares(test.loc[day_of == d, "raw_risk"], sref["edges"])), 4),
                    "mean_score": round(float(test.loc[day_of == d, "risk"].mean()), 4)} for d in days],
    }
    score["cumulative"] = cumulative(lambda g: psi(sref["proportions"], _shares(g["raw_risk"], sref["edges"])))
    score["status"] = judged(score["cumulative"])

    # Calendar weeks (ISO). The held-out period spans only two part-weeks of three appointment days each
    iso = test["appointment_date"].dt.isocalendar()
    weekly = []
    for (yr, wk), g in test.groupby([iso["year"], iso["week"]]):
        perf = window_performance(g["no_show"], g["risk"], g["raw_risk"])
        weekly.append({"week": f"{yr}-W{int(wk):02d}", "start": g["appointment_date"].min().strftime("%Y-%m-%d"),
                       "end": g["appointment_date"].max().strftime("%Y-%m-%d"), "days": int(g["appointment_date"].nunique()),
                       **perf})
    daily = []
    for d in days:
        g = test[day_of == d]
        perf = window_performance(g["no_show"], g["risk"], g["raw_risk"])
        daily.append({"date": d, **perf})

    br = reference["base_rate"]
    report = {
        "simulation": True, "simulation_note": SIMULATION_NOTE,
        "window": {"start": days[0], "end": days[-1], "days": len(days), "rows": int(len(test))},
        "thresholds": thresholds,
        "features": features, "score": score,
        "performance": {"weekly": weekly, "daily": daily},
        "base_rate": br,
    }
    report["alerts"] = evaluate(report, thresholds)
    return report
