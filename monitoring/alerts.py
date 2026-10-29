"""Alert rules. Thresholds are policy choices (PSI 0.1 / 0.2 are the conventional cut-offs)."""

THRESHOLDS = {
    "psi_warn": 0.1,         # moderate shift
    "psi_alert": 0.2,        # significant shift
    "auc_min": 0.70,         # below this the ranking is no longer clearly useful (baseline 0.73)
    "ece_max": 0.05,         # calibration off by more than 5 points on average
    "base_rate_shift": 0.03, # no-show rate 3+ points away from the training rate
    # Drift is judged on the window accumulated so far, and only once it covers this many appointment
    # days. Single days are too sensitive: in the training data a lone Tuesday or Thursday already scores
    # PSI 0.55-0.75 on lead time against the all-days reference, because the weekday mix shapes booking lead time.
    "min_days_for_drift": 5,
}


def psi_status(value, t=THRESHOLDS):
    return "alert" if value > t["psi_alert"] else "warning" if value > t["psi_warn"] else "ok"


def evaluate(report: dict, t=THRESHOLDS) -> list:
    """Turn a monitoring report into a list of alerts, worst first. Each carries the date (or window)
    it was first seen so the page can show when a real system would have fired."""
    out = []

    def add(level, rule, subject, when, message, value):
        out.append({"level": level, "rule": rule, "subject": subject, "when": when,
                    "message": message, "value": value})

    ready = lambda rows: [d for d in rows if d["days"] >= t["min_days_for_drift"]]
    for feat in report["features"]:
        for d in ready(feat["cumulative"]):
            if d["psi"] > t["psi_warn"]:
                level = psi_status(d["psi"], t)
                add(level, "feature_psi", feat["label"], d["date"],
                    f"{feat['label']} drifted from the training data (PSI {d['psi']:.2f}, "
                    f"{'above 0.2' if level == 'alert' else 'above 0.1'}).", d["psi"])
                break  # first day it crossed a threshold is enough
    for d in ready(report["score"]["cumulative"]):
        if d["psi"] > t["psi_warn"]:
            level = psi_status(d["psi"], t)
            add(level, "score_psi", "Score distribution", d["date"],
                f"The score distribution moved away from the validation reference (PSI {d['psi']:.2f}).", d["psi"])
            break
    for w in report["performance"]["weekly"]:
        label = f"{w['start']} to {w['end']}"
        if w["roc_auc"] is not None and w["roc_auc"] < t["auc_min"]:
            add("alert", "auc_below_min", "ROC AUC", w["end"],
                f"ROC AUC {w['roc_auc']:.3f} for {label} is below {t['auc_min']:.2f}.", w["roc_auc"])
        if w["ece"] is not None and w["ece"] > t["ece_max"]:
            add("alert", "ece_above_max", "Calibration (ECE)", w["end"],
                f"Calibration error {w['ece']:.3f} for {label} is above {t['ece_max']:.2f}.", w["ece"])
        shift = abs(w["observed_rate"] - report["base_rate"]["train"])
        if shift > t["base_rate_shift"]:
            add("warning", "base_rate_shift", "No-show rate", w["end"],
                f"No-show rate {w['observed_rate']:.1%} for {label} is {shift * 100:.1f} points from training "
                f"({report['base_rate']['train']:.1%}); calibration may need refreshing.", shift)
    order = {"alert": 0, "warning": 1}
    return sorted(out, key=lambda a: (order[a["level"]], a["when"]))
