"""PSI, drift on the fixture data, performance windows and alert rules."""
import copy

import numpy as np
import pandas as pd

from monitoring.alerts import THRESHOLDS, evaluate, psi_status
from monitoring.stats import (categorical_reference, ece, feature_psi, numeric_reference, psi, window_performance)


def test_psi_is_zero_for_identical_shares_and_grows_with_shift():
    p = [0.25, 0.25, 0.25, 0.25]
    assert psi(p, p) == 0
    assert 0 < psi(p, [0.3, 0.25, 0.25, 0.2]) < psi(p, [0.6, 0.2, 0.1, 0.1])
    assert psi(p, [0.0, 0.5, 0.25, 0.25]) < 10          # an empty bin is floored, not infinite


def test_numeric_psi_detects_a_shifted_distribution():
    rng = np.random.default_rng(1)
    ref = numeric_reference(rng.normal(30, 10, 20000))
    assert feature_psi(ref, rng.normal(30, 10, 5000)) < 0.02
    assert feature_psi(ref, rng.normal(40, 10, 5000)) > 0.2


def test_numeric_reference_survives_heavily_tied_values():
    ref = numeric_reference([0] * 600 + list(range(1, 400)))      # many zeros collapse quantile edges
    assert abs(sum(ref["proportions"]) - 1) < 1e-9 and len(ref["edges"]) == len(set(ref["edges"]))


def test_categorical_psi_flags_an_unseen_category():
    ref = categorical_reference(["A"] * 50 + ["B"] * 50)
    assert feature_psi(ref, ["A"] * 50 + ["B"] * 50) < 1e-6
    assert feature_psi(ref, ["A"] * 50 + ["Brand new"] * 50) > 0.2


def test_ece_perfect_and_biased():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.5, 40000)
    y = (rng.random(40000) < p).astype(int)
    assert ece(y, p) < 0.01
    assert ece(y, np.clip(p + 0.1, 0, 1)) > 0.08


def test_window_performance_handles_a_single_class():
    perf = window_performance(np.zeros(50), np.full(50, 0.2), np.full(50, 0.2))
    assert perf["roc_auc"] is None and perf["ece"] is None and perf["rows"] == 50


def test_replayed_report_on_the_fixture(client):
    m = client.get("/api/monitoring").json()
    assert all(f["status"] == "ok" for f in m["features"])        # the replay is drawn from the same process
    assert len(m["performance"]["daily"]) == 6
    assert all(w["roc_auc"] > 0.5 for w in m["performance"]["weekly"])
    assert m["features"][0]["cumulative"][-1]["days"] == 6


# --- alert rules ----------------------------------------------------------------------------------
def report(**over):
    cum = lambda v: [{"date": f"2016-06-0{i + 1}", "days": i + 1, "psi": v} for i in range(6)]
    r = {"features": [{"feature": "age", "label": "Age", "psi": 0.01, "cumulative": cum(0.01), "by_day": []}],
         "score": {"cumulative": cum(0.01), "by_day": []},
         "performance": {"weekly": [{"start": "2016-06-01", "end": "2016-06-03", "roc_auc": 0.73, "ece": 0.02,
                                     "observed_rate": 0.2}]},
         "base_rate": {"train": 0.21}}
    r.update(over)
    return r


def test_no_alerts_when_everything_is_inside_thresholds():
    assert evaluate(report()) == []


def test_psi_status_boundaries():
    assert [psi_status(v) for v in (0.1, 0.11, 0.2, 0.21)] == ["ok", "warning", "warning", "alert"]


def test_feature_drift_alert_fires_only_once_enough_days_have_accumulated():
    r = report()
    cum = [{"date": f"2016-06-0{i + 1}", "days": i + 1, "psi": 0.6 if i < 4 else 0.01} for i in range(6)]
    r["features"][0]["cumulative"] = cum                            # noisy early days, settled by day 5
    assert evaluate(r) == []
    cum = [{"date": f"2016-06-0{i + 1}", "days": i + 1, "psi": 0.3 if i >= 4 else 0.0} for i in range(6)]
    r["features"][0]["cumulative"] = cum
    a = evaluate(r)
    assert len(a) == 1 and a[0]["level"] == "alert" and a[0]["rule"] == "feature_psi" and a[0]["when"] == "2016-06-05"


def test_score_drift_warning():
    r = report()
    r["score"]["cumulative"] = [{"date": "2016-06-05", "days": 5, "psi": 0.15}]
    a = evaluate(r)
    assert [x["rule"] for x in a] == ["score_psi"] and a[0]["level"] == "warning"


def test_auc_and_calibration_alerts():
    r = report()
    r["performance"]["weekly"][0].update(roc_auc=0.69, ece=0.06)
    rules = {a["rule"]: a for a in evaluate(r)}
    assert rules["auc_below_min"]["level"] == "alert" and rules["ece_above_max"]["level"] == "alert"
    r["performance"]["weekly"][0].update(roc_auc=THRESHOLDS["auc_min"], ece=THRESHOLDS["ece_max"])
    assert evaluate(r) == []                                         # exactly on the threshold is not an alert
    r["performance"]["weekly"][0].update(roc_auc=None, ece=None)    # too few rows or a single class
    assert evaluate(r) == []


def test_base_rate_shift_warning_and_ordering():
    r = report()
    r["performance"]["weekly"][0].update(observed_rate=0.17, roc_auc=0.6)
    a = evaluate(r)
    assert [x["level"] for x in a] == ["alert", "warning"]           # alerts sort before warnings
    assert a[1]["rule"] == "base_rate_shift"


def test_alert_rules_do_not_mutate_the_report():
    r = report()
    before = copy.deepcopy(r)
    evaluate(r)
    assert r == before
