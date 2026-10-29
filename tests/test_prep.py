"""
Tests for the data-prep logic. The most important one protects against target leakage:
a patient's outcome on one day must never be visible to their other appointments that day.

Run:  pytest -q
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prep_noshow import (OTHER, add_history_features, group_rare_neighbourhoods,  # noqa: E402
                         time_split, to_other)


def make_frame(rows):
    df = pd.DataFrame(rows, columns=["appointment_id", "patient_id", "appointment_date", "no_show"])
    df["appointment_date"] = pd.to_datetime(df["appointment_date"])
    return df


def history(rows):
    out = add_history_features(make_frame(rows))
    return out.set_index("appointment_id")


def test_same_day_outcomes_do_not_leak():
    # Patient 1 has TWO appointments on 1 June (one missed, one attended).
    out = history([
        (1, 1, "2016-06-01", 1),
        (2, 1, "2016-06-01", 0),
    ])
    # Neither row may see the other's outcome, and neither has any earlier history.
    assert out.loc[[1, 2], "prior_appts"].tolist() == [0, 0]
    assert out.loc[[1, 2], "prior_no_shows"].tolist() == [0, 0]
    assert out.loc[[1, 2], "is_first_appt"].tolist() == [1, 1]
    assert out.loc[[1, 2], "multiple_appts_same_day"].tolist() == [1, 1]


def test_history_counts_only_earlier_dates():
    out = history([
        (1, 7, "2016-06-01", 1),   # first day, missed
        (2, 7, "2016-06-01", 0),   # same day, attended
        (3, 7, "2016-06-03", 0),   # sees the two earlier appointments, one missed
        (4, 7, "2016-06-08", 1),   # sees three earlier appointments, one missed
    ])
    assert out.loc[3, "prior_appts"] == 2
    assert out.loc[3, "prior_no_shows"] == 1
    assert out.loc[3, "prior_no_show_rate"] == 0.5
    assert out.loc[3, "days_since_last_appt"] == 2
    assert out.loc[4, "prior_appts"] == 3
    assert out.loc[4, "prior_no_shows"] == 1
    assert out.loc[4, "days_since_last_appt"] == 5


def test_first_appointment_uses_sentinel_values():
    out = history([(1, 1, "2016-06-01", 0)])
    assert out.loc[1, "prior_no_show_rate"] == -1.0
    assert out.loc[1, "days_since_last_appt"] == -1


def test_patients_do_not_share_history():
    out = history([
        (1, 1, "2016-06-01", 1),
        (2, 2, "2016-06-02", 0),   # a different patient: must not inherit patient 1's missed visit
    ])
    assert out.loc[2, "prior_appts"] == 0
    assert out.loc[2, "prior_no_shows"] == 0


def test_time_split_has_no_date_overlap_and_test_is_later():
    dates = pd.date_range("2016-05-01", periods=20)
    df = pd.DataFrame({"appointment_date": np.repeat(dates, 3), "no_show": 0})
    train, test = time_split(df, test_frac=0.2)
    assert train["appointment_date"].max() < test["appointment_date"].min()
    assert len(train) + len(test) == len(df)
    assert test["appointment_date"].nunique() == 4   # 20% of 20 dates


def test_rare_neighbourhoods_are_counted_on_training_rows_only():
    # 20 days: dates 0-13 are train, 14-16 validation, 17-19 test (80% / 85% of 80%)
    dates = pd.date_range("2016-05-01", periods=20)
    rows = []
    for i, d in enumerate(dates):
        rows += [{"appointment_date": d, "neighbourhood": "Big", "no_show": 0}] * 5
        if i < 3:
            rows.append({"appointment_date": d, "neighbourhood": "TrainRare", "no_show": 0})
    # Plenty of rows, but all in the test period: must still be grouped, because training never saw it
    rows += [{"appointment_date": dates[19], "neighbourhood": "TestOnly", "no_show": 0}] * 60
    df = pd.DataFrame(rows)
    out = group_rare_neighbourhoods(df, min_rows=10)
    assert set(out["neighbourhood"].astype(str)) == {"Big", OTHER}
    assert out["neighbourhood"].dtype == "category"
    assert (out.loc[df["neighbourhood"] == "TrainRare", "neighbourhood"] == OTHER).all()
    assert len(out) == len(df)


def test_threshold_is_inclusive_at_min_rows():
    dates = pd.date_range("2016-05-01", periods=20)
    df = pd.DataFrame({"appointment_date": np.repeat(dates, 5), "neighbourhood": "Big", "no_show": 0})
    df.loc[df.index[:3], "neighbourhood"] = "Small"        # 3 rows, all in training
    assert "Small" in group_rare_neighbourhoods(df, min_rows=3)["neighbourhood"].cat.categories
    assert "Small" not in group_rare_neighbourhoods(df, min_rows=4)["neighbourhood"].cat.categories


def test_prediction_time_mapping_sends_unknown_to_other():
    cats = ["Jardim Camburi", OTHER]
    assert to_other("Jardim Camburi", cats) == "Jardim Camburi"
    assert to_other("Somewhere New", cats) == OTHER
    assert to_other(None, cats) == OTHER
