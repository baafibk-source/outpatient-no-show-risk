"""band(), action_for(), build_row() and the request-validation limits."""
import pytest
from pydantic import ValidationError

from dashboard.model import action_for, band, build_row, fmt_value
from dashboard.schemas import AppointmentIn, ChatIn, ContextIn
from tests.conftest import FEATURES, HOODS


# --- band: whole-percent, round half up, exactly as the UI shows it --------------------------------
@pytest.mark.parametrize("p,expected", [
    (0.0, "Low"), (0.194, "Low"), (0.1949, "Low"),
    (0.195, "Medium"), (0.20, "Medium"), (0.3449, "Medium"),   # 19.5% displays as 20%, so Medium
    (0.345, "High"), (0.35, "High"), (0.99, "High"),            # 34.5% displays as 35%, so High
])
def test_band_matches_displayed_percentage(p, expected):
    assert band(p) == expected


# --- action_for -----------------------------------------------------------------------------------
def test_low_risk_is_standard_process_whatever_the_lead_time():
    assert {action_for("Low", d) for d in (0, 1, 3, 30)} == {"Standard process. No extra contact needed."}


@pytest.mark.parametrize("lead,fragment", [(0, "today"), (1, "tomorrow"), (2, "Short notice"), (3, "Short notice")])
def test_short_notice_actions_replace_the_default(lead, fragment):
    for level in ("High", "Medium"):
        assert fragment in action_for(level, lead)


def test_high_risk_with_time_to_spare_is_a_call_and_medium_an_sms():
    assert action_for("High", 14).startswith("Phone")
    assert action_for("Medium", 14).startswith("Send an SMS")


def test_high_same_day_is_a_call_and_medium_same_day_is_an_sms():
    assert "call the patient now" in action_for("High", 0)
    assert "send an SMS now" in action_for("Medium", 0)


# --- build_row ------------------------------------------------------------------------------------
def row(**kw):
    return build_row(AppointmentIn(**kw), FEATURES, HOODS)


def test_build_row_has_exactly_the_model_features_in_order():
    r = row()
    assert list(r.columns) == FEATURES and len(r) == 1
    assert str(r["neighbourhood"].dtype) == "category" and list(r["neighbourhood"].cat.categories) == HOODS


def test_build_row_derived_fields():
    r = row(lead_time_days=0, weekday="Friday", conditions=["Diabetes"], prior_appts=4, prior_no_shows=1,
            days_since_last_appt=9)
    x = r.iloc[0]
    assert x["same_day_booking"] == 1 and x["appt_weekday"] == 4
    assert x["diabetes"] == 1 and x["hypertension"] == 0 and x["alcoholism"] == 0
    assert x["prior_no_show_rate"] == 0.25 and x["is_first_appt"] == 0 and x["days_since_last_appt"] == 9


def test_build_row_without_history_uses_sentinels():
    x = row(prior_appts=0, prior_no_shows=0, days_since_last_appt=50).iloc[0]
    assert x["is_first_appt"] == 1 and x["prior_no_show_rate"] == -1.0 and x["days_since_last_appt"] == -1


def test_build_row_caps_no_shows_at_appointments():
    x = row(prior_appts=2, prior_no_shows=9).iloc[0]
    assert x["prior_no_shows"] == 2 and x["prior_no_show_rate"] == 1.0


@pytest.mark.parametrize("hood", [None, "Somewhere New", "not-a-place"])
def test_unknown_neighbourhood_maps_to_other(hood):
    assert row(neighbourhood=hood).iloc[0]["neighbourhood"] == "Other"


def test_known_neighbourhood_is_kept():
    assert row(neighbourhood="Centro").iloc[0]["neighbourhood"] == "Centro"


def test_removed_sensitive_inputs_are_not_in_the_row():
    assert not {"is_male", "welfare_programme", "has_handicap", "n_conditions"} & set(row().columns)


def test_fmt_value():
    assert fmt_value("prior_no_show_rate", -1.0) == "n/a"
    assert fmt_value("lead_time_days", 5) == "5 days"
    assert fmt_value("sms_received", 1) == "Yes"
    assert fmt_value("appt_weekday", 0) == "Monday"


# --- validation limits ----------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [
    {"lead_time_days": -1}, {"lead_time_days": 366}, {"age": 111}, {"age": -1}, {"booked_hour": 24},
    {"prior_appts": 501}, {"prior_no_shows": -1}, {"days_since_last_appt": 3651},
    {"weekday": "Sunday"}, {"conditions": ["Disability"]}, {"conditions": ["Diabetes"] * 4},
    {"neighbourhood": "x" * 101}, {"sex": "Male"}, {"welfare_programme": True}, {"surprise": 1},
])
def test_appointment_limits(bad):
    with pytest.raises(ValidationError):
        AppointmentIn(**bad)


def test_appointment_limits_accept_the_edges():
    AppointmentIn(lead_time_days=365, age=110, booked_hour=23, prior_appts=500, days_since_last_appt=3650,
                  conditions=["Hypertension", "Diabetes", "Alcoholism"], neighbourhood="x" * 100)


def test_context_needs_exactly_one_source():
    with pytest.raises(ValidationError):
        ContextIn()
    with pytest.raises(ValidationError):
        ContextIn(appointment_id=1, appointment={})
    assert ContextIn(appointment_id=1).appointment is None
    assert ContextIn(appointment={"age": 30}).appointment.age == 30


def test_context_rejects_client_supplied_prediction_text():
    with pytest.raises(ValidationError):
        ContextIn(risk=0.9, band="High", factors=[])


def test_chat_limits():
    ctx = {"appointment_id": 1}
    ok = {"role": "user", "content": "hello"}
    ChatIn(context=ctx, messages=[ok])
    for messages in ([], [ok] * 9, [{"role": "user", "content": "x" * 501}],
                     [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]):
        with pytest.raises(ValidationError):
            ChatIn(context=ctx, messages=messages)
    # an assistant reply may be longer than a staff question
    ChatIn(context=ctx, messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a" * 600},
                                  {"role": "user", "content": "q2"}])
