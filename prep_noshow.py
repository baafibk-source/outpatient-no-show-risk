"""
Outpatient No-Show (DNA) Prediction - Data Cleaning & Feature Engineering
Proof of concept on public data: Kaggle "Medical Appointment No Shows"
(file: KaggleV2-May-2016.csv, ~110k appointments, Brazil, 2016)

Usage:
    python prep_noshow.py
Outputs:
    noshow_features.parquet  (cleaned, model-ready table; categories and dates preserved)
"""

import numpy as np
import pandas as pd

RAW_PATH = "KaggleV2-May-2016.csv"
OUT_PATH = "noshow_features.parquet"
MIN_NEIGHBOURHOOD_ROWS = 50
OTHER = "Other"


# -----------------------------------------------------------------------------
# 1. LOAD & STANDARDISE
# -----------------------------------------------------------------------------
def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Fix the original typos and use consistent snake_case names
    df = df.rename(columns={
        "PatientId": "patient_id",
        "AppointmentID": "appointment_id",
        "Gender": "gender",
        "ScheduledDay": "scheduled_ts",
        "AppointmentDay": "appointment_date",
        "Age": "age",
        "Neighbourhood": "neighbourhood",
        "Scholarship": "welfare_programme",   # Bolsa Familia enrolment
        "Hipertension": "hypertension",
        "Diabetes": "diabetes",
        "Alcoholism": "alcoholism",
        "Handcap": "handicap_level",
        "SMS_received": "sms_received",
        "No-show": "no_show",
    })

    # PatientId is stored as a float in the raw file - make it a clean integer key
    df["patient_id"] = df["patient_id"].astype("int64")

    # Dates: ScheduledDay has a real time; AppointmentDay is date-only (always 00:00)
    df["scheduled_ts"] = pd.to_datetime(df["scheduled_ts"]).dt.tz_localize(None)
    df["appointment_date"] = pd.to_datetime(df["appointment_date"]).dt.tz_localize(None).dt.normalize()
    df["scheduled_date"] = df["scheduled_ts"].dt.normalize()

    # Target: "Yes" in the raw file means the patient DID NOT attend
    df["no_show"] = (df["no_show"].str.strip().str.lower() == "yes").astype(int)

    return df


# -----------------------------------------------------------------------------
# 2. DATA QUALITY CHECKS & CLEANING
# -----------------------------------------------------------------------------
def profile_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    n_start = len(df)
    print(f"Rows loaded: {n_start:,}")
    print(f"Unique patients: {df['patient_id'].nunique():,}")
    print(f"No-show rate: {df['no_show'].mean():.1%}")
    print(f"Appointment dates: {df['appointment_date'].min().date()} "
          f"to {df['appointment_date'].max().date()}")
    print(f"Nulls per column:\n{df.isna().sum()[df.isna().sum() > 0]}")

    # Duplicate appointment IDs
    dupes = df["appointment_id"].duplicated().sum()
    print(f"Duplicate appointment IDs: {dupes}")
    df = df.drop_duplicates("appointment_id")

    # Impossible ages (there is at least one -1 in the raw file)
    bad_age = (df["age"] < 0) | (df["age"] > 110)
    print(f"Removing {bad_age.sum()} rows with impossible ages")
    df = df[~bad_age]

    # Appointments dated BEFORE they were booked (data entry errors)
    df["lead_time_days"] = (df["appointment_date"] - df["scheduled_date"]).dt.days
    bad_lead = df["lead_time_days"] < 0
    print(f"Removing {bad_lead.sum()} rows booked after the appointment date")
    df = df[~bad_lead]

    # handicap_level ranges 0-4 but higher levels are very rare - keep a binary flag too
    df["has_handicap"] = (df["handicap_level"] > 0).astype(int)

    # Standardise text categories
    df["gender"] = df["gender"].str.strip().str.upper()
    df["neighbourhood"] = df["neighbourhood"].str.strip().str.title()

    print(f"Rows after cleaning: {len(df):,} ({n_start - len(df):,} removed)\n")
    return df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# -----------------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["same_day_booking"] = (df["lead_time_days"] == 0).astype(int)
    df["appt_weekday"] = df["appointment_date"].dt.day_of_week      # 0 = Monday
    df["booked_hour"] = df["scheduled_ts"].dt.hour

    # Lead-time bands are easier to explain to clinical/operational teams
    df["lead_time_band"] = pd.cut(
        df["lead_time_days"],
        bins=[-1, 0, 7, 14, 30, 60, np.inf],
        labels=["same_day", "1-7d", "8-14d", "15-30d", "31-60d", "60d+"],
    )
    return df


def add_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    df["is_male"] = (df["gender"] == "M").astype(int)
    df["age_band"] = pd.cut(
        df["age"],
        bins=[-1, 12, 17, 29, 44, 64, 120],
        labels=["0-12", "13-17", "18-29", "30-44", "45-64", "65+"],
    )
    df["n_conditions"] = df[["hypertension", "diabetes", "alcoholism", "has_handicap"]].sum(axis=1)
    return df


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Patient attendance history, using ONLY appointments on EARLIER dates.

    Leakage guard: a patient can have several appointments on the same day.
    If we used a simple row-level cumulative sum, same-day outcomes would leak
    into each other. So we aggregate to patient-day first, then shift.
    """
    daily = (
        df.groupby(["patient_id", "appointment_date"])
          .agg(day_appts=("appointment_id", "count"),
               day_no_shows=("no_show", "sum"))
          .reset_index()
          .sort_values(["patient_id", "appointment_date"])
    )

    g = daily.groupby("patient_id")
    daily["prior_appts"] = g["day_appts"].cumsum() - daily["day_appts"]
    daily["prior_no_shows"] = g["day_no_shows"].cumsum() - daily["day_no_shows"]
    daily["days_since_last_appt"] = g["appointment_date"].diff().dt.days

    df = df.merge(
        daily[["patient_id", "appointment_date", "day_appts",
               "prior_appts", "prior_no_shows", "days_since_last_appt"]],
        on=["patient_id", "appointment_date"],
        how="left",
    )

    df["is_first_appt"] = (df["prior_appts"] == 0).astype(int)
    # Rate is undefined for first-time patients; -1 lets tree models treat it separately
    df["prior_no_show_rate"] = np.where(
        df["prior_appts"] > 0, df["prior_no_shows"] / df["prior_appts"], -1.0
    )
    df["days_since_last_appt"] = df["days_since_last_appt"].fillna(-1)
    df["multiple_appts_same_day"] = (df["day_appts"] > 1).astype(int)
    return df.drop(columns="day_appts")


# -----------------------------------------------------------------------------
# 4. TIME-BASED TRAIN/TEST SPLIT
# -----------------------------------------------------------------------------
def time_split(df: pd.DataFrame, test_frac: float = 0.2, verbose: bool = True):
    """
    Train on earlier appointment dates, test on later ones.
    This mirrors real use: a Trust would score FUTURE appointments
    using a model trained on past attendance.
    """
    dates = np.sort(df["appointment_date"].unique())
    cutoff = dates[int(len(dates) * (1 - test_frac))]
    train = df[df["appointment_date"] < cutoff]
    test = df[df["appointment_date"] >= cutoff]
    if verbose:
        print(f"Split cutoff: {pd.Timestamp(cutoff).date()}")
        print(f"Train: {len(train):,} rows, no-show rate {train['no_show'].mean():.1%}")
        print(f"Test:  {len(test):,} rows, no-show rate {test['no_show'].mean():.1%}")
    return train, test


def training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The rows the model is fitted on: the first 85% of the first 80% of appointment dates.
    Mirrors the two time_split calls in model_noshow.py (test, then validation)."""
    train_full, _ = time_split(df, test_frac=0.2, verbose=False)
    train, _ = time_split(train_full, test_frac=0.15, verbose=False)
    return train


def group_rare_neighbourhoods(df: pd.DataFrame, min_rows: int = MIN_NEIGHBOURHOOD_ROWS) -> pd.DataFrame:
    """Map neighbourhoods with fewer than `min_rows` TRAINING rows to "Other", then make the column
    categorical. Counting on training rows only means nothing is learned from validation or test;
    applying the mapping to every row means train, test and the dashboard share one category list."""
    counts = training_rows(df)["neighbourhood"].value_counts()
    keep = set(counts[counts >= min_rows].index)
    df = df.copy()
    df["neighbourhood"] = df["neighbourhood"].where(df["neighbourhood"].isin(keep), OTHER).astype("category")
    return df


def to_other(name, categories):
    """Prediction-time counterpart of group_rare_neighbourhoods: anything the model has not
    seen as its own category (unknown or rare) is "Other"."""
    return name if name in categories else OTHER


FEATURES = [
    # Booking / timing
    "lead_time_days", "same_day_booking", "appt_weekday", "booked_hour", "sms_received",
    # Patient
    "age", "is_male", "welfare_programme",
    "hypertension", "diabetes", "alcoholism", "has_handicap", "n_conditions",
    # History
    "prior_appts", "prior_no_shows", "prior_no_show_rate",
    "is_first_appt", "days_since_last_appt", "multiple_appts_same_day",
    # High-cardinality category (use XGBoost enable_categorical=True, or target-encode on TRAIN only)
    "neighbourhood",
]
TARGET = "no_show"


if __name__ == "__main__":
    df = load_raw(RAW_PATH)
    df = profile_and_clean(df)
    df = add_time_features(df)
    df = add_patient_features(df)
    df = add_history_features(df)

    n_before = df["neighbourhood"].nunique()
    df = group_rare_neighbourhoods(df)
    n_other = int((df["neighbourhood"] == OTHER).sum())
    print(f"Neighbourhoods: {n_before} before grouping, {df['neighbourhood'].nunique()} categories after "
          f"(rare ones, fewer than {MIN_NEIGHBOURHOOD_ROWS} training rows, now '{OTHER}': {n_other:,} rows)\n")

    # Quick sanity checks worth screenshotting for your README
    print("No-show rate by lead-time band:")
    print(df.groupby("lead_time_band", observed=True)["no_show"].mean().round(3), "\n")
    print("No-show rate: first appointment vs returning patient:")
    print(df.groupby("is_first_appt")["no_show"].mean().round(3), "\n")

    train, test = time_split(df)

    df.to_parquet(OUT_PATH, index=False)  # pyarrow keeps dtypes (categories, dates) intact
    print(f"\nSaved {len(df):,} rows to {OUT_PATH} ({len(FEATURES)} candidate features; the shipped model uses 16, see model_noshow.py --drop-sensitive)")
