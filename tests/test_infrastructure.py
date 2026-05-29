"""Smoke tests for the test infrastructure delivered in Task 1.

These validate that the shared Hypothesis strategies generate raw-schema
frames as documented and that the configuration constants are wired up. They
are NOT the correctness property tests (Properties 1-14) - those arrive with
their respective component tasks.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

import config
from tests import strategies as strat


# ─── Config constants ───────────────────────────────────────────

def test_config_exposes_new_constants():
    assert config.SEED == 42
    assert config.TEST_TOD_SLOT_MIN == 9
    assert config.TEST_TOD_SLOT_MAX == 55
    assert config.TE_SMOOTHING_ALPHA == 20.0
    assert config.TE_N_FOLDS == 5
    assert config.RECORDED_ONLINE_SCORE == 83.13
    assert config.LEADERBOARD_TOP == 93.13
    assert hasattr(config, "N_ESTIMATORS")
    assert hasattr(config, "EARLY_STOPPING")
    assert callable(config.seed_everything)


def test_config_drops_long_history_knobs():
    for removed in ("MAX_LAGS", "ROLLING_WINDOWS", "HOLDOUT_FRAC"):
        assert not hasattr(config, removed), f"{removed} should be removed"


def test_seed_everything_is_deterministic():
    config.seed_everything(config.SEED)
    a = np.random.rand(5)
    config.seed_everything(config.SEED)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


# ─── slot/timestamp helper ──────────────────────────────────────

def test_slot_to_timestamp_known_values():
    assert strat.slot_to_timestamp(0) == "0:0"
    assert strat.slot_to_timestamp(9) == "2:15"
    assert strat.slot_to_timestamp(55) == "13:45"
    assert strat.slot_to_timestamp(95) == "23:45"


# ─── raw_frame strategy ─────────────────────────────────────────

@given(df=strat.raw_frame())
@settings(max_examples=50)
def test_raw_frame_has_raw_schema(df):
    assert list(df.columns) == strat.RAW_COLUMNS
    assert len(df) >= 1
    assert df["day"].isin([48, 49]).all()
    assert df["NumberofLanes"].isin(strat.NUM_LANES).all()
    # demand is a continuous target in [0, 1]
    assert df["demand"].between(0.0, 1.0).all()
    # timestamps parse back to slots in 0..95
    for ts in df["timestamp"]:
        h, m = ts.split(":")
        slot = (int(h) * 60 + int(m)) // 15
        assert 0 <= slot <= 95


@given(df=strat.raw_frame(include_demand=False, nullable=False))
@settings(max_examples=25)
def test_raw_frame_test_like_has_no_demand_and_no_nulls(df):
    assert "demand" not in df.columns
    for col in strat.NULLABLE_CONTEXT_COLS:
        assert df[col].notna().all()


# ─── specialized frame strategies ───────────────────────────────

@given(df=strat.two_day_frame())
@settings(max_examples=25)
def test_two_day_frame_contains_both_days(df):
    days = set(df["day"].unique())
    assert days == {48, 49}


@given(df=strat.real_task_frame())
@settings(max_examples=25)
def test_real_task_frame_has_day49_in_test_window(df):
    d49 = df[df["day"] == 49]
    slots = ((d49["timestamp"].str.split(":").str[0].astype(int) * 60
              + d49["timestamp"].str.split(":").str[1].astype(int)) // 15)
    assert ((slots >= 9) & (slots <= 55)).any()


@given(df=strat.morning_only_day49_frame())
@settings(max_examples=25)
def test_morning_only_frame_has_no_day49_test_window_rows(df):
    d49 = df[df["day"] == 49]
    slots = ((d49["timestamp"].str.split(":").str[0].astype(int) * 60
              + d49["timestamp"].str.split(":").str[1].astype(int)) // 15)
    assert (slots <= 8).all()


@given(pair=strat.train_test_pair())
@settings(max_examples=25)
def test_train_test_pair_has_unseen_geohash(pair):
    train_df, test_df = pair
    assert "demand" in train_df.columns
    assert "demand" not in test_df.columns
    unseen = set(test_df["geohash"]) - set(train_df["geohash"])
    assert len(unseen) >= 1


@given(df=strat.frame_with_injected_nulls())
@settings(max_examples=25)
def test_injected_nulls_present(df):
    null_count = df[strat.NULLABLE_CONTEXT_COLS].isna().sum().sum()
    assert null_count >= 1


# ─── prediction-vector strategies ───────────────────────────────

@given(arr=strat.prediction_array())
@settings(max_examples=25)
def test_prediction_array_is_finite(arr):
    assert arr.ndim == 1 and arr.size >= 1
    assert np.isfinite(arr).all()


@given(arr=strat.prediction_array_with_nans())
@settings(max_examples=25)
def test_prediction_array_with_nans_has_bad_value(arr):
    assert (~np.isfinite(arr)).any()


@given(pair=strat.test_frame_with_predictions())
@settings(max_examples=25)
def test_test_frame_with_predictions_aligns(pair):
    test_df, preds = pair
    assert len(test_df) == len(preds)
    assert test_df["Index"].is_unique
