"""Property-based test for Task 12.3 (Property 14).

Validates that the prediction path is **deterministic** given a fixed seed and
fixed inputs (Requirements 6.5, 6.4): two successive executions of the path,
each re-seeded through the single ``config.seed_everything`` entry point, must
produce *identical* predicted demand vectors, and every value must lie within
``[0, train_max]``.

Rather than driving the heavyweight ``run.py`` end-to-end (which loads the real
CSVs and trains 5000-tree models), the test exercises the same deterministic
prediction path using the library components directly on a *small synthetic
two-day frame* (CPU, tiny ``n_estimators``):

    seed_everything(SEED)
      -> build_day48_daytime_holdout            (surrogate holdout)
      -> TargetEncoder.fit / fit_oof            (leak-free geohash encodings)
      -> fit_tod_curve                          (day-48 time-of-day curve)
      -> build_imputers                         (train-only imputation values)
      -> build_features                         (train / eval / test frames)
      -> select_feature_cols                    (leak-free feature set)
      -> train_lightgbm (cpu, ~12 trees)        (single GBM, deterministic seed)
      -> postprocess(preds, "identity", ...)    (clip to [0, train_max])

The whole path runs twice per example (re-seeding each time) and the two final
prediction vectors are compared with ``np.array_equal`` (exact equality, the
strongest possible determinism check). ``train_max`` is obtained the way the
real pipeline obtains it — ``postprocess`` reads it from a train CSV — so the
synthetic train frame is written to a temporary CSV and that path is passed in.

Per-example cost note: each example performs two full prediction-path runs and
each ``train_lightgbm`` call trains ``n_folds + 1`` tiny LightGBM models. To keep
the suite fast while still exercising 100 generated inputs, the synthetic frame
is small and the trainer uses ``n_folds=2`` with ``n_estimators=12`` on CPU. The
example count is the spec-mandated minimum of 100; ``deadline=None`` disables
Hypothesis' per-example deadline because GBM training time per example is
variable (and irrelevant to the determinism property).
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import config
from src.features import build_features, build_imputers, select_feature_cols
from src.models import train_lightgbm
from src.postprocess import postprocess
from src.spatial import TargetEncoder, fit_tod_curve
from src.validation import STATUS_OK, build_day48_daytime_holdout
from tests import strategies as strat

_TARGET = "demand"
_SLOT_COL = "tod_slot"
_CATEGORICAL_COLS = ["RoadType", "LargeVehicles", "Landmarks", "Weather"]
_LOCATION_COLS = ["geohash"]

#: Shared geohash pool; the test frame also appends one geohash unseen in train.
_GEOHASHES = ["qp02z1", "qp02zt", "qp08bj", "qp0r0a", "qp0r0b"]

#: Day-48 slots: a fixed spread that straddles the test daytime window [9, 55]
#: so the surrogate holdout always has BOTH a non-empty eval partition (slots in
#: [9, 55]) and a non-empty training complement (slots outside it).
_DAY48_SLOTS = (0, 5, 10, 20, 30, 45, 55, 70, 90)
#: Day-49 morning slots (mirror train.csv's morning-only day-49 rows).
_DAY49_MORNING_SLOTS = (0, 4, 8)
#: Day-49 daytime slots for the (label-free) test frame, all inside [9, 55].
_TEST_SLOTS = (10, 20, 30, 40, 50)


def _row(idx: int, gh: str, gh_i: int, day: int, slot: int, demand: float | None):
    """Build one model-ready row (raw columns + parsed time fields).

    Mirrors the columns produced by ``src.data.load_data`` (``hour``/``minute``/
    ``tod_slot``/``abs_time``) plus the six contextual columns, so the row can be
    fed straight into ``build_features``.
    """
    minutes = slot * 15
    hour = minutes // 60
    minute = minutes % 60
    rec = {
        "Index": idx,
        "geohash": gh,
        "day": day,
        "timestamp": strat.slot_to_timestamp(slot),
        "RoadType": strat.ROAD_TYPES[(gh_i + slot) % len(strat.ROAD_TYPES)],
        "NumberofLanes": strat.NUM_LANES[gh_i % len(strat.NUM_LANES)],
        "LargeVehicles": strat.LARGE_VEHICLES[slot % 2],
        "Landmarks": strat.LANDMARKS[gh_i % 2],
        "Temperature": 20.0 + (slot % 7),
        "Weather": strat.WEATHERS[(gh_i + slot) % len(strat.WEATHERS)],
        "hour": hour,
        "minute": minute,
        "tod_slot": slot,
        "abs_time": day * 1440 + hour * 60 + minute,
    }
    if demand is not None:
        rec[_TARGET] = demand
    return rec


def _build_frames(n_geohashes: int, demand_seed: int):
    """Construct a deterministic synthetic (train_df, test_df) pair.

    The structure (slot layout / partitioning) is fixed so the surrogate
    holdout is always valid; the geohash count and the synthetic demand signal
    vary per Hypothesis example so determinism is checked across many inputs.

    Args:
        n_geohashes: Number of geohashes to include in the train frame.
        demand_seed: Seed for the synthetic demand signal (kept independent of
            the pipeline seed so the *input data* varies between examples).

    Returns:
        ``(train_df, test_df)`` — train carries ``demand``; test does not.
    """
    rng = np.random.default_rng(demand_seed)
    geohashes = _GEOHASHES[:n_geohashes]

    train_records = []
    idx = 0
    for gh_i, gh in enumerate(geohashes):
        for slot in _DAY48_SLOTS:
            demand = float(
                np.clip(0.10 + 0.08 * gh_i + 0.004 * slot + rng.normal(0, 0.01),
                        0.0, 1.0)
            )
            train_records.append(_row(idx, gh, gh_i, 48, slot, demand))
            idx += 1
        for slot in _DAY49_MORNING_SLOTS:
            demand = float(
                np.clip(0.08 + 0.08 * gh_i + 0.004 * slot + rng.normal(0, 0.01),
                        0.0, 1.0)
            )
            train_records.append(_row(idx, gh, gh_i, 49, slot, demand))
            idx += 1
    train_df = pd.DataFrame(train_records)

    # Test frame: day-49 daytime, no demand, plus one unseen geohash.
    test_records = []
    tidx = 10_000
    for gh_i, gh in enumerate(geohashes + ["zzzzzz"]):
        for slot in _TEST_SLOTS:
            test_records.append(_row(tidx, gh, gh_i, 49, slot, None))
            tidx += 1
    test_df = pd.DataFrame(test_records)

    return train_df, test_df


def _run_prediction_path(train_df: pd.DataFrame, test_df: pd.DataFrame,
                         train_csv_path: str) -> np.ndarray:
    """Execute the deterministic prediction path once and return final preds.

    Re-seeds through the single ``seed_everything`` entry point (Req 6.4), then
    runs the surrogate-holdout fit → leak-free features → single LightGBM (CPU,
    tiny) → ``postprocess`` path, returning the bounded final demand vector.
    """
    config.seed_everything(config.SEED)

    split = build_day48_daytime_holdout(train_df)
    assert split.status == STATUS_OK
    train_part = train_df.loc[split.train_idx].reset_index(drop=True)
    eval_part = train_df.loc[split.eval_idx].reset_index(drop=True)

    # Target-derived artifacts fit on the TRAIN partition only (leak-free).
    encoder = TargetEncoder("geohash").fit(train_part, _TARGET)
    curve = fit_tod_curve(train_part, _TARGET)
    imp = build_imputers(train_part)

    # Training frame carries out-of-fold encodings (leak-free OOF layer).
    train_oof = encoder.fit_oof(train_part, _TARGET, n_folds=2, seed=config.SEED)

    def _feats(df):
        return build_features(
            df, encoder=encoder, tod_curve=curve,
            categorical_cols=_CATEGORICAL_COLS, location_cols=_LOCATION_COLS,
            impute_values=imp, slot_col=_SLOT_COL,
        )

    train_feats = _feats(train_oof)
    eval_feats = _feats(eval_part)
    test_feats = _feats(test_df)

    feature_cols = select_feature_cols(
        train_feats, test_feats,
        timestamp_col="timestamp", target_col=_TARGET,
        id_col="Index", location_cols=_LOCATION_COLS,
    )

    result = train_lightgbm(
        train_feats, feature_cols, _TARGET,
        None, None, eval_feats, eval_part[_TARGET], test_feats,
        n_folds=2, n_estimators=12, early_stop=4,
        seed=config.SEED, lgb_device="cpu",
    )

    return postprocess(
        result.test_preds, "identity", train_csv_path, _TARGET
    )


# Feature: demand-prediction-overhaul, Property 14: Predictions are deterministic given seed and inputs
@given(
    n_geohashes=st.integers(min_value=2, max_value=len(_GEOHASHES)),
    demand_seed=st.integers(min_value=0, max_value=10_000),
)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_prediction_path_is_deterministic(n_geohashes: int, demand_seed: int) -> None:
    """Two re-seeded runs of the prediction path produce identical, bounded preds."""
    train_df, test_df = _build_frames(n_geohashes, demand_seed)

    # train_max is obtained exactly as the real pipeline obtains it: postprocess
    # reads it from a train CSV. Write the synthetic train frame to a temp CSV.
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    )
    try:
        train_df.to_csv(tmp.name, index=False)
        tmp.close()
        train_max = float(train_df[_TARGET].max())

        # Suppress the components' verbose stdout to keep the run fast/quiet.
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            preds_a = _run_prediction_path(train_df, test_df, tmp.name)
            preds_b = _run_prediction_path(train_df, test_df, tmp.name)
    finally:
        os.unlink(tmp.name)

    # Same length, one prediction per test row.
    assert preds_a.shape == preds_b.shape == (len(test_df),)

    # Determinism: identical vectors (exact equality, atol == 0).
    assert np.array_equal(preds_a, preds_b), (
        "prediction path is non-deterministic: the two re-seeded runs differ"
    )

    # Both vectors are finite and bounded to [0, train_max] (Req 6.5 / 5.4-5.5).
    for preds in (preds_a, preds_b):
        assert np.all(np.isfinite(preds))
        assert np.all(preds >= 0.0)
        assert np.all(preds <= train_max)
