"""Unit/edge test for Task 4.4: the ``FAILED_NO_MATCHING_SLOTS`` validator path.

When ``train.csv``'s day-49 rows are morning-only (slots ``0..8``) none of them
fall inside the test daytime window ``[9, 55]``, so the aligned day-48 → day-49
validator :func:`src.validation.build_real_task_holdout` must report status
``FAILED_NO_MATCHING_SLOTS`` and **proceed without raising** (the pipeline logs
the absence and falls back to the day-48 daytime surrogate). This is the
real-data behavior described in the design's validation section.

This is a unit/edge test (not one of the numbered Properties 1-14), so it is
deliberately *not* tagged with a ``Property N`` comment. It combines a small
Hypothesis-driven check over the shared ``morning_only_day49_frame`` strategy
with an explicit concrete example.

Requirements: 2.4
"""

from __future__ import annotations

import pandas as pd
from hypothesis import HealthCheck, given, settings

from src.validation import (
    KIND_REAL_TASK,
    STATUS_FAILED_NO_MATCHING_SLOTS,
    build_real_task_holdout,
)
from tests import strategies as strat

# Inclusive test daytime window (quarter-hour slots) the validator aligns to.
_TEST_MIN, _TEST_MAX = strat.TEST_WINDOW_SLOTS  # (9, 55)


def _add_tod_slot(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ``tod_slot = (hour*60 + minute) // 15`` from the raw timestamp.

    The shared strategies (and the real CSVs) carry an un-padded ``"H:M"``
    ``timestamp`` string rather than a precomputed ``tod_slot`` column, so the
    validator's slot column must be derived before calling it.
    """
    out = df.copy()
    parts = out["timestamp"].str.split(":", expand=True)
    hour = parts[0].astype(int)
    minute = parts[1].astype(int)
    out["tod_slot"] = (hour * 60 + minute) // 15
    return out


@given(frame=strat.morning_only_day49_frame())
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.data_too_large, HealthCheck.too_slow],
)
def test_morning_only_day49_reports_failed_no_matching_slots(frame):
    """A morning-only day-49 frame yields FAILED_NO_MATCHING_SLOTS, no raise."""
    df = _add_tod_slot(frame)

    # Sanity: the generated frame really has day-49 rows, all in morning slots
    # 0..8 (so none can fall inside the test window [9, 55]).
    day49 = df[df["day"] == 49]
    assert not day49.empty, "strategy must produce day-49 rows"
    assert day49["tod_slot"].between(0, 8).all()
    assert not day49["tod_slot"].between(_TEST_MIN, _TEST_MAX).any()

    # Must not raise; must return the failure status with an empty eval fold.
    split = build_real_task_holdout(df)

    assert split.status == STATUS_FAILED_NO_MATCHING_SLOTS
    assert split.kind == KIND_REAL_TASK
    assert split.eval_idx.size == 0

    # Train partition is day-48-only and non-empty (the strategy always emits
    # day-48 rows). Pipeline-proceed contract: usable train fold survives.
    train_rows = df.loc[split.train_idx]
    assert not train_rows.empty
    assert (train_rows["day"] == 48).all()

    # The absence of matching slots is reported in the coverage note.
    note = split.coverage_note.lower()
    assert STATUS_FAILED_NO_MATCHING_SLOTS.lower() in note
    assert "window" in note or "slot" in note


def test_concrete_morning_only_frame_does_not_raise():
    """Explicit hand-built example mirroring train.csv's morning-only day 49."""
    rows = [
        # geohash, day, slot, demand
        ("qp02z1", 48, 0, 0.05),
        ("qp02z1", 48, 30, 0.40),
        ("qp02z1", 48, 95, 0.22),
        ("qp02zt", 48, 12, 0.10),
        ("qp02zt", 48, 55, 0.31),   # day-48 in-window row stays in TRAIN here
        ("qp02z1", 49, 0, 0.06),    # day-49 morning slots only (0..8)
        ("qp02zt", 49, 4, 0.09),
        ("qp02z1", 49, 8, 0.11),
    ]
    records = []
    for i, (gh, day, slot, dem) in enumerate(rows):
        records.append({
            "Index": i,
            "geohash": gh,
            "day": day,
            "timestamp": strat.slot_to_timestamp(slot),
            "demand": dem,
            "RoadType": "Residential",
            "NumberofLanes": 2,
            "LargeVehicles": "Not Allowed",
            "Landmarks": "No",
            "Temperature": 25.0,
            "Weather": "Sunny",
        })
    df = _add_tod_slot(pd.DataFrame(records, columns=strat.RAW_COLUMNS))

    split = build_real_task_holdout(df)

    assert split.status == STATUS_FAILED_NO_MATCHING_SLOTS
    assert split.eval_idx.size == 0

    # Train fold = exactly the five day-48 rows; no day-49 row leaks in.
    train_rows = df.loc[split.train_idx]
    assert len(train_rows) == 5
    assert (train_rows["day"] == 48).all()

    # Reported note names the failure and the empty test window.
    assert STATUS_FAILED_NO_MATCHING_SLOTS in split.coverage_note
    assert f"[{_TEST_MIN}, {_TEST_MAX}]" in split.coverage_note
