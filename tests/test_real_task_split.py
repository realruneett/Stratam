"""Property-based test for Task 4.2 (Property 8).

Validates the structure of the aligned day-48 → day-49-daytime holdout built by
:func:`src.validation.build_real_task_holdout`: when day-49 rows include slots
inside the test window ``[9, 55]``, the train partition must be day-48-only and
the eval partition must be exactly the day-49 rows whose ``tod_slot`` falls in
``[9, 55]``, with status ``"OK"`` (Requirements 2.1, 2.2, 2.3).

The shared ``real_task_frame()`` strategy emits raw-schema frames carrying
``day`` and a ``"H:M"`` ``timestamp`` string but no ``tod_slot`` column, so this
test derives ``tod_slot = (hour * 60 + minute) // 15`` from the timestamp before
calling the validator (which expects a ``tod_slot`` column).
"""

from __future__ import annotations

import pandas as pd
from hypothesis import HealthCheck, given, settings

from src.validation import (
    KIND_REAL_TASK,
    STATUS_OK,
    build_real_task_holdout,
)
from tests import strategies as strat

# Test daytime window (quarter-hour slots), 2:15-13:45 inclusive.
_TEST_TOD_MIN = 9
_TEST_TOD_MAX = 55
_EVAL_DAY = 49
_TRAIN_DAY = 48


def _derive_tod_slot(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``tod_slot`` column derived from timestamp.

    ``timestamp`` is the un-padded ``"H:M"`` format (e.g. ``"2:15"``); the slot
    is ``(hour * 60 + minute) // 15`` in ``0..95``.
    """
    parts = df["timestamp"].str.split(":", expand=True)
    hour = parts[0].astype(int)
    minute = parts[1].astype(int)
    out = df.copy()
    out["tod_slot"] = (hour * 60 + minute) // 15
    return out


# Feature: demand-prediction-overhaul, Property 8: Real-task split trains on day 48 and evaluates on aligned day-49 daytime rows
@given(df=strat.real_task_frame())
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.data_too_large, HealthCheck.too_slow],
)
def test_real_task_split_structure(df):
    """Train = day-48-only; eval = day-49 rows with tod_slot in [9, 55]; OK."""
    df = _derive_tod_slot(df)

    split = build_real_task_holdout(
        df,
        day_col="day",
        slot_col="tod_slot",
        test_tod_min=_TEST_TOD_MIN,
        test_tod_max=_TEST_TOD_MAX,
    )

    # The strategy guarantees at least one day-49 row inside [9, 55], so the
    # split is usable.
    assert split.status == STATUS_OK
    assert split.kind == KIND_REAL_TASK

    train_idx = set(split.train_idx.tolist())
    eval_idx = set(split.eval_idx.tolist())

    # Train partition is day-48-only.
    assert (df.loc[df.index.isin(train_idx), "day"] == _TRAIN_DAY).all()

    # Eval partition is exactly the day-49 rows whose tod_slot is in [9, 55].
    expected_eval = set(
        df.index[
            (df["day"] == _EVAL_DAY)
            & (df["tod_slot"] >= _TEST_TOD_MIN)
            & (df["tod_slot"] <= _TEST_TOD_MAX)
        ].tolist()
    )
    assert eval_idx == expected_eval

    # Train and eval partitions are disjoint.
    assert train_idx.isdisjoint(eval_idx)
