"""
Task-mirroring validation holdouts for the day-48 → day-49-daytime task.

The dataset is a two-day snapshot: ``train.csv`` holds all 96 quarter-hour
slots of day 48 plus day-49 morning slots, while ``test.csv`` holds day-49
*daytime* slots (``2:15``–``13:45`` → ``tod_slot ∈ [9, 55]``). The previous
``chronological_split`` / ``time_kfold_split`` validators folded within the
mixed day-48 + day-49-morning data and never reproduced the real task, which
inflated the reported R². They are replaced here by two task-mirroring
holdouts:

  - :func:`build_real_task_holdout` — train on day 48, evaluate on the day-49
    rows that fall inside the test daytime window. On the actual data the
    day-49 rows are morning-only (slots ``0..8``), so this validator reports
    ``FAILED_NO_MATCHING_SLOTS`` and the pipeline proceeds with the surrogate
    below (it does **not** raise / abort).
  - :func:`build_day48_daytime_holdout` — the **primary** model-selection
    surrogate. Evaluate on day-48 rows inside the test daytime window; train
    on the complement (day-48 outside the window + all day-49 rows). This
    matches the test time-of-day distribution.

Both functions return a :class:`HoldoutSplit`. The ``train_idx`` and
``eval_idx`` arrays hold the DataFrame's index *labels* (via
``df.index[mask].to_numpy()``) so callers can ``df.loc[...]`` them directly.
For a given split the train and eval partitions are **disjoint** by
construction. Every target-derived artifact (encodings, time-of-day curve,
imputers, transform choice, early stopping) used to score the eval partition
must be fit on the ``train_idx`` partition only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config

# Day labels present in the two-day snapshot.
_TRAIN_DAY = 48
_EVAL_DAY = 49

# Holdout-kind discriminators.
KIND_REAL_TASK = "real_task"
KIND_DAY48_SURROGATE = "day48_daytime_surrogate"

# Status discriminators.
STATUS_OK = "OK"
STATUS_FAILED_NO_MATCHING_SLOTS = "FAILED_NO_MATCHING_SLOTS"


@dataclass
class HoldoutSplit:
    """A task-mirroring train/eval holdout.

    Attributes:
        train_idx: DataFrame index labels of the training partition.
        eval_idx: DataFrame index labels of the evaluation partition. Disjoint
            from ``train_idx``. Empty when ``status`` is
            ``FAILED_NO_MATCHING_SLOTS``.
        status: ``"OK"`` when the eval partition is usable, otherwise
            ``"FAILED_NO_MATCHING_SLOTS"``.
        coverage_note: Human-readable explanation of how the eval partition was
            built and, on failure, why no matching slots were found.
        kind: ``"real_task"`` or ``"day48_daytime_surrogate"``.
    """

    train_idx: np.ndarray
    eval_idx: np.ndarray
    status: str
    coverage_note: str
    kind: str


def build_real_task_holdout(
    df: pd.DataFrame,
    day_col: str = "day",
    slot_col: str = "tod_slot",
    test_tod_min: int = config.TEST_TOD_SLOT_MIN,
    test_tod_max: int = config.TEST_TOD_SLOT_MAX,
) -> HoldoutSplit:
    """Build the aligned day-48 → day-49-daytime holdout (Req 2.1–2.4).

    Train on all day-48 rows; evaluate on day-49 rows whose ``tod_slot`` falls
    inside the test daytime window ``[test_tod_min, test_tod_max]``.

    Behavior:
      - Eval = day-49 rows with slot in ``[min, max]``. When that set is
        non-empty the status is ``"OK"`` (Req 2.3).
      - When day-49 rows exist but none fall inside the window, the status is
        ``"FAILED_NO_MATCHING_SLOTS"`` with the absence reported in
        ``coverage_note`` and an empty eval partition (Req 2.4). The function
        does **not** raise or abort — the pipeline logs and proceeds with the
        surrogate holdout.

    Args:
        df: Loaded train frame with ``day_col`` and ``slot_col`` columns.
        day_col: Name of the day column (48/49). Defaults to ``"day"``.
        slot_col: Name of the time-of-day slot column (0..95). Defaults to
            ``"tod_slot"``.
        test_tod_min: Inclusive lower slot bound of the test window.
        test_tod_max: Inclusive upper slot bound of the test window.

    Returns:
        HoldoutSplit with ``kind="real_task"``.
    """
    day = df[day_col]
    slot = df[slot_col]

    train_mask = (day == _TRAIN_DAY).to_numpy()
    day49_mask = (day == _EVAL_DAY).to_numpy()
    in_window = ((slot >= test_tod_min) & (slot <= test_tod_max)).to_numpy()
    eval_mask = day49_mask & in_window

    train_idx = df.index[train_mask].to_numpy()
    eval_idx = df.index[eval_mask].to_numpy()

    if eval_idx.size > 0:
        status = STATUS_OK
        coverage_note = (
            f"real_task holdout OK: train on {train_idx.size} day-{_TRAIN_DAY} "
            f"rows, evaluate on {eval_idx.size} day-{_EVAL_DAY} rows with "
            f"tod_slot in [{test_tod_min}, {test_tod_max}]."
        )
    else:
        status = STATUS_FAILED_NO_MATCHING_SLOTS
        day49_slots = slot[day49_mask]
        if day49_slots.size > 0:
            slot_lo = int(day49_slots.min())
            slot_hi = int(day49_slots.max())
            coverage_note = (
                f"FAILED_NO_MATCHING_SLOTS: day-{_EVAL_DAY} rows present "
                f"(slots {slot_lo}..{slot_hi}) but none within test window "
                f"[{test_tod_min}, {test_tod_max}]. Falling back to the "
                f"day-48 daytime surrogate for model selection."
            )
        else:
            coverage_note = (
                f"FAILED_NO_MATCHING_SLOTS: no day-{_EVAL_DAY} rows present, so "
                f"no aligned eval slots in [{test_tod_min}, {test_tod_max}]. "
                f"Falling back to the day-48 daytime surrogate for model "
                f"selection."
            )
        # No usable eval partition.
        eval_idx = np.array([], dtype=train_idx.dtype)

    return HoldoutSplit(
        train_idx=train_idx,
        eval_idx=eval_idx,
        status=status,
        coverage_note=coverage_note,
        kind=KIND_REAL_TASK,
    )


def build_day48_daytime_holdout(
    df: pd.DataFrame,
    day_col: str = "day",
    slot_col: str = "tod_slot",
    test_tod_min: int = config.TEST_TOD_SLOT_MIN,
    test_tod_max: int = config.TEST_TOD_SLOT_MAX,
) -> HoldoutSplit:
    """Build the day-48 daytime surrogate holdout (Req 2.1, 2.5 — PRIMARY).

    This is the **primary model-selection metric**: it matches the test
    time-of-day distribution (the same clock window as ``test.csv``) without
    needing day-49 labels.

      - Eval = day-48 rows whose ``tod_slot`` falls in
        ``[test_tod_min, test_tod_max]``.
      - Train = the complement: day-48 rows outside the window plus all day-49
        rows.

    All target-derived artifacts that score this holdout must be fit on the
    ``train_idx`` partition only (Req 2.5). The train and eval partitions are
    disjoint by construction (eval is exactly the in-window day-48 rows; train
    is everything else).

    Status is ``"OK"`` on populated data; if there are zero day-48 daytime rows
    the status is ``"FAILED_NO_MATCHING_SLOTS"`` and the absence is reported,
    but on the real data this window is populated.

    Args:
        df: Loaded train frame with ``day_col`` and ``slot_col`` columns.
        day_col: Name of the day column (48/49). Defaults to ``"day"``.
        slot_col: Name of the time-of-day slot column (0..95). Defaults to
            ``"tod_slot"``.
        test_tod_min: Inclusive lower slot bound of the test window.
        test_tod_max: Inclusive upper slot bound of the test window.

    Returns:
        HoldoutSplit with ``kind="day48_daytime_surrogate"``.
    """
    day = df[day_col]
    slot = df[slot_col]

    in_window = ((slot >= test_tod_min) & (slot <= test_tod_max)).to_numpy()
    eval_mask = (day == _TRAIN_DAY).to_numpy() & in_window
    # Train is the strict complement of the eval partition, so the two are
    # disjoint and together cover every row.
    train_mask = ~eval_mask

    train_idx = df.index[train_mask].to_numpy()
    eval_idx = df.index[eval_mask].to_numpy()

    if eval_idx.size > 0:
        status = STATUS_OK
        coverage_note = (
            f"day48_daytime_surrogate (PRIMARY): evaluate on {eval_idx.size} "
            f"day-{_TRAIN_DAY} rows with tod_slot in "
            f"[{test_tod_min}, {test_tod_max}]; train on the {train_idx.size}-row "
            f"complement (day-{_TRAIN_DAY} outside the window + all "
            f"day-{_EVAL_DAY} rows)."
        )
    else:
        status = STATUS_FAILED_NO_MATCHING_SLOTS
        coverage_note = (
            f"FAILED_NO_MATCHING_SLOTS: no day-{_TRAIN_DAY} rows with tod_slot "
            f"in [{test_tod_min}, {test_tod_max}]; surrogate holdout is empty."
        )

    return HoldoutSplit(
        train_idx=train_idx,
        eval_idx=eval_idx,
        status=status,
        coverage_note=coverage_note,
        kind=KIND_DAY48_SURROGATE,
    )
