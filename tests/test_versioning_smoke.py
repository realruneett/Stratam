"""End-to-end versioning + smoke test for the demand-prediction-overhaul.

Task 12.4 (integration / smoke — NOT a Property 1-14 test). This drives the real
``run.py`` orchestrator end-to-end on a *tiny* synthetic dataset (CPU only,
minimal data so the GBMs early-stop quickly) and asserts the versioning and
submission contracts hold:

  * the versioned run id increments (a pre-existing ``submission_1.csv`` forces
    ``RUN_ID = 2``), exercising the ``max(existing) + 1`` logic in ``run.py``;
  * ``submission_2.csv`` and ``metrics_2.json`` are written, plus the
    ``submission.csv`` / ``metrics.json`` copies;
  * a valid submission is produced — exactly the test-row count with the two
    columns ``Index`` and ``demand`` and no nulls;
  * ``metrics`` records ``leaderboard_top == 93.13`` and the honest-reporting
    keys (``local_score``, ``cv_lb_gap``, ``ensemble_holdout_r2``,
    ``real_task_validation_status``).

The 99.99 ceiling is a documented non-goal (Requirement 6.3) and carries no
automated assertion — only the ``leaderboard_top`` target of 93.13 is recorded.

``run.py`` is a script (it executes on import) and reads ``./data/{train,test}.csv``
and writes ``./submission_*.csv`` relative to the *current working directory*, so
it is launched as a subprocess with ``cwd`` set to a throwaway temp dir. Running
``python <project>/run.py`` puts the project root on ``sys.path`` (so ``import
config`` / ``import src.*`` resolve) while keeping all file IO inside the temp
dir — nothing in the real repo is touched.

Validates: Requirements 5.7, 5.8, 6.2, 6.3
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from tests.strategies import (
    RAW_COLUMNS,
    RAW_COLUMNS_NO_DEMAND,
    slot_to_timestamp,
)

# Generous ceiling: tiny data early-stops fast, but the three GBMs each run an
# OOF loop plus a final fit, so allow plenty of headroom on a cold CPU.
_SUBPROCESS_TIMEOUT_S = 600

# Geohashes shared by train + test (so encoders see overlapping groups).
_GEOHASHES = ["qp02z1", "qp02zt", "qp08bj", "qp090x"]
# A geohash present only in test, to exercise the unseen-geohash fallback path.
_UNSEEN_GEOHASH = "zzzzzz"

# Day-48 daytime slots (in the test window [9, 55]) -> surrogate EVAL partition.
_DAY48_DAYTIME_SLOTS = [9, 14, 20, 27, 34, 41, 48, 55]
# Day-48 slots OUTSIDE the window -> part of the surrogate TRAIN complement.
_DAY48_OUTSIDE_SLOTS = [0, 2, 4, 6, 8, 60, 72, 84, 95]
# Day-49 morning slots (0..8) -> train rows; none fall in [9, 55] so the
# real-task holdout reports FAILED_NO_MATCHING_SLOTS and the pipeline proceeds.
_DAY49_MORNING_SLOTS = [0, 3, 6, 8]
# Day-49 daytime slots (the real test window) -> test.csv rows.
_TEST_SLOTS = [9, 18, 27, 36, 45, 55]

_ROAD_TYPES = ["Highway", "Residential", "Street"]
_WEATHERS = ["Sunny", "Rainy", "Foggy", "Snowy"]
_LARGE_VEHICLES = ["Allowed", "Not Allowed"]
_LANDMARKS = ["Yes", "No"]


def _demand(rng: np.random.Generator, base: float, slot: int) -> float:
    """A continuous, mildly-structured demand value in ``(0, 1]``.

    The slot-dependent term gives the day-48 time-of-day curve and the GBMs a
    bit of real signal so early stopping converges quickly on tiny data.
    """
    value = base + 0.3 * (slot / 95.0) + rng.normal(0.0, 0.02)
    return float(np.clip(value, 0.01, 1.0))


def _context(rng: np.random.Generator, i: int, *, nullable: bool) -> dict:
    """Build the contextual columns for one row, optionally injecting nulls."""
    road = _ROAD_TYPES[i % len(_ROAD_TYPES)]
    weather = _WEATHERS[i % len(_WEATHERS)]
    temp = float(round(10.0 + (i % 30), 2))
    # Inject a sparse scattering of nulls into the nullable columns (Req 3.6)
    # without ever nulling a whole column.
    if nullable and i % 11 == 0:
        road = None
    if nullable and i % 13 == 0:
        weather = None
    if nullable and i % 17 == 0:
        temp = None
    return {
        "RoadType": road,
        "NumberofLanes": int(1 + (i % 5)),
        "LargeVehicles": _LARGE_VEHICLES[i % len(_LARGE_VEHICLES)],
        "Landmarks": _LANDMARKS[i % len(_LANDMARKS)],
        "Temperature": temp,
        "Weather": weather,
    }


def _build_synthetic_train() -> pd.DataFrame:
    """A tiny train frame: day-48 (in/out of window) + day-49 morning rows."""
    rng = np.random.default_rng(0)
    bases = {gh: 0.1 + 0.12 * j for j, gh in enumerate(_GEOHASHES)}
    rows: list[dict] = []
    i = 0
    for gh in _GEOHASHES:
        # Day-48 daytime rows -> surrogate eval partition.
        for slot in _DAY48_DAYTIME_SLOTS:
            rows.append(_row(i, gh, 48, slot, _demand(rng, bases[gh], slot)))
            i += 1
        # Day-48 outside-window rows -> surrogate train complement.
        for slot in _DAY48_OUTSIDE_SLOTS:
            rows.append(_row(i, gh, 48, slot, _demand(rng, bases[gh], slot)))
            i += 1
        # Day-49 morning rows -> train; keep the real-task window empty.
        for slot in _DAY49_MORNING_SLOTS:
            rows.append(_row(i, gh, 49, slot, _demand(rng, bases[gh], slot)))
            i += 1
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def _build_synthetic_test() -> pd.DataFrame:
    """A tiny test frame: day-49 daytime rows (no demand), with one unseen gh."""
    rows: list[dict] = []
    i = 0
    for gh in _GEOHASHES:
        for slot in _TEST_SLOTS:
            rows.append(_row(i, gh, 49, slot, None, include_demand=False))
            i += 1
    # One unseen-geohash row to exercise the train-derived fallback (Req 3.7).
    rows.append(_row(i, _UNSEEN_GEOHASH, 49, 27, None, include_demand=False))
    return pd.DataFrame(rows, columns=RAW_COLUMNS_NO_DEMAND)


def _row(
    index: int,
    geohash: str,
    day: int,
    slot: int,
    demand: float | None,
    *,
    include_demand: bool = True,
) -> dict:
    """Assemble a single raw-schema row matching ``train.csv`` / ``test.csv``."""
    rng = np.random.default_rng(index + 1)
    row = {
        "Index": index,
        "geohash": geohash,
        "day": day,
        "timestamp": slot_to_timestamp(slot),
    }
    if include_demand:
        row["demand"] = demand
    row.update(_context(rng, index, nullable=True))
    return row


def _venv_python(project_root: str) -> str | None:
    """Resolve the interpreter for the subprocess.

    Prefers the current interpreter (the WSL venv python already running the
    tests); falls back to the documented ``./.venv/bin/python``. Returns
    ``None`` when neither is usable so the caller can skip gracefully.
    """
    if sys.executable and os.path.exists(sys.executable):
        return sys.executable
    candidate = os.path.join(project_root, ".venv", "bin", "python")
    return candidate if os.path.exists(candidate) else None


def test_versioning_and_end_to_end_smoke(tmp_path):
    """Run ``run.py`` on tiny data; assert versioning + submission contracts."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_script = os.path.join(project_root, "run.py")
    assert os.path.exists(run_script), f"run.py not found at {run_script}"

    python_exe = _venv_python(project_root)
    if python_exe is None:
        pytest.skip("No usable Python interpreter (venv) found for the subprocess.")

    # ── Lay out the throwaway working directory: ./data/{train,test}.csv ──
    work_dir = tmp_path
    data_dir = work_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_df = _build_synthetic_train()
    test_df = _build_synthetic_test()
    train_df.to_csv(data_dir / "train.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)
    n_test_rows = len(test_df)

    # Pre-seed a dummy submission_1.csv so RUN_ID must increment to 2 (Req 5.7).
    (work_dir / "submission_1.csv").write_text("Index,demand\n0,0.5\n")

    # ── Launch run.py in the temp dir (CPU). PYTHONPATH set for robustness. ──
    env = dict(os.environ)
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    # Hide any GPU so the pipeline exercises the documented CPU fallback.
    env["CUDA_VISIBLE_DEVICES"] = ""

    proc = subprocess.run(
        [python_exe, run_script],
        cwd=str(work_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_S,
    )

    if proc.returncode != 0:
        raise AssertionError(
            "run.py failed (returncode "
            f"{proc.returncode}).\n--- STDOUT tail ---\n"
            f"{proc.stdout[-3000:]}\n--- STDERR tail ---\n{proc.stderr[-3000:]}"
        )

    # ── Versioned outputs incremented to run 2 (Req 5.7) ──
    submission_2 = work_dir / "submission_2.csv"
    metrics_2 = work_dir / "metrics_2.json"
    assert submission_2.exists(), (
        "submission_2.csv was not written (run id did not increment).\n"
        f"STDOUT tail:\n{proc.stdout[-2000:]}"
    )
    assert metrics_2.exists(), "metrics_2.json was not written."

    # ── submission.csv / metrics.json copies (Req 5.8) ──
    assert (work_dir / "submission.csv").exists(), "submission.csv copy missing."
    assert (work_dir / "metrics.json").exists(), "metrics.json copy missing."

    # ── Valid submission: row count, columns, null-free (Req 5.1-5.3) ──
    submission = pd.read_csv(submission_2)
    assert list(submission.columns) == ["Index", "demand"], (
        f"Unexpected submission columns: {list(submission.columns)}"
    )
    assert len(submission) == n_test_rows, (
        f"Submission has {len(submission)} rows; expected {n_test_rows}."
    )
    assert submission.isnull().sum().sum() == 0, "Submission contains nulls."

    # ── Metrics: leaderboard_top recorded + honest-reporting keys (Req 6.2/6.3) ──
    metrics = json.loads(metrics_2.read_text())
    assert metrics["leaderboard_top"] == 93.13, (
        f"leaderboard_top = {metrics.get('leaderboard_top')!r}; expected 93.13."
    )
    for key in (
        "local_score",
        "cv_lb_gap",
        "ensemble_holdout_r2",
        "real_task_validation_status",
        "run_id",
    ):
        assert key in metrics, f"metrics is missing required key {key!r}."

    # Run id recorded in metrics matches the incremented version (Req 5.7).
    assert metrics["run_id"] == 2, (
        f"metrics run_id = {metrics.get('run_id')!r}; expected 2."
    )
