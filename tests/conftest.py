"""Shared pytest fixtures for the demand-prediction-overhaul test suite.

Fixtures here provide:

* project path / import access (so ``config`` and ``src`` import cleanly),
* deterministic seeding via the single ``seed_everything`` entry point (Req 6.4),
* small concrete raw-schema example frames built from the shared strategies,
  for unit tests that want a quick fixed input without driving Hypothesis.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

# Make the project root importable so ``import config`` and ``import src.*``
# resolve regardless of pytest's rootdir handling.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tests import strategies as strat  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> str:
    """Absolute path to the project root directory."""
    return _PROJECT_ROOT


@pytest.fixture(autouse=True)
def _seed_each_test():
    """Seed every test deterministically through the single seeding entry point.

    Keeps property/unit tests reproducible (Req 6.4) without each test having
    to remember to call ``seed_everything`` itself.
    """
    from config import SEED, seed_everything
    seed_everything(SEED)
    yield


@pytest.fixture
def raw_train_example() -> pd.DataFrame:
    """A small, fixed raw-schema training frame (day 48 + day-49 morning).

    Built with concrete values (not via Hypothesis) for quick unit tests.
    Mirrors ``train.csv`` columns and the un-padded ``H:M`` timestamp format.
    """
    rows = [
        # geohash, day, slot, demand, RoadType, lanes, lv, lm, temp, weather
        ("qp02z1", 48, 0, 0.05, "Residential", 1, "Not Allowed", "No", 25.0, "Sunny"),
        ("qp02z1", 48, 40, 0.42, "Residential", 1, "Not Allowed", "No", 26.0, "Sunny"),
        ("qp02zt", 48, 0, 0.12, "Highway", 3, "Allowed", "Yes", 31.0, "Rainy"),
        ("qp02zt", 48, 55, 0.30, "Highway", 3, "Allowed", "Yes", 30.0, "Foggy"),
        ("qp08bj", 48, 12, 0.03, None, 2, "Not Allowed", "No", None, None),
        ("qp02z1", 49, 4, 0.08, "Residential", 1, "Not Allowed", "No", 24.0, "Snowy"),
        ("qp02zt", 49, 8, 0.21, "Street", 2, "Allowed", "No", 22.0, "Sunny"),
    ]
    records = []
    for i, (gh, day, slot, dem, rt, lanes, lv, lm, temp, wx) in enumerate(rows):
        records.append({
            "Index": i,
            "geohash": gh,
            "day": day,
            "timestamp": strat.slot_to_timestamp(slot),
            "demand": dem,
            "RoadType": rt,
            "NumberofLanes": lanes,
            "LargeVehicles": lv,
            "Landmarks": lm,
            "Temperature": temp,
            "Weather": wx,
        })
    return pd.DataFrame(records, columns=strat.RAW_COLUMNS)


@pytest.fixture
def raw_test_example() -> pd.DataFrame:
    """A small, fixed raw-schema test frame (day-49 daytime, no demand).

    Includes one geohash unseen in ``raw_train_example`` ("zzzzzz") to
    exercise the unseen-geohash fallback path.
    """
    rows = [
        ("qp02z1", 49, 9, "Residential", 1, "Not Allowed", "No", 18.0, "Sunny"),
        ("qp02zt", 49, 20, "Highway", 3, "Allowed", "Yes", 15.0, "Rainy"),
        ("zzzzzz", 49, 55, None, 2, "Not Allowed", "No", None, None),
    ]
    records = []
    for i, (gh, day, slot, rt, lanes, lv, lm, temp, wx) in enumerate(rows):
        records.append({
            "Index": i,
            "geohash": gh,
            "day": day,
            "timestamp": strat.slot_to_timestamp(slot),
            "RoadType": rt,
            "NumberofLanes": lanes,
            "LargeVehicles": lv,
            "Landmarks": lm,
            "Temperature": temp,
            "Weather": wx,
        })
    return pd.DataFrame(records, columns=strat.RAW_COLUMNS_NO_DEMAND)
