"""Property-based test for cyclical time-of-day encoding (Task 5.5, Property 4).

This file implements the single property-based test for design Property 4 of the
demand-prediction-overhaul spec. It drives the real feature builder
(``src.features.build_features``) and validates that the time-of-day cyclical
encodings it emits form a unit-circle, periodic mapping.

The shared strategies emit *raw-schema* frames with a ``day`` column and a raw
``timestamp`` (``"H:M"``) string, but no parsed ``hour``/``minute``/``tod_slot``.
``load_data`` would normally derive ``hour``, ``minute`` and
``tod_slot = (hour*60 + minute) // 15``; here we reproduce that derivation in the
test before calling ``build_features`` so the test validates the builder's real
output.

The property has two sub-claims, both asserted below:

  (a) Unit circle: for every produced row, ``tod_sin² + tod_cos² ≈ 1`` and
      ``minute_of_day_sin² + minute_of_day_cos² ≈ 1`` (atol 1e-9). The builder
      output is also pinned to the exact formula it documents
      (``sin/cos(2π·tod_slot/96)`` and ``sin/cos(2π·minute_of_day/1440)``).
  (b) Periodicity: slots that differ by a full period map to the same point.
      For each row's ``tod_slot`` s the encoding of s equals that of ``s + 96``
      (period 96), and for each row's ``minute_of_day`` m the encoding of m
      equals that of ``m + 1440`` (period 1440), within tolerance.

To guarantee coverage of *every* slot ``0..95`` (rather than only the slots a
generated frame happens to hit), an explicit deterministic frame containing one
row per slot on day 48 is also built and checked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings

from src.features import build_features, build_imputers
from src.spatial import TargetEncoder, fit_tod_curve
from tests import strategies as strat

_ATOL = 1e-9

# Periods of the two cyclical encodings produced by build_features.
_SLOTS_PER_DAY = 96      # period of tod_slot  -> tod_sin / tod_cos
_MINUTES_PER_DAY = 1440  # period of minute_of_day -> minute_of_day_sin / cos

# The string categorical columns build_features integer-encodes, and the raw
# location column whose signal enters only via the leak-free encodings.
_CATEGORICAL_COLS = ["RoadType", "LargeVehicles", "Landmarks", "Weather"]
_LOCATION_COLS = ["geohash"]


def _add_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ``hour``/``minute``/``tod_slot`` from the raw ``"H:M"`` timestamp.

    Mirrors ``load_data``'s derivation so the raw-schema strategy frames can be
    fed straight into ``build_features``.
    """
    out = df.copy()
    parts = out["timestamp"].str.split(":", n=1, expand=True)
    out["hour"] = parts[0].astype(int)
    out["minute"] = parts[1].astype(int)
    out["tod_slot"] = (out["hour"] * 60 + out["minute"]) // 15
    return out


def _build(df: pd.DataFrame) -> pd.DataFrame:
    """Fit the leak-free artifacts and run the real ``build_features``.

    Returns the feature-engineered frame so the test inspects the builder's
    actual cyclical-encoding output (not a re-implementation).
    """
    encoder = TargetEncoder("geohash").fit(df, target_col="demand")
    tod_curve = fit_tod_curve(df, target_col="demand")
    impute_values = build_imputers(df)

    return build_features(
        df,
        encoder=encoder,
        tod_curve=tod_curve,
        categorical_cols=_CATEGORICAL_COLS,
        location_cols=_LOCATION_COLS,
        impute_values=impute_values,
    )


def _assert_unit_circle_and_periodic(feats: pd.DataFrame) -> None:
    """Assert both sub-claims of Property 4 on a feature-engineered frame."""
    slot = feats["tod_slot"].to_numpy(dtype=float)
    minute_of_day = (
        feats["hour"].to_numpy(dtype=float) * 60
        + feats["minute"].to_numpy(dtype=float)
    )

    tod_sin = feats["tod_sin"].to_numpy(dtype=float)
    tod_cos = feats["tod_cos"].to_numpy(dtype=float)
    mod_sin = feats["minute_of_day_sin"].to_numpy(dtype=float)
    mod_cos = feats["minute_of_day_cos"].to_numpy(dtype=float)

    # ── Sub-claim (a): unit-circle mapping ───────────────────────
    # Every produced (sin, cos) pair lies on the unit circle.
    np.testing.assert_allclose(
        tod_sin ** 2 + tod_cos ** 2,
        np.ones_like(tod_sin),
        atol=_ATOL,
        err_msg="tod_sin^2 + tod_cos^2 != 1 for some slot",
    )
    np.testing.assert_allclose(
        mod_sin ** 2 + mod_cos ** 2,
        np.ones_like(mod_sin),
        atol=_ATOL,
        err_msg="minute_of_day_sin^2 + minute_of_day_cos^2 != 1 for some row",
    )

    # Pin the builder output to its documented formula so the periodicity check
    # below (computed directly via numpy) is about the SAME mapping the builder
    # uses, not an unrelated one.
    np.testing.assert_allclose(
        tod_sin, np.sin(2 * np.pi * slot / _SLOTS_PER_DAY), atol=_ATOL
    )
    np.testing.assert_allclose(
        tod_cos, np.cos(2 * np.pi * slot / _SLOTS_PER_DAY), atol=_ATOL
    )
    np.testing.assert_allclose(
        mod_sin, np.sin(2 * np.pi * minute_of_day / _MINUTES_PER_DAY), atol=_ATOL
    )
    np.testing.assert_allclose(
        mod_cos, np.cos(2 * np.pi * minute_of_day / _MINUTES_PER_DAY), atol=_ATOL
    )

    # ── Sub-claim (b): full-period invariance ────────────────────
    # Slots differing by a full period (96) map to the same point; likewise for
    # minute_of_day differing by a full period (1440). Since real slots are
    # 0..95, this validates the periodicity of the encoding function used by the
    # builder (pinned to be identical above) for each present slot.
    np.testing.assert_allclose(
        np.sin(2 * np.pi * slot / _SLOTS_PER_DAY),
        np.sin(2 * np.pi * (slot + _SLOTS_PER_DAY) / _SLOTS_PER_DAY),
        atol=_ATOL,
        err_msg="tod_sin not invariant under a full +96-slot period shift",
    )
    np.testing.assert_allclose(
        np.cos(2 * np.pi * slot / _SLOTS_PER_DAY),
        np.cos(2 * np.pi * (slot + _SLOTS_PER_DAY) / _SLOTS_PER_DAY),
        atol=_ATOL,
        err_msg="tod_cos not invariant under a full +96-slot period shift",
    )
    np.testing.assert_allclose(
        np.sin(2 * np.pi * minute_of_day / _MINUTES_PER_DAY),
        np.sin(2 * np.pi * (minute_of_day + _MINUTES_PER_DAY) / _MINUTES_PER_DAY),
        atol=_ATOL,
        err_msg="minute_of_day_sin not invariant under a full +1440-minute period shift",
    )
    np.testing.assert_allclose(
        np.cos(2 * np.pi * minute_of_day / _MINUTES_PER_DAY),
        np.cos(2 * np.pi * (minute_of_day + _MINUTES_PER_DAY) / _MINUTES_PER_DAY),
        atol=_ATOL,
        err_msg="minute_of_day_cos not invariant under a full +1440-minute period shift",
    )


def _all_slots_frame() -> pd.DataFrame:
    """A deterministic raw-schema frame with one row per slot ``0..95`` (day 48).

    Guarantees every time-of-day slot is exercised by the unit-circle check,
    independent of what the Hypothesis generator happens to draw.
    """
    rows = []
    for slot in range(_SLOTS_PER_DAY):
        rows.append({
            "Index": slot,
            "geohash": "qp02z1",
            "day": 48,
            "timestamp": strat.slot_to_timestamp(slot),
            "demand": 0.25,
            "RoadType": "Residential",
            "NumberofLanes": 2,
            "LargeVehicles": "Not Allowed",
            "Landmarks": "No",
            "Temperature": 25.0,
            "Weather": "Sunny",
        })
    return pd.DataFrame(rows, columns=strat.RAW_COLUMNS)


# Feature: demand-prediction-overhaul, Property 4: Time-of-day cyclical encoding is a unit-circle, periodic mapping
def test_cyclical_encoding_covers_all_96_slots() -> None:
    """Every slot in ``0..95`` maps onto the unit circle and is periodic."""
    df = _add_time_fields(_all_slots_frame())
    feats = _build(df)

    # All 96 slots are present exactly once.
    assert sorted(feats["tod_slot"].tolist()) == list(range(_SLOTS_PER_DAY))

    _assert_unit_circle_and_periodic(feats)


# Feature: demand-prediction-overhaul, Property 4: Time-of-day cyclical encoding is a unit-circle, periodic mapping
@given(
    df=strat.raw_frame(
        min_rows=1,
        max_rows=40,
        days=(48, 49),
        slot_min=0,
        slot_max=95,
        include_demand=True,
        nullable=True,
    )
)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_cyclical_encoding_is_unit_circle_and_periodic(df: pd.DataFrame) -> None:
    df = _add_time_fields(df.reset_index(drop=True))
    feats = _build(df)
    _assert_unit_circle_and_periodic(feats)
