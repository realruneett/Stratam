"""Property-based test for the day-48 time-of-day curve (Task 3.5, Property 3).

This file implements the single property-based test for design Property 3 of the
demand-prediction-overhaul spec. It exercises ``src.spatial.fit_tod_curve`` across
many generated raw-schema frames that span BOTH day 48 and day 49 (via the shared
``two_day_frame`` strategy), so there are day-49 rows available to mutate.

The shared strategies emit *raw-schema* frames with a ``day`` column and a raw
``timestamp`` (``"H:M"``) string, but no parsed ``tod_slot``. ``load_data`` would
normally derive ``tod_slot = (hour*60 + minute) // 15``; here we reproduce that
derivation in the test before calling ``fit_tod_curve``.

The property has two sub-claims, both asserted in the one test below:

  (a) Correctness: for each slot present in the returned curve, the curve value
      equals the mean demand of the fitting partition's day-48 rows at that slot.
  (b) Leak-free: mutating ALL day-49 target values does not change the curve (the
      curve is fit from day-48 rows only, so day-49 targets cannot leak in).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings

from src.spatial import TOD_CURVE_FIT_DAY, fit_tod_curve
from tests import strategies as strat

_ATOL = 1e-9


def _add_tod_slot(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the integer ``tod_slot`` (0..95) from the raw ``"H:M"`` timestamp.

    Mirrors ``load_data``'s ``tod_slot = (hour*60 + minute) // 15`` derivation so
    the raw-schema strategy frames can be fed straight into ``fit_tod_curve``.
    """
    out = df.copy()
    parts = out["timestamp"].str.split(":", expand=True)
    hour = parts[0].astype(int)
    minute = parts[1].astype(int)
    out["tod_slot"] = (hour * 60 + minute) // 15
    return out


# Feature: demand-prediction-overhaul, Property 3: Day-48 time-of-day curve is correct and leak-free
@given(df=strat.two_day_frame(max_rows_per_day=15))
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_tod_curve_is_correct_and_leak_free(df: pd.DataFrame) -> None:
    df = _add_tod_slot(df.reset_index(drop=True))

    curve = fit_tod_curve(df, target_col="demand")

    # ── Sub-claim (a): correctness ───────────────────────────────
    # Each curve value equals the mean day-48 demand at that slot.
    day48 = df[df["day"] == TOD_CURVE_FIT_DAY]
    for slot, value in curve.items():
        expected = day48.loc[day48["tod_slot"] == slot, "demand"].mean()
        assert np.isclose(value, expected, atol=_ATOL), (
            f"curve[{slot}]={value} != mean day-48 demand {expected} at that slot"
        )

    # The curve only covers slots actually observed on day 48.
    assert set(curve.index) == set(day48["tod_slot"].unique())

    # ── Sub-claim (b): leak-free w.r.t. day-49 targets ───────────
    # Mutating ALL day-49 demand values must not change the curve.
    df_mut = df.copy()
    day49_mask = (df_mut["day"] != TOD_CURVE_FIT_DAY).to_numpy()
    # Push day-49 targets far outside the [0, 1] target range so any leak would
    # be obvious; this exercises the mutation regardless of original values.
    df_mut.loc[day49_mask, "demand"] = df_mut.loc[day49_mask, "demand"] + 1000.0

    curve_mut = fit_tod_curve(df_mut, target_col="demand")

    # Identical index (same slots) and identical values within tolerance.
    assert list(curve.index) == list(curve_mut.index), (
        "mutating day-49 targets changed the curve's slot index"
    )
    np.testing.assert_allclose(
        curve.to_numpy(),
        curve_mut.to_numpy(),
        atol=_ATOL,
        err_msg="mutating day-49 targets changed the day-48 time-of-day curve",
    )
