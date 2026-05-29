"""Property-based test for train-derived, null-only imputation (Task 5.7, Property 6).

This file implements the single property-based test for design Property 6 of the
demand-prediction-overhaul spec, driving the real feature builder
(``src.features.build_features``) and the real imputer fitter
(``src.features.build_imputers``) across many generated raw-schema frames that
are guaranteed (via ``tests.strategies.frame_with_injected_nulls``) to contain at
least one injected null in a nullable contextual column
(``RoadType`` / ``Temperature`` / ``Weather``).

Property 6 (from design.md):
    For any frame with injected nulls in ``RoadType``, ``Temperature``, or
    ``Weather``, after imputation those columns SHALL contain no nulls, every
    originally non-null cell SHALL be unchanged, and every imputed cell SHALL
    equal the train-derived imputation value (the ``"Missing"`` category for
    ``RoadType`` / ``Weather``, the train median for ``Temperature``).

Design of the test (mirrors the leakage boundary in the pipeline):

  1. Generate a raw frame with injected nulls and derive the time-of-day fields
     (``hour`` / ``minute`` / ``tod_slot``) exactly as ``src.data.load_data`` does.
  2. Record the null masks for ``Temperature`` / ``RoadType`` / ``Weather`` and
     the original (non-null) values BEFORE any imputation runs.
  3. Fit the imputers with ``build_imputers`` on this frame — the returned
     ``Temperature_median`` / ``"Missing"`` category are therefore TRAIN-DERIVED
     values, independent of the per-row cell being filled.
  4. Fit the (also train-derived) ``TargetEncoder`` and ``fit_tod_curve`` that
     ``build_features`` requires, then run ``build_features`` on a COPY of the
     frame to obtain the encoded output.

Because ``build_features`` integer-encodes the categoricals AFTER imputing, the
imputed ``"Missing"`` cells for ``RoadType`` / ``Weather`` surface as the integer
code of ``"Missing"`` in the supplied ``category_maps``. The assertions below
therefore compare encoded codes against ``category_maps[col].index(...)``:

  * originally-null ``RoadType`` / ``Weather`` cell -> code of ``"Missing"``
  * originally-non-null cell                        -> code of its raw value

``Temperature`` stays numeric (it is not coded), so it is checked directly:
originally-null cells equal the train median; originally-non-null cells are
unchanged within tolerance. ``NumberofLanes`` is complete (never imputed) and is
asserted to be unchanged entirely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings

from config import TE_SMOOTHING_ALPHA
from src.features import MISSING_CATEGORY, build_features, build_imputers
from src.spatial import TargetEncoder, fit_tod_curve
from tests import strategies as strat

_ATOL = 1e-9

# The string categoricals passed to build_features. ``geohash`` is intentionally
# excluded (its signal enters only via the leak-free encodings, not a code).
_CATEGORICAL_COLS = ["RoadType", "LargeVehicles", "Landmarks", "Weather"]
_LOCATION_COLS = ["geohash"]


def _add_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ``hour`` / ``minute`` / ``abs_time`` / ``tod_slot`` from ``timestamp``.

    Mirrors ``src.data.load_data`` so the raw-schema strategy frame carries the
    exact time-of-day fields ``build_features`` and ``fit_tod_curve`` expect.
    Row order / index are reset and preserved so the recorded null masks stay
    aligned with the encoded output.
    """
    df = df.reset_index(drop=True).copy()
    parts = df["timestamp"].astype(str).str.split(":", n=1, expand=True)
    df["hour"] = parts[0].astype(int)
    df["minute"] = parts[1].astype(int)
    minute_of_day = df["hour"] * 60 + df["minute"]
    df["abs_time"] = df["day"] * 1440 + minute_of_day
    df["tod_slot"] = minute_of_day // 15
    return df


# Feature: demand-prediction-overhaul, Property 6: Imputation uses train-derived values and touches only null cells
@given(df=strat.frame_with_injected_nulls(max_rows=30))
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.function_scoped_fixture,
    ],
)
def test_imputation_is_train_derived_and_null_only(df: pd.DataFrame) -> None:
    raw = _add_time_fields(df)

    # ── Record null masks + original values BEFORE build_features ──
    temp_null_mask = raw["Temperature"].isna().to_numpy()
    roadtype_null_mask = raw["RoadType"].isna().to_numpy()
    weather_null_mask = raw["Weather"].isna().to_numpy()

    orig_temp = pd.to_numeric(raw["Temperature"], errors="coerce").to_numpy()
    orig_roadtype = raw["RoadType"].tolist()
    orig_weather = raw["Weather"].tolist()
    orig_lanes = raw["NumberofLanes"].to_numpy()

    # ── Train-derived imputers (the values come from build_imputers, not the
    #    per-row cell being filled) ──────────────────────────────
    impute_values = build_imputers(raw)
    temp_median = impute_values["Temperature_median"]
    assert impute_values["missing_category"] == MISSING_CATEGORY
    # Cross-check the train-derived median against the partition's own median.
    expected_median = pd.to_numeric(raw["Temperature"], errors="coerce").median()
    if pd.notna(expected_median):
        assert temp_median == float(expected_median)

    # ── Consistent category maps, "Missing" appended for nullable cats ──
    category_maps = {
        "RoadType": list(strat.ROAD_TYPES) + [MISSING_CATEGORY],
        "Weather": list(strat.WEATHERS) + [MISSING_CATEGORY],
        "LargeVehicles": list(strat.LARGE_VEHICLES),
        "Landmarks": list(strat.LANDMARKS),
    }

    # ── Fit the (train-derived) encoder + ToD curve required by build_features ──
    encoder = TargetEncoder("geohash").fit(raw, "demand", TE_SMOOTHING_ALPHA)
    curve = fit_tod_curve(raw, "demand")

    # ── Build features on a COPY → encoded frame ─────────────────
    feats = build_features(
        raw.copy(),
        encoder,
        curve,
        categorical_cols=_CATEGORICAL_COLS,
        location_cols=_LOCATION_COLS,
        impute_values=impute_values,
        category_maps=category_maps,
    )

    # ── Temperature: numeric, not encoded ────────────────────────
    temp_out = pd.to_numeric(feats["Temperature"], errors="coerce")
    assert temp_out.notna().all(), "Temperature still contains nulls after imputation"
    temp_vals = temp_out.to_numpy()

    if temp_null_mask.any():
        # Imputed cells equal the train-derived median.
        np.testing.assert_allclose(
            temp_vals[temp_null_mask],
            temp_median,
            atol=_ATOL,
            err_msg="Imputed Temperature cells != train-derived median",
        )
    if (~temp_null_mask).any():
        # Originally non-null cells are unchanged.
        np.testing.assert_allclose(
            temp_vals[~temp_null_mask],
            orig_temp[~temp_null_mask],
            atol=_ATOL,
            err_msg="Originally non-null Temperature cell was modified",
        )

    # ── RoadType / Weather: integer-encoded, "Missing" for null cells ──
    for col, null_mask, orig_vals in (
        ("RoadType", roadtype_null_mask, orig_roadtype),
        ("Weather", weather_null_mask, orig_weather),
    ):
        codes = feats[col].to_numpy()
        # No nulls: every cell maps to a real category code (never the -1
        # "value not in categories" sentinel produced by pd.Categorical).
        assert (codes != -1).all(), f"{col} has unmapped (null) cells after imputation"

        missing_code = category_maps[col].index(MISSING_CATEGORY)
        for i in range(len(raw)):
            if null_mask[i]:
                # Imputed cell equals the train-derived "Missing" category code.
                assert codes[i] == missing_code, (
                    f"{col} originally-null row {i} did not map to 'Missing' "
                    f"(got code {codes[i]}, expected {missing_code})"
                )
            else:
                # Originally non-null cell is unchanged (maps to its own value via
                # the SAME pd.Categorical(categories=cats).codes mapping).
                expected = category_maps[col].index(orig_vals[i])
                assert codes[i] == expected, (
                    f"{col} originally-non-null row {i} changed value "
                    f"(got code {codes[i]}, expected {expected} for {orig_vals[i]!r})"
                )

    # ── NumberofLanes: complete column, never imputed → unchanged entirely ──
    np.testing.assert_array_equal(
        feats["NumberofLanes"].to_numpy(),
        orig_lanes,
        err_msg="NumberofLanes (a complete column) was modified by imputation",
    )
