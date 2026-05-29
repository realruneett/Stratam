"""Unit test for Task 5.8: contextual feature presence (Requirement 3.4).

Asserts that the six contextual columns

    RoadType, NumberofLanes, LargeVehicles, Landmarks, Temperature, Weather

appear in the feature set produced by :func:`src.features.build_features` and
survive into the model feature list returned by
:func:`src.features.select_feature_cols`.

This is a plain unit test, NOT one of the numbered correctness Properties
(Properties 1-14), so it intentionally carries no ``# ... Property N`` tag. It
includes a concrete hand-built example plus a light Hypothesis check over
generated raw frames.
"""

from __future__ import annotations

import pandas as pd
from hypothesis import HealthCheck, given, settings

from src.features import (
    CONTEXTUAL_COLS,
    build_features,
    build_imputers,
    select_feature_cols,
)
from src.spatial import fit_tod_curve, TargetEncoder
from tests import strategies as strat

# String categorical contextual columns handed to ``build_features`` for
# consistent integer encoding. ``NumberofLanes`` (complete int) and
# ``Temperature`` (float) are NOT integer-encoded; they pass through / are
# imputed as numeric features.
_CATEGORICAL_COLS = ["RoadType", "LargeVehicles", "Landmarks", "Weather"]


def _derive_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``hour``/``minute``/``tod_slot`` columns.

    Mirrors ``data.load_data``: the un-padded ``"H:M"`` ``timestamp`` is parsed
    into ``hour`` and ``minute``, and ``tod_slot = (hour*60 + minute) // 15`` in
    ``0..95``. ``build_features`` needs all three columns.
    """
    parts = df["timestamp"].astype(str).str.split(":", n=1, expand=True)
    out = df.copy()
    out["hour"] = parts[0].astype(int)
    out["minute"] = parts[1].astype(int)
    out["tod_slot"] = (out["hour"] * 60 + out["minute"]) // 15
    return out


def _fit_and_build(df: pd.DataFrame):
    """Fit encoder / curve / imputers on ``df`` and build its feature frame.

    Returns ``(feats, encoder, curve, impute_values)`` so callers can reuse the
    train-fitted artifacts to build a matching test feature frame.
    """
    encoder = TargetEncoder("geohash").fit(df, "demand")
    curve = fit_tod_curve(df, "demand")
    impute_values = build_imputers(df)
    feats = build_features(
        df,
        encoder=encoder,
        tod_curve=curve,
        categorical_cols=_CATEGORICAL_COLS,
        location_cols=["geohash"],
        impute_values=impute_values,
    )
    return feats, encoder, curve, impute_values


def _assert_contextual_cols_present_and_selected(
    feats: pd.DataFrame,
    test_feats: pd.DataFrame,
) -> None:
    """Assert the six contextual columns are in ``feats`` and survive selection."""
    for col in CONTEXTUAL_COLS:
        assert col in feats.columns, f"{col} missing from build_features output"

    selected = select_feature_cols(
        feats,
        test_feats,
        timestamp_col="timestamp",
        target_col="demand",
        id_col="Index",
        location_cols=["geohash"],
    )
    for col in CONTEXTUAL_COLS:
        assert col in selected, f"{col} dropped from select_feature_cols output"


def test_contextual_columns_present_concrete():
    """Concrete example: all six contextual columns are built and selected."""
    rows = [
        # geohash, day, slot, demand, RoadType, lanes, lv, lm, temp, weather
        ("qp02z1", 48, 0, 0.05, "Residential", 1, "Not Allowed", "No", 25.0, "Sunny"),
        ("qp02z1", 48, 40, 0.42, "Residential", 1, "Not Allowed", "No", 26.0, "Sunny"),
        ("qp02zt", 48, 9, 0.12, "Highway", 3, "Allowed", "Yes", 31.0, "Rainy"),
        ("qp02zt", 48, 55, 0.30, "Highway", 3, "Allowed", "Yes", 30.0, "Foggy"),
        # nulls in RoadType / Temperature / Weather exercise imputation.
        ("qp08bj", 48, 12, 0.03, None, 2, "Not Allowed", "No", None, None),
        ("qp02z1", 49, 4, 0.08, "Street", 1, "Not Allowed", "No", 24.0, "Snowy"),
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
    df = _derive_time_fields(pd.DataFrame(records, columns=strat.RAW_COLUMNS))

    feats, encoder, curve, impute_values = _fit_and_build(df)

    # A matching test frame: the same rows without the target column. Building
    # it via the train-fitted encoder/curve/imputers mirrors the submission
    # context and gives select_feature_cols a frame to intersect against.
    test_df = df.drop(columns=["demand"])
    test_feats = build_features(
        test_df,
        encoder=encoder,
        tod_curve=curve,
        categorical_cols=_CATEGORICAL_COLS,
        location_cols=["geohash"],
        impute_values=impute_values,
    )

    _assert_contextual_cols_present_and_selected(feats, test_feats)


# Light Hypothesis check: the property holds across generated raw frames too.
@given(df=strat.raw_frame(min_rows=4, max_rows=24, include_demand=True))
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.data_too_large, HealthCheck.too_slow],
)
def test_contextual_columns_present_generated(df):
    """For any generated raw frame, the six contextual columns are built/selected."""
    df = _derive_time_fields(df)

    feats, encoder, curve, impute_values = _fit_and_build(df)

    test_df = df.drop(columns=["demand"])
    test_feats = build_features(
        test_df,
        encoder=encoder,
        tod_curve=curve,
        categorical_cols=_CATEGORICAL_COLS,
        location_cols=["geohash"],
        impute_values=impute_values,
    )

    _assert_contextual_cols_present_and_selected(feats, test_feats)
