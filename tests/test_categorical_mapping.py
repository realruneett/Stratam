"""Property-based test for consistent categorical mapping (Task 5.6, Property 5).

This file implements the single property-based test for design Property 5 of the
demand-prediction-overhaul spec: *categorical mapping is consistent across train
and test*. It drives the real feature builder
(:func:`src.features.build_features`) over many generated ``(train_df, test_df)``
pairs that share categorical vocabularies (the shared Hypothesis strategy
``train_test_pair`` draws every categorical column from the same fixed vocab
lists, so category overlap between the two frames is guaranteed).

Property statement
------------------
*For any* pair of train and test frames, every category value present in BOTH
frames SHALL map to an identical integer code in the encoded train frame and the
encoded test frame.

How the property is exercised
-----------------------------
``build_features`` integer-encodes the string categorical columns
(``RoadType``, ``LargeVehicles``, ``Landmarks``, ``Weather`` — NOT ``geohash``)
from a ``category_maps`` dict via ``pd.Categorical(df[col], categories=cats).codes``.
A category's code is therefore its position in the shared ordered category list.

This test builds a single shared ``category_maps`` over the UNION of the train
and test categories for each categorical column (mirroring what ``run.py`` /
Task 12 will do) and passes the SAME map to ``build_features`` for both frames.
It then asserts that, for every category value appearing in BOTH (imputed)
frames — including the ``"Missing"`` sentinel that null ``RoadType`` / ``Weather``
cells are imputed to — the integer code produced for the train frame equals the
code produced for the test frame, and both equal the expected index in the
shared map.

The shared strategy emits *raw-schema* frames carrying a ``day`` column and a
raw ``timestamp`` (``"H:M"``) string but no parsed ``hour`` / ``minute`` /
``tod_slot``. ``load_data`` would normally derive those; here they are
reproduced before calling ``build_features`` so the test drives the builder's
real output.
"""

from __future__ import annotations

import pandas as pd
from hypothesis import HealthCheck, given, settings

from src.features import (
    MISSING_CATEGORY,
    MISSING_CATEGORY_COLS,
    build_features,
    build_imputers,
)
from src.spatial import TargetEncoder, fit_tod_curve
from tests import strategies as strat

# The string categorical columns build_features integer-encodes. The raw
# ``geohash`` location column is intentionally excluded (its signal enters via
# the leak-free encodings, not as a category code).
CATEGORICAL_COLS = ["RoadType", "LargeVehicles", "Landmarks", "Weather"]
LOCATION_COLS = ["geohash"]


def _derive_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ``hour`` / ``minute`` / ``abs_time`` / ``tod_slot`` from ``timestamp``.

    Mirrors ``src.data.load_data`` so the generated raw frames carry exactly the
    time-of-day fields ``build_features`` and ``fit_tod_curve`` expect, without
    invoking the CSV loader / schema detection.
    """
    out = df.copy()
    parts = out["timestamp"].astype(str).str.split(":", n=1, expand=True)
    out["hour"] = parts[0].astype(int)
    out["minute"] = parts[1].astype(int)
    minute_of_day = out["hour"] * 60 + out["minute"]
    out["abs_time"] = out["day"] * 1440 + minute_of_day
    out["tod_slot"] = minute_of_day // 15
    return out


def _build_shared_category_maps(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> dict[str, list]:
    """Build ``col -> ordered categories`` over the UNION of train+test values.

    Mirrors the Task 12 / run.py construction: categories are the sorted union of
    the non-null string values seen in train and test, with the ``"Missing"``
    sentinel appended for the nullable string categoricals (``RoadType`` /
    ``Weather``) so imputed cells get a stable code shared across both frames.
    """
    category_maps: dict[str, list] = {}
    for col in CATEGORICAL_COLS:
        cats = sorted(
            set(train_df[col].dropna().astype(str))
            | set(test_df[col].dropna().astype(str))
        )
        if col in MISSING_CATEGORY_COLS and MISSING_CATEGORY not in cats:
            cats = cats + [MISSING_CATEGORY]
        category_maps[col] = cats
    return category_maps


def _imputed_categories(raw_df: pd.DataFrame, col: str) -> pd.Series:
    """Return the post-imputation category value per row (as strings).

    Mirrors ``build_features``' null handling so the test can reason about the
    categories that actually reach the integer encoder: null ``RoadType`` /
    ``Weather`` cells become the ``"Missing"`` sentinel; everything else is the
    raw string value.
    """
    series = raw_df[col]
    if col in MISSING_CATEGORY_COLS:
        series = series.fillna(MISSING_CATEGORY)
    return series.astype(str)


# Feature: demand-prediction-overhaul, Property 5: Categorical mapping is consistent across train and test
@given(pair=strat.train_test_pair())
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_categorical_mapping_consistent_across_train_and_test(
    pair: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    raw_train, raw_test = pair
    raw_train = _derive_time_fields(raw_train).reset_index(drop=True)
    raw_test = _derive_time_fields(raw_test).reset_index(drop=True)

    # Single shared category map over the union of train+test categories (Req 3.5).
    category_maps = _build_shared_category_maps(raw_train, raw_test)

    # Fit target-derived artifacts on the train frame only (leak-free). These are
    # required arguments of build_features but do not affect the category codes.
    encoder = TargetEncoder("geohash").fit(raw_train, target_col="demand")
    tod_curve = fit_tod_curve(raw_train, target_col="demand")
    impute_values = build_imputers(raw_train)

    # Encode BOTH frames with the SAME category_maps and impute values.
    feats_train = build_features(
        raw_train,
        encoder=encoder,
        tod_curve=tod_curve,
        categorical_cols=CATEGORICAL_COLS,
        location_cols=LOCATION_COLS,
        impute_values=impute_values,
        category_maps=category_maps,
    )
    feats_test = build_features(
        raw_test,
        encoder=encoder,
        tod_curve=tod_curve,
        categorical_cols=CATEGORICAL_COLS,
        location_cols=LOCATION_COLS,
        impute_values=impute_values,
        category_maps=category_maps,
    )

    for col in CATEGORICAL_COLS:
        # Post-imputation categories per row (handles the "Missing" sentinel for
        # null RoadType / Weather cells, so missingness is itself a shared code).
        train_cats = _imputed_categories(raw_train, col)
        test_cats = _imputed_categories(raw_test, col)

        # Categories present in BOTH (imputed) frames — the property's scope.
        common = set(train_cats) & set(test_cats)

        for value in common:
            # Expected code = the value's index in the shared ordered map.
            assert value in category_maps[col], (
                f"col {col!r}: imputed category {value!r} missing from the shared "
                f"category map {category_maps[col]}"
            )
            expected_code = category_maps[col].index(value)

            # Every train row holding this (imputed) category got exactly this code.
            train_codes = set(feats_train.loc[train_cats == value, col].tolist())
            assert train_codes == {expected_code}, (
                f"train col {col!r} value {value!r}: encoded codes {train_codes} "
                f"!= expected map index {expected_code}"
            )

            # Every test row holding this (imputed) category got exactly this code.
            test_codes = set(feats_test.loc[test_cats == value, col].tolist())
            assert test_codes == {expected_code}, (
                f"test col {col!r} value {value!r}: encoded codes {test_codes} "
                f"!= expected map index {expected_code}"
            )

            # The core property: identical integer code in both encoded frames.
            assert train_codes == test_codes, (
                f"col {col!r} value {value!r} maps to {train_codes} in train but "
                f"{test_codes} in test"
            )
