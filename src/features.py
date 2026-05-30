"""
Leak-free feature builder for the demand-prediction-overhaul.

This module was gutted of every history-based / target-shifted feature group
(the root cause of the train/test leakage). The old ``build_compute_df``
history-tail mechanism and the lag / rolling / EWM / lag-derived interaction
features are gone, as is the synthetic-calendar temporal/cyclical block.

What remains is a single, leak-free ``build_features`` that assembles model
inputs from signal that genuinely exists at prediction time:

  * **Time-of-day cyclical** encodings derived from the real clock
    (``tod_slot``, ``hour``, ``minute`` provided by ``data.load_data``):
    ``tod_sin``/``tod_cos`` (period 96), ``minute_of_day_sin``/``cos``
    (period 1440), ``is_peak_hour``, ``is_night``, and ``day``.
  * **Leak-free geohash encodings** merged from a fitted
    :class:`src.spatial.TargetEncoder` (or already-present out-of-fold columns),
    with the train-derived Unseen_Geohash fallback.
  * The **day-48 time-of-day demand curve** (``tod_curve_mean``) joined on
    ``tod_slot`` via :func:`src.spatial.transform_tod_curve`.
  * The **contextual columns** ``RoadType``, ``NumberofLanes``,
    ``LargeVehicles``, ``Landmarks``, ``Temperature``, ``Weather`` as row
    features with consistent integer category codes shared across train/test.

All target-derived artifacts (geohash encodings, the time-of-day curve, the
imputation values) are fit on a fitting partition *outside* this module and
passed in, so ``build_features`` never reads the input frame's own target. It
therefore succeeds and produces the same feature columns whether ``demand`` is
present, absent, or all-NaN (Requirement 1.6 / design Property 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.spatial import (
    ENCODING_COLS,
    InteractionEncoder,
    PerGeohashTodCurve,
    PG_CURVE_COL,
    TargetEncoder,
    interaction_feature_cols,
    transform_tod_curve,
)

# ─── Feature configuration ─────────────────────────────────────

#: The six contextual columns kept as ordinary row features (Requirement 3.4).
CONTEXTUAL_COLS = [
    "RoadType",
    "NumberofLanes",
    "LargeVehicles",
    "Landmarks",
    "Temperature",
    "Weather",
]

#: Contextual columns that are imputed with the dedicated ``"Missing"`` category
#: when null (Requirement 3.6). These are the nullable *string* categoricals.
MISSING_CATEGORY_COLS = ["RoadType", "Weather"]

#: Default sentinel category used for missing nullable categorical values.
MISSING_CATEGORY = "Missing"

#: Hours treated as peak / night for the boolean time-of-day flags.
_PEAK_HOURS = (7, 8, 9, 17, 18, 19, 20)
_NIGHT_HOURS = (23, 0, 1, 2, 3, 4)

#: Quarter-hour slots per day (period of the ``tod_slot`` cycle).
_SLOTS_PER_DAY = 96
#: Minutes per day (period of the ``minute_of_day`` cycle).
_MINUTES_PER_DAY = 1440


# ─── Imputer fitting ───────────────────────────────────────────


def build_imputers(
    train_df: pd.DataFrame,
    categorical_cols: list[str] | None = None,
    target_col: str | None = None,
) -> dict:
    """Compute train-only imputation values for ``build_features`` (Req 3.6).

    Imputation values are derived from the **fitting partition only** (the
    holdout's training complement in the validation context, or the full
    ``train.csv`` in the final context). They are then applied inside
    :func:`build_features` to the null cells of any frame.

    Imputation policy:
      * ``Temperature`` → the train-only **median** (a finite float; falls back
        to ``0.0`` when the partition has no non-null Temperature values).
      * ``RoadType`` / ``Weather`` → a dedicated ``"Missing"`` category so that
        missingness itself becomes a signal.
      * ``NumberofLanes`` is complete in the data and is **not** imputed.

    Args:
        train_df: The fitting-partition frame to derive imputation values from.
        categorical_cols: Accepted for API symmetry with the Task 12 wiring;
            the set of categorical columns imputed with ``"Missing"`` is fixed
            by :data:`MISSING_CATEGORY_COLS`.
        target_col: Accepted for API symmetry; unused (imputers are not
            target-derived).

    Returns:
        A dict of imputation values::

            {
                "Temperature_median": float,   # train-only median
                "missing_category": "Missing", # for RoadType / Weather
            }
    """
    temperature_median = 0.0
    if "Temperature" in train_df.columns:
        median = pd.to_numeric(train_df["Temperature"], errors="coerce").median()
        if pd.notna(median):
            temperature_median = float(median)

    return {
        "Temperature_median": temperature_median,
        "missing_category": MISSING_CATEGORY,
    }


# ─── Main feature builder ──────────────────────────────────────


def build_features(
    df: pd.DataFrame,
    encoder: TargetEncoder | None,
    tod_curve: pd.Series,
    categorical_cols: list[str],
    location_cols: list[str],
    impute_values: dict,
    category_maps: dict[str, list] | None = None,
    slot_col: str = "tod_slot",
    oof_encoded: bool | None = None,
    interaction_encoder: "InteractionEncoder | None" = None,
    pg_curve: "PerGeohashTodCurve | None" = None,
) -> pd.DataFrame:
    """Assemble the reduced, leak-free model-input feature set.

    The function is leak-free by construction: it never reads ``df``'s target
    column. All target-derived signal enters through the pre-fitted ``encoder``
    (or already-present out-of-fold encoding columns) and ``tod_curve``, and all
    imputation values come from ``impute_values`` (fit on a separate partition).
    As a result the produced feature columns are identical whether the target
    column is present, absent, or all-NaN (Requirement 1.6 / design Property 1).

    Feature groups produced:
      * Time-of-day cyclical (Req 3.2): ``tod_sin``/``tod_cos`` (period 96),
        ``minute_of_day_sin``/``minute_of_day_cos`` (period 1440),
        ``is_peak_hour``, ``is_night``, and ``day`` (already present).
      * Geohash encodings (Req 3.1, 3.7): ``geohash_mean``/``geohash_median``/
        ``geohash_std``/``geohash_rank`` merged with the Unseen_Geohash fallback.
      * Day-48 time-of-day curve (Req 3.3): ``tod_curve_mean`` joined on
        ``tod_slot``.
      * Contextual columns (Req 3.4) with consistent integer category codes
        (Req 3.5) and train-derived, null-only imputation (Req 3.6).

    Geohash-encoding source (two contexts):
      * **Training frame (out-of-fold):** the caller passes a frame that already
        carries the OOF encoding columns from
        :meth:`src.spatial.TargetEncoder.fit_oof`. Those columns are detected and
        reused as-is so a row's encoding never uses its own target.
      * **Eval / test frame:** the caller passes a fitted ``encoder`` and no
        encoding columns; ``encoder.transform(df)`` merges them (with the
        train-derived unseen-geohash fallback).

    The behavior can be forced via ``oof_encoded``; when left as ``None`` it is
    auto-detected by the presence of all :data:`src.spatial.ENCODING_COLS`.

    Row order and index are preserved (no sorting), so the output aligns 1:1
    with the input frame and with any OOF encodings computed alongside it.

    Args:
        df: Input frame (post-``load_data``: has ``hour``, ``minute``,
            ``tod_slot``, ``day``, ``geohash`` and the contextual columns).
        encoder: A fitted :class:`src.spatial.TargetEncoder`. May be ``None``
            only when the encoding columns are already present (OOF path).
        tod_curve: The day-48 time-of-day curve Series from
            :func:`src.spatial.fit_tod_curve`.
        categorical_cols: Names of the string categorical columns to integer
            encode (e.g. ``RoadType``, ``LargeVehicles``, ``Landmarks``,
            ``Weather``). The raw ``geohash`` location column is intentionally
            NOT in this list — its signal enters via the encodings only.
        location_cols: Location column names (e.g. ``["geohash"]``); used to
            locate the geohash column for encoding and to keep the raw string
            column out of the numeric feature set.
        impute_values: Dict from :func:`build_imputers`.
        category_maps: Optional ``col -> ordered categories`` mapping built over
            the union of train+test categories so codes are consistent across
            frames (Req 3.5). When provided for a nullable categorical column,
            ``"Missing"`` is appended if absent so it maps to a stable code.
        slot_col: Name of the quarter-hour slot column. Defaults to ``"tod_slot"``.
        oof_encoded: Force the OOF path (``True``) or the ``encoder.transform``
            path (``False``). ``None`` (default) auto-detects.

    Returns:
        A copy of ``df`` enriched with the leak-free feature columns. The target
        column (if present) is passed through untouched for downstream training;
        :func:`select_feature_cols` excludes it from the model inputs.

    Raises:
        RuntimeError: If geohash encodings are neither present nor obtainable
            (``oof_encoded`` is False / unset and ``encoder`` is ``None``).
    """
    df = df.copy()

    # ── Time-of-day cyclical features (Req 3.2) ──────────────────
    slot = df[slot_col].astype(float)
    df["tod_sin"] = np.sin(2 * np.pi * slot / _SLOTS_PER_DAY)
    df["tod_cos"] = np.cos(2 * np.pi * slot / _SLOTS_PER_DAY)

    minute_of_day = (df["hour"].astype(float) * 60 + df["minute"].astype(float))
    df["minute_of_day_sin"] = np.sin(2 * np.pi * minute_of_day / _MINUTES_PER_DAY)
    df["minute_of_day_cos"] = np.cos(2 * np.pi * minute_of_day / _MINUTES_PER_DAY)

    df["is_peak_hour"] = df["hour"].isin(_PEAK_HOURS).astype(int)
    df["is_night"] = df["hour"].isin(_NIGHT_HOURS).astype(int)
    # ``day`` is already a column from load_data and is retained as a feature.

    # ── Geohash encodings merge (Req 3.1, 3.7) ───────────────────
    have_encodings = all(col in df.columns for col in ENCODING_COLS)
    use_oof = have_encodings if oof_encoded is None else oof_encoded

    if use_oof:
        if not have_encodings:
            raise RuntimeError(
                "build_features: oof_encoded=True but the out-of-fold encoding "
                f"columns {ENCODING_COLS} are not present on the frame."
            )
        # Encodings already present (OOF) — reuse as-is (leak-free by construction).
    else:
        if encoder is None:
            raise RuntimeError(
                "build_features: no encoder provided and the encoding columns "
                f"{ENCODING_COLS} are absent. Pass a fitted TargetEncoder or a "
                "frame already carrying OOF encodings."
            )
        df = encoder.transform(df)

    # ── Day-48 time-of-day curve merge (Req 3.3) ─────────────────
    df = transform_tod_curve(df, tod_curve, slot_col=slot_col)

    # ── Leak-free geohash×context interaction means (validated lever) ──
    # Captures that demand depends on the INTERACTION of location with
    # time-of-day / road type / weather, not just each separately. For the
    # training frame the caller supplies the OOF columns already on `df`
    # (auto-detected); for eval/test a fitted ``interaction_encoder`` merges
    # them with the geohash/prior fallback chain. No-op when neither is given,
    # so older callers/tests remain valid.
    _inter_cols = interaction_feature_cols(
        interaction_encoder.other_keys if interaction_encoder is not None else None
    )
    have_inter = all(c in df.columns for c in _inter_cols) and len(_inter_cols) > 0
    if have_inter:
        pass  # OOF interaction columns already present — reuse as-is (leak-free).
    elif interaction_encoder is not None:
        df = interaction_encoder.transform(df)

    # ── Per-geohash day-48 time-of-day curve (sharp temporal signal) ──
    # Like the geohash encodings: if the OOF column is already present (training
    # frame) reuse it; otherwise a fitted ``pg_curve`` merges it. No-op when
    # neither is supplied, preserving older callers/tests.
    if PG_CURVE_COL in df.columns:
        pass  # OOF per-geohash curve already present — reuse as-is (leak-free).
    elif pg_curve is not None:
        df = pg_curve.transform(df, slot_col=slot_col)

    # ── Null handling, train-derived & null-only (Req 3.6) ───────
    # Done BEFORE categorical encoding so "Missing" becomes its own code.
    _apply_imputation(df, impute_values)

    # ── Consistent categorical integer codes (Req 3.5) ───────────
    missing_cat = impute_values.get("missing_category", MISSING_CATEGORY)
    _encode_categoricals(df, categorical_cols, category_maps, missing_cat)

    return df


# ─── Helpers ────────────────────────────────────────────────────


def _apply_imputation(df: pd.DataFrame, impute_values: dict) -> None:
    """Impute only the null cells of the nullable contextual columns in-place.

    ``Temperature`` nulls → the train-derived median; ``RoadType`` / ``Weather``
    nulls → the ``"Missing"`` category. ``pandas.Series.fillna`` touches only
    null cells, so non-null values are left unchanged (Requirement 3.6).
    ``NumberofLanes`` is complete and is not imputed.

    Args:
        df: Frame to impute (modified in-place).
        impute_values: Dict from :func:`build_imputers`.

    Returns:
        None
    """
    temp_median = impute_values.get("Temperature_median")
    if "Temperature" in df.columns and temp_median is not None:
        df["Temperature"] = pd.to_numeric(
            df["Temperature"], errors="coerce"
        ).fillna(float(temp_median))

    missing_cat = impute_values.get("missing_category", MISSING_CATEGORY)
    for col in MISSING_CATEGORY_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(missing_cat)


def _encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: list[str],
    category_maps: dict[str, list] | None,
    missing_cat: str,
) -> None:
    """Map string categorical columns to consistent integer codes in-place.

    Codes come from ``category_maps`` (built over the union of train+test
    categories) so a category present in both frames maps to the same integer
    in both (Requirement 3.5). For nullable categorical columns the
    ``"Missing"`` category is ensured in the category list so imputed cells get
    a stable code. When no map is supplied for a column, codes are derived from
    that column's own sorted categories (single-frame fallback).

    Args:
        df: Frame to encode (modified in-place).
        categorical_cols: String categorical column names to encode.
        category_maps: Optional ``col -> ordered categories`` mapping.
        missing_cat: The sentinel category to ensure for nullable categoricals.

    Returns:
        None
    """
    if category_maps is None:
        category_maps = {}

    for col in categorical_cols:
        if col not in df.columns:
            continue

        if col in category_maps:
            cats = list(category_maps[col])
        else:
            cats = sorted(df[col].dropna().astype(str).unique().tolist())

        # Ensure the "Missing" sentinel has a stable code for nullable cats.
        if col in MISSING_CATEGORY_COLS and missing_cat not in cats:
            cats = cats + [missing_cat]

        df[col] = pd.Categorical(df[col], categories=cats).codes


# ─── Memory optimization ───────────────────────────────────────


def reduce_memory(df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
    """
    Downcast numeric columns to save RAM.

    Skips the target column (kept float64) and any column whose
    max > 1e15 (would overflow float32).

    Args:
        df: Input DataFrame.
        target_col: Column to keep as float64.

    Returns:
        Memory-optimized DataFrame (same object, modified in-place).
    """
    mem_before = df.memory_usage(deep=True).sum() / 1e6

    for col in df.select_dtypes(include=["int", "int64", "int32", "int16", "int8"]).columns:
        if col == target_col:
            continue
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in df.select_dtypes(include=["float", "float64"]).columns:
        if col == target_col:
            continue
        if df[col].max() > 1e15:
            continue
        df[col] = pd.to_numeric(df[col], downcast="float")

    mem_after = df.memory_usage(deep=True).sum() / 1e6
    pct = 100 * (mem_before - mem_after) / mem_before if mem_before > 0 else 0
    print(f"  Memory: {mem_before:.1f} MB → {mem_after:.1f} MB ({pct:.1f}% reduction)")

    return df


# ─── Feature column selection ──────────────────────────────────


def select_feature_cols(
    train_feats: pd.DataFrame,
    test_feats: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    id_col: str | None,
    location_cols: list[str] | None = None,
) -> list[str]:
    """
    Determine which columns to use as model inputs.

    Excludes the timestamp, target, ``_is_test`` flag, ID, the ordering-only
    ``abs_time`` index, the raw (string) location columns, and any remaining
    non-numeric column. The raw ``geohash`` string is therefore never a feature
    — its signal enters only through the leak-free encodings.

    Args:
        train_feats: Feature-engineered training DataFrame.
        test_feats: Feature-engineered test DataFrame.
        timestamp_col: Timestamp column name.
        target_col: Target column name.
        id_col: ID column name (or None).
        location_cols: Raw location column names to exclude (e.g. ``geohash``).

    Returns:
        Sorted list of feature column names present in both frames.
    """
    exclude = {timestamp_col, target_col, "_is_test", "abs_time"}
    if id_col is not None:
        exclude.add(id_col)
    if location_cols:
        exclude.update(location_cols)

    cols = []
    for col in train_feats.columns:
        if col in exclude:
            continue
        if not pd.api.types.is_numeric_dtype(train_feats[col]):
            continue
        cols.append(col)

    # Drop any that are missing in test
    missing = [c for c in cols if c not in test_feats.columns]
    if missing:
        print(f"⚠ {len(missing)} features missing in test — dropping: {missing}")
        cols = [c for c in cols if c in test_feats.columns]

    print(f"Total features: {len(cols)}")
    return cols
