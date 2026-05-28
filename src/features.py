"""
Feature engineering: temporal, cyclical, spatial merge, lags,
rolling / EWM, interactions, null handling, and memory reduction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─── Computation Frame ──────────────────────────────────────────


def build_compute_df(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    timestamp_col: str,
    location_cols: list[str],
    target_col: str,
    max_lags: int,
) -> pd.DataFrame:
    """
    Concatenate a history tail from train onto test so that
    lag / rolling features can be computed for test rows.

    Args:
        train_df: Training data.
        test_df: Test data.
        timestamp_col: Timestamp column name.
        location_cols: Location column names.
        target_col: Target column name.
        max_lags: Number of historical rows per location.

    Returns:
        Combined DataFrame with ``_is_test`` flag column.
    """
    if not location_cols:
        history = train_df.tail(max_lags).copy()
        history["_is_test"] = 0
        test_part = test_df.copy()
        test_part["_is_test"] = 1
        if target_col not in test_part.columns:
            test_part[target_col] = np.nan
        out = pd.concat([history, test_part], ignore_index=True)
        return out.sort_values(timestamp_col).reset_index(drop=True)

    loc_col = location_cols[0]
    parts = []
    for loc in test_df[loc_col].unique():
        tail = train_df[train_df[loc_col] == loc].tail(max_lags).copy()
        parts.append(tail)

    history = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=train_df.columns)
    history["_is_test"] = 0

    test_part = test_df.copy()
    test_part["_is_test"] = 1
    if target_col not in test_part.columns:
        test_part[target_col] = np.nan

    out = pd.concat([history, test_part], ignore_index=True)
    return out.sort_values([timestamp_col, loc_col]).reset_index(drop=True)


# ─── Main Feature Builder ──────────────────────────────────────


def build_features(
    df: pd.DataFrame,
    spatial_stats: dict[str, pd.DataFrame],
    timestamp_col: str,
    location_cols: list[str],
    target_col: str,
    global_train_median: float,
    rolling_windows: list[int],
    categorical_cols: list[str],
) -> pd.DataFrame:
    """
    Build all features. Works on both the full training set
    and the compute frame (history-tail + test).

    Feature groups:
        A. Temporal   B. Cyclical   C. Spatial merge
        D. Lags       E. Rolling / EWM
        F. Interactions   G. Null handling

    Args:
        df: Input DataFrame.
        spatial_stats: Pre-computed spatial statistics dict.
        timestamp_col: Timestamp column name.
        location_cols: Location column names.
        target_col: Target column name.
        global_train_median: Global target median from train.
        rolling_windows: Window sizes for rolling features.
        categorical_cols: Categorical column names.

    Returns:
        Feature-enriched DataFrame (copy of input).
    """
    df = df.copy()
    ts = df[timestamp_col]

    # ── A. Temporal ──────────────────────────────────────────────
    df["year"]         = ts.dt.year
    df["month"]        = ts.dt.month
    df["day"]          = ts.dt.day
    df["hour"]         = ts.dt.hour
    df["minute"]       = ts.dt.minute
    df["day_of_week"]  = ts.dt.dayofweek
    df["day_of_year"]  = ts.dt.dayofyear
    df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    df["quarter"]      = ts.dt.quarter
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19, 20]).astype(int)
    df["is_night"]     = df["hour"].isin([23, 0, 1, 2, 3, 4]).astype(int)

    # ── B. Cyclical ──────────────────────────────────────────────
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"]   = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]   = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # ── C. Spatial stats merge ───────────────────────────────────
    for loc_col in location_cols:
        if loc_col not in spatial_stats:
            continue
        stats_df = spatial_stats[loc_col]
        df = df.merge(stats_df, left_on=loc_col, right_index=True, how="left")
        for sc in stats_df.columns:
            df[sc] = df[sc].fillna(global_train_median)

    # ── D. Lag features ──────────────────────────────────────────
    lag_values = [1, 2, 3, 6, 12, 24, 48, 168]
    if location_cols:
        loc_col = location_cols[0]
        df = df.sort_values([loc_col, timestamp_col]).reset_index(drop=True)
        for n in lag_values:
            df[f"lag_{n}"] = df.groupby(loc_col)[target_col].shift(n)
    else:
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        for n in lag_values:
            df[f"lag_{n}"] = df[target_col].shift(n)

    # ── E. Rolling / EWM features ────────────────────────────────
    if location_cols:
        loc_col = location_cols[0]
        shifted = df.groupby(loc_col)[target_col].shift(1)
    else:
        shifted = df[target_col].shift(1)

    for w in rolling_windows:
        if location_cols:
            loc_col = location_cols[0]
            roll = shifted.groupby(df[loc_col]).rolling(window=w, min_periods=1)
            df[f"rolling_mean_{w}"] = roll.mean().reset_index(level=0, drop=True)
            df[f"rolling_std_{w}"]  = roll.std().reset_index(level=0, drop=True)
            df[f"rolling_max_{w}"]  = roll.max().reset_index(level=0, drop=True)
            df[f"rolling_min_{w}"]  = roll.min().reset_index(level=0, drop=True)
        else:
            roll = shifted.rolling(window=w, min_periods=1)
            df[f"rolling_mean_{w}"] = roll.mean()
            df[f"rolling_std_{w}"]  = roll.std()
            df[f"rolling_max_{w}"]  = roll.max()
            df[f"rolling_min_{w}"]  = roll.min()

    # EWM span 24
    if location_cols:
        loc_col = location_cols[0]
        df["ewm_mean_24"] = (
            shifted.groupby(df[loc_col])
            .apply(lambda x: x.ewm(span=24, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
    else:
        df["ewm_mean_24"] = shifted.ewm(span=24, min_periods=1).mean()

    # ── F. Interaction features ──────────────────────────────────
    if location_cols:
        loc_col = location_cols[0]
        mean_c = f"{loc_col}_mean_demand"
        rank_c = f"{loc_col}_demand_rank"
        if mean_c in df.columns:
            df["demand_vs_loc_mean"] = df["lag_1"] / (df[mean_c] + 1e-5)
        if rank_c in df.columns:
            df["hour_x_loc_rank"] = df["hour"] * df[rank_c]

    # Haversine to Bengaluru centre
    _add_haversine(df)

    # Encode categoricals as integer codes
    for cat_col in categorical_cols:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype("category").cat.codes
    for loc_col in location_cols:
        if df[loc_col].dtype == "object" or df[loc_col].dtype.name == "category":
            df[loc_col] = df[loc_col].astype("category").cat.codes

    # ── G. Null handling ─────────────────────────────────────────
    _handle_nulls(df, lag_values, rolling_windows, location_cols)

    return df


# ─── Helpers ────────────────────────────────────────────────────


def _add_haversine(df: pd.DataFrame) -> None:
    """Add haversine distance to Bengaluru centre if lat/lon exist.

    Args:
        df: DataFrame (modified in-place).

    Returns:
        None
    """
    lat_col = lon_col = None
    for col in df.columns:
        cl = col.lower()
        if "lat" in cl and lat_col is None:
            lat_col = col
        if ("lon" in cl or "lng" in cl) and lon_col is None:
            lon_col = col

    if lat_col is None or lon_col is None:
        return

    R = 6371.0
    lat_r = np.radians(df[lat_col].astype(float))
    lon_r = np.radians(df[lon_col].astype(float))
    c_lat = np.radians(12.9716)
    c_lon = np.radians(77.5946)
    dlat = lat_r - c_lat
    dlon = lon_r - c_lon
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r) * np.cos(c_lat) * np.sin(dlon / 2) ** 2
    df["dist_to_center_km"] = 2 * R * np.arcsin(np.sqrt(a))


def _handle_nulls(
    df: pd.DataFrame,
    lag_values: list[int],
    rolling_windows: list[int],
    location_cols: list[str],
) -> None:
    """Fill NaNs and replace infinities in-place.

    Steps (in order):
        G1. Per-location median fill for lag/rolling columns.
        G2. Global median fill for remaining numeric NaNs.
        G3. Replace ±inf with 99th / 1st percentile.
        G4. Final sweep: fill stragglers with 0.
        G5. Assert zero nulls.

    Args:
        df: DataFrame (modified in-place).
        lag_values: Lag step sizes.
        rolling_windows: Rolling window sizes.
        location_cols: Location columns.

    Returns:
        None

    Raises:
        AssertionError: If any null remains.
    """
    lr_cols = (
        [f"lag_{n}" for n in lag_values]
        + [f"rolling_mean_{w}" for w in rolling_windows]
        + [f"rolling_std_{w}" for w in rolling_windows]
        + [f"rolling_max_{w}" for w in rolling_windows]
        + [f"rolling_min_{w}" for w in rolling_windows]
        + ["ewm_mean_24"]
    )
    if "demand_vs_loc_mean" in df.columns:
        lr_cols.append("demand_vs_loc_mean")

    # G1
    if location_cols:
        loc = location_cols[0]
        for col in lr_cols:
            if col in df.columns:
                medians = df.groupby(loc)[col].transform("median")
                df[col] = df[col].fillna(medians)

    # G2
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # G3
    for col in df.select_dtypes(include=[np.number]).columns:
        if np.isinf(df[col]).any():
            finite = df.loc[np.isfinite(df[col]), col]
            df[col] = df[col].replace([np.inf], finite.quantile(0.99))
            df[col] = df[col].replace([-np.inf], finite.quantile(0.01))

    # G4
    df.fillna(0, inplace=True)

    # G5
    null_counts = df.isnull().sum()
    assert null_counts.sum() == 0, f"Null leak: {null_counts[null_counts > 0]}"


# ─── Memory Optimization ───────────────────────────────────────


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


# ─── Feature Column Selection ──────────────────────────────────


def select_feature_cols(
    train_feats: pd.DataFrame,
    test_feats: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    id_col: str | None,
) -> list[str]:
    """
    Determine which columns to use as model inputs.

    Excludes timestamp, target, ``_is_test`` flag, ID, and any
    non-numeric columns.

    Args:
        train_feats: Feature-engineered training DataFrame.
        test_feats: Feature-engineered test DataFrame.
        timestamp_col: Timestamp column name.
        target_col: Target column name.
        id_col: ID column name (or None).

    Returns:
        Sorted list of feature column names present in both frames.
    """
    exclude = {timestamp_col, target_col, "_is_test"}
    if id_col is not None:
        exclude.add(id_col)

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
