"""
Adaptive schema detection for demand-forecasting datasets.

Automatically identifies timestamp, location, target, categorical,
and ID columns without any hardcoded column names.
"""

import numpy as np
import pandas as pd


def detect_schema(df: pd.DataFrame) -> dict:
    """
    Detect the schema of a demand-forecasting dataset.

    Detection priority per column role:

    TIMESTAMP:  datetime64 → parseable string → name match
    ID:         exact name match → monotonically increasing int
    TARGET:     numeric, not timestamp/id/location/latlon, highest variance
    LOCATION:   int/str with cardinality 2–10 000, name contains spatial keyword
    CATEGORICAL: object/string with cardinality < 300

    Args:
        df: Input dataframe (train or test).

    Returns:
        Schema dict with keys:
            timestamp_col  (str)
            location_cols  (List[str])
            target_col     (str)
            categorical_cols (List[str])
            id_col         (str | None)

    Raises:
        ValueError: If timestamp or target column cannot be determined.
    """
    schema = {
        "timestamp_col": None,
        "location_cols": [],
        "target_col": None,
        "categorical_cols": [],
        "id_col": None,
    }
    used_cols: set = set()

    # ── TIMESTAMP ────────────────────────────────────────────────
    ts_candidates = _detect_timestamp(df)
    if not ts_candidates:
        raise ValueError(
            f"No timestamp column detected. Columns: {list(df.columns)}"
        )
    schema["timestamp_col"] = ts_candidates[0]
    used_cols.add(schema["timestamp_col"])

    # ── ID ───────────────────────────────────────────────────────
    schema["id_col"] = _detect_id(df, used_cols)
    if schema["id_col"]:
        used_cols.add(schema["id_col"])

    # ── lat/lon exclusion set ────────────────────────────────────
    latlon_cols = _detect_latlon(df)

    # ── LOCATION ─────────────────────────────────────────────────
    schema["location_cols"] = _detect_location(df, used_cols, latlon_cols)
    used_cols.update(schema["location_cols"])

    # ── TARGET ───────────────────────────────────────────────────
    schema["target_col"] = _detect_target(df, used_cols, latlon_cols)
    used_cols.add(schema["target_col"])

    # ── CATEGORICAL ──────────────────────────────────────────────
    schema["categorical_cols"] = _detect_categorical(df, used_cols, latlon_cols)

    # ── Validate ─────────────────────────────────────────────────
    assert schema["timestamp_col"] is not None, "Timestamp detection failed"
    assert schema["target_col"] is not None, "Target detection failed"

    return schema


# ─── Private Helpers ────────────────────────────────────────────


def _detect_timestamp(df: pd.DataFrame) -> list:
    """Return candidate timestamp column names.

    Args:
        df: Input DataFrame.

    Returns:
        List of candidate timestamp column names.
    """
    candidates = []

    # Priority 1: existing datetime64
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            candidates.append(col)
    if candidates:
        return candidates

    # Priority 2: parseable object/string
    for col in df.select_dtypes(include=["object", "string"]).columns:
        try:
            parsed = pd.to_datetime(df[col], infer_datetime_format=True, utc=False)
            if parsed.notna().sum() > len(df) * 0.5:
                candidates.append(col)
        except (ValueError, TypeError, OverflowError):
            continue
    if candidates:
        return candidates

    # Priority 3: name heuristic
    keywords = ["time", "date", "hour", "period", "timestamp", "ts"]
    for col in df.columns:
        if any(kw in col.lower() for kw in keywords):
            candidates.append(col)

    return candidates


def _detect_id(df: pd.DataFrame, used: set) -> str | None:
    """Return the ID column name or None.

    Args:
        df: Input DataFrame.
        used: Already-claimed column names.

    Returns:
        ID column name or None.
    """
    # Exact name match
    id_names = {"id", "row_id", "trip_id", "record_id", "index"}
    for col in df.columns:
        if col.lower().strip() in id_names:
            return col

    # Monotonically increasing integer
    for col in df.select_dtypes(include=["int", "int64", "int32"]).columns:
        if col in used:
            continue
        vals = df[col].values
        if len(vals) > 1 and np.all(np.diff(vals) > 0):
            return col

    return None


def _detect_latlon(df: pd.DataFrame) -> set:
    """Identify lat/lon columns to exclude from location / target.

    Args:
        df: Input DataFrame.

    Returns:
        Set of column names identified as latitude/longitude.
    """
    latlon = set()
    for col in df.columns:
        low = col.lower()
        if "lat" in low or "lon" in low or "lng" in low:
            latlon.add(col)
    return latlon


_LOC_KEYWORDS = [
    "grid", "zone", "junction", "sector", "area",
    "region", "location", "cluster", "node",
]


def _detect_location(df: pd.DataFrame, used: set, latlon: set) -> list:
    """Return location column names.

    Args:
        df: Input DataFrame.
        used: Already-claimed column names.
        latlon: Lat/lon columns to exclude.

    Returns:
        List of location column names.
    """
    candidates = []
    for col in df.columns:
        if col in used or col in latlon:
            continue
        if df[col].dtype in ["object", "string"] or pd.api.types.is_integer_dtype(df[col]):
            card = df[col].nunique()
            if 2 <= card <= 10_000:
                if any(kw in col.lower() for kw in _LOC_KEYWORDS):
                    candidates.append(col)

    # Fallback: int columns with cardinality < 1000
    if not candidates:
        for col in df.columns:
            if col in used or col in latlon:
                continue
            if pd.api.types.is_integer_dtype(df[col]):
                card = df[col].nunique()
                if 2 <= card < 1000:
                    candidates.append(col)

    return candidates


_TARGET_EXCLUDE = [
    "lat", "lon", "grid", "zone", "junction", "sector",
    "area", "region", "location", "id",
]


def _detect_target(df: pd.DataFrame, used: set, latlon: set) -> str:
    """Return the target column name.

    Args:
        df: Input DataFrame.
        used: Already-claimed column names.
        latlon: Lat/lon columns to exclude.

    Returns:
        Target column name.

    Raises:
        ValueError: If no target column can be determined.
    """
    candidates = []
    for col in df.columns:
        if col in used or col in latlon:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if any(sub in col.lower() for sub in _TARGET_EXCLUDE):
            continue
        candidates.append(col)

    if len(candidates) == 0:
        raise ValueError("No target column detected after filtering.")
    if len(candidates) == 1:
        return candidates[0]

    # Ambiguity → pick highest variance
    variances = {c: df[c].var() for c in candidates}
    return max(variances, key=variances.get)


def _detect_categorical(df: pd.DataFrame, used: set, latlon: set) -> list:
    """Return categorical column names.

    Args:
        df: Input DataFrame.
        used: Already-claimed column names.
        latlon: Lat/lon columns to exclude.

    Returns:
        List of categorical column names.
    """
    cats = []
    for col in df.columns:
        if col in used or col in latlon:
            continue
        if df[col].dtype in ["object", "string", "category"]:
            if df[col].nunique() < 300:
                cats.append(col)
    return cats
