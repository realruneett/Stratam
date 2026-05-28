"""
Data loading, parsing, sorting, and target transformation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.schema import detect_schema


def load_data(
    train_path: str,
    test_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load train/test CSVs, detect schema, parse timestamps, and sort.

    Args:
        train_path: Path to train.csv.
        test_path: Path to test.csv.

    Returns:
        (train_df, test_df, schema_dict)

    Raises:
        FileNotFoundError: If CSV files are missing.
        ValueError: If schema detection fails.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    schema = detect_schema(train_df)
    ts_col = schema["timestamp_col"]
    loc_cols = schema["location_cols"]

    # ── Parse timestamps ────────────────────────────────────────
    for df in [train_df, test_df]:
        if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
            df[ts_col] = pd.to_datetime(
                df[ts_col], infer_datetime_format=True, utc=False
            )
        # Strip timezone if present
        if hasattr(df[ts_col].dt, "tz") and df[ts_col].dt.tz is not None:
            df[ts_col] = df[ts_col].dt.tz_localize(None)

    # ── Sort ────────────────────────────────────────────────────
    sort_cols = [ts_col] + (loc_cols[:1] if loc_cols else [])
    train_df = train_df.sort_values(sort_cols).reset_index(drop=True)
    test_df = test_df.sort_values(
        [c for c in sort_cols if c in test_df.columns]
    ).reset_index(drop=True)

    return train_df, test_df, schema


def print_diagnostics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema: dict,
) -> None:
    """
    Print dataset diagnostics to stdout.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        schema: Schema dictionary.

    Returns:
        None
    """
    ts_col = schema["timestamp_col"]
    tgt_col = schema["target_col"]
    loc_cols = schema["location_cols"]

    print("=" * 60)
    print("SCHEMA REPORT")
    print("=" * 60)
    for key, val in schema.items():
        print(f"  {key:<18s}: {val}")

    print(f"\n  Train shape     : {train_df.shape}")
    print(f"  Test  shape     : {test_df.shape}")
    print(f"  Train date range: {train_df[ts_col].min()} → {train_df[ts_col].max()}")
    print(f"  Test  date range: {test_df[ts_col].min()} → {test_df[ts_col].max()}")

    if loc_cols:
        print(f"  Unique locations: {train_df[loc_cols[0]].nunique()}")

    t = train_df[tgt_col]
    print(f"\n  Target stats:")
    print(f"    mean    : {t.mean():.4f}")
    print(f"    std     : {t.std():.4f}")
    print(f"    min     : {t.min():.4f}")
    print(f"    max     : {t.max():.4f}")
    print(f"    skew    : {t.skew():.4f}")
    print(f"    kurtosis: {t.kurtosis():.4f}")
    print(f"    nulls   : {t.isnull().sum()}")

    nulls = train_df.isnull().sum()
    print(f"\n  Null counts per column (train):")
    print(nulls[nulls > 0].to_string() if nulls.sum() > 0 else "    None")

    # Temporal overlap check
    train_ts = set(train_df[ts_col].unique())
    test_ts = set(test_df[ts_col].unique())
    overlap = train_ts & test_ts
    if overlap:
        print(f"\n  ⚠ WARNING: {len(overlap)} test timestamps overlap with train!")
    else:
        print("\n  ✓ CONFIRMED FUTURE: Test timestamps are entirely after train.")
    print("=" * 60)


def apply_target_transform(
    train_df: pd.DataFrame,
    target_col: str,
) -> bool:
    """
    Conditionally apply log1p transform to the target column in-place.

    Args:
        train_df: Training DataFrame (modified in-place).
        target_col: Name of the target column.

    Returns:
        True if log1p was applied, False otherwise.
    """
    skew = train_df[target_col].skew()
    print(f"Original target skew: {skew:.4f}")

    if abs(skew) > 1.0:
        train_df[target_col] = np.log1p(train_df[target_col].clip(lower=0))
        new_skew = train_df[target_col].skew()
        print(f"Log1p transform applied. New skew: {new_skew:.3f}")
        return True

    print(f"No transform needed. Skew: {skew:.3f}")
    return False
