"""
Chronological validation split and time-based K-fold CV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def chronological_split(
    train_feats: pd.DataFrame,
    timestamp_col: str,
    holdout_frac: float,
    feature_cols: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
    """
    Split on UNIQUE TIMESTAMPS — not row index — to prevent
    same-timestamp rows being split across train/val.

    Args:
        train_feats: Feature-engineered training DataFrame.
        timestamp_col: Timestamp column name.
        holdout_frac: Fraction of unique timestamps for validation.
        feature_cols: Feature column names.
        target_col: Target column name.

    Returns:
        (X_tr, y_tr, X_val, y_val, tr_mask, val_mask)
    """
    unique_ts = np.sort(train_feats[timestamp_col].unique())
    cutoff = int(len(unique_ts) * (1 - holdout_frac))
    train_times = unique_ts[:cutoff]
    val_times   = unique_ts[cutoff:]

    tr_mask  = train_feats[timestamp_col].isin(train_times).values
    val_mask = train_feats[timestamp_col].isin(val_times).values

    X_tr  = train_feats.loc[tr_mask,  feature_cols]
    y_tr  = train_feats.loc[tr_mask,  target_col]
    X_val = train_feats.loc[val_mask, feature_cols]
    y_val = train_feats.loc[val_mask, target_col]

    print(f"Train size : {X_tr.shape}")
    print(f"Val   size : {X_val.shape}")
    print(f"Val range  : "
          f"{train_feats.loc[val_mask, timestamp_col].min()} → "
          f"{train_feats.loc[val_mask, timestamp_col].max()}")

    return X_tr, y_tr, X_val, y_val, tr_mask, val_mask


def time_kfold_split(
    df: pd.DataFrame,
    timestamp_col: str,
    n_splits: int,
):
    """
    Chronological K-fold generator.

    Guarantees:
      • No same-timestamp rows are split across folds.
      • Train set is *strictly* before the validation set.
      • First fold (no history) is skipped automatically.

    Args:
        df: Input DataFrame with a timestamp column.
        timestamp_col: Name of the timestamp column.
        n_splits: Number of folds.

    Yields:
        (train_indices, val_indices) as numpy arrays.
    """
    unique_ts = np.sort(df[timestamp_col].unique())
    fold_size = len(unique_ts) // n_splits

    for i in range(n_splits):
        val_ts   = unique_ts[i * fold_size : (i + 1) * fold_size]
        train_ts = unique_ts[: i * fold_size]          # strictly past

        if len(train_ts) == 0:
            continue  # first fold has no history

        train_idx = df.index[df[timestamp_col].isin(train_ts)].values
        val_idx   = df.index[df[timestamp_col].isin(val_ts)].values

        yield train_idx, val_idx
