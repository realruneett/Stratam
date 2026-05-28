"""
Train-only spatial (per-location) statistics.

Computed once on the training set and stored globally so they can
be merged into test features without data leakage.
"""

from __future__ import annotations

import pandas as pd


def compute_spatial_stats(
    train_df: pd.DataFrame,
    location_cols: list[str],
    target_col: str,
    timestamp_col: str,
) -> dict[str, pd.DataFrame]:
    """
    Compute per-location demand statistics on TRAIN data only.

    For each location column the following aggregates are produced:
        mean, median, std, max, min, demand_rank, peak_hour.

    Args:
        train_df: Training DataFrame.
        location_cols: Location column names.
        target_col: Target column name.
        timestamp_col: Timestamp column name.

    Returns:
        Dict keyed by location column name → DataFrame of stats
        indexed by location value.
    """
    spatial_stats: dict[str, pd.DataFrame] = {}

    for loc_col in location_cols:
        grp = train_df.groupby(loc_col)[target_col]

        stats = pd.DataFrame({
            f"{loc_col}_mean_demand":   grp.mean(),
            f"{loc_col}_median_demand": grp.median(),
            f"{loc_col}_std_demand":    grp.std().fillna(0),
            f"{loc_col}_max_demand":    grp.max(),
            f"{loc_col}_min_demand":    grp.min(),
        })

        # Rank: 1 = highest demand (dense)
        stats[f"{loc_col}_demand_rank"] = (
            stats[f"{loc_col}_mean_demand"]
            .rank(ascending=False, method="dense")
            .astype(int)
        )

        # Peak hour: hour of day with max avg demand
        _hours = train_df[timestamp_col].dt.hour
        peak_hour = (
            train_df.assign(_hour=_hours)
            .groupby([loc_col, "_hour"])[target_col]
            .mean()
            .reset_index()
            .sort_values(target_col, ascending=False)
            .drop_duplicates(subset=[loc_col])
            .set_index(loc_col)["_hour"]
            .rename(f"{loc_col}_peak_hour")
        )
        stats = stats.join(peak_hour)

        spatial_stats[loc_col] = stats
        print(f"  Spatial stats for '{loc_col}': "
              f"{stats.shape[0]} unique locations")

    return spatial_stats
