"""
Leak-free spatial (per-geohash) target encoding and the time-of-day curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from config import SEED, TE_N_FOLDS, TE_SMOOTHING_ALPHA, IE_SMOOTHING_ALPHA

# Stable output column names.
GEOHASH_MEAN_COL = "geohash_mean"
GEOHASH_MEDIAN_COL = "geohash_median"
GEOHASH_STD_COL = "geohash_std"
GEOHASH_RANK_COL = "geohash_rank"

ENCODING_COLS = [
    GEOHASH_MEAN_COL,
    GEOHASH_MEDIAN_COL,
    GEOHASH_STD_COL,
    GEOHASH_RANK_COL,
]

TOD_CURVE_MEAN_COL = "tod_curve_mean"
TOD_CURVE_FIT_DAY = 48


def compute_spatial_stats(
    train_df: pd.DataFrame,
    location_cols: list[str],
    target_col: str,
    timestamp_col: str,
) -> dict[str, pd.DataFrame]:
    """Compute per-location demand statistics on TRAIN data only (legacy)."""
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

        stats[f"{loc_col}_demand_rank"] = (
            stats[f"{loc_col}_mean_demand"]
            .rank(ascending=False, method="dense")
            .astype(int)
        )

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


class TargetEncoder:
    """Leak-free, Bayesian-smoothed per-geohash demand target encoder."""

    def __init__(self, geohash_col: str = "geohash") -> None:
        self.geohash_col = geohash_col
        self.encodings_: pd.DataFrame | None = None
        self.global_prior_mean: float = 0.0
        self.neutral_rank: int = 1
        self.alpha_: float = TE_SMOOTHING_ALPHA

    def fit(
        self,
        df_fit: pd.DataFrame,
        target_col: str,
        alpha: float | None = None,
    ) -> "TargetEncoder":
        if alpha is None:
            alpha = TE_SMOOTHING_ALPHA
        self.alpha_ = float(alpha)

        geo = self.geohash_col

        if len(df_fit) == 0:
            self.global_prior_mean = 0.0
        else:
            prior = df_fit[target_col].mean()
            self.global_prior_mean = float(prior) if pd.notna(prior) else 0.0

        grp = df_fit.groupby(geo)[target_col]
        group_sum = grp.sum()
        group_count = grp.count()

        smoothed_mean = (
            (group_sum + self.global_prior_mean * self.alpha_)
            / (group_count + self.alpha_)
        )

        group_median = grp.median()
        group_std = grp.std().fillna(0.0)

        encodings = pd.DataFrame({
            GEOHASH_MEAN_COL: smoothed_mean,
            GEOHASH_MEDIAN_COL: group_median,
            GEOHASH_STD_COL: group_std,
        })

        if len(encodings) > 0:
            encodings[GEOHASH_RANK_COL] = (
                smoothed_mean.rank(ascending=False, method="dense").astype(int)
            )
        else:
            encodings[GEOHASH_RANK_COL] = pd.Series(dtype=int)

        self.encodings_ = encodings

        distinct_means = pd.unique(smoothed_mean.dropna())
        self.neutral_rank = int(1 + np.sum(distinct_means > self.global_prior_mean))

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.encodings_ is None:
            raise RuntimeError("TargetEncoder.transform called before fit().")

        out = df.copy()
        gh = out[self.geohash_col]
        enc = self.encodings_

        out[GEOHASH_MEAN_COL] = (
            gh.map(enc[GEOHASH_MEAN_COL]).astype(float).fillna(self.global_prior_mean)
        )
        out[GEOHASH_MEDIAN_COL] = (
            gh.map(enc[GEOHASH_MEDIAN_COL]).astype(float).fillna(self.global_prior_mean)
        )
        out[GEOHASH_STD_COL] = (
            gh.map(enc[GEOHASH_STD_COL]).astype(float).fillna(0.0)
        )
        out[GEOHASH_RANK_COL] = (
            gh.map(enc[GEOHASH_RANK_COL]).fillna(self.neutral_rank).astype(int)
        )

        return out

    def fit_oof(
        self,
        df_train: pd.DataFrame,
        target_col: str,
        alpha: float | None = None,
        n_folds: int | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        if alpha is None:
            alpha = TE_SMOOTHING_ALPHA
        if n_folds is None:
            n_folds = TE_N_FOLDS
        if seed is None:
            seed = SEED

        out = df_train.copy()
        n = len(df_train)

        mean_arr = np.empty(n, dtype=float)
        median_arr = np.empty(n, dtype=float)
        std_arr = np.empty(n, dtype=float)
        rank_arr = np.empty(n, dtype=int)

        if n == 0:
            out[GEOHASH_MEAN_COL] = pd.Series(dtype=float)
            out[GEOHASH_MEDIAN_COL] = pd.Series(dtype=float)
            out[GEOHASH_STD_COL] = pd.Series(dtype=float)
            out[GEOHASH_RANK_COL] = pd.Series(dtype=int)
            return out

        positions = np.arange(n)

        if n < 2:
            folds = [(np.array([], dtype=int), positions)]
        else:
            effective_folds = min(n_folds, n)
            kf = KFold(n_splits=effective_folds, shuffle=True, random_state=seed)
            folds = list(kf.split(positions))

        for train_pos, eval_pos in folds:
            fold_fit = df_train.iloc[train_pos]
            fold_encoder = TargetEncoder(self.geohash_col).fit(
                fold_fit, target_col, alpha
            )
            encoded = fold_encoder.transform(df_train.iloc[eval_pos])

            mean_arr[eval_pos] = encoded[GEOHASH_MEAN_COL].to_numpy()
            median_arr[eval_pos] = encoded[GEOHASH_MEDIAN_COL].to_numpy()
            std_arr[eval_pos] = encoded[GEOHASH_STD_COL].to_numpy()
            rank_arr[eval_pos] = encoded[GEOHASH_RANK_COL].to_numpy()

        out[GEOHASH_MEAN_COL] = mean_arr
        out[GEOHASH_MEDIAN_COL] = median_arr
        out[GEOHASH_STD_COL] = std_arr
        out[GEOHASH_RANK_COL] = rank_arr

        return out


# ── Day-48 time-of-day demand curve ──────────────────────────────


def fit_tod_curve(
    df: pd.DataFrame,
    target_col: str,
    day_col: str = "day",
    slot_col: str = "tod_slot",
) -> pd.Series:
    """Fit the day-48 time-of-day demand curve on a fitting partition."""
    day48 = df[df[day_col] == TOD_CURVE_FIT_DAY]

    if len(day48) == 0:
        return pd.Series(dtype=float, name=TOD_CURVE_MEAN_COL).rename_axis(slot_col)

    curve = (
        day48.groupby(slot_col)[target_col]
        .mean()
        .sort_index()
        .rename(TOD_CURVE_MEAN_COL)
    )
    curve.index.name = slot_col
    return curve


def transform_tod_curve(
    df: pd.DataFrame,
    curve: pd.Series,
    slot_col: str = "tod_slot",
    out_col: str = TOD_CURVE_MEAN_COL,
    fallback: float | None = None,
) -> pd.DataFrame:
    """Map a fitted time-of-day curve onto df by tod_slot."""
    if fallback is None:
        if len(curve) > 0:
            overall = curve.mean()
            fallback = float(overall) if pd.notna(overall) else 0.0
        else:
            fallback = 0.0

    out = df.copy()
    out[out_col] = (
        out[slot_col].map(curve).astype(float).fillna(float(fallback))
    )
    return out


# ── Leak-free interaction encoder (geohash × context / time) ─────

# FIX 6: Added "NumberofLanes" — lanes 4&5 have 7× higher mean demand than lanes 1-3
# (0.60 vs 0.09) but had importance only 91. Adding it as a geohash interaction
# will let the model learn per-location lane-demand patterns properly.
INTERACTION_KEYS = ["tod_slot", "Weather", "RoadType", "LargeVehicles", "Landmarks", "NumberofLanes"]


def _interaction_col(other_key: str) -> str:
    """Stable output column name for a geohash×other_key interaction mean."""
    return f"gh_x_{other_key}_mean"


class InteractionEncoder:
    """Leak-free, Bayesian-smoothed geohash×context interaction encoder."""

    def __init__(
        self,
        geohash_col: str = "geohash",
        other_keys: list[str] | None = None,
        fit_day: int = TOD_CURVE_FIT_DAY,
    ) -> None:
        self.geohash_col = geohash_col
        self.other_keys = list(other_keys) if other_keys is not None else list(INTERACTION_KEYS)
        self.fit_day = fit_day
        self.tables_: dict[str, pd.DataFrame] = {}
        self.geohash_fallback_: pd.Series | None = None
        self.global_prior_mean_: float = 0.0
        self.alpha_: float = IE_SMOOTHING_ALPHA
        self._active_keys: list[str] = []

    def fit(
        self,
        df_fit: pd.DataFrame,
        target_col: str,
        alpha: float | None = None,
        day_col: str = "day",
    ) -> "InteractionEncoder":
        if alpha is None:
            alpha = IE_SMOOTHING_ALPHA
        self.alpha_ = float(alpha)
        geo = self.geohash_col

        day_fit = df_fit[df_fit[day_col] == self.fit_day] if day_col in df_fit.columns else df_fit
        if len(day_fit) == 0:
            day_fit = df_fit

        prior = day_fit[target_col].mean()
        self.global_prior_mean_ = float(prior) if pd.notna(prior) else 0.0

        gstat = day_fit.groupby(geo)[target_col].agg(["sum", "count"])
        gh_alpha = IE_SMOOTHING_ALPHA
        self.geohash_fallback_ = (
            (gstat["sum"] + self.global_prior_mean_ * gh_alpha)
            / (gstat["count"] + gh_alpha)
        )

        self._active_keys = [k for k in self.other_keys if k in day_fit.columns]
        self.tables_ = {}
        for key in self._active_keys:
            grp = day_fit.groupby([geo, key], dropna=False)[target_col].agg(["sum", "count"])
            grp = grp.reset_index()
            gh_fb = grp[geo].map(self.geohash_fallback_).fillna(self.global_prior_mean_)
            grp["_val"] = (grp["sum"] + gh_fb * self.alpha_) / (grp["count"] + self.alpha_)
            self.tables_[key] = grp[[geo, key, "_val"]]

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        geo = self.geohash_col
        gh_fb_series = out[geo].map(self.geohash_fallback_) if self.geohash_fallback_ is not None else None
        for key in self._active_keys:
            col = _interaction_col(key)
            table = self.tables_.get(key)
            if table is None or key not in out.columns:
                if gh_fb_series is not None:
                    out[col] = gh_fb_series.fillna(self.global_prior_mean_).to_numpy()
                else:
                    out[col] = self.global_prior_mean_
                continue
            merged = out[[geo, key]].merge(table, on=[geo, key], how="left")
            vals = merged["_val"].to_numpy()
            if gh_fb_series is not None:
                fb = gh_fb_series.fillna(self.global_prior_mean_).to_numpy()
            else:
                fb = np.full(len(out), self.global_prior_mean_)
            vals = np.where(np.isnan(vals), fb, vals)
            out[col] = vals
        return out

    def fit_oof(
        self,
        df_train: pd.DataFrame,
        target_col: str,
        alpha: float | None = None,
        n_folds: int | None = None,
        seed: int | None = None,
        day_col: str = "day",
    ) -> pd.DataFrame:
        if alpha is None:
            alpha = IE_SMOOTHING_ALPHA
        if n_folds is None:
            n_folds = TE_N_FOLDS
        if seed is None:
            seed = SEED

        out = df_train.copy()
        n = len(df_train)
        active = [k for k in self.other_keys if k in df_train.columns]
        cols = {k: _interaction_col(k) for k in active}
        buffers = {k: np.full(n, np.nan, dtype=float) for k in active}

        if n == 0:
            for k in active:
                out[cols[k]] = pd.Series(dtype=float)
            return out

        positions = np.arange(n)
        if n < 2:
            folds = [(np.array([], dtype=int), positions)]
        else:
            effective_folds = min(n_folds, n)
            kf = KFold(n_splits=effective_folds, shuffle=True, random_state=seed)
            folds = list(kf.split(positions))

        for train_pos, eval_pos in folds:
            enc = InteractionEncoder(self.geohash_col, self.other_keys, self.fit_day)
            enc.fit(df_train.iloc[train_pos], target_col, alpha, day_col)
            encoded = enc.transform(df_train.iloc[eval_pos])
            for k in active:
                buffers[k][eval_pos] = encoded[cols[k]].to_numpy()

        full = InteractionEncoder(self.geohash_col, self.other_keys, self.fit_day)
        full.fit(df_train, target_col, alpha, day_col)
        full_enc = full.transform(df_train)
        for k in active:
            mask = np.isnan(buffers[k])
            if mask.any():
                buffers[k][mask] = full_enc[cols[k]].to_numpy()[mask]
            out[cols[k]] = buffers[k]
        return out


def interaction_feature_cols(other_keys: list[str] | None = None) -> list[str]:
    """Return the interaction output column names for the given keys."""
    keys = other_keys if other_keys is not None else INTERACTION_KEYS
    return [_interaction_col(k) for k in keys]


# ── Per-geohash day-48 time-of-day curve (sharp temporal signal) ─

PG_CURVE_COL = "pg_tod_curve_mean"


class PerGeohashTodCurve:
    """Leak-free per-geohash day-48 time-of-day demand curve.

    FIX 5: transform() now uses nearest-slot interpolation instead of
    falling back directly to geohash mean for unmatched (geohash, tod_slot)
    pairs. This improves prediction for the 11% of test rows that have a
    geohash in train but at a slot not observed on day 48.
    """

    def __init__(self, geohash_col: str = "geohash", fit_day: int = TOD_CURVE_FIT_DAY) -> None:
        self.geohash_col = geohash_col
        self.fit_day = fit_day
        self.cell_: pd.DataFrame | None = None
        self.geohash_mean_: pd.Series | None = None
        self.global_prior_: float = 0.0
        self.alpha_: float = 25.0
        # Built at fit time for fast nearest-slot lookup in transform.
        self._geo_slot_lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, df_fit: pd.DataFrame, target_col: str, alpha: float = 25.0,
            day_col: str = "day", slot_col: str = "tod_slot") -> "PerGeohashTodCurve":
        self.alpha_ = float(alpha)
        geo = self.geohash_col
        d48 = df_fit[df_fit[day_col] == self.fit_day] if day_col in df_fit.columns else df_fit
        if len(d48) == 0:
            d48 = df_fit
        p = d48[target_col].mean()
        self.global_prior_ = float(p) if pd.notna(p) else 0.0
        self.geohash_mean_ = d48.groupby(geo)[target_col].mean()
        cell = d48.groupby([geo, slot_col])[target_col].agg(["sum", "count"]).reset_index()
        fb = cell[geo].map(self.geohash_mean_).fillna(self.global_prior_)
        cell["_v"] = (cell["sum"] + fb * self.alpha_) / (cell["count"] + self.alpha_)
        self.cell_ = cell[[geo, slot_col, "_v"]]
        self._slot_col = slot_col

        # Build per-geohash sorted (slots, values) arrays for nearest-slot lookup.
        self._geo_slot_lookup = {}
        for g, grp in self.cell_.groupby(geo):
            sorted_grp = grp.sort_values(slot_col)
            self._geo_slot_lookup[g] = (
                sorted_grp[slot_col].to_numpy(dtype=int),
                sorted_grp["_v"].to_numpy(dtype=float),
            )

        return self

    def transform(self, df: pd.DataFrame, slot_col: str = "tod_slot",
                  out_col: str = PG_CURVE_COL) -> pd.DataFrame:
        """Merge per-geohash curve onto df with nearest-slot fallback.

        Lookup chain:
          1. Exact (geohash, tod_slot) match → smoothed cell mean.
          2. Nearest observed tod_slot for same geohash → nearest cell value.
          3. Geohash mean (for geohashes with no day-48 data at all).
          4. Global prior (for geohashes never seen in training).
        """
        out = df.copy()
        geo = self.geohash_col

        if self.cell_ is None or len(self.cell_) == 0:
            fb_series = out[geo].map(self.geohash_mean_).fillna(self.global_prior_) \
                if self.geohash_mean_ is not None else pd.Series(self.global_prior_, index=out.index)
            out[out_col] = fb_series.to_numpy()
            return out

        merged = out[[geo, slot_col]].merge(self.cell_, on=[geo, slot_col], how="left")
        fb = (out[geo].map(self.geohash_mean_).fillna(self.global_prior_).to_numpy()
              if self.geohash_mean_ is not None
              else np.full(len(out), self.global_prior_))
        vals = merged["_v"].to_numpy(dtype=float).copy()  # .copy() ensures writability for in-place nearest-slot fill

        # FIX 5: nearest-slot interpolation for unmatched (geohash, slot) pairs.
        unmatched_mask = np.isnan(vals)
        if unmatched_mask.any():
            query_geos = out[geo].to_numpy()
            query_slots = out[slot_col].to_numpy(dtype=int)
            for i in np.where(unmatched_mask)[0]:
                g = query_geos[i]
                s = query_slots[i]
                if g in self._geo_slot_lookup:
                    slots_arr, vals_arr = self._geo_slot_lookup[g]
                    # Find the nearest observed slot (searchsorted for O(log n)).
                    pos = int(np.searchsorted(slots_arr, s))
                    if pos == 0:
                        nearest_idx = 0
                    elif pos >= len(slots_arr):
                        nearest_idx = len(slots_arr) - 1
                    else:
                        left_dist = abs(int(slots_arr[pos - 1]) - s)
                        right_dist = abs(int(slots_arr[pos]) - s)
                        nearest_idx = pos - 1 if left_dist <= right_dist else pos
                    vals[i] = vals_arr[nearest_idx]
                # else: geohash not in lookup → keep NaN, filled by fb below.

        out[out_col] = np.where(np.isnan(vals), fb, vals)
        return out

    def fit_oof(self, df_train: pd.DataFrame, target_col: str, alpha: float = 25.0,
                n_folds: int | None = None, seed: int | None = None,
                day_col: str = "day", slot_col: str = "tod_slot",
                out_col: str = PG_CURVE_COL) -> pd.DataFrame:
        if n_folds is None:
            n_folds = TE_N_FOLDS
        if seed is None:
            seed = SEED
        out = df_train.copy()
        n = len(df_train)
        buf = np.full(n, np.nan)
        if n == 0:
            out[out_col] = pd.Series(dtype=float)
            return out
        pos = np.arange(n)
        folds = ([(np.array([], dtype=int), pos)] if n < 2
                 else list(KFold(min(n_folds, n), shuffle=True, random_state=seed).split(pos)))
        for tri, evi in folds:
            c = PerGeohashTodCurve(self.geohash_col, self.fit_day).fit(
                df_train.iloc[tri], target_col, alpha, day_col, slot_col)
            buf[evi] = c.transform(df_train.iloc[evi], slot_col, out_col)[out_col].to_numpy()
        full = PerGeohashTodCurve(self.geohash_col, self.fit_day).fit(
            df_train, target_col, alpha, day_col, slot_col)
        fenc = full.transform(df_train, slot_col, out_col)[out_col].to_numpy()
        mask = np.isnan(buf)
        buf[mask] = fenc[mask]
        out[out_col] = buf
        return out