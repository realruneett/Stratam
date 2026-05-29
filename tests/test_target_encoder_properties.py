"""Property-based test for leak-free geohash target encoding (Property 2).

This file implements the single property-based test for design Property 2 of the
demand-prediction-overhaul spec. It exercises ``src.spatial.TargetEncoder`` across
many generated raw-schema frames (built from the shared Hypothesis strategies),
covering frames that span both day 48 and day 49 and that contain single-row
geohash groups.

The property has three sub-claims, all asserted in the one test below:

  (a) Leak-free OOF: each row's out-of-fold encoding equals the encoding produced
      by a fresh encoder fit on the complement of that row's fold, using the same
      ``KFold(shuffle=True, random_state=seed)`` split that ``fit_oof`` uses.
  (b) Mutation-invariance: mutating ONLY one row's target does not change THAT
      row's own OOF encoding (its encoding comes from the complement of its fold,
      which never contains the row itself).
  (c) Shrinkage: for the non-OOF ``fit`` path, every smoothed geohash mean lies
      between the global prior mean and the raw geohash group mean (inclusive).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from sklearn.model_selection import KFold

from config import TE_SMOOTHING_ALPHA
from src.spatial import (
    ENCODING_COLS,
    GEOHASH_MEAN_COL,
    GEOHASH_MEDIAN_COL,
    GEOHASH_RANK_COL,
    GEOHASH_STD_COL,
    TargetEncoder,
)
from tests import strategies as strat

# Fixed OOF configuration shared between fit_oof and the reconstruction so the
# KFold split matches exactly.
_N_FOLDS = 4
_SEED = 123
_ALPHA = TE_SMOOTHING_ALPHA
_ATOL = 1e-9


def _reconstruct_oof_folds(n: int, n_folds: int, seed: int):
    """Mirror TargetEncoder.fit_oof's fold assignment exactly.

    Returns a list of ``(train_pos, eval_pos)`` index arrays identical to the
    ones ``fit_oof`` iterates over, so the reconstruction below fits on the same
    complements.
    """
    positions = np.arange(n)
    if n < 2:
        return [(np.array([], dtype=int), positions)]
    effective_folds = min(n_folds, n)
    kf = KFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    return list(kf.split(positions))


# Feature: demand-prediction-overhaul, Property 2: Geohash target encodings are leak-free, present, and shrink toward the prior
@given(df=strat.two_day_frame(max_rows_per_day=15))
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_geohash_encodings_leak_free_present_and_shrink(df: pd.DataFrame) -> None:
    df = df.reset_index(drop=True)
    n = len(df)

    # ── Sub-claim (a): leak-free OOF reconstruction ──────────────
    oof = TargetEncoder("geohash").fit_oof(
        df, target_col="demand", alpha=_ALPHA, n_folds=_N_FOLDS, seed=_SEED
    )

    # The geohash encoding columns are present in the OOF feature set.
    for col in ENCODING_COLS:
        assert col in oof.columns

    folds = _reconstruct_oof_folds(n, _N_FOLDS, _SEED)
    for train_pos, eval_pos in folds:
        complement = df.iloc[train_pos]
        fresh = TargetEncoder("geohash").fit(complement, "demand", _ALPHA)
        encoded = fresh.transform(df.iloc[eval_pos])

        for col in (GEOHASH_MEAN_COL, GEOHASH_MEDIAN_COL, GEOHASH_STD_COL):
            np.testing.assert_allclose(
                oof[col].to_numpy()[eval_pos],
                encoded[col].to_numpy(),
                atol=_ATOL,
                err_msg=f"OOF {col} differs from complement-fit encoding",
            )
        # Rank is an integer code and must match exactly.
        np.testing.assert_array_equal(
            oof[GEOHASH_RANK_COL].to_numpy()[eval_pos],
            encoded[GEOHASH_RANK_COL].to_numpy(),
        )

    # ── Sub-claim (b): mutation-invariance of a row's own encoding ──
    victim = n // 2  # any fixed position
    original = float(df.loc[victim, "demand"])
    # Guaranteed-different in-range value.
    mutated_val = original + 0.5 if original <= 0.5 else original - 0.5

    df_mut = df.copy()
    df_mut.loc[victim, "demand"] = mutated_val
    oof_mut = TargetEncoder("geohash").fit_oof(
        df_mut, target_col="demand", alpha=_ALPHA, n_folds=_N_FOLDS, seed=_SEED
    )

    for col in (GEOHASH_MEAN_COL, GEOHASH_MEDIAN_COL, GEOHASH_STD_COL):
        assert np.isclose(
            oof[col].to_numpy()[victim],
            oof_mut[col].to_numpy()[victim],
            atol=_ATOL,
        ), f"Mutating row {victim}'s target changed its own OOF {col}"
    assert (
        oof[GEOHASH_RANK_COL].to_numpy()[victim]
        == oof_mut[GEOHASH_RANK_COL].to_numpy()[victim]
    ), f"Mutating row {victim}'s target changed its own OOF rank"

    # ── Sub-claim (c): shrinkage toward the prior (non-OOF fit) ──
    encoder = TargetEncoder("geohash").fit(df, "demand", _ALPHA)
    prior = encoder.global_prior_mean
    raw_group_mean = df.groupby("geohash")["demand"].mean()

    # The encoding columns are present after a plain fit/transform as well.
    transformed = encoder.transform(df)
    for col in ENCODING_COLS:
        assert col in transformed.columns

    for geohash, smoothed in encoder.encodings_[GEOHASH_MEAN_COL].items():
        raw = float(raw_group_mean.loc[geohash])
        lo, hi = min(prior, raw), max(prior, raw)
        assert lo - _ATOL <= smoothed <= hi + _ATOL, (
            f"smoothed mean {smoothed} for {geohash!r} not between prior "
            f"{prior} and raw group mean {raw}"
        )
