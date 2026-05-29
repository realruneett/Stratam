"""Property-based test for Task 3.3 (Property 7).

Validates that geohashes appearing in the test frame but absent from the
training frame receive the train-derived fallback encoding from
:class:`src.spatial.TargetEncoder`: the global prior mean for the smoothed
mean/median, a zero std, and the neutral fallback rank (Requirement 3.7).
"""

from __future__ import annotations

import math

from hypothesis import HealthCheck, given, settings

from src.spatial import (
    GEOHASH_MEAN_COL,
    GEOHASH_MEDIAN_COL,
    GEOHASH_RANK_COL,
    GEOHASH_STD_COL,
    TargetEncoder,
)
from tests import strategies as strat

_TOL = 1e-9


# Feature: demand-prediction-overhaul, Property 7: Unseen geohashes receive the train-derived fallback encoding
@given(pair=strat.train_test_pair(with_unseen_geohash=True))
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.data_too_large, HealthCheck.too_slow],
)
def test_unseen_geohash_receives_train_derived_fallback(pair):
    """Unseen-geohash test rows get global_prior_mean + neutral fallback rank."""
    train_df, test_df = pair

    encoder = TargetEncoder().fit(train_df, target_col="demand")
    encoded = encoder.transform(test_df)

    seen_geohashes = set(train_df["geohash"].unique())
    unseen_mask = ~test_df["geohash"].isin(seen_geohashes).to_numpy()

    # The strategy stamps at least one unseen geohash onto a test row, so the
    # fallback path is guaranteed to be exercised.
    assert unseen_mask.any(), "strategy must produce at least one unseen-geohash row"

    unseen_rows = encoded.loc[unseen_mask]

    for value in unseen_rows[GEOHASH_MEAN_COL]:
        assert math.isclose(
            value, encoder.global_prior_mean, rel_tol=0.0, abs_tol=_TOL
        )
    for value in unseen_rows[GEOHASH_MEDIAN_COL]:
        assert math.isclose(
            value, encoder.global_prior_mean, rel_tol=0.0, abs_tol=_TOL
        )
    for value in unseen_rows[GEOHASH_STD_COL]:
        assert math.isclose(value, 0.0, rel_tol=0.0, abs_tol=_TOL)
    for value in unseen_rows[GEOHASH_RANK_COL]:
        assert int(value) == encoder.neutral_rank
