"""Property-based test for leak-free fitted artifacts (Task 4.3, Property 9).

This file implements the single property-based test for design Property 9 of the
demand-prediction-overhaul spec. It verifies the leakage boundary enforced by the
validators in ``src.validation``: every target-derived artifact used to *score* a
holdout's eval partition is fit on the ``train_idx`` partition only, so mutating
the eval-partition targets cannot change any fitted artifact.

Concretely it uses :func:`src.validation.build_day48_daytime_holdout` (the primary
surrogate, whose eval = day-48 daytime rows in ``[9, 55]`` and whose train = the
complement) and the two target-derived artifacts that are implemented today:

  * :class:`src.spatial.TargetEncoder` (geohash encodings + ``global_prior_mean`` +
    ``neutral_rank``), and
  * :func:`src.spatial.fit_tod_curve` (the day-48 time-of-day demand curve).

The split structure depends only on ``day`` / ``tod_slot`` (never on the target),
so rebuilding the split on the mutated frame yields an identical ``train_idx``;
because that partition excludes every eval row, the artifacts fit on it must be
byte-for-byte identical before and after the eval targets are mutated.

The shared strategies emit *raw-schema* frames with a ``day`` column and a raw
``timestamp`` (``"H:M"``) string but no parsed ``tod_slot``; the validator needs a
``tod_slot`` column, so we reproduce ``load_data``'s
``tod_slot = (hour*60 + minute) // 15`` derivation here first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, assume, given, settings

from src.spatial import (
    ENCODING_COLS,
    GEOHASH_MEAN_COL,
    TargetEncoder,
    fit_tod_curve,
)
from src.validation import STATUS_OK, build_day48_daytime_holdout
from tests import strategies as strat

_ATOL = 1e-9
# A target perturbation large enough that any leak from the eval partition into a
# train-fit artifact would be unmistakable.
_MUTATION = 1000.0


def _add_tod_slot(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the integer ``tod_slot`` (0..95) from the raw ``"H:M"`` timestamp.

    Mirrors ``load_data``'s ``tod_slot = (hour*60 + minute) // 15`` derivation so
    the raw-schema strategy frames carry the ``tod_slot`` column the validator
    needs to build a holdout.
    """
    out = df.copy()
    parts = out["timestamp"].str.split(":", expand=True)
    hour = parts[0].astype(int)
    minute = parts[1].astype(int)
    out["tod_slot"] = (hour * 60 + minute) // 15
    return out


# Feature: demand-prediction-overhaul, Property 9: Validator excludes eval-fold targets from all fitted artifacts
@given(df=strat.two_day_frame(max_rows_per_day=15))
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_fitted_artifacts_exclude_eval_fold_targets(df: pd.DataFrame) -> None:
    df = _add_tod_slot(df.reset_index(drop=True))

    # Build the surrogate holdout: eval = day-48 daytime rows in [9, 55],
    # train = the complement. Artifacts that score eval are fit on train only.
    split = build_day48_daytime_holdout(df)

    # Only meaningful when there are eval rows whose targets we can mutate;
    # an empty eval partition would make the mutation a no-op (vacuously true).
    assume(split.status == STATUS_OK)
    assume(split.eval_idx.size > 0)

    train_part = df.loc[split.train_idx]

    # ── Fit the target-derived artifacts on the TRAIN partition only ──
    encoder = TargetEncoder("geohash").fit(train_part, target_col="demand")
    curve = fit_tod_curve(train_part, target_col="demand")

    # ── Mutate ONLY the eval-partition targets ───────────────────
    df_mut = df.copy()
    df_mut.loc[split.eval_idx, "demand"] = (
        df_mut.loc[split.eval_idx, "demand"] + _MUTATION
    )

    # Rebuilding the split on the mutated frame must yield the SAME partitions:
    # the split depends only on day / tod_slot, never on the target.
    split_mut = build_day48_daytime_holdout(df_mut)
    np.testing.assert_array_equal(
        split.train_idx, split_mut.train_idx,
        err_msg="train_idx changed after mutating eval-partition targets",
    )
    np.testing.assert_array_equal(
        split.eval_idx, split_mut.eval_idx,
        err_msg="eval_idx changed after mutating eval-partition targets",
    )

    # Re-fit on the (re-derived) train partition of the mutated frame.
    train_part_mut = df_mut.loc[split_mut.train_idx]
    encoder_mut = TargetEncoder("geohash").fit(train_part_mut, target_col="demand")
    curve_mut = fit_tod_curve(train_part_mut, target_col="demand")

    # ── Assert every fitted artifact is identical ────────────────
    # Geohash encoding table (mean/median/std/rank per geohash).
    pd.testing.assert_frame_equal(
        encoder.encodings_[ENCODING_COLS],
        encoder_mut.encodings_[ENCODING_COLS],
        check_like=True,
        check_exact=False,
        atol=_ATOL,
        obj="encoder.encodings_",
    )

    # Global prior mean (the unseen-geohash fallback) is train-derived only.
    assert np.isclose(
        encoder.global_prior_mean, encoder_mut.global_prior_mean, atol=_ATOL
    ), "global_prior_mean changed after mutating eval-partition targets"

    # Neutral fallback rank depends only on the fitted smoothed means.
    assert encoder.neutral_rank == encoder_mut.neutral_rank, (
        "neutral_rank changed after mutating eval-partition targets"
    )

    # The smoothed-mean column is the artifact actually merged onto eval rows;
    # assert it explicitly in addition to the whole-frame check above.
    np.testing.assert_allclose(
        encoder.encodings_[GEOHASH_MEAN_COL].sort_index().to_numpy(),
        encoder_mut.encodings_[GEOHASH_MEAN_COL].sort_index().to_numpy(),
        atol=_ATOL,
        err_msg="geohash_mean encodings changed after mutating eval targets",
    )

    # Day-48 time-of-day demand curve.
    pd.testing.assert_series_equal(
        curve.sort_index(),
        curve_mut.sort_index(),
        check_exact=False,
        atol=_ATOL,
        obj="tod_curve",
    )
