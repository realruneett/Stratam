"""Unit tests for Task 6.3: transform-selection edge cases.

Covers the two documented edge behaviors of :func:`src.data.select_transform`:

* Requirement 4.3 -- an exact Holdout_R2 tie resolves to ``"identity"``.
* Requirement 4.4 -- when both candidates yield Holdout_R2 <= 0 the run fails
  with :class:`src.data.TransformSelectionError`, reporting that no transform
  achieved a positive Holdout_R2.

Plus a sanity check on the reversible ``apply_transform`` / ``invert_transform``
pair (``log1p`` round-trips on the non-negative demand domain).

These are example-based unit tests (not one of the numbered Properties 1-14),
so they carry no ``Property N`` comment tag. ``select_transform`` is
model-agnostic: each test supplies a ``fit_predict_fn`` crafted to force the
desired per-transform Holdout_R2 outcome.

Requirements: 4.3, 4.4
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data import (
    TransformSelectionError,
    apply_transform,
    invert_transform,
    select_transform,
)


def _make_transform_aware(pred_original, y_train_part):
    """fit_predict_fn returning a fixed original-space prediction per candidate.

    Identifies the active transform by matching the received training target
    against the precomputed identity / log1p forward images of ``y_train_part``,
    then returns that transform's forward image of ``pred_original`` so the
    inverse-transform inside ``select_transform`` recovers exactly
    ``pred_original`` for BOTH candidates (an exact Holdout_R2 tie).
    """
    pred_original = np.asarray(pred_original, dtype=float)
    images = {
        "identity": apply_transform(y_train_part, "identity"),
        "log1p": apply_transform(y_train_part, "log1p"),
    }

    def fit_predict(y_train_transformed):
        arr = np.asarray(y_train_transformed, dtype=float)
        transform = "identity"
        for name, img in images.items():
            if img.shape == arr.shape and np.allclose(img, arr):
                transform = name
                break
        return apply_transform(pred_original, transform)

    return fit_predict


def test_exact_tie_selects_identity():
    """Identical Holdout_R2 for both candidates -> identity is chosen (4.3)."""
    y_train = np.array([0.2, 0.4, 0.6, 0.8])
    y_eval = np.array([0.1, 0.5, 0.9])
    # A decent (positive-R2) but identical prediction for both transforms.
    pred_original = np.array([0.12, 0.52, 0.88])

    fit_predict = _make_transform_aware(pred_original, y_train)
    chosen, details = select_transform(y_train, y_eval, fit_predict)

    assert details["identity_r2"] == pytest.approx(details["log1p_r2"])
    assert details["identity_r2"] > 0.0          # positive -> does not raise
    assert chosen == "identity"
    assert details["chosen"] == "identity"


def test_both_non_positive_r2_raises_transform_selection_error():
    """Both candidates Holdout_R2 <= 0 -> TransformSelectionError (4.4)."""
    y_train = np.array([0.2, 0.4, 0.6])
    y_eval = np.array([0.1, 0.5, 0.9])
    # A constant prediction far above the eval target scores R2 < 0 in both
    # transform spaces (after inversion it is the same constant either way).
    far_constant = float(np.max(y_eval)) + 10.0
    pred_original = np.full(y_eval.shape, far_constant)

    fit_predict = _make_transform_aware(pred_original, y_train)

    with pytest.raises(TransformSelectionError) as exc_info:
        select_transform(y_train, y_eval, fit_predict)

    message = str(exc_info.value).lower()
    assert "positive" in message
    assert "transform" in message


def test_log1p_round_trip_recovers_original():
    """invert_transform(apply_transform(y, 'log1p'), 'log1p') ~= y for y >= 0."""
    y = np.array([0.0, 6e-7, 0.05, 0.42, 1.0])
    round_tripped = invert_transform(apply_transform(y, "log1p"), "log1p")
    np.testing.assert_allclose(round_tripped, y, rtol=1e-9, atol=1e-12)


def test_identity_round_trip_is_exact():
    """The identity transform and its inverse are no-ops on the demand domain."""
    y = np.array([0.0, 0.13, 0.77, 1.0])
    np.testing.assert_array_equal(
        invert_transform(apply_transform(y, "identity"), "identity"),
        y,
    )
