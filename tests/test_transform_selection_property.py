"""Property-based test for Task 6.2 (Property 10).

Validates that :func:`src.data.select_transform` returns the candidate target
transform with the higher Holdout_R2 (identity on an exact tie), per
Requirement 4.2.

Approach: ``select_transform`` is model-agnostic -- it delegates fitting to a
caller-supplied ``fit_predict_fn(y_train_transformed) -> y_eval_pred_transformed``
and scores the inverse-transformed predictions against the original eval target.
The test controls that callback so each candidate transform yields a *known*
Holdout_R2: the favored transform receives a perfect prediction (R2 == 1.0) and
the other a constant far from the eval target (R2 < 0, but the favored positive
score keeps the run from raising). The callback recovers which transform space
it is being called in by matching its input against the forward transform of the
(fixed) training target, so it never relies on call order. Selection must then
return the favored transform; an engineered exact tie (both perfect) must
resolve to ``"identity"``.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from src.data import (
    VALID_TRANSFORMS,
    apply_transform,
    select_transform,
)
from tests import strategies as strat


def _detect_transform(y_train_transformed, transformed_inputs: dict) -> str:
    """Recover which transform produced ``y_train_transformed`` (order-free).

    ``select_transform`` calls the callback once per candidate transform, each
    time with that transform's forward image of the (fixed) training target.
    Matching the received array against the precomputed forward images recovers
    the active transform without depending on iteration order. A positive
    training value guarantees the identity and ``log1p`` images differ, so the
    match is unambiguous.
    """
    arr = np.asarray(y_train_transformed, dtype=float)
    matches = [
        name
        for name, vals in transformed_inputs.items()
        if vals.shape == arr.shape and np.allclose(vals, arr)
    ]
    if len(matches) == 1:
        return matches[0]
    return "identity"


def _make_fit_predict(y_train_part, y_eval_original, favored: str):
    """Build a deterministic ``fit_predict_fn`` with controlled per-transform R2.

    For the ``favored`` transform the callback returns a perfect prediction
    (after inversion, exactly ``y_eval_original`` -> R2 == 1.0); for the other
    it returns a constant well above the eval target (R2 < 0). When ``favored``
    is ``"tie"`` both transforms get the perfect prediction, producing an exact
    Holdout_R2 tie.
    """
    y_train_part = np.asarray(y_train_part, dtype=float)
    y_eval_original = np.asarray(y_eval_original, dtype=float)
    transformed_inputs = {
        name: apply_transform(y_train_part, name) for name in VALID_TRANSFORMS
    }
    bad_const = float(np.max(y_eval_original)) + 5.0

    def fit_predict(y_train_transformed):
        transform = _detect_transform(y_train_transformed, transformed_inputs)
        if favored == "tie" or transform == favored:
            desired_original = y_eval_original          # perfect -> R2 == 1.0
        else:
            desired_original = np.full(
                y_eval_original.shape, bad_const, dtype=float
            )                                            # off-target -> R2 < 0
        # Return predictions in the transform's own space; select_transform
        # inverts them back to original space before scoring.
        return apply_transform(desired_original, transform)

    return fit_predict


# Feature: demand-prediction-overhaul, Property 10: Transform selection chooses the higher Holdout_R2
@given(
    y_train=st.lists(strat.demand_value(), min_size=2, max_size=12),
    y_eval=st.lists(strat.demand_value(), min_size=2, max_size=12),
    favored=st.sampled_from(["identity", "log1p", "tie"]),
)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_select_transform_chooses_higher_holdout_r2(y_train, y_eval, favored):
    """select_transform returns the higher-R2 transform; identity on a tie."""
    # The identity and log1p forward images of the training target must be
    # *distinguishable* for the order-free callback to recover the active
    # transform. For tiny values log1p(x) ≈ x, so the two images collapse to
    # within floating-point tolerance and the transform becomes unrecoverable
    # (which would spuriously hand both candidates a failing R2). Require the
    # two forward images to differ by more than the detection tolerance so the
    # callback is always unambiguous. Eval variance keeps r2_score well-defined.
    y_train_arr = np.asarray(y_train, dtype=float)
    identity_img = apply_transform(y_train_arr, "identity")
    log1p_img = apply_transform(y_train_arr, "log1p")
    assume(not np.allclose(identity_img, log1p_img, atol=1e-8))
    assume(float(np.std(y_eval)) > 1e-6)

    fit_predict = _make_fit_predict(y_train, y_eval, favored)
    chosen, details = select_transform(y_train, y_eval, fit_predict)

    assert chosen in VALID_TRANSFORMS
    assert set(details) >= {"identity_r2", "log1p_r2", "chosen", "r2_scores"}
    assert details["chosen"] == chosen

    if favored == "tie":
        # Both candidates score an identical (perfect) Holdout_R2 -> identity.
        assert details["identity_r2"] == pytest.approx(details["log1p_r2"])
        assert chosen == "identity"
    else:
        # The favored transform was handed the strictly higher Holdout_R2.
        other = "log1p" if favored == "identity" else "identity"
        assert details[f"{favored}_r2"] == pytest.approx(1.0)
        assert details[f"{favored}_r2"] > details[f"{other}_r2"]
        assert chosen == favored
