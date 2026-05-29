"""Unit test for Task 10.2: OOF-only Ridge meta-learner + baseline floor.

This is a plain unit test (NOT one of the numbered correctness Properties
1-14), so it intentionally carries no ``# ... Property N`` tag.

It pins down the two guarantees ``src.ensemble.stack_predictions`` makes for the
stacking design (Requirement 4.5, design "src/ensemble.py — stacking with a
floor"):

  1. **OOF-only meta-learner.** The Ridge meta-learner is fit on out-of-fold
     predictions only — with the leak-free ``KFold`` OOF scheme every training
     row now feeds the meta-learner, so ``info["oof_mask"]`` is all-``True`` and
     sums to ``len(y_train)``. The fitted ``info["ridge"]`` exposes one
     coefficient per *stacked* model (the GBMs plus the appended baseline
     column), and ``info["ensemble_r2"]`` is the R² of the chosen OOF blend
     against ``y_train``.

  2. **Baseline floor.** When the GBM stack scores worse (OOF R²) than the
     robust baseline alone, the final predictions fall back to the baseline's
     test predictions (``method="baseline_floor"``, ``floor_applied=True``).
     When the GBM stack meets or beats the baseline the floor stays inactive and
     the ensemble R² is at least the baseline R².

The inputs are tiny, hand-built synthetic :class:`~src.models.ModelResult`
objects (no real training), so the test is fast and fully deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import r2_score

from src.ensemble import stack_predictions
from src.models import ModelResult


def _signal(n: int) -> np.ndarray:
    """A smooth, non-constant target of length ``n`` (deterministic)."""
    t = np.linspace(0.0, 4.0 * np.pi, n)
    return 5.0 + 2.0 * np.sin(t) + 0.5 * np.cos(3.0 * t)


def _make_result(
    name: str, oof: np.ndarray, test_preds: np.ndarray
) -> ModelResult:
    """Build a synthetic ModelResult carrying just ``oof`` / ``test_preds``."""
    return ModelResult(
        name=name,
        oof=np.asarray(oof, dtype=float),
        test_preds=np.asarray(test_preds, dtype=float),
    )


@pytest.fixture
def rng() -> np.random.Generator:
    """A deterministic RNG for the synthetic OOF / test arrays."""
    return np.random.default_rng(20240517)


def test_meta_learner_is_fit_on_oof_rows_only(rng: np.random.Generator) -> None:
    """Ridge is fit on every (finite) OOF row; coefficients cover all stacked models."""
    n, m = 150, 30
    y = _signal(n)
    std = float(np.std(y))

    # Three GBMs whose OOF closely tracks y (independent small noise each).
    gbms = [
        _make_result(
            f"gbm{i}",
            oof=y + rng.normal(0.0, 0.05 * std, n),
            test_preds=rng.normal(5.0, 1.0, m),
        )
        for i in range(3)
    ]
    # A decent-but-weaker baseline (larger noise) so it never floors the stack.
    baseline = _make_result(
        "baseline",
        oof=y + rng.normal(0.0, 0.5 * std, n),
        test_preds=rng.normal(5.0, 1.0, m),
    )

    final_preds, info = stack_predictions(gbms, y, baseline_result=baseline)

    # ── The meta-learner used OOF rows only — here, all rows. ────
    mask = info["oof_mask"]
    assert mask.dtype == bool
    assert mask.shape == (n,)
    assert bool(np.all(mask))
    assert int(mask.sum()) == n

    # ── Ridge is fitted with one coefficient per stacked model. ──
    # Stack order is GBMs first, then the appended baseline column.
    ridge = info["ridge"]
    assert ridge is not None
    assert hasattr(ridge, "coef_")
    assert len(ridge.coef_) == len(gbms) + 1  # 3 GBMs + baseline

    # ── ensemble_r2 is the R² of the chosen OOF blend vs y_train. ─
    assert info["floor_applied"] is False
    assert info["method"] in {"ridge", "weighted_avg"}
    expected_r2 = r2_score(y, info["oof_blend"])
    assert info["ensemble_r2"] == pytest.approx(expected_r2)

    assert final_preds.shape == (m,)
    assert np.all(np.isfinite(final_preds))


def test_baseline_floor_applied_when_stack_underperforms(
    rng: np.random.Generator,
) -> None:
    """A perfect baseline + noisy GBMs trigger the floor → final == baseline test preds."""
    n, m = 150, 30
    y = _signal(n)

    # GBMs are pure noise → poor OOF predictors of y.
    gbms = [
        _make_result(
            f"gbm{i}",
            oof=rng.normal(5.0, 3.0, n),
            test_preds=rng.normal(5.0, 1.0, m),
        )
        for i in range(2)
    ]
    # Baseline OOF is the target itself (R² == 1) with distinct test predictions.
    baseline_test = rng.normal(7.0, 2.0, m)
    baseline = _make_result("baseline", oof=y.copy(), test_preds=baseline_test)

    final_preds, info = stack_predictions(gbms, y, baseline_result=baseline)

    assert info["floor_applied"] is True
    assert info["method"] == "baseline_floor"
    # The baseline scores perfectly; the regularized stack cannot beat it.
    assert info["baseline_r2"] == pytest.approx(1.0)
    assert info["ensemble_r2"] <= info["baseline_r2"]
    # Final predictions fall back to the baseline's test predictions exactly.
    np.testing.assert_array_equal(final_preds, baseline_test)


def test_baseline_floor_not_applied_when_stack_beats_baseline(
    rng: np.random.Generator,
) -> None:
    """Strong GBMs + a noisy baseline keep the floor inactive (ens R² >= baseline R²)."""
    n, m = 150, 30
    y = _signal(n)
    std = float(np.std(y))

    # GBMs track y well.
    gbms = [
        _make_result(
            f"gbm{i}",
            oof=y + rng.normal(0.0, 0.05 * std, n),
            test_preds=rng.normal(5.0, 1.0, m),
        )
        for i in range(2)
    ]
    # Baseline is noisy → low OOF R².
    baseline = _make_result(
        "baseline",
        oof=y + rng.normal(0.0, 2.0 * std, n),
        test_preds=rng.normal(5.0, 1.0, m),
    )

    final_preds, info = stack_predictions(gbms, y, baseline_result=baseline)

    assert info["floor_applied"] is False
    assert info["method"] in {"ridge", "weighted_avg"}
    assert info["baseline_r2"] is not None
    assert info["ensemble_r2"] >= info["baseline_r2"]
    assert final_preds.shape == (m,)
    assert np.all(np.isfinite(final_preds))


def test_negative_ridge_weight_falls_back_to_non_negative_weighted_avg(
    rng: np.random.Generator,
) -> None:
    """An anti-correlated model forces a negative Ridge coef → weighted_avg fallback."""
    n, m = 150, 30
    y = _signal(n)

    # A single GBM whose OOF is anti-correlated with y (mirror image). Stacked
    # with a near-perfect baseline, the Ridge solution drives the GBM coef
    # negative, which must trip the clipped non-negative weighted-average path.
    gbm = _make_result("gbm_anti", oof=-y, test_preds=rng.normal(5.0, 1.0, m))
    baseline = _make_result(
        "baseline",
        oof=y + rng.normal(0.0, 1e-3, n),
        test_preds=rng.normal(5.0, 1.0, m),
    )

    _final_preds, info = stack_predictions([gbm], y, baseline_result=baseline)

    # The raw Ridge fit produced at least one negative coefficient (the trigger).
    assert np.any(info["ridge"].coef_ < 0)
    # ...so the function fell back to clipped, normalized non-negative weights.
    assert info["method"] == "weighted_avg"
    weights = np.asarray(info["weights"], dtype=float)
    assert np.all(weights >= 0.0)
    assert float(weights.sum()) == pytest.approx(1.0)
