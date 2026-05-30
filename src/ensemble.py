"""
Non-Negative Least Squares stacking meta-learner over base-model OOF
predictions, with a baseline floor (Requirement 4.5, design
"src/ensemble.py — stacking with a floor").

The meta-learner is fit on **out-of-fold predictions only**. With the leak-free
``KFold`` OOF scheme in ``src.models`` every training row now receives an OOF
prediction, so the old ``oof != 0`` heuristic (a workaround for the removed
``time_kfold_split`` which skipped fold 0) is no longer needed — all training
rows feed the meta-learner.

When a ``baseline_result`` is supplied, the robust baseline (geohash mean ×
time-of-day curve blend) participates in two ways:

1. **In the stack** — its OOF column is added as an extra meta-feature so the
   meta-learner can lean on it.
2. **As a floor** — after blending, the stack's OOF R² is compared against the
   baseline's OOF R². If the stack scores *worse* than the baseline alone on the
   surrogate holdout, the final predictions fall back to the baseline's test
   predictions (``method="baseline_floor"``). This guarantees the ensemble can
   never score worse than the robust baseline on the surrogate holdout.

When ``baseline_result`` is ``None`` the function behaves exactly as before
(GBM models only), preserving backward compatibility with the current
``run.py`` call site.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from src.models import ModelResult, rmse


def stack_predictions(
    results: list[ModelResult],
    y_train: np.ndarray,
    baseline_result: ModelResult | None = None,
) -> tuple[np.ndarray, dict]:
    """Blend base-model predictions via True Non-Negative Least Squares Stacking."""
    n = len(y_train)
    y_train = np.asarray(y_train, dtype=float)

    # ── Rows usable by the meta-learner ──────────────────────────
    oof_mask = np.ones(n, dtype=bool)
    for r in results:
        oof_mask &= np.isfinite(r.oof)
    if baseline_result is not None:
        oof_mask &= np.isfinite(baseline_result.oof)

    # ── Models feeding the meta-learner ──────────────────────────
    stack_models = list(results)
    if baseline_result is not None:
        stack_models = list(results) + [baseline_result]

    X_meta = np.column_stack([r.oof[oof_mask] for r in stack_models])
    y_meta = y_train[oof_mask]

    # ── Fit True Non-Negative Linear Regression ──────────────────
    # positive=True activates a bounded solver instead of unconstrained fitting + clipping
    meta_learner = LinearRegression(positive=True, fit_intercept=True)
    meta_learner.fit(X_meta, y_meta)

    # Normalize weights so they sum to 1.0 to preserve the target scale bounds
    raw_coefs = meta_learner.coef_
    coef_sum = raw_coefs.sum()
    if coef_sum > 0:
        weights = raw_coefs / coef_sum
    else:
        weights = np.full(len(stack_models), 1.0 / len(stack_models))

    print("\n" + "=" * 60)
    print("META-LEARNER ACTIVE-SET BOUNDED WEIGHTS")
    print("=" * 60)
    for model, weight in zip(stack_models, weights):
        print(f"  Model: {model.name:<15s} -> True Stack Weight: {weight:.5f}")
    print(f"  Meta Intercept : {meta_learner.intercept_:.6f}\n")

    # ── Build the true stacked test predictions + OOF blend ──────
    # Compute convex combination to guarantee continuous mapping integration
    final_preds = sum(w * r.test_preds for w, r in zip(weights, stack_models))
    oof_blend = sum(w * r.oof[oof_mask] for w, r in zip(weights, stack_models))

    final_preds = np.asarray(final_preds, dtype=float)
    oof_blend = np.asarray(oof_blend, dtype=float)

    ens_rmse = rmse(y_meta, oof_blend)
    ens_mae  = mean_absolute_error(y_meta, oof_blend)
    ens_r2   = r2_score(y_meta, oof_blend)

    method = "nnls_bounded_stack"

    print(f"Ensemble OOF RMSE: {ens_rmse:.5f}")
    print(f"Ensemble OOF MAE : {ens_mae:.5f}")
    print(f"Ensemble OOF R²  : {ens_r2:.5f}")
    print(f"Competition score : {max(0, 100 * ens_r2):.2f} / 100")

    # ── Baseline floor guard ─────────────────────────────────────
    baseline_r2: float | None = None
    floor_applied = False
    if baseline_result is not None:
        baseline_oof = np.asarray(baseline_result.oof[oof_mask], dtype=float)
        baseline_r2 = r2_score(y_meta, baseline_oof)
        print(f"  {'Baseline':<10s} OOF R²: {baseline_r2:.5f}")

        if ens_r2 < baseline_r2:
            print("\n⚠ Stack OOF R² is below the baseline — applying baseline floor.")
            final_preds = np.asarray(baseline_result.test_preds, dtype=float)
            oof_blend = baseline_oof
            ens_rmse = rmse(y_meta, oof_blend)
            ens_mae  = mean_absolute_error(y_meta, oof_blend)
            ens_r2   = r2_score(y_meta, oof_blend)
            method = "baseline_floor"
            floor_applied = True
            print(f"  Floored Ensemble OOF R²: {ens_r2:.5f}")
        else:
            print("✓ Stack meets or beats the baseline floor.")

    info = dict(
        method=method,
        weights=weights,
        oof_blend=oof_blend,
        oof_mask=oof_mask,
        ensemble_rmse=ens_rmse,
        ensemble_mae=ens_mae,
        ensemble_r2=ens_r2,
        ridge=meta_learner,
        baseline_r2=baseline_r2,
        floor_applied=floor_applied,
    )
    return final_preds, info
