"""
Ridge stacking meta-learner over base-model OOF predictions.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from src.models import ModelResult, rmse


def stack_predictions(
    results: list[ModelResult],
    y_train: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Blend base-model predictions via Ridge regression on OOF values.

    If any Ridge coefficient is negative the function falls back to
    clipped non-negative weighted averaging.

    Args:
        results: List of ModelResult objects (one per base model).
        y_train: Full training target array (same length as OOF).

    Returns:
        (final_test_preds, info_dict)

        info_dict keys:
            method          – "ridge" or "weighted_avg"
            weights         – per-model weights / coefficients
            oof_blend       – blended OOF predictions (masked)
            oof_mask        – boolean mask of rows that received OOF
            ensemble_rmse   – OOF RMSE of the blend
            ensemble_mae    – OOF MAE of the blend
    """
    n = len(y_train)

    # Identify rows that received OOF predictions (fold 0 is skipped)
    oof_mask = np.zeros(n, dtype=bool)
    for r in results:
        oof_mask |= (r.oof != 0)  # rows that got predictions

    X_meta = np.column_stack([r.oof[oof_mask] for r in results])
    y_meta = y_train[oof_mask]

    # Fit Ridge
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(X_meta, y_meta)

    print(f"Ridge coefficients: {ridge.coef_}")
    print(f"Ridge intercept   : {ridge.intercept_:.6f}")

    use_weighted = False
    weights = ridge.coef_

    if np.any(ridge.coef_ < 0):
        print("\n⚠ WARNING: Negative Ridge weight detected — "
              "using clipped non-negative weights instead.")
        weights = np.maximum(ridge.coef_, 0)
        weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(results), 1/len(results))
        use_weighted = True
        print(f"  Adjusted weights: {weights}")
    else:
        print("✓ All Ridge coefficients non-negative.")

    # Build final test predictions
    X_test_meta = np.column_stack([r.test_preds for r in results])
    if use_weighted:
        final_preds = sum(w * r.test_preds for w, r in zip(weights, results))
        oof_blend = sum(w * r.oof[oof_mask] for w, r in zip(weights, results))
    else:
        final_preds = ridge.predict(X_test_meta)
        oof_blend = ridge.predict(X_meta)

    ens_rmse = rmse(y_meta, oof_blend)
    ens_mae  = mean_absolute_error(y_meta, oof_blend)

    print(f"\nEnsemble OOF RMSE: {ens_rmse:.5f}")
    print(f"Ensemble OOF MAE : {ens_mae:.5f}")
    for r in results:
        print(f"  {r.name:<10s} OOF RMSE: {rmse(y_meta, r.oof[oof_mask]):.5f}  "
              f"MAE: {mean_absolute_error(y_meta, r.oof[oof_mask]):.5f}")

    info = dict(
        method="weighted_avg" if use_weighted else "ridge",
        weights=weights,
        oof_blend=oof_blend,
        oof_mask=oof_mask,
        ensemble_rmse=ens_rmse,
        ensemble_mae=ens_mae,
        ridge=ridge,
    )
    return final_preds, info
