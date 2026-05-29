"""
Ridge stacking meta-learner over base-model OOF predictions, with a
baseline floor (Requirement 4.5, design "src/ensemble.py — stacking with a
floor").

The meta-learner is fit on **out-of-fold predictions only**. With the leak-free
``KFold`` OOF scheme in ``src.models`` every training row now receives an OOF
prediction, so the old ``oof != 0`` heuristic (a workaround for the removed
``time_kfold_split`` which skipped fold 0) is no longer needed — all training
rows feed the meta-learner.

When a ``baseline_result`` is supplied, the robust baseline (geohash mean ×
time-of-day curve blend) participates in two ways:

1. **In the stack** — its OOF column is added as an extra meta-feature so Ridge
   can lean on it.
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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from src.models import ModelResult, rmse


def stack_predictions(
    results: list[ModelResult],
    y_train: np.ndarray,
    baseline_result: ModelResult | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Blend base-model predictions via Ridge regression on OOF values, with an
    optional baseline floor.

    The Ridge meta-learner is fit on out-of-fold predictions only. If any Ridge
    coefficient is negative the function falls back to clipped non-negative
    weighted averaging. When ``baseline_result`` is provided, the baseline is
    added as an extra stacking input and a floor guard ensures the final blend
    cannot score worse (OOF R²) than the baseline alone.

    Args:
        results: List of GBM ModelResult objects (one per base model). Each
            carries a full-length leak-free ``oof`` vector and ``test_preds``.
        y_train: Full training target array (same length as each ``oof``).
        baseline_result: Optional robust-baseline ModelResult. When supplied its
            ``oof`` is added as a meta-feature column and used as a floor. When
            ``None`` the function blends the GBM models only (legacy behaviour).

    Returns:
        (final_test_preds, info_dict)

        info_dict keys:
            method          – "ridge", "weighted_avg", or "baseline_floor"
            weights         – per-model weights / coefficients (stack order:
                              GBM models, then baseline if included)
            oof_blend       – blended OOF predictions of the chosen blend
            oof_mask        – boolean mask of rows that received OOF (now all
                              rows that have finite OOF across stacked models)
            ensemble_rmse   – OOF RMSE of the FINAL chosen blend
            ensemble_mae    – OOF MAE of the FINAL chosen blend
            ensemble_r2     – OOF R² of the FINAL chosen blend
            ridge           – the fitted Ridge meta-learner
            baseline_r2     – baseline OOF R² (None when no baseline supplied)
            floor_applied   – True when the baseline floor overrode the stack
    """
    n = len(y_train)
    y_train = np.asarray(y_train, dtype=float)

    # ── Rows usable by the meta-learner ──────────────────────────
    # Every training row now receives an OOF prediction (KFold), so the legacy
    # `oof != 0` heuristic is dropped. Use all rows whose OOF is finite across
    # every stacked model (in practice all rows). The key is retained for
    # diagnostics / backward compatibility.
    oof_mask = np.ones(n, dtype=bool)
    for r in results:
        oof_mask &= np.isfinite(r.oof)
    if baseline_result is not None:
        oof_mask &= np.isfinite(baseline_result.oof)

    # ── Models feeding the Ridge meta-learner ────────────────────
    # GBMs first; the baseline (when present) is appended as an extra column so
    # the stack can lean on the robust floor (design 4.5 "baseline in the stack").
    stack_models = list(results)
    if baseline_result is not None:
        stack_models = list(results) + [baseline_result]

    X_meta = np.column_stack([r.oof[oof_mask] for r in stack_models])
    y_meta = y_train[oof_mask]

    # ── Fit Ridge on OOF predictions only ────────────────────────
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(X_meta, y_meta)

    print(f"Ridge coefficients: {ridge.coef_}")
    print(f"Ridge intercept   : {ridge.intercept_:.6f}")
    if baseline_result is not None:
        print("  (last coefficient corresponds to the baseline meta-feature)")

    # ── Non-negative-weight fallback (retained) ──────────────────
    use_weighted = False
    weights = ridge.coef_

    if np.any(ridge.coef_ < 0):
        print("\n⚠ WARNING: Negative Ridge weight detected — "
              "using clipped non-negative weights instead.")
        weights = np.maximum(ridge.coef_, 0)
        weights = (weights / weights.sum() if weights.sum() > 0
                   else np.full(len(stack_models), 1 / len(stack_models)))
        use_weighted = True
        print(f"  Adjusted weights: {weights}")
    else:
        print("✓ All Ridge coefficients non-negative.")

    # ── Build the stacked test predictions + OOF blend ───────────
    X_test_meta = np.column_stack([r.test_preds for r in stack_models])
    if use_weighted:
        final_preds = sum(w * r.test_preds for w, r in zip(weights, stack_models))
        oof_blend = sum(w * r.oof[oof_mask] for w, r in zip(weights, stack_models))
    else:
        final_preds = ridge.predict(X_test_meta)
        oof_blend = ridge.predict(X_meta)

    final_preds = np.asarray(final_preds, dtype=float)
    oof_blend = np.asarray(oof_blend, dtype=float)

    ens_rmse = rmse(y_meta, oof_blend)
    ens_mae  = mean_absolute_error(y_meta, oof_blend)
    ens_r2   = r2_score(y_meta, oof_blend)

    method = "weighted_avg" if use_weighted else "ridge"

    print(f"\nEnsemble OOF RMSE: {ens_rmse:.5f}")
    print(f"Ensemble OOF MAE : {ens_mae:.5f}")
    print(f"Ensemble OOF R²  : {ens_r2:.5f}")
    print(f"Competition score : {max(0, 100 * ens_r2):.2f} / 100")
    for r in results:
        r2_i = r2_score(y_meta, r.oof[oof_mask])
        print(f"  {r.name:<10s} OOF RMSE: {rmse(y_meta, r.oof[oof_mask]):.5f}  "
              f"MAE: {mean_absolute_error(y_meta, r.oof[oof_mask]):.5f}  "
              f"R²: {r2_i:.5f}")

    # ── Baseline floor guard ─────────────────────────────────────
    # The robust baseline gives an ensemble floor: if the stacked blend scores
    # worse (OOF R²) than the baseline alone on the surrogate holdout, fall back
    # to the baseline's predictions so the ensemble can never underperform it.
    baseline_r2: float | None = None
    floor_applied = False
    if baseline_result is not None:
        baseline_oof = np.asarray(baseline_result.oof[oof_mask], dtype=float)
        baseline_r2 = r2_score(y_meta, baseline_oof)
        print(f"  {'Baseline':<10s} OOF R²: {baseline_r2:.5f}")

        if ens_r2 < baseline_r2:
            print("\n⚠ Stack OOF R² is below the baseline — applying baseline "
                  "floor (final predictions = baseline).")
            final_preds = np.asarray(baseline_result.test_preds, dtype=float)
            oof_blend = baseline_oof
            ens_rmse = rmse(y_meta, oof_blend)
            ens_mae  = mean_absolute_error(y_meta, oof_blend)
            ens_r2   = r2_score(y_meta, oof_blend)  # == baseline_r2
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
        ridge=ridge,
        baseline_r2=baseline_r2,
        floor_applied=floor_applied,
    )
    return final_preds, info
