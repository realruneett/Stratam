"""
Honest reporting for the demand-prediction-overhaul (Task 11.1).

This module is reworked from the OLD pipeline, where ``run_diagnostics`` rebuilt
a validation R² from ``val_mask`` + ``train_feats`` index alignment + the
stacking ``oof``. That recomputation mirrored the removed ``chronological_split``
validator and is gone.

The NEW pipeline scores everything on the **day-48 daytime surrogate holdout**
(``validation.build_day48_daytime_holdout``) — the primary model-selection
metric that matches the test time-of-day distribution (Requirement 2). Holdout
R² is therefore computed directly from explicit predictions on that surrogate
holdout's eval partition rather than reconstructed from OOF masks.

Reporting contract (Requirements 2.6, 2.7, 6.1)
-----------------------------------------------
``run_diagnostics`` prints and returns a :class:`RunMetrics`-shaped dict with:

  * ``baseline_holdout_r2``         – robust baseline R² on the surrogate holdout
  * ``model_holdout_r2``            – ``{LightGBM, XGBoost, CatBoost}`` R² each
  * ``ensemble_holdout_r2``         – stacked-ensemble R² on the surrogate holdout
  * ``local_score``                 – ``max(0, 100 * ensemble_holdout_r2)`` (2.6)
  * ``recorded_online_score``       – ``config.RECORDED_ONLINE_SCORE`` (83.13)
  * ``cv_lb_gap``                   – ``|local_score − recorded_online_score|`` (2.7)
  * ``leaderboard_top``             – ``config.LEADERBOARD_TOP`` (93.13)
  * ``real_task_validation_status`` – the ``build_real_task_holdout`` status
  * ``transform_selected``          – ``"identity"`` | ``"log1p"``

Feature importance (LightGBM ``feature_importances_`` → CSV + PNG) and the
optional SHAP summary are retained, plus per-model timing and peak GPU VRAM.

Design contract for Task 12 (the orchestrator)
----------------------------------------------
``run_diagnostics`` is driven by **explicit holdout predictions** rather than by
``val_mask``. Task 12 computes, on the surrogate holdout's eval partition
(``X_holdout`` = the eval-partition feature matrix, ``holdout_eval_y`` = its true
target):

    holdout_model_preds = {r.name: r.final_model.predict(X_holdout)
                           for r in results}                      # GBM models
    baseline_holdout_preds = baseline_result.final_model.predict(X_holdout)
    ensemble_holdout_preds = <stack/floor applied to those holdout preds>

and passes them in. R² for each is computed here. As a convenience for callers
that would rather not pre-compute predictions, when an explicit prediction
argument is omitted but ``X_holdout`` and the corresponding fitted model are
supplied, ``run_diagnostics`` will predict inside (see ``_holdout_preds``). The
explicit-predictions path is the recommended, cleanest wiring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

from config import RECORDED_ONLINE_SCORE, LEADERBOARD_TOP
from src.models import ModelResult, rmse


def _r2_or_none(y_true, y_pred) -> float | None:
    """R² of ``y_pred`` against ``y_true``, or ``None`` when not computable.

    R² is undefined for fewer than two points, length-mismatched vectors, or
    non-finite inputs, so those cases return ``None`` rather than raising.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted values, aligned to ``y_true``.

    Returns:
        The R² as a float, or ``None`` when it cannot be computed.
    """
    if y_true is None or y_pred is None:
        return None
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.size != yp.size or yt.size < 2:
        return None
    if not (np.all(np.isfinite(yt)) and np.all(np.isfinite(yp))):
        return None
    return float(r2_score(yt, yp))


def _holdout_preds(
    explicit_preds,
    model,
    X_holdout: pd.DataFrame | None,
) -> np.ndarray | None:
    """Resolve holdout predictions, predicting inside only as a fallback.

    Prefers the caller-supplied ``explicit_preds``. When those are absent but a
    fitted ``model`` and the holdout feature matrix ``X_holdout`` are available,
    predicts inside. Returns ``None`` when neither path is possible.

    Args:
        explicit_preds: Predictions already computed by the caller, or ``None``.
        model: A fitted model exposing ``.predict`` (e.g. ``final_model``), or
            ``None``.
        X_holdout: The surrogate-holdout eval feature matrix, or ``None``.

    Returns:
        A float ``np.ndarray`` of predictions, or ``None``.
    """
    if explicit_preds is not None:
        return np.asarray(explicit_preds, dtype=float).ravel()
    if model is not None and X_holdout is not None:
        return np.asarray(model.predict(X_holdout), dtype=float).ravel()
    return None


def _write_feature_importance(
    lgb_result: ModelResult,
    feature_cols: list[str],
    importance_csv: str,
    importance_png: str,
) -> None:
    """Write the LightGBM feature-importance table (CSV) and bar chart (PNG).

    Args:
        lgb_result: LightGBM ModelResult whose ``final_model`` exposes
            ``feature_importances_``.
        feature_cols: Ordered feature column names aligned to the importances.
        importance_csv: Output CSV path.
        importance_png: Output PNG path.

    Returns:
        None
    """
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": lgb_result.final_model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    imp_df.to_csv(importance_csv, index=False)
    print(f"✓ {importance_csv} saved")
    print(f"\nTop 30 features:\n{imp_df.head(30).to_string()}")

    fig, ax = plt.subplots(figsize=(10, 8))
    top30 = imp_df.head(30)
    ax.barh(top30["feature"][::-1], top30["importance"][::-1], color="#2196F3")
    ax.set_xlabel("Gain Importance")
    ax.set_title("Top 30 Feature Importances (LightGBM)")
    plt.tight_layout()
    plt.savefig(importance_png, dpi=150)
    plt.close()
    print(f"✓ {importance_png} saved")


def _write_shap_summary(
    lgb_result: ModelResult,
    X_holdout: pd.DataFrame | None,
    shap_png: str,
) -> None:
    """Write the optional SHAP summary plot on the surrogate-holdout features.

    Skipped silently (with a printed note) when ``shap`` is not installed,
    ``X_holdout`` is unavailable, or SHAP raises.

    Args:
        lgb_result: LightGBM ModelResult whose ``final_model`` is explained.
        X_holdout: Surrogate-holdout eval feature matrix to explain, or ``None``.
        shap_png: Output SHAP PNG path.

    Returns:
        None
    """
    if X_holdout is None or len(X_holdout) == 0:
        print("SHAP skipped — no holdout features provided.")
        return
    try:
        import shap
        sample_n = min(500, len(X_holdout))
        idx = np.random.choice(len(X_holdout), sample_n, replace=False)
        X_shap = X_holdout.iloc[idx]
        explainer = shap.TreeExplainer(lgb_result.final_model)
        shap_values = explainer.shap_values(X_shap)
        shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(shap_png, dpi=150)
        plt.close()
        print(f"✓ {shap_png} saved")
    except ImportError:
        print("SHAP skipped — shap package not installed.")
    except Exception as e:  # noqa: BLE001 — SHAP is a best-effort diagnostic
        print(f"SHAP skipped — error: {e}")


def _print_timing(results: list[ModelResult]) -> None:
    """Print per-model wall-clock training time and peak GPU VRAM if available.

    Args:
        results: GBM ModelResult objects carrying ``elapsed`` seconds.

    Returns:
        None
    """
    print(f"\n{'Model':<12} {'Time (s)':>10}")
    print("─" * 22)
    for r in results:
        print(f"  {r.name:<10s} {r.elapsed:>10.1f}")

    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.max_memory_reserved(0) / 1e9
            print(f"\nPeak GPU VRAM: {vram:.2f} GB")
    except ImportError:
        pass


def run_diagnostics(
    lgb_result: ModelResult,
    results: list[ModelResult],
    baseline_result: ModelResult | None,
    feature_cols: list[str],
    holdout_eval_y,
    holdout_model_preds: dict[str, np.ndarray],
    ensemble_holdout_preds,
    real_task_status: str,
    transform_selected: str,
    importance_csv: str,
    importance_png: str,
    shap_png: str,
    X_holdout: pd.DataFrame | None = None,
    baseline_holdout_preds=None,
    recorded_online_score: float | None = RECORDED_ONLINE_SCORE,
    leaderboard_top: float = LEADERBOARD_TOP,
) -> dict:
    """Report honest surrogate-holdout metrics and write diagnostic artifacts.

    Computes R² directly from explicit predictions on the day-48 daytime
    surrogate holdout's eval partition (Requirement 2) — the OLD ``val_mask`` /
    OOF reconstruction is removed. Writes the LightGBM feature-importance
    CSV/PNG and an optional SHAP summary, prints per-model timing, and returns a
    :class:`RunMetrics`-shaped dict for ``run.py`` to merge and write
    (Requirements 2.6, 2.7, 6.1).

    Args:
        lgb_result: LightGBM ModelResult (source of feature importance + SHAP).
        results: All GBM ModelResult objects (used for timing prints).
        baseline_result: Robust-baseline ModelResult, or ``None``. Used to
            predict baseline holdout preds when ``baseline_holdout_preds`` is not
            supplied and ``X_holdout`` is available.
        feature_cols: Ordered feature column names aligned to the importances.
        holdout_eval_y: True target of the surrogate-holdout eval partition.
        holdout_model_preds: Mapping of model name → predictions on the
            surrogate-holdout eval partition (typically the three GBM models).
        ensemble_holdout_preds: Stacked-ensemble predictions on the
            surrogate-holdout eval partition.
        real_task_status: Status string from ``build_real_task_holdout``
            (``"OK"`` | ``"FAILED_NO_MATCHING_SLOTS"``).
        transform_selected: Selected target transform (``"identity"`` |
            ``"log1p"``).
        importance_csv: Output feature-importance CSV path.
        importance_png: Output feature-importance PNG path.
        shap_png: Output SHAP summary PNG path.
        X_holdout: Surrogate-holdout eval feature matrix. Used for SHAP and, as a
            fallback, to predict holdout preds inside when explicit ones are
            omitted.
        baseline_holdout_preds: Explicit baseline predictions on the surrogate
            holdout. When ``None``, falls back to predicting
            ``baseline_result.final_model`` on ``X_holdout``.
        recorded_online_score: Recorded leaderboard score for the CV_LB_Gap
            (``config.RECORDED_ONLINE_SCORE`` = 83.13). ``None`` yields a
            ``None`` gap.
        leaderboard_top: Leaderboard top score (``config.LEADERBOARD_TOP`` =
            93.13).

    Returns:
        A metrics dict (RunMetrics) with keys ``baseline_holdout_r2``,
        ``model_holdout_r2``, ``ensemble_holdout_r2``, ``local_score``,
        ``recorded_online_score``, ``cv_lb_gap``, ``leaderboard_top``,
        ``real_task_validation_status``, and ``transform_selected``.
    """
    # ── Artifacts: feature importance + optional SHAP ────────────
    _write_feature_importance(
        lgb_result, feature_cols, importance_csv, importance_png
    )
    _write_shap_summary(lgb_result, X_holdout, shap_png)

    # ── Timing / VRAM ────────────────────────────────────────────
    _print_timing(results)

    # ── Holdout R² from explicit surrogate-holdout predictions ───
    y_eval = np.asarray(holdout_eval_y, dtype=float).ravel()

    model_holdout_r2: dict[str, float | None] = {}
    for name, preds in (holdout_model_preds or {}).items():
        model_holdout_r2[name] = _r2_or_none(y_eval, preds)

    baseline_preds = _holdout_preds(
        baseline_holdout_preds,
        getattr(baseline_result, "final_model", None),
        X_holdout,
    )
    baseline_holdout_r2 = _r2_or_none(y_eval, baseline_preds)

    ensemble_holdout_r2 = _r2_or_none(y_eval, ensemble_holdout_preds)

    # ── Local score + CV_LB_Gap (Req 2.6, 2.7) ───────────────────
    ens_r2 = ensemble_holdout_r2 if ensemble_holdout_r2 is not None else 0.0
    local_score = max(0.0, 100.0 * ens_r2)
    cv_lb_gap = (
        abs(local_score - recorded_online_score)
        if recorded_online_score is not None
        else None
    )

    metrics = {
        "transform_selected": transform_selected,
        "baseline_holdout_r2": baseline_holdout_r2,
        "model_holdout_r2": model_holdout_r2,
        "ensemble_holdout_r2": ensemble_holdout_r2,
        "local_score": local_score,
        "recorded_online_score": recorded_online_score,
        "cv_lb_gap": cv_lb_gap,
        "leaderboard_top": leaderboard_top,
        "real_task_validation_status": real_task_status,
    }

    # ── Honest report (Req 6.1) ──────────────────────────────────
    _print_report(metrics)
    return metrics


def _fmt(value) -> str:
    """Format a metric value for the report (``n/a`` for ``None``).

    Args:
        value: A float metric value or ``None``.

    Returns:
        A fixed-precision string, or ``"n/a"`` when ``value`` is ``None``.
    """
    return "n/a" if value is None else f"{value:.5f}"


def _print_report(metrics: dict) -> None:
    """Print the honest surrogate-holdout scoring report.

    Args:
        metrics: The RunMetrics dict assembled by :func:`run_diagnostics`.

    Returns:
        None
    """
    print("\n" + "=" * 60)
    print("HONEST REPORTING — day-48 daytime surrogate holdout")
    print("=" * 60)
    print(f"Transform selected          : {metrics['transform_selected']}")
    print(f"Real-task validation status : "
          f"{metrics['real_task_validation_status']}")
    print(f"Baseline Holdout R²         : {_fmt(metrics['baseline_holdout_r2'])}")
    for name, r2 in metrics["model_holdout_r2"].items():
        print(f"  {name:<10s} Holdout R²     : {_fmt(r2)}")
    print(f"Ensemble Holdout R²         : {_fmt(metrics['ensemble_holdout_r2'])}")
    print(f"Local score (max 0,100·r²)  : {metrics['local_score']:.2f} / 100")
    online = metrics["recorded_online_score"]
    print(f"Recorded online score       : "
          f"{'n/a' if online is None else f'{online:.2f}'}")
    gap = metrics["cv_lb_gap"]
    print(f"CV_LB_Gap (primary metric)  : "
          f"{'n/a' if gap is None else f'{gap:.2f}'}")
    print(f"Leaderboard top (target)    : {metrics['leaderboard_top']:.2f}")
