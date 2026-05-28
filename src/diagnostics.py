"""
Feature importance plots, SHAP analysis, timing, and MAPE
diagnostic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models import ModelResult, rmse


def run_diagnostics(
    lgb_result: ModelResult,
    results: list[ModelResult],
    feature_cols: list[str],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    train_feats: pd.DataFrame,
    val_mask: np.ndarray,
    stack_info: dict,
    importance_csv: str,
    importance_png: str,
    shap_png: str,
) -> None:
    """
    Produce feature importance table/chart, optional SHAP summary,
    per-model timing, GPU VRAM, and validation MAPE.

    Args:
        lgb_result: LightGBM ModelResult (for feature importance).
        results: All ModelResult objects.
        feature_cols: Feature column names.
        X_val: Holdout validation features.
        y_val: Holdout validation target.
        train_feats: Full feature-engineered train DataFrame.
        val_mask: Boolean mask for validation rows.
        stack_info: Dict returned by stack_predictions().
        importance_csv: Output CSV path.
        importance_png: Output PNG path.
        shap_png: Output SHAP PNG path.

    Returns:
        None
    """
    # ── Feature importance ──────────────────────────────────────
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": lgb_result.final_model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    imp_df.to_csv(importance_csv, index=False)
    print(f"✓ {importance_csv} saved")
    print(f"\nTop 30 features:\n{imp_df.head(30).to_string()}")

    fig, ax = plt.subplots(figsize=(10, 8))
    top30 = imp_df.head(30)
    ax.barh(top30["feature"][::-1], top30["importance"][::-1],
            color="#2196F3")
    ax.set_xlabel("Gain Importance")
    ax.set_title("Top 30 Feature Importances (LightGBM)")
    plt.tight_layout()
    plt.savefig(importance_png, dpi=150)
    plt.close()
    print(f"✓ {importance_png} saved")

    # ── SHAP (optional) ────────────────────────────────────────
    try:
        import shap
        sample_n = min(500, len(X_val))
        idx = np.random.choice(len(X_val), sample_n, replace=False)
        X_shap = X_val.iloc[idx]
        explainer = shap.TreeExplainer(lgb_result.final_model)
        shap_values = explainer.shap_values(X_shap)
        shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(shap_png, dpi=150)
        plt.close()
        print(f"✓ {shap_png} saved")
    except ImportError:
        print("SHAP skipped — shap package not installed.")
    except Exception as e:
        print(f"SHAP skipped — error: {e}")

    # ── Timing ─────────────────────────────────────────────────
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

    # ── MAPE on validation ─────────────────────────────────────
    oof_mask = stack_info["oof_mask"]
    val_indices = train_feats.index[val_mask]
    val_has_oof = np.isin(val_indices.values, np.where(oof_mask)[0])

    if val_has_oof.sum() > 0:
        val_oof_idx = val_indices[val_has_oof]
        y_val_oof = train_feats.loc[val_oof_idx, y_val.name].values

        method = stack_info["method"]
        weights = stack_info["weights"]
        ridge = stack_info["ridge"]

        if method == "weighted_avg":
            blend = sum(
                w * r.oof[val_oof_idx]
                for w, r in zip(weights, results)
            )
        else:
            X_vm = np.column_stack([r.oof[val_oof_idx] for r in results])
            blend = ridge.predict(X_vm)

        eps = 1e-5
        mape = np.mean(
            np.abs(y_val_oof - blend) / np.maximum(np.abs(y_val_oof), eps)
        ) * 100
        print(f"\nValidation MAPE: {mape:.2f}%")
    else:
        print("\nNo OOF predictions for validation set.")
