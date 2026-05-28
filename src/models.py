"""
Model training: LightGBM, XGBoost, CatBoost.

Each model has:
  • GPU → CPU fallback chain
  • Pass A: OOF generation via time-based K-fold
  • Pass B: Final model on 100 % training data
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost

from src.validation import time_kfold_split

from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error.

    Args:
        y_true: Ground truth.
        y_pred: Predictions.

    Returns:
        RMSE value.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ─── Result Container ──────────────────────────────────────────


@dataclass
class ModelResult:
    """Container for one model's training outputs.

    Attributes:
        name: Model name (e.g. "LightGBM").
        oof: OOF predictions array (full training-set length).
        test_preds: Test-set predictions.
        fold_rmses: Per-fold RMSE values.
        best_iters: Per-fold best iteration counts.
        elapsed: Wall-clock training time in seconds.
        final_model: The fitted final model object.
    """
    name: str
    oof: np.ndarray
    test_preds: np.ndarray
    fold_rmses: list[float] = field(default_factory=list)
    best_iters: list[int]   = field(default_factory=list)
    elapsed: float = 0.0
    final_model: object = None


# ─── LightGBM ──────────────────────────────────────────────────


def _fit_lgb(params, X_tr, y_tr, X_va, y_va, early_stop):
    """Fit LGB with fallback chain: cuda → gpu → cpu.

    Args:
        params: LightGBM parameter dict.
        X_tr: Training features.
        y_tr: Training target.
        X_va: Validation features.
        y_va: Validation target.
        early_stop: Early-stopping rounds.

    Returns:
        Fitted LGBMRegressor.

    Raises:
        RuntimeError: If all devices fail.
    """
    chain = [params["device"]]
    if params["device"] == "cuda":
        chain += ["gpu", "cpu"]
    elif params["device"] == "gpu":
        chain += ["cpu"]

    for dev in chain:
        try:
            p = {**params, "device": dev}
            m = lgb.LGBMRegressor(**p)
            m.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[
                    lgb.early_stopping(early_stop),
                    lgb.log_evaluation(500),
                ],
            )
            if dev != params["device"]:
                print(f"  ⚠ LightGBM fell back to device='{dev}'")
            return m
        except Exception as e:
            print(f"  LightGBM device='{dev}' failed: {e}")
    raise RuntimeError("LightGBM: all device options failed.")


def train_lightgbm(
    train_feats, feature_cols, target_col, timestamp_col,
    X_full, y_full, X_val, y_val, X_test,
    n_folds, n_estimators, early_stop, seed, lgb_device,
) -> ModelResult:
    """
    Full LightGBM training: OOF (Pass A) + final model (Pass B).

    Args:
        train_feats: Feature-engineered training DataFrame.
        feature_cols: Feature column names.
        target_col: Target column name.
        timestamp_col: Timestamp column name.
        X_full: Full training features.
        y_full: Full training target.
        X_val: Holdout validation features.
        y_val: Holdout validation target.
        X_test: Test features.
        n_folds: Number of CV folds.
        n_estimators: Max boosting rounds.
        early_stop: Early-stopping patience.
        seed: Random seed.
        lgb_device: LightGBM device string.

    Returns:
        ModelResult with OOF, test preds, and final model.
    """
    print("=" * 60)
    print("MODEL 1: LightGBM")
    print("=" * 60)
    t0 = time.perf_counter()

    params = dict(
        device=lgb_device, objective="regression_l1",
        metric=["rmse", "mae"], n_estimators=n_estimators,
        learning_rate=0.02, num_leaves=255, min_child_samples=20,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        reg_alpha=0.1, reg_lambda=0.1, verbose=-1, seed=seed,
    )

    res = ModelResult(name="LightGBM", oof=np.zeros(len(train_feats)),
                      test_preds=np.zeros(len(X_test)))

    # Pass A — OOF
    for fold_i, (tr_idx, va_idx) in enumerate(
        time_kfold_split(train_feats, timestamp_col, n_folds)
    ):
        print(f"\n  LGB Fold {fold_i+1}: "
              f"train={len(tr_idx)}, val={len(va_idx)}")
        m = _fit_lgb(
            params,
            train_feats.loc[tr_idx, feature_cols],
            train_feats.loc[tr_idx, target_col],
            train_feats.loc[va_idx, feature_cols],
            train_feats.loc[va_idx, target_col],
            early_stop,
        )
        res.oof[va_idx] = m.predict(train_feats.loc[va_idx, feature_cols])
        res.fold_rmses.append(
            rmse(train_feats.loc[va_idx, target_col], res.oof[va_idx]))
        res.best_iters.append(m.best_iteration_)
        print(f"    RMSE: {res.fold_rmses[-1]:.5f} | "
              f"Best iter: {m.best_iteration_}")

    mean_iter = int(np.mean(res.best_iters))
    print(f"\n  LGB OOF mean RMSE : {np.mean(res.fold_rmses):.5f}")
    print(f"  LGB mean best_iter: {mean_iter}")

    # Pass B — Final
    print("\n  Training final LGB model on full data...")
    final_p = {**params, "n_estimators": min(int(mean_iter * 1.1), 10_000)}
    res.final_model = _fit_lgb(final_p, X_full, y_full, X_val, y_val, early_stop)
    res.test_preds = res.final_model.predict(X_test)

    res.elapsed = time.perf_counter() - t0
    print(f"  LGB total time: {res.elapsed:.1f}s")
    return res


# ─── XGBoost ────────────────────────────────────────────────────


def _fit_xgb(params, X_tr, y_tr, X_va, y_va, early_stop):
    """Fit XGB with fallback: cuda → cpu.

    Args:
        params: XGBoost parameter dict.
        X_tr: Training features.
        y_tr: Training target.
        X_va: Validation features.
        y_va: Validation target.
        early_stop: Early-stopping rounds.

    Returns:
        Fitted XGBRegressor.

    Raises:
        RuntimeError: If all devices fail.
    """
    for dev in [params["device"], "cpu"]:
        try:
            p = {**params, "device": dev}
            m = xgb.XGBRegressor(**p)
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  early_stopping_rounds=early_stop, verbose=500)
            if dev != params["device"]:
                print(f"  ⚠ XGBoost fell back to device='{dev}'")
            return m
        except Exception as e:
            print(f"  XGBoost device='{dev}' failed: {e}")
    raise RuntimeError("XGBoost: all device options failed.")


def train_xgboost(
    train_feats, feature_cols, target_col, timestamp_col,
    X_full, y_full, X_val, y_val, X_test,
    n_folds, n_estimators, early_stop, seed,
) -> ModelResult:
    """
    Full XGBoost training: OOF (Pass A) + final model (Pass B).

    Args:
        train_feats: Feature-engineered training DataFrame.
        feature_cols: Feature column names.
        target_col: Target column name.
        timestamp_col: Timestamp column name.
        X_full: Full training features.
        y_full: Full training target.
        X_val: Holdout validation features.
        y_val: Holdout validation target.
        X_test: Test features.
        n_folds: Number of CV folds.
        n_estimators: Max boosting rounds.
        early_stop: Early-stopping patience.
        seed: Random seed.

    Returns:
        ModelResult with OOF, test preds, and final model.
    """
    print("=" * 60)
    print("MODEL 2: XGBoost")
    print("=" * 60)
    t0 = time.perf_counter()

    params = dict(
        tree_method="hist", device="cuda",
        objective="reg:absoluteerror", eval_metric=["rmse", "mae"],
        n_estimators=n_estimators, learning_rate=0.02, max_depth=8,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.05, reg_lambda=0.1, seed=seed,
    )

    res = ModelResult(name="XGBoost", oof=np.zeros(len(train_feats)),
                      test_preds=np.zeros(len(X_test)))

    for fold_i, (tr_idx, va_idx) in enumerate(
        time_kfold_split(train_feats, timestamp_col, n_folds)
    ):
        print(f"\n  XGB Fold {fold_i+1}: "
              f"train={len(tr_idx)}, val={len(va_idx)}")
        m = _fit_xgb(
            params,
            train_feats.loc[tr_idx, feature_cols],
            train_feats.loc[tr_idx, target_col],
            train_feats.loc[va_idx, feature_cols],
            train_feats.loc[va_idx, target_col],
            early_stop,
        )
        res.oof[va_idx] = m.predict(train_feats.loc[va_idx, feature_cols])
        res.fold_rmses.append(
            rmse(train_feats.loc[va_idx, target_col], res.oof[va_idx]))
        res.best_iters.append(m.best_iteration)
        print(f"    RMSE: {res.fold_rmses[-1]:.5f} | "
              f"Best iter: {m.best_iteration}")

    mean_iter = int(np.mean(res.best_iters))
    print(f"\n  XGB OOF mean RMSE : {np.mean(res.fold_rmses):.5f}")
    print(f"  XGB mean best_iter: {mean_iter}")

    print("\n  Training final XGB model on full data...")
    final_p = {**params, "n_estimators": min(int(mean_iter * 1.1), 10_000)}
    res.final_model = _fit_xgb(final_p, X_full, y_full, X_val, y_val, early_stop)
    res.test_preds = res.final_model.predict(X_test)

    res.elapsed = time.perf_counter() - t0
    print(f"  XGB total time: {res.elapsed:.1f}s")
    return res


# ─── CatBoost ──────────────────────────────────────────────────


def _fit_cat(params, X_tr, y_tr, X_va, y_va, cat_feats, early_stop):
    """Fit CatBoost with fallback: GPU → CPU.

    Args:
        params: CatBoost parameter dict.
        X_tr: Training features.
        y_tr: Training target.
        X_va: Validation features.
        y_va: Validation target.
        cat_feats: Categorical feature names.
        early_stop: Early-stopping rounds.

    Returns:
        Fitted CatBoostRegressor.

    Raises:
        RuntimeError: If all task types fail.
    """
    for task in [params["task_type"], "CPU"]:
        try:
            p = {**params, "task_type": task}
            if task == "CPU":
                p.pop("devices", None)
            m = catboost.CatBoostRegressor(**p)
            m.fit(X_tr, y_tr,
                  cat_features=cat_feats or None,
                  eval_set=(X_va, y_va),
                  early_stopping_rounds=early_stop,
                  use_best_model=True)
            if task != params["task_type"]:
                print(f"  ⚠ CatBoost fell back to task_type='{task}'")
            return m
        except Exception as e:
            print(f"  CatBoost task_type='{task}' failed: {e}")
    raise RuntimeError("CatBoost: all task_type options failed.")


def train_catboost(
    train_feats, feature_cols, target_col, timestamp_col,
    X_full, y_full, X_val, y_val, X_test,
    categorical_cols,
    n_folds, n_estimators, early_stop, seed,
) -> ModelResult:
    """
    Full CatBoost training: OOF (Pass A) + final model (Pass B).

    Args:
        train_feats: Feature-engineered training DataFrame.
        feature_cols: Feature column names.
        target_col: Target column name.
        timestamp_col: Timestamp column name.
        X_full: Full training features.
        y_full: Full training target.
        X_val: Holdout validation features.
        y_val: Holdout validation target.
        X_test: Test features.
        categorical_cols: Categorical column names from schema.
        n_folds: Number of CV folds.
        n_estimators: Max boosting iterations.
        early_stop: Early-stopping patience.
        seed: Random seed.

    Returns:
        ModelResult with OOF, test preds, and final model.
    """
    print("=" * 60)
    print("MODEL 3: CatBoost")
    print("=" * 60)
    t0 = time.perf_counter()

    cat_feats = [c for c in feature_cols if c in categorical_cols]
    print(f"  CatBoost cat_features: {cat_feats}")

    params = dict(
        task_type="GPU", devices="0", loss_function="MAE",
        eval_metric="RMSE", iterations=n_estimators,
        learning_rate=0.02, depth=8, l2_leaf_reg=3,
        random_seed=seed, verbose=500,
    )

    res = ModelResult(name="CatBoost", oof=np.zeros(len(train_feats)),
                      test_preds=np.zeros(len(X_test)))

    for fold_i, (tr_idx, va_idx) in enumerate(
        time_kfold_split(train_feats, timestamp_col, n_folds)
    ):
        print(f"\n  CAT Fold {fold_i+1}: "
              f"train={len(tr_idx)}, val={len(va_idx)}")
        m = _fit_cat(
            params,
            train_feats.loc[tr_idx, feature_cols],
            train_feats.loc[tr_idx, target_col],
            train_feats.loc[va_idx, feature_cols],
            train_feats.loc[va_idx, target_col],
            cat_feats, early_stop,
        )
        res.oof[va_idx] = m.predict(train_feats.loc[va_idx, feature_cols])
        res.fold_rmses.append(
            rmse(train_feats.loc[va_idx, target_col], res.oof[va_idx]))
        res.best_iters.append(m.get_best_iteration())
        print(f"    RMSE: {res.fold_rmses[-1]:.5f} | "
              f"Best iter: {m.get_best_iteration()}")

    mean_iter = int(np.mean(res.best_iters))
    print(f"\n  CAT OOF mean RMSE : {np.mean(res.fold_rmses):.5f}")
    print(f"  CAT mean best_iter: {mean_iter}")

    print("\n  Training final CatBoost model on full data...")
    final_p = {**params, "iterations": min(int(mean_iter * 1.1), 10_000)}
    res.final_model = _fit_cat(
        final_p, X_full, y_full, X_val, y_val, cat_feats, early_stop)
    res.test_preds = res.final_model.predict(X_test)

    res.elapsed = time.perf_counter() - t0
    print(f"  CAT total time: {res.elapsed:.1f}s")
    return res
