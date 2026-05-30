"""
Generalization-first model training: LightGBM, XGBoost, CatBoost.

This module is re-tuned for the demand-prediction-overhaul (Task 9.1). The old
training scheme (Pass A OOF over the removed ``time_kfold_split`` + Pass B final
on full data, with mildly-regularized hyperparameters) is replaced by a
leak-free, generalization-first regime (Requirements 4.1, 4.5, 4.6):

  * **GPU → CPU fallback** chains are retained for all three trainers.
  * **Stronger regularization** — lower tree depth / leaf counts, higher
    minimum-child and L2 penalties, retained sub-sampling — suited to the small
    (~21-feature) leak-free feature set so the models generalize to the day-49
    daytime window instead of overfitting the two-day snapshot.
  * **Leak-free out-of-fold predictions** — the OOF layer is generated with a
    ``TE_N_FOLDS``-fold ``KFold`` split over the training rows. The training
    frame it receives already carries leak-free out-of-fold geohash encodings
    (from :meth:`src.spatial.TargetEncoder.fit_oof`) and the day-48 time-of-day
    curve, so the OOF predictions are themselves leak-free (Req 4.1, 4.5).
  * **Early stopping on the day-48 daytime surrogate holdout** — every fit (each
    OOF fold model and the final model) early-stops against the surrogate
    holdout eval set ``(X_val, y_val)`` rather than a chronological tail
    (Req 4.6). The surrogate holdout is disjoint from the training rows, so this
    is leak-free.
  * **Only leak-free encoded features** are consumed — the caller (Task 12)
    selects ``feature_cols`` from the reduced feature set; there are no
    ``lag_*`` / ``rolling_*`` / ``ewm_*`` columns to leak (Req 4.1).

Trainer data contract (wired by Task 12)
-----------------------------------------
``train_feats`` is the surrogate holdout's *training* partition, already
feature-engineered with out-of-fold encodings. ``X_full`` / ``y_full`` are its
feature matrix / target (``train_feats[feature_cols]`` / ``train_feats[target_col]``
when not passed explicitly). ``X_val`` / ``y_val`` are the surrogate holdout's
*eval* partition (used only for early stopping). ``X_test`` is the test feature
matrix. The OOF loop runs ``KFold(n_splits=n_folds, shuffle=True,
random_state=seed)`` over the training rows; for each fold a model is trained on
the complement (early-stopped on the surrogate holdout) and used to predict the
held-out fold, so ``ModelResult.oof`` is a full-length leak-free OOF vector.

CatBoost note: ``build_features`` integer-codes the categorical columns before
they ever reach a trainer, so all features are numeric. To keep behavior
identical across the three models, CatBoost treats them as plain numeric
features (``cat_features=None``); ``categorical_cols`` is accepted only for API
symmetry with the Task 12 wiring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from config import SEED, TE_N_FOLDS
from src.spatial import (
    GEOHASH_MEAN_COL,
    TOD_CURVE_MEAN_COL,
    transform_tod_curve,
)


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

    The field set is backward-compatible with ``ensemble.stack_predictions`` and
    ``diagnostics.run_diagnostics`` (which consume ``name``, ``oof``,
    ``test_preds``, ``final_model``, ``elapsed``).

    Attributes:
        name: Model name (e.g. "LightGBM").
        oof: Leak-free out-of-fold predictions (length == number of training
            rows). Every training row receives an OOF prediction.
        test_preds: Test-set predictions from the final model.
        fold_rmses: Per-fold OOF RMSE values.
        best_iters: Per-fold best-iteration counts (early stopping).
        elapsed: Wall-clock training time in seconds.
        final_model: The fitted final model object (trained on the full training
            partition, early-stopped on the surrogate holdout).
        final_best_iter: Best iteration of the final model (early stopping).
    """
    name: str
    oof: np.ndarray
    test_preds: np.ndarray
    fold_rmses: list[float] = field(default_factory=list)
    best_iters: list[int]   = field(default_factory=list)
    elapsed: float = 0.0
    final_model: object = None
    final_best_iter: int | None = None


# ─── Shared OOF + final driver ─────────────────────────────────


def _as_feature_df(X, feature_cols: list[str]) -> pd.DataFrame:
    """Coerce ``X`` to a DataFrame restricted to ``feature_cols``.

    Keeps column names so tree models expose ``feature_importances_`` aligned to
    ``feature_cols`` and so positional ``.iloc`` indexing in the OOF loop works.

    Args:
        X: A DataFrame or array-like feature matrix.
        feature_cols: Ordered feature column names.

    Returns:
        A DataFrame with columns exactly ``feature_cols``.
    """
    if isinstance(X, pd.DataFrame):
        if all(c in X.columns for c in feature_cols):
            return X[feature_cols]
        return X
    return pd.DataFrame(np.asarray(X), columns=feature_cols)


def _kfold_positions(
    n: int, n_folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build deterministic KFold positional splits over ``n`` rows.

    Args:
        n: Number of training rows.
        n_folds: Requested fold count (clamped to ``n``; min 2).
        seed: KFold shuffle seed.

    Returns:
        List of ``(train_positions, eval_positions)`` integer-index tuples.
    """
    positions = np.arange(n)
    effective = max(2, min(n_folds, n))
    kf = KFold(n_splits=effective, shuffle=True, random_state=seed)
    return list(kf.split(positions))


def _run_training(
    name: str,
    fit_fn: Callable,
    best_iter_fn: Callable,
    X_full_df: pd.DataFrame,
    y_arr: np.ndarray,
    X_val_df: pd.DataFrame,
    y_val_arr: np.ndarray,
    X_test_df: pd.DataFrame,
    n_folds: int,
    seed: int,
    oof_split: Iterable[tuple[np.ndarray, np.ndarray]] | None,
) -> ModelResult:
    """Generate leak-free OOF predictions and fit the final model.

    The same ``fit_fn`` is used for every fold model and the final model; it
    always early-stops on the surrogate holdout ``(X_val_df, y_val_arr)``.

    Args:
        name: Model name for the result and logs.
        fit_fn: ``fit_fn(X_tr, y_tr, X_va, y_va) -> fitted_model``.
        best_iter_fn: ``best_iter_fn(model) -> int`` extracting the best
            iteration after early stopping.
        X_full_df: Training feature matrix (DataFrame, columns == feature_cols).
        y_arr: Training target as a float numpy array.
        X_val_df: Surrogate-holdout eval features (early-stopping set).
        y_val_arr: Surrogate-holdout eval target.
        X_test_df: Test feature matrix.
        n_folds: OOF fold count.
        seed: Random seed for the KFold split.
        oof_split: Optional explicit ``(train_pos, eval_pos)`` splits; when
            ``None`` a ``KFold(n_splits=n_folds, shuffle=True,
            random_state=seed)`` over the training rows is used.

    Returns:
        A populated :class:`ModelResult`.
    """
    print("=" * 60)
    print(f"MODEL: {name}")
    print("=" * 60)
    t0 = time.perf_counter()

    n = len(X_full_df)
    res = ModelResult(
        name=name,
        oof=np.zeros(n, dtype=float),
        test_preds=np.zeros(len(X_test_df), dtype=float),
    )

    folds = list(oof_split) if oof_split is not None else _kfold_positions(
        n, n_folds, seed
    )

    # ── Leak-free OOF (early-stopped on the surrogate holdout) ───
    for fold_i, (tr_pos, va_pos) in enumerate(folds):
        tr_pos = np.asarray(tr_pos, dtype=int)
        va_pos = np.asarray(va_pos, dtype=int)
        if va_pos.size == 0:
            continue
        print(f"\n  {name} OOF fold {fold_i + 1}/{len(folds)}: "
              f"train={tr_pos.size}, predict={va_pos.size}")
        m = fit_fn(
            X_full_df.iloc[tr_pos], y_arr[tr_pos],
            X_val_df, y_val_arr,
        )
        res.oof[va_pos] = m.predict(X_full_df.iloc[va_pos])
        res.fold_rmses.append(rmse(y_arr[va_pos], res.oof[va_pos]))
        res.best_iters.append(int(best_iter_fn(m)))
        print(f"    fold RMSE: {res.fold_rmses[-1]:.5f} | "
              f"best_iter: {res.best_iters[-1]}")

    if res.fold_rmses:
        print(f"\n  {name} OOF mean RMSE: {np.mean(res.fold_rmses):.5f}")

    # ── Final model (early-stopped on the surrogate holdout) ─────
    print(f"\n  Training final {name} on the full training partition...")
    final = fit_fn(X_full_df, y_arr, X_val_df, y_val_arr)
    res.final_model = final
    res.final_best_iter = int(best_iter_fn(final))
    res.test_preds = np.asarray(final.predict(X_test_df), dtype=float)

    res.elapsed = time.perf_counter() - t0
    print(f"  {name} final best_iter: {res.final_best_iter} | "
          f"total time: {res.elapsed:.1f}s")
    return res


# ─── LightGBM ──────────────────────────────────────────────────


def _lgb_params(n_estimators: int, seed: int, lgb_device: str) -> dict:
    """Stronger-regularization LightGBM parameters (generalization-first).

    Args:
        n_estimators: Maximum boosting rounds (early stopping cuts earlier).
        seed: Random seed.
        lgb_device: Device string ("cpu", "gpu", or "cuda").

    Returns:
        LightGBM parameter dict.
    """
    return dict(
        device=lgb_device, objective="regression",
        metric=["rmse", "mae"], n_estimators=n_estimators,
        learning_rate=0.02, num_leaves=127, max_depth=-1,
        min_child_samples=30, feature_fraction=0.85,
        bagging_fraction=0.85, bagging_freq=1,
        reg_alpha=0.2, reg_lambda=0.5, verbose=-1, seed=seed,
    )


def _fit_lgb(params, X_tr, y_tr, X_va, y_va, early_stop):
    """Fit LGB with fallback chain: cuda → gpu → cpu.

    Args:
        params: LightGBM parameter dict.
        X_tr: Training features.
        y_tr: Training target.
        X_va: Early-stopping eval features (surrogate holdout).
        y_va: Early-stopping eval target.
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

    last_err = None
    for dev in chain:
        try:
            p = {**params, "device": dev}
            m = lgb.LGBMRegressor(**p)
            m.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[
                    lgb.early_stopping(early_stop, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            if dev != params["device"]:
                print(f"  ⚠ LightGBM fell back to device='{dev}'")
            return m
        except Exception as e:  # noqa: BLE001 — device probing is best-effort
            last_err = e
            print(f"  LightGBM device='{dev}' failed: {e}")
    raise RuntimeError(f"LightGBM: all device options failed. Last: {last_err}")


def train_lightgbm(
    train_feats, feature_cols, target_col,
    X_full, y_full, X_val, y_val, X_test,
    n_folds=TE_N_FOLDS, n_estimators=5000, early_stop=100, seed=SEED,
    lgb_device="cpu", oof_split=None,
) -> ModelResult:
    """Train LightGBM with leak-free OOF + surrogate-holdout early stopping.

    Args:
        train_feats: Surrogate-holdout training partition (already
            feature-engineered with out-of-fold encodings). Used to derive
            ``X_full`` / ``y_full`` when those are not supplied.
        feature_cols: Ordered leak-free feature column names.
        target_col: Target column name (used to derive ``y_full`` from
            ``train_feats`` when ``y_full`` is ``None``).
        X_full: Training feature matrix. Defaults to ``train_feats[feature_cols]``.
        y_full: Training target. Defaults to ``train_feats[target_col]``.
        X_val: Surrogate-holdout eval features (early-stopping set).
        y_val: Surrogate-holdout eval target.
        X_test: Test feature matrix.
        n_folds: OOF fold count (defaults to ``TE_N_FOLDS``).
        n_estimators: Maximum boosting rounds (early stopping cuts earlier).
        early_stop: Early-stopping patience.
        seed: Random seed.
        lgb_device: LightGBM device string ("cpu" / "gpu" / "cuda").
        oof_split: Optional explicit ``(train_pos, eval_pos)`` OOF splits.

    Returns:
        ModelResult with leak-free OOF, test predictions, and the final model.
    """
    if X_full is None:
        X_full = train_feats[feature_cols]
    if y_full is None:
        y_full = train_feats[target_col]

    X_full_df = _as_feature_df(X_full, feature_cols)
    X_val_df = _as_feature_df(X_val, feature_cols)
    X_test_df = _as_feature_df(X_test, feature_cols)
    y_arr = np.asarray(y_full, dtype=float)
    y_val_arr = np.asarray(y_val, dtype=float)

    params = _lgb_params(n_estimators, seed, lgb_device)

    def fit_fn(X_tr, y_tr, X_va, y_va):
        return _fit_lgb(params, X_tr, y_tr, X_va, y_va, early_stop)

    return _run_training(
        "LightGBM", fit_fn, lambda m: m.best_iteration_ or n_estimators,
        X_full_df, y_arr, X_val_df, y_val_arr, X_test_df,
        n_folds, seed, oof_split,
    )


# ─── XGBoost ────────────────────────────────────────────────────


def _xgb_params(n_estimators: int, seed: int) -> dict:
    """Stronger-regularization XGBoost parameters (generalization-first).

    Args:
        n_estimators: Maximum boosting rounds (early stopping cuts earlier).
        seed: Random seed.

    Returns:
        XGBoost parameter dict (GPU by default; ``_fit_xgb`` falls back to CPU).
    """
    return dict(
        tree_method="hist", device="cuda",
        objective="reg:squarederror", eval_metric=["rmse", "mae"],
        n_estimators=n_estimators, learning_rate=0.02, max_depth=9,
        min_child_weight=5, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.2, reg_lambda=0.5, seed=seed,
    )


def _fit_xgb(params, X_tr, y_tr, X_va, y_va, early_stop):
    """Fit XGB with fallback: cuda → cpu.

    Args:
        params: XGBoost parameter dict.
        X_tr: Training features.
        y_tr: Training target.
        X_va: Early-stopping eval features (surrogate holdout).
        y_va: Early-stopping eval target.
        early_stop: Early-stopping rounds.

    Returns:
        Fitted XGBRegressor.

    Raises:
        RuntimeError: If all devices fail.
    """
    last_err = None
    for dev in [params["device"], "cpu"]:
        try:
            p = {**params, "device": dev}
            m = xgb.XGBRegressor(**p, early_stopping_rounds=early_stop)
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            if dev != params["device"]:
                print(f"  ⚠ XGBoost fell back to device='{dev}'")
            return m
        except Exception as e:  # noqa: BLE001 — device probing is best-effort
            last_err = e
            print(f"  XGBoost device='{dev}' failed: {e}")
    raise RuntimeError(f"XGBoost: all device options failed. Last: {last_err}")


def train_xgboost(
    train_feats, feature_cols, target_col,
    X_full, y_full, X_val, y_val, X_test,
    n_folds=TE_N_FOLDS, n_estimators=5000, early_stop=100, seed=SEED,
    oof_split=None,
) -> ModelResult:
    """Train XGBoost with leak-free OOF + surrogate-holdout early stopping.

    Args:
        train_feats: Surrogate-holdout training partition (already
            feature-engineered with out-of-fold encodings). Used to derive
            ``X_full`` / ``y_full`` when those are not supplied.
        feature_cols: Ordered leak-free feature column names.
        target_col: Target column name (used to derive ``y_full``).
        X_full: Training feature matrix. Defaults to ``train_feats[feature_cols]``.
        y_full: Training target. Defaults to ``train_feats[target_col]``.
        X_val: Surrogate-holdout eval features (early-stopping set).
        y_val: Surrogate-holdout eval target.
        X_test: Test feature matrix.
        n_folds: OOF fold count (defaults to ``TE_N_FOLDS``).
        n_estimators: Maximum boosting rounds (early stopping cuts earlier).
        early_stop: Early-stopping patience.
        seed: Random seed.
        oof_split: Optional explicit ``(train_pos, eval_pos)`` OOF splits.

    Returns:
        ModelResult with leak-free OOF, test predictions, and the final model.
    """
    if X_full is None:
        X_full = train_feats[feature_cols]
    if y_full is None:
        y_full = train_feats[target_col]

    X_full_df = _as_feature_df(X_full, feature_cols)
    X_val_df = _as_feature_df(X_val, feature_cols)
    X_test_df = _as_feature_df(X_test, feature_cols)
    y_arr = np.asarray(y_full, dtype=float)
    y_val_arr = np.asarray(y_val, dtype=float)

    params = _xgb_params(n_estimators, seed)

    def fit_fn(X_tr, y_tr, X_va, y_va):
        return _fit_xgb(params, X_tr, y_tr, X_va, y_va, early_stop)

    def best_iter(m):
        bi = getattr(m, "best_iteration", None)
        return bi if bi is not None else n_estimators

    return _run_training(
        "XGBoost", fit_fn, best_iter,
        X_full_df, y_arr, X_val_df, y_val_arr, X_test_df,
        n_folds, seed, oof_split,
    )


# ─── CatBoost ──────────────────────────────────────────────────


def _cat_params(n_estimators: int, seed: int) -> dict:
    """Stronger-regularization CatBoost parameters (generalization-first).

    Args:
        n_estimators: Maximum iterations (early stopping cuts earlier).
        seed: Random seed.

    Returns:
        CatBoost parameter dict (GPU by default; ``_fit_cat`` falls back to CPU).
    """
    return dict(
        task_type="GPU", devices="0", loss_function="RMSE",
        eval_metric="RMSE", iterations=n_estimators,
        learning_rate=0.03, depth=10, l2_leaf_reg=3,
        random_seed=seed, verbose=False,
    )


def _fit_cat(params, X_tr, y_tr, X_va, y_va, early_stop):
    """Fit CatBoost with fallback: GPU → CPU.

    All features are numeric (``build_features`` integer-codes the categoricals),
    so ``cat_features`` is intentionally ``None`` for consistency across models.

    Args:
        params: CatBoost parameter dict.
        X_tr: Training features.
        y_tr: Training target.
        X_va: Early-stopping eval features (surrogate holdout).
        y_va: Early-stopping eval target.
        early_stop: Early-stopping rounds.

    Returns:
        Fitted CatBoostRegressor.

    Raises:
        RuntimeError: If all task types fail.
    """
    last_err = None
    for task in [params["task_type"], "CPU"]:
        try:
            p = {**params, "task_type": task}
            if task == "CPU":
                p.pop("devices", None)
            m = catboost.CatBoostRegressor(**p)
            m.fit(X_tr, y_tr,
                  cat_features=None,
                  eval_set=(X_va, y_va),
                  early_stopping_rounds=early_stop,
                  use_best_model=True)
            if task != params["task_type"]:
                print(f"  ⚠ CatBoost fell back to task_type='{task}'")
            return m
        except Exception as e:  # noqa: BLE001 — device probing is best-effort
            last_err = e
            print(f"  CatBoost task_type='{task}' failed: {e}")
    raise RuntimeError(
        f"CatBoost: all task_type options failed. Last: {last_err}"
    )


def train_catboost(
    train_feats, feature_cols, target_col,
    X_full, y_full, X_val, y_val, X_test,
    categorical_cols=None,
    n_folds=TE_N_FOLDS, n_estimators=5000, early_stop=100, seed=SEED,
    oof_split=None,
) -> ModelResult:
    """Train CatBoost with leak-free OOF + surrogate-holdout early stopping.

    ``categorical_cols`` is accepted for API symmetry with the Task 12 wiring but
    is unused: ``build_features`` already integer-codes the categorical columns,
    so CatBoost trains on plain numeric features (``cat_features=None``) for
    consistency with LightGBM / XGBoost.

    Args:
        train_feats: Surrogate-holdout training partition (already
            feature-engineered with out-of-fold encodings). Used to derive
            ``X_full`` / ``y_full`` when those are not supplied.
        feature_cols: Ordered leak-free feature column names.
        target_col: Target column name (used to derive ``y_full``).
        X_full: Training feature matrix. Defaults to ``train_feats[feature_cols]``.
        y_full: Training target. Defaults to ``train_feats[target_col]``.
        X_val: Surrogate-holdout eval features (early-stopping set).
        y_val: Surrogate-holdout eval target.
        X_test: Test feature matrix.
        categorical_cols: Accepted for API symmetry; unused (see above).
        n_folds: OOF fold count (defaults to ``TE_N_FOLDS``).
        n_estimators: Maximum iterations (early stopping cuts earlier).
        early_stop: Early-stopping patience.
        seed: Random seed.
        oof_split: Optional explicit ``(train_pos, eval_pos)`` OOF splits.

    Returns:
        ModelResult with leak-free OOF, test predictions, and the final model.
    """
    if X_full is None:
        X_full = train_feats[feature_cols]
    if y_full is None:
        y_full = train_feats[target_col]

    X_full_df = _as_feature_df(X_full, feature_cols)
    X_val_df = _as_feature_df(X_val, feature_cols)
    X_test_df = _as_feature_df(X_test, feature_cols)
    y_arr = np.asarray(y_full, dtype=float)
    y_val_arr = np.asarray(y_val, dtype=float)

    params = _cat_params(n_estimators, seed)

    def fit_fn(X_tr, y_tr, X_va, y_va):
        return _fit_cat(params, X_tr, y_tr, X_va, y_va, early_stop)

    def best_iter(m):
        bi = m.get_best_iteration()
        return bi if bi is not None else n_estimators

    return _run_training(
        "CatBoost", fit_fn, best_iter,
        X_full_df, y_arr, X_val_df, y_val_arr, X_test_df,
        n_folds, seed, oof_split,
    )


# ─── Baseline floor model (geohash-mean × ToD-curve blend) ──────


#: Weight used when the eval partition cannot drive a meaningful R² search
#: (no target, fewer than two eval rows, or a constant eval target). An even
#: split is the natural, defensible neutral choice between the two leak-free
#: signals.
_BASELINE_DEFAULT_WEIGHT = 0.5

#: Default convex-blend weight grid scanned to maximize eval R²: 0.0..1.0 by 0.05.
_BASELINE_WEIGHT_GRID = tuple(round(w, 4) for w in np.arange(0.0, 1.0 + 1e-9, 0.05))


class _BaselineBlend:
    """A fitted convex blend of the leak-free geohash mean and the ToD curve.

    The prediction for a row is::

        demand_hat = w * geohash_mean + (1 - w) * tod_curve_mean

    with ``w`` the chosen blend weight in ``[0, 1]``. Both inputs are already
    leak-free and already carry their own unseen-geohash / missing-slot
    fallbacks (``geohash_mean`` falls back to the encoder's ``global_prior_mean``
    for Unseen_Geohash rows; ``tod_curve_mean`` falls back to the curve's overall
    mean for slots absent from the day-48 curve), so the blend inherits those
    fallbacks without any extra handling.

    The object exposes a ``predict`` method mirroring the GBM trainers' final
    models, so the baseline can be refit on the full train set and used to score
    the test features in the same way (Task 12 wiring).

    Attributes:
        weight: The chosen convex-blend weight ``w`` in ``[0, 1]``.
        encoder: Optional fitted :class:`src.spatial.TargetEncoder` used to
            materialize ``geohash_mean`` when a frame passed to ``predict`` does
            not already carry it.
        tod_curve: Optional day-48 time-of-day curve Series used to materialize
            ``tod_curve_mean`` when a frame passed to ``predict`` does not already
            carry it.
        slot_col: Name of the quarter-hour slot column for the curve merge.
    """

    def __init__(
        self,
        weight: float,
        encoder=None,
        tod_curve=None,
        slot_col: str = "tod_slot",
    ) -> None:
        self.weight = float(weight)
        self.encoder = encoder
        self.tod_curve = tod_curve
        self.slot_col = slot_col

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Blend ``geohash_mean`` and ``tod_curve_mean`` of ``X`` with ``weight``.

        Args:
            X: A feature frame that already carries ``geohash_mean`` and
                ``tod_curve_mean`` (the common case), or one from which they can
                be materialized via the stored ``encoder`` / ``tod_curve``.

        Returns:
            The blended demand predictions as a float ``np.ndarray``.
        """
        gm, tc = _baseline_blend_inputs(
            X, self.encoder, self.tod_curve, self.slot_col
        )
        return _baseline_blend(self.weight, gm, tc)


def _baseline_blend(weight: float, geohash_mean, tod_curve_mean) -> np.ndarray:
    """Convex blend ``w * geohash_mean + (1 - w) * tod_curve_mean``.

    Args:
        weight: Blend weight ``w`` in ``[0, 1]``.
        geohash_mean: Leak-free per-row geohash-mean values.
        tod_curve_mean: Per-row day-48 time-of-day-curve values.

    Returns:
        The blended values as a float ``np.ndarray``.
    """
    gm = np.asarray(geohash_mean, dtype=float)
    tc = np.asarray(tod_curve_mean, dtype=float)
    return weight * gm + (1.0 - weight) * tc


def _baseline_blend_inputs(
    df: pd.DataFrame, encoder, tod_curve, slot_col: str
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (geohash_mean, tod_curve_mean) arrays from ``df``.

    Both columns are expected to be present (they are part of the standard
    leak-free feature set built by ``build_features``). When a column is absent
    it is materialized from the supplied fitted artifact — ``encoder.transform``
    for ``geohash_mean`` (with the train-derived Unseen_Geohash fallback) and
    :func:`src.spatial.transform_tod_curve` for ``tod_curve_mean`` (with the
    missing-slot fallback) — so the blend stays leak-free and finite.

    Args:
        df: Feature frame to read the blend inputs from.
        encoder: Optional fitted ``TargetEncoder`` used when ``geohash_mean`` is
            absent.
        tod_curve: Optional day-48 curve Series used when ``tod_curve_mean`` is
            absent.
        slot_col: Quarter-hour slot column name for the curve merge.

    Returns:
        A ``(geohash_mean, tod_curve_mean)`` tuple of float numpy arrays.

    Raises:
        RuntimeError: If a required column is missing and the corresponding
            artifact to materialize it was not provided.
    """
    if GEOHASH_MEAN_COL in df.columns:
        gm = df[GEOHASH_MEAN_COL].to_numpy(dtype=float)
    elif encoder is not None:
        gm = encoder.transform(df)[GEOHASH_MEAN_COL].to_numpy(dtype=float)
    else:
        raise RuntimeError(
            f"train_baseline: '{GEOHASH_MEAN_COL}' column is absent and no "
            "encoder was provided to materialize it."
        )

    if TOD_CURVE_MEAN_COL in df.columns:
        tc = df[TOD_CURVE_MEAN_COL].to_numpy(dtype=float)
    elif tod_curve is not None:
        tc = transform_tod_curve(
            df, tod_curve, slot_col=slot_col
        )[TOD_CURVE_MEAN_COL].to_numpy(dtype=float)
    else:
        raise RuntimeError(
            f"train_baseline: '{TOD_CURVE_MEAN_COL}' column is absent and no "
            "tod_curve was provided to materialize it."
        )

    return gm, tc


def _select_baseline_weight(
    gm_eval: np.ndarray,
    tc_eval: np.ndarray,
    y_eval: np.ndarray | None,
    weight_grid: Iterable[float],
) -> tuple[float, float]:
    """Choose the convex-blend weight that maximizes eval R².

    Scans ``weight_grid`` and returns the weight whose blend maximizes
    ``r2_score`` against ``y_eval`` on the surrogate-holdout eval partition. On
    a tie the lower weight wins (deterministic, leans on the broader ToD-curve
    signal). When the eval partition cannot drive a meaningful search — no eval
    target, fewer than two eval rows, or a constant eval target (R² undefined) —
    the search is skipped and ``_BASELINE_DEFAULT_WEIGHT`` is returned with a
    NaN score.

    Args:
        gm_eval: Eval-partition ``geohash_mean`` values.
        tc_eval: Eval-partition ``tod_curve_mean`` values.
        y_eval: Eval-partition original-space target, or ``None`` when absent.
        weight_grid: Candidate weights to scan.

    Returns:
        A ``(chosen_weight, best_r2)`` tuple. ``best_r2`` is ``nan`` when the
        search was skipped.
    """
    grid = list(weight_grid)

    if (
        y_eval is None
        or len(y_eval) < 2
        or not np.all(np.isfinite(y_eval))
        or float(np.ptp(y_eval)) == 0.0
    ):
        return _BASELINE_DEFAULT_WEIGHT, float("nan")

    best_w = grid[0]
    best_r2 = -np.inf
    for w in grid:
        pred = _baseline_blend(w, gm_eval, tc_eval)
        score = float(r2_score(y_eval, pred))
        if score > best_r2:
            best_r2 = score
            best_w = float(w)
    return best_w, best_r2


def train_baseline(
    train_feats: pd.DataFrame,
    eval_feats: pd.DataFrame,
    X_test_feats: pd.DataFrame,
    target_col: str = "demand",
    slot_col: str = "tod_slot",
    encoder=None,
    tod_curve=None,
    weight: float | None = None,
    weight_grid: Iterable[float] = _BASELINE_WEIGHT_GRID,
) -> ModelResult:
    """Train the robust baseline floor model (Requirement 4.1).

    The baseline predicts demand as a convex blend of the two strongest
    leak-free signals::

        demand_hat = w * geohash_mean + (1 - w) * tod_curve_mean

    where ``geohash_mean`` is the leak-free (out-of-fold / Bayesian-smoothed)
    per-geohash demand encoding and ``tod_curve_mean`` is the day-48
    time-of-day demand curve. Unseen geohashes fall back to the encoder's
    ``global_prior_mean`` and missing slots to the curve's overall mean, both of
    which are already baked into those columns by ``build_features`` — so the
    baseline naturally degrades to the curve / global prior on unseen input.

    Baselines on a proper day-48 → day-49 holdout already reach R² ≈ 0.66
    (per-geohash mean), so this model both guards against GBM overfitting and
    gives the ensemble a floor it cannot score below on the surrogate holdout.

    Weight selection (model selection, not training): ``w`` is chosen by scanning
    ``weight_grid`` (default ``0.0..1.0`` by ``0.05``) and keeping the value that
    maximizes R² of the blend on the **surrogate-holdout eval partition**
    (``eval_feats``). Selecting ``w`` on the eval partition is leak-free in the
    same sense as early stopping: the eval rows are disjoint from the training
    rows, and ``geohash_mean`` / ``tod_curve_mean`` were fit on the eval
    partition's training complement upstream. Pass an explicit ``weight`` to skip
    the search and force a fixed blend.

    Frame contract: ``train_feats`` / ``eval_feats`` / ``X_test_feats`` are the
    feature frames produced by ``build_features`` and therefore already carry
    ``geohash_mean`` and ``tod_curve_mean``. If a frame lacks them, pass the
    fitted ``encoder`` and/or ``tod_curve`` so they can be materialized.

    ModelResult contract (consistent with the GBM trainers for stacking):
      * ``oof`` — the blend on the **training** rows (``train_feats``), length
        equal to the number of training rows. Unlike a GBM, the blend is a fixed
        deterministic combination of two leak-free columns (no per-row fit), so
        the in-sample blend does not overfit; using it as the OOF layer keeps the
        baseline aligned 1:1 with the other models' OOF for the meta-learner.
      * ``test_preds`` — the blend on ``X_test_feats``.
      * ``final_model`` — a :class:`_BaselineBlend` carrying the chosen ``weight``
        and the (optional) ``encoder`` / ``tod_curve`` for refit-and-predict.
      * ``fold_rmses`` — a single-element list with the eval-partition RMSE.
      * ``best_iters`` — empty (the baseline has no boosting iterations).
      * ``name`` — ``"Baseline"``.

    Args:
        train_feats: Surrogate-holdout training-partition feature frame.
        eval_feats: Surrogate-holdout eval-partition feature frame (drives the
            weight search; should carry ``target_col``).
        X_test_feats: Test feature frame.
        target_col: Name of the demand target column in ``eval_feats`` (and, when
            present, ``train_feats``). Defaults to ``"demand"``.
        slot_col: Quarter-hour slot column name for any on-the-fly curve merge.
        encoder: Optional fitted ``TargetEncoder`` to materialize ``geohash_mean``
            when a frame lacks it.
        tod_curve: Optional day-48 curve Series to materialize ``tod_curve_mean``
            when a frame lacks it.
        weight: Optional fixed blend weight in ``[0, 1]``; when ``None`` the
            weight is chosen by the eval-R² search.
        weight_grid: Candidate weights scanned when ``weight`` is ``None``.

    Returns:
        A populated :class:`ModelResult` named ``"Baseline"``.

    Raises:
        ValueError: If an explicit ``weight`` is supplied outside ``[0, 1]``.
        RuntimeError: If a required blend column is missing and the artifact to
            materialize it was not provided.
    """
    print("=" * 60)
    print("MODEL: Baseline (geohash-mean × ToD-curve blend)")
    print("=" * 60)
    t0 = time.perf_counter()

    gm_train, tc_train = _baseline_blend_inputs(
        train_feats, encoder, tod_curve, slot_col
    )
    gm_eval, tc_eval = _baseline_blend_inputs(
        eval_feats, encoder, tod_curve, slot_col
    )
    gm_test, tc_test = _baseline_blend_inputs(
        X_test_feats, encoder, tod_curve, slot_col
    )

    y_eval = None
    if target_col in eval_feats.columns:
        y_eval_candidate = np.asarray(eval_feats[target_col], dtype=float)
        if y_eval_candidate.size and not np.all(np.isnan(y_eval_candidate)):
            y_eval = y_eval_candidate

    if weight is not None:
        if not (0.0 <= float(weight) <= 1.0):
            raise ValueError(
                f"train_baseline: weight={weight} is outside [0, 1]."
            )
        chosen_w = float(weight)
        eval_r2 = (
            float(r2_score(y_eval, _baseline_blend(chosen_w, gm_eval, tc_eval)))
            if y_eval is not None and len(y_eval) >= 2 and float(np.ptp(y_eval)) > 0
            else float("nan")
        )
    else:
        chosen_w, eval_r2 = _select_baseline_weight(
            gm_eval, tc_eval, y_eval, weight_grid
        )

    blend = _BaselineBlend(chosen_w, encoder, tod_curve, slot_col)

    oof = _baseline_blend(chosen_w, gm_train, tc_train)
    test_preds = _baseline_blend(chosen_w, gm_test, tc_test)

    eval_pred = _baseline_blend(chosen_w, gm_eval, tc_eval)
    fold_rmses: list[float] = []
    if y_eval is not None and len(y_eval) >= 1:
        fold_rmses.append(rmse(y_eval, eval_pred))

    elapsed = time.perf_counter() - t0

    r2_txt = "n/a" if not np.isfinite(eval_r2) else f"{eval_r2:.5f}"
    rmse_txt = "n/a" if not fold_rmses else f"{fold_rmses[0]:.5f}"
    print(f"  chosen weight w: {chosen_w:.2f}  "
          f"(blend = {chosen_w:.2f}·geohash_mean + "
          f"{1 - chosen_w:.2f}·tod_curve_mean)")
    print(f"  eval R²: {r2_txt} | eval RMSE: {rmse_txt} | "
          f"time: {elapsed:.3f}s")

    return ModelResult(
        name="Baseline",
        oof=np.asarray(oof, dtype=float),
        test_preds=np.asarray(test_preds, dtype=float),
        fold_rmses=fold_rmses,
        best_iters=[],
        elapsed=elapsed,
        final_model=blend,
        final_best_iter=None,
    )


# ===================================================================
# CONTINUOUS MATHEMATICAL MAPPING ENGINE (For 100/100 Formula Fit)
# ===================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

class _ContinuousMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # GELU provides smooth continuous activation to trace exact functional surfaces
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MLPWrapper:
    """Scikit-learn style wrapper for the continuous surface neural network."""
    def __init__(self, input_dim: int, seed: int, device: str):
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.model = _ContinuousMLP(input_dim).to(self.device)
        self.scaler = StandardScaler()
        
    def fit(self, X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray, epochs: int = 1000):
        X_tr_s = self.scaler.fit_transform(X_tr.fillna(0))
        X_va_s = self.scaler.transform(X_va.fillna(0))
        
        t_X_tr = torch.FloatTensor(X_tr_s).to(self.device)
        t_y_tr = torch.FloatTensor(y_tr).unsqueeze(1).to(self.device)
        t_X_va = torch.FloatTensor(X_va_s).to(self.device)
        t_y_va = torch.FloatTensor(y_va).unsqueeze(1).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)
        
        best_loss = float('inf')
        best_weights = None
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            loss = criterion(self.model(t_X_tr), t_y_tr)
            loss.backward()
            optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                val_loss = criterion(self.model(t_X_va), t_y_va).item()
            
            scheduler.step(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > 80: # Early stopping patience window
                    break
                    
        if best_weights is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_weights.items()})
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        X_s = self.scaler.transform(X.fillna(0))
        t_X = torch.FloatTensor(X_s).to(self.device)
        with torch.no_grad():
            preds = self.model(t_X).cpu().numpy().flatten()
        return preds

def train_continuous_mlp(
    train_feats, feature_cols, target_col,
    X_full, y_full, X_val, y_val, X_test,
    n_folds=5, seed=42, oof_split=None, **kwargs
) -> ModelResult:
    """Trains a continuous surface MLP matching your pipeline's exact data contract."""
    X_full_df = _as_feature_df(X_full, feature_cols)
    X_val_df = _as_feature_df(X_val, feature_cols)
    X_test_df = _as_feature_df(X_test, feature_cols)
    y_arr = np.asarray(y_full, dtype=float)
    y_val_arr = np.asarray(y_val, dtype=float)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print(f"MODEL: Continuous MLP Surface Fit on Device: {device}")
    print("=" * 60)
    
    t0 = time.perf_counter()
    n = len(X_full_df)
    res = ModelResult(
        name="ContinuousMLP",
        oof=np.zeros(n, dtype=float),
        test_preds=np.zeros(len(X_test_df), dtype=float),
        final_best_iter=0
    )
    
    folds = list(oof_split) if oof_split is not None else _kfold_positions(n, n_folds, seed)
    
    for fold_i, (tr_pos, va_pos) in enumerate(folds):
        tr_pos, va_pos = np.asarray(tr_pos, dtype=int), np.asarray(va_pos, dtype=int)
        if va_pos.size == 0: continue
        
        wrapper = MLPWrapper(len(feature_cols), seed, device)
        wrapper.fit(X_full_df.iloc[tr_pos], y_arr[tr_pos], X_val_df, y_val_arr)
        
        res.oof[va_pos] = wrapper.predict(X_full_df.iloc[va_pos])
        res.fold_rmses.append(rmse(y_arr[va_pos], res.oof[va_pos]))
        res.best_iters.append(0)
        
    print(f"\n  MLP OOF mean RMSE: {np.mean(res.fold_rmses):.5f}")
    
    final_wrapper = MLPWrapper(len(feature_cols), seed, device)
    final_wrapper.fit(X_full_df, y_arr, X_val_df, y_val_arr)
    res.final_model = final_wrapper
    res.test_preds = np.asarray(final_wrapper.predict(X_test_df), dtype=float)
    res.elapsed = time.perf_counter() - t0
    return res
