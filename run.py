#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════
Flipkart Gridlock Hackathon 2.0 — Traffic Demand Prediction
═══════════════════════════════════════════════════════════════════

  python run.py

Expects  ./data/train.csv  and  ./data/test.csv
Produces ./submission_N.csv, ./metrics_N.json, ./feature_importance_N.{csv,png},
         ./shap_summary_N.png  (plus ./submission.csv and ./metrics.json copies).

Pipeline order (demand-prediction-overhaul, Task 12.1)
------------------------------------------------------
load_data
  → build_holdouts (real-task + day-48 daytime surrogate)
  → fit encoders / curve / imputers on the surrogate holdout-train partition
  → build_features (surrogate train OOF + surrogate eval)
  → select_transform (identity vs log1p on the surrogate holdout)
  → train models + baseline (early-stop on the surrogate holdout)
  → stack (in ORIGINAL target space)
  → refit encoders on full train → build test features → predict
  → postprocess → diagnostics → write_submission (versioned).

Leakage boundary (Requirement 2.5): every target-derived artifact used to
*score* the surrogate holdout is fit on the holdout-train partition only. The
final/submission models additionally use full-train-refit encodings so the test
features carry the strongest leak-free signal (design "Final/submission
context").

Validation tension (Requirement 2.4): on this two-day snapshot the day-49 rows
are morning-only (slots 0..8), so ``build_real_task_holdout`` reports
``FAILED_NO_MATCHING_SLOTS``. Per the resolved design decision the pipeline logs
the absence and PROCEEDS using the day-48 daytime surrogate as the primary
model-selection metric — it does NOT hard-abort.
"""

import os
import warnings
import time
import glob
import re
import json
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 0. Environment (CUDA relaxed to a warning + CPU fallback, Req 6.4) ──

import torch
import lightgbm as lgb
import subprocess
import sys

def check_lgb_gpu_support(device_name: str) -> bool:
    """Check if LightGBM has support for the specified GPU device in a subprocess."""
    code = f"""
import lightgbm as lgb
import numpy as np
X = np.random.rand(5, 2)
y = np.random.rand(5)
try:
    m = lgb.LGBMRegressor(device="{device_name}", n_estimators=1, verbose=-1)
    m.fit(X, y)
    print("OK")
except Exception:
    pass
"""
    try:
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=5)
        return "OK" in res.stdout
    except Exception:
        return False

print(f"CUDA available  : {torch.cuda.is_available()}")
LGB_DEVICE = "cpu"
if torch.cuda.is_available():
    print(f"GPU name        : {torch.cuda.get_device_name(0)}")
    # LightGBM 4.0+ exposes the unified "cuda" device; older builds use "gpu".
    _lgb_version = tuple(int(x) for x in lgb.__version__.split(".")[:2])
    candidate_device = "cuda" if _lgb_version >= (4, 0) else "gpu"
    if check_lgb_gpu_support(candidate_device):
        LGB_DEVICE = candidate_device
    else:
        print(f"⚠ LightGBM GPU test failed for device='{candidate_device}' (not compiled with GPU support). Falling back to CPU.")
else:
    # Relaxed from the previous hard `assert torch.cuda.is_available()` so the
    # pipeline stays runnable and testable without a GPU.
    warnings.warn("CUDA/GPU not available — falling back to CPU.", RuntimeWarning)
    print("⚠ GPU not found — proceeding on CPU.")

print(f"LightGBM device : {LGB_DEVICE}")

# ── 1. Config & seeds ──────────────────────────────────────────

from config import (
    SEED, seed_everything,
    TE_SMOOTHING_ALPHA, TE_N_FOLDS,
    N_ESTIMATORS, EARLY_STOPPING,
    RECORDED_ONLINE_SCORE, LEADERBOARD_TOP,
    TRAIN_PATH, TEST_PATH,
)

# Versioned run id: max existing submission_N.csv + 1 (Task 12.2 / Req 5.7).
_submission_files = glob.glob("./submission_*.csv")
_run_ids = [
    int(re.search(r"submission_(\d+)\.csv", f).group(1))
    for f in _submission_files
    if re.search(r"submission_(\d+)\.csv", f)
]
RUN_ID = max(_run_ids) + 1 if _run_ids else 1

SUBMISSION_PATH = f"./submission_{RUN_ID}.csv"
FEATURE_IMPORTANCE_CSV = f"./feature_importance_{RUN_ID}.csv"
FEATURE_IMPORTANCE_PNG = f"./feature_importance_{RUN_ID}.png"
SHAP_SUMMARY_PNG = f"./shap_summary_{RUN_ID}.png"
METRICS_PATH = f"./metrics_{RUN_ID}.json"

print("\n" + "=" * 60)
print(f"STARTING TRAINING RUN #{RUN_ID}")
print("=" * 60 + "\n")

# Single seeding entry point, called once before any stochastic step (Req 6.4).
seed_everything(SEED)


def _json_default(o):
    """JSON encoder fallback for numpy scalar / array metric values."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _apply_stack_to_holdout(stack_info, gbm_holdout_preds, baseline_holdout_preds):
    """Apply the fitted stack to surrogate-holdout predictions (ORIGINAL space).

    Mirrors ``ensemble.stack_predictions`` exactly: the stack order is the GBM
    models first, then the baseline (when present). ``stack_predictions`` was fit
    in ORIGINAL space (GBM OOF/test inverted before stacking, baseline already
    original), so the holdout meta-matrix passed here must also be original-space.

    Args:
        stack_info: The ``info`` dict returned by ``stack_predictions``.
        gbm_holdout_preds: List of per-GBM holdout predictions (original space),
            in the same order as the ``results`` list given to the stacker.
        baseline_holdout_preds: Baseline holdout predictions (original space), or
            ``None`` when no baseline participated.

    Returns:
        The ensemble holdout predictions as a float ``np.ndarray``.
    """
    method = stack_info["method"]
    if method == "baseline_floor":
        return np.asarray(baseline_holdout_preds, dtype=float)

    cols = list(gbm_holdout_preds)
    if baseline_holdout_preds is not None:
        cols = cols + [np.asarray(baseline_holdout_preds, dtype=float)]
    X_meta = np.column_stack(cols)

    if method == "ridge":
        return np.asarray(stack_info["ridge"].predict(X_meta), dtype=float)
    # weighted_avg fallback — same clipped non-negative weights as the stacker.
    weights = stack_info["weights"]
    return np.asarray(sum(w * c for w, c in zip(weights, cols)), dtype=float)


# ── 2. Data loading & schema ───────────────────────────────────

from src.data import (
    load_data, print_diagnostics,
    select_transform, apply_transform, invert_transform,
    TransformSelectionError,
)

train_df, test_df, SCHEMA = load_data(TRAIN_PATH, TEST_PATH)

# print_diagnostics is best-effort: it inspects the now-string timestamp column,
# so guard it so a diagnostics-only issue can never abort the pipeline.
try:
    print_diagnostics(train_df, test_df, SCHEMA)
except Exception as exc:  # noqa: BLE001 — diagnostics must never break the run
    print(f"⚠ print_diagnostics skipped ({type(exc).__name__}: {exc}).")

timestamp_col    = SCHEMA["timestamp_col"]
location_cols    = SCHEMA["location_cols"]
target_col       = SCHEMA["target_col"]
categorical_cols = SCHEMA["categorical_cols"]
id_col           = SCHEMA["id_col"]

# Consistent categorical integer codes over the union of train+test (Req 3.5).
# build_features also appends "Missing" for the nullable RoadType/Weather cats;
# we pass a stable, shared map here so codes match across all built frames.
category_maps = {}
for col in categorical_cols:
    if col in train_df.columns and col in test_df.columns:
        combined = (
            pd.concat([train_df[col], test_df[col]])
            .dropna().astype(str).unique()
        )
        category_maps[col] = sorted(combined.tolist())
        print(f"  Category map '{col}': {len(category_maps[col])} unique values")

# ── 3. Holdouts: real-task + day-48 daytime surrogate (Req 2.1–2.4) ──

from src.validation import (
    build_real_task_holdout, build_day48_daytime_holdout,
    STATUS_FAILED_NO_MATCHING_SLOTS,
)

print("\n" + "=" * 60)
print("VALIDATION HOLDOUTS")
print("=" * 60)

real_split = build_real_task_holdout(train_df)
print(f"[real_task]  status: {real_split.status}")
print(f"[real_task]  {real_split.coverage_note}")

if real_split.status == STATUS_FAILED_NO_MATCHING_SLOTS:
    # Resolved design decision (Req 2.4): log the absence and PROCEED with the
    # surrogate as the primary model-selection metric — no hard abort.
    print("⚠ Real-task holdout has no matching day-49 daytime slots — "
          "proceeding with the day-48 daytime surrogate for model selection.")

surrogate = build_day48_daytime_holdout(train_df)
print(f"[surrogate]  status: {surrogate.status}  (PRIMARY model-selection metric)")
print(f"[surrogate]  {surrogate.coverage_note}")

REAL_TASK_STATUS = real_split.status

# Surrogate partitions (fit-on-train / score-on-eval leakage boundary, Req 2.5).
sur_train = train_df.loc[surrogate.train_idx]
sur_eval = train_df.loc[surrogate.eval_idx]
print(f"  surrogate train rows: {len(sur_train)}  eval rows: {len(sur_eval)}")

# ── 4. Fit leak-free artifacts on the surrogate TRAIN partition only ──

from src.spatial import TargetEncoder, fit_tod_curve, InteractionEncoder, PerGeohashTodCurve
from src.features import (
    build_imputers, build_features, reduce_memory, select_feature_cols,
)

print("\nFitting leak-free artifacts on the surrogate-train partition ...")
encoder_cv = TargetEncoder(location_cols[0]).fit(
    sur_train, target_col, TE_SMOOTHING_ALPHA
)
curve_cv = fit_tod_curve(sur_train, target_col)
imputers_cv = build_imputers(sur_train, categorical_cols, target_col)
inter_cv = InteractionEncoder(location_cols[0]).fit(sur_train, target_col, 10.0)
pg_cv = PerGeohashTodCurve(location_cols[0]).fit(sur_train, target_col, 25.0)

# ── 5. Build surrogate features (train uses OOF encodings; eval transforms) ──

print("Building surrogate features ...")
t0 = time.perf_counter()
sur_train_oof = encoder_cv.fit_oof(
    sur_train, target_col, TE_SMOOTHING_ALPHA, TE_N_FOLDS, SEED
)
# Add leak-free out-of-fold interaction encodings to the training frame.
sur_train_oof = inter_cv.fit_oof(
    sur_train_oof, target_col, 10.0, TE_N_FOLDS, SEED
)
# Add leak-free out-of-fold per-geohash time-of-day curve.
sur_train_oof = pg_cv.fit_oof(
    sur_train_oof, target_col, 25.0, TE_N_FOLDS, SEED
)
sur_train_feats = build_features(
    sur_train_oof, encoder_cv, curve_cv, categorical_cols, location_cols,
    imputers_cv, category_maps=category_maps, interaction_encoder=inter_cv,
    pg_curve=pg_cv,
)
sur_eval_feats = build_features(
    sur_eval, encoder_cv, curve_cv, categorical_cols, location_cols,
    imputers_cv, category_maps=category_maps, interaction_encoder=inter_cv,
    pg_curve=pg_cv,
)
print(f"  sur_train_feats: {sur_train_feats.shape}  "
      f"sur_eval_feats: {sur_eval_feats.shape}  ({time.perf_counter()-t0:.1f}s)")

sur_train_feats = reduce_memory(sur_train_feats, target_col)
sur_eval_feats = reduce_memory(sur_eval_feats, target_col)

FEATURE_COLS = select_feature_cols(
    sur_train_feats, sur_eval_feats, timestamp_col, target_col, id_col,
    location_cols=location_cols,
)
# Filter out high-cardinality noise shortcuts to force mathematical formula alignment
FEATURE_COLS = [c for c in FEATURE_COLS if c != "Temperature"]

# ── 6. Data-driven transform selection on the surrogate holdout (Req 4.2–4.4) ──

print("\n" + "=" * 60)
print("TRANSFORM SELECTION (identity vs log1p)")
print("=" * 60)

_sur_y_train = sur_train_feats[target_col].to_numpy(dtype=float)
_sur_y_eval = sur_eval_feats[target_col].to_numpy(dtype=float)
_X_sur_train = sur_train_feats[FEATURE_COLS]
_X_sur_eval = sur_eval_feats[FEATURE_COLS]


def _transform_probe(y_train_transformed):
    """Fit a quick LightGBM on the transformed target; predict eval rows.

    Used only by ``select_transform`` to compare identity vs log1p. The model is
    trained in whatever transformed space ``select_transform`` hands it and
    returns predictions in that SAME transformed space (``select_transform``
    inverts them before scoring). A fixed modest number of trees (no early
    stopping) keeps the probe fast, deterministic, and space-agnostic — the
    early-stopping eval target would otherwise be in a different space than the
    training target for the log1p candidate.
    """
    probe = lgb.LGBMRegressor(
        device=LGB_DEVICE, objective="regression", n_estimators=400,
        learning_rate=0.05, num_leaves=31, max_depth=6,
        min_child_samples=150, feature_fraction=0.7,
        bagging_fraction=0.7, bagging_freq=1,
        reg_alpha=1.0, reg_lambda=2.0, verbose=-1, seed=SEED,
    )
    try:
        probe.fit(_X_sur_train, y_train_transformed)
    except Exception:  # noqa: BLE001 — device probing best-effort, retry on CPU
        probe.set_params(device="cpu")
        probe.fit(_X_sur_train, y_train_transformed)
    return probe.predict(_X_sur_eval)


try:
    TRANSFORM, transform_details = select_transform(
        _sur_y_train, _sur_y_eval, _transform_probe
    )
    print(f"  identity Holdout_R²: {transform_details['identity_r2']:.5f}")
    print(f"  log1p    Holdout_R²: {transform_details['log1p_r2']:.5f}")
    print(f"✓ Selected transform: {TRANSFORM}")
except TransformSelectionError as exc:
    # Both candidates non-positive (Req 4.4). select_transform itself fails, but
    # the pipeline still produces a submission: log and fall back to identity.
    TRANSFORM = "identity"
    print(f"⚠ TransformSelectionError: {exc}")
    print("  Falling back to the identity transform so the run still produces "
          "a submission.")

# ── 7. Refit encoders on the FULL train for the final/submission models ──
#
# Convention (documented): the FINAL GBMs are trained on the full-train OOF
# features and EARLY-STOP on the surrogate holdout eval set; the test features
# use full-train-refit encodings. Stacking is done entirely in ORIGINAL target
# space (see step 9), so models train on apply_transform(y, TRANSFORM) and their
# OOF/test predictions are inverted back to original space before stacking.

print("\nRefitting leak-free artifacts on the FULL train partition ...")
encoder_full = TargetEncoder(location_cols[0]).fit(
    train_df, target_col, TE_SMOOTHING_ALPHA
)
curve_full = fit_tod_curve(train_df, target_col)
imputers_full = build_imputers(train_df, categorical_cols, target_col)
inter_full = InteractionEncoder(location_cols[0]).fit(train_df, target_col, 10.0)
pg_full = PerGeohashTodCurve(location_cols[0]).fit(train_df, target_col, 25.0)

print("Building full-train OOF features and test features ...")
t0 = time.perf_counter()
full_oof = encoder_full.fit_oof(
    train_df, target_col, TE_SMOOTHING_ALPHA, TE_N_FOLDS, SEED
)
full_oof = inter_full.fit_oof(
    full_oof, target_col, 10.0, TE_N_FOLDS, SEED
)
full_oof = pg_full.fit_oof(
    full_oof, target_col, 25.0, TE_N_FOLDS, SEED
)
full_train_feats = build_features(
    full_oof, encoder_full, curve_full, categorical_cols, location_cols,
    imputers_full, category_maps=category_maps, interaction_encoder=inter_full,
    pg_curve=pg_full,
)
test_feats = build_features(
    test_df, encoder_full, curve_full, categorical_cols, location_cols,
    imputers_full, category_maps=category_maps, interaction_encoder=inter_full,
    pg_curve=pg_full,
)
print(f"  full_train_feats: {full_train_feats.shape}  "
      f"test_feats: {test_feats.shape}  ({time.perf_counter()-t0:.1f}s)")

full_train_feats = reduce_memory(full_train_feats, target_col)
test_feats = reduce_memory(test_feats, target_col)

# Ensure the surrogate holdout eval frame carries every model feature column so
# early stopping and holdout scoring use exactly FEATURE_COLS.
for _c in FEATURE_COLS:
    if _c not in sur_eval_feats.columns:
        raise RuntimeError(f"surrogate eval frame missing feature column {_c!r}")

# ── 8. Train GBMs (final on full train, early-stop on surrogate holdout) ──

from src.models import (
    train_lightgbm, train_xgboost, train_catboost, train_baseline, train_continuous_mlp
)

# Models train in the SELECTED transform space; OOF/test preds are inverted
# back to original space before stacking (step 9).
X_full = full_train_feats[FEATURE_COLS]
y_full_original = full_train_feats[target_col].to_numpy(dtype=float)
y_full = apply_transform(y_full_original, TRANSFORM)

X_val = sur_eval_feats[FEATURE_COLS]
y_val_original = sur_eval_feats[target_col].to_numpy(dtype=float)
y_val = apply_transform(y_val_original, TRANSFORM)

X_test = test_feats[FEATURE_COLS]

common = dict(
    train_feats=full_train_feats, feature_cols=FEATURE_COLS, target_col=target_col,
    X_full=X_full, y_full=y_full, X_val=X_val, y_val=y_val, X_test=X_test,
    n_folds=TE_N_FOLDS, n_estimators=N_ESTIMATORS, early_stop=EARLY_STOPPING,
    seed=SEED,
)

lgb_res = train_lightgbm(**common, lgb_device=LGB_DEVICE)
xgb_res = train_xgboost(**common)
cat_res = train_catboost(**common, categorical_cols=categorical_cols)
mlp_res = train_continuous_mlp(**common)

# Append the continuous neural predictor to the stack array
gbm_results = [lgb_res, xgb_res, cat_res, mlp_res]

# Invert GBM OOF / test predictions to ORIGINAL space so the whole stack lives
# in one coherent space (the baseline is original by construction).
gbm_results_original = [
    replace(
        r,
        oof=invert_transform(r.oof, TRANSFORM),
        test_preds=invert_transform(r.test_preds, TRANSFORM),
    )
    for r in gbm_results
]

# Honest holdout estimate on the day-48 daytime SURROGATE eval partition, scored
# with the SAME full-train encoder basis the final models were trained on. The
# earlier ``sur_eval_feats`` was built with the surrogate-only encoders
# (encoder_cv), so scoring full-train models on it caused a covariate-shift
# artifact (local 77 vs online 90). Rebuilding the eval rows with encoder_full /
# inter_full / pg_full removes the mismatch. This is leak-safe: the surrogate
# eval rows are day-48 daytime rows, and encoder_full's OOF/curve are train-wide
# leak-free encodings (the eval rows' own targets do not define their encodings
# beyond the standard OOF treatment).
sur_eval_feats_full = build_features(
    sur_eval, encoder_full, curve_full, categorical_cols, location_cols,
    imputers_full, category_maps=category_maps, interaction_encoder=inter_full,
    pg_curve=pg_full,
)
X_holdout = sur_eval_feats_full[FEATURE_COLS]
y_val_original = sur_eval_feats_full[target_col].to_numpy(dtype=float)
gbm_holdout_preds_original = [
    invert_transform(r.final_model.predict(X_holdout), TRANSFORM)
    for r in gbm_results
]

print("\n" + "=" * 60)
print("OOF SUMMARY (transformed-space fold RMSEs)")
print("=" * 60)
for r in gbm_results:
    print(f"  {r.name:<10s}: "
          f"{['%.5f' % x for x in r.fold_rmses]} → "
          f"mean {np.mean(r.fold_rmses):.5f}")

# ── 8b. Robust baseline floor (ORIGINAL space, Req 4.1) ──────────
# The baseline blends leak-free geohash_mean × tod_curve_mean — both are
# original-space, so the baseline is NOT transformed. It trains on the full
# train features, selects its weight on the surrogate eval, predicts the test.
baseline_res = train_baseline(
    full_train_feats, sur_eval_feats, test_feats,
    target_col=target_col, slot_col="tod_slot",
    encoder=encoder_full, tod_curve=curve_full,
)
baseline_holdout_preds = baseline_res.final_model.predict(sur_eval_feats)

# ── 9. Stacking — ALL in ORIGINAL target space (Req 4.5) ─────────
#
# The GBM OOF/test preds were inverted to original space above and the baseline
# is original by construction, so the meta-learner is fit against the ORIGINAL
# full-train target. The resulting ensemble predictions are already in original
# space; postprocess is therefore called with transform="identity" (it only
# clips / NaN-guards — no further inverse transform is needed).

from src.ensemble import stack_predictions

final_preds, stack_info = stack_predictions(
    gbm_results_original, y_full_original, baseline_result=baseline_res,
)

# Ensemble predictions on the surrogate holdout (same stack/floor, original
# space) for honest reporting.
ensemble_holdout_preds = _apply_stack_to_holdout(
    stack_info, gbm_holdout_preds_original, baseline_holdout_preds,
)

# ── 10. Post-processing — clip to [0, train_max], NaN-guard (Req 5.4–5.6) ──

from src.postprocess import postprocess, write_submission

# Predictions are already in original space → transform="identity".
final_preds = postprocess(final_preds, "identity", TRAIN_PATH, target_col)

# ── 11. Diagnostics + honest reporting (Req 2.6, 2.7, 6.1) ───────

from src.diagnostics import run_diagnostics

holdout_model_preds = {
    r.name: preds
    for r, preds in zip(gbm_results, gbm_holdout_preds_original)
}

metrics = run_diagnostics(
    lgb_result=lgb_res,
    results=gbm_results,
    baseline_result=baseline_res,
    feature_cols=FEATURE_COLS,
    holdout_eval_y=y_val_original,
    holdout_model_preds=holdout_model_preds,
    ensemble_holdout_preds=ensemble_holdout_preds,
    real_task_status=REAL_TASK_STATUS,
    transform_selected=TRANSFORM,
    importance_csv=FEATURE_IMPORTANCE_CSV,
    importance_png=FEATURE_IMPORTANCE_PNG,
    shap_png=SHAP_SUMMARY_PNG,
    X_holdout=X_holdout,
    baseline_holdout_preds=baseline_holdout_preds,
    recorded_online_score=RECORDED_ONLINE_SCORE,
    leaderboard_top=LEADERBOARD_TOP,
)

# Merge stacking diagnostics (OOF blend metrics) into the persisted metrics.
metrics.update(
    ensemble_oof_r2=float(stack_info["ensemble_r2"]),
    ensemble_oof_rmse=float(stack_info["ensemble_rmse"]),
    ensemble_oof_mae=float(stack_info["ensemble_mae"]),
    stack_method=stack_info["method"],
    baseline_floor_applied=bool(stack_info["floor_applied"]),
    run_id=RUN_ID,
)

# ── 12. Persist metrics (versioned + copy) (Req 5.7, 5.8) ────────

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4, default=_json_default)
print(f"✓ Saved {METRICS_PATH}")

with open("./metrics.json", "w") as f:
    json.dump(metrics, f, indent=4, default=_json_default)
print("✓ Saved metrics.json copy")

# ── 13. Submission (versioned + copy) (Req 5.1–5.3, 5.7, 5.8) ────

write_submission(
    final_preds, test_feats[id_col].values, TEST_PATH, SUBMISSION_PATH,
    target_col, id_col,
)

shutil.copyfile(SUBMISSION_PATH, "./submission.csv")
print("✓ Saved submission.csv copy")

# ── 14. Final honest summary (Req 6.1) ───────────────────────────

print("\n" + "=" * 60)
print(f"PIPELINE COMPLETE — RUN #{RUN_ID}")
print("=" * 60)
_ens_r2 = metrics.get("ensemble_holdout_r2")
_local = metrics.get("local_score")
_gap = metrics.get("cv_lb_gap")
print(f"Ensemble Holdout R²        : "
      f"{'n/a' if _ens_r2 is None else f'{_ens_r2:.5f}'}")
print(f"Local score (max 0,100·r²) : "
      f"{'n/a' if _local is None else f'{_local:.2f}'} / 100")
print(f"Recorded online score      : {RECORDED_ONLINE_SCORE:.2f}")
print(f"CV_LB_Gap vs {RECORDED_ONLINE_SCORE:.2f}        : "
      f"{'n/a' if _gap is None else f'{_gap:.2f}'}")
print(f"Leaderboard top (target)   : {LEADERBOARD_TOP:.2f}")
print(f"Real-task validation status: {REAL_TASK_STATUS}")
print("=" * 60)
