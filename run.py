#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════
Flipkart Gridlock Hackathon 2.0 — Traffic Demand Prediction
═══════════════════════════════════════════════════════════════════

  python run.py

Expects  ./data/train.csv  and  ./data/test.csv
Produces ./submission.csv, ./feature_importance.csv, and plots.
"""

import warnings
import time

import numpy as np

warnings.filterwarnings("ignore")

# ── 0. Environment ─────────────────────────────────────────────

import torch
import lightgbm as lgb

print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name        : {torch.cuda.get_device_name(0)}")
assert torch.cuda.is_available(), "GPU not found. Check CUDA drivers."

lgb_version = tuple(int(x) for x in lgb.__version__.split(".")[:2])
LGB_DEVICE = "cuda" if lgb_version >= (4, 0) else "gpu"
print(f"LightGBM device : {LGB_DEVICE}")

# ── 1. Config & Seeds ──────────────────────────────────────────

from config import (
    SEED, seed_everything,
    MAX_LAGS, ROLLING_WINDOWS, HOLDOUT_FRAC,
    N_CV_FOLDS, N_ESTIMATORS, EARLY_STOPPING,
    TRAIN_PATH, TEST_PATH, SUBMISSION_PATH,
    FEATURE_IMPORTANCE_CSV, FEATURE_IMPORTANCE_PNG, SHAP_SUMMARY_PNG,
)

seed_everything(SEED)

# ── 2–3. Data Loading & Schema ─────────────────────────────────

from src.data import load_data, print_diagnostics, apply_target_transform

train_df, test_df, SCHEMA = load_data(TRAIN_PATH, TEST_PATH)
print_diagnostics(train_df, test_df, SCHEMA)

timestamp_col    = SCHEMA["timestamp_col"]
location_cols    = SCHEMA["location_cols"]
target_col       = SCHEMA["target_col"]
categorical_cols = SCHEMA["categorical_cols"]
id_col           = SCHEMA["id_col"]

# ── 4. Target Transform ───────────────────────────────────────

USE_LOG_TRANSFORM = apply_target_transform(train_df, target_col)

# ── 5. Spatial Statistics ──────────────────────────────────────

from src.spatial import compute_spatial_stats

SPATIAL_STATS = compute_spatial_stats(
    train_df, location_cols, target_col, timestamp_col
)
GLOBAL_TRAIN_MEDIAN = train_df[target_col].median()
print(f"Global train median: {GLOBAL_TRAIN_MEDIAN:.4f}")

# ── 6. Feature Engineering ────────────────────────────────────

from src.features import (
    build_compute_df, build_features,
    reduce_memory, select_feature_cols,
)

compute_df = build_compute_df(
    train_df, test_df, timestamp_col,
    location_cols, target_col, MAX_LAGS,
)
print(f"compute_df shape: {compute_df.shape}")

train_df["_is_test"] = 0

print("Building features on train_df ...")
t0 = time.perf_counter()
train_feats = build_features(
    train_df, SPATIAL_STATS, timestamp_col, location_cols,
    target_col, GLOBAL_TRAIN_MEDIAN, ROLLING_WINDOWS, categorical_cols,
)
print(f"  train_feats: {train_feats.shape} ({time.perf_counter()-t0:.1f}s)")

print("Building features on compute_df ...")
t0 = time.perf_counter()
compute_feats = build_features(
    compute_df, SPATIAL_STATS, timestamp_col, location_cols,
    target_col, GLOBAL_TRAIN_MEDIAN, ROLLING_WINDOWS, categorical_cols,
)
print(f"  compute_feats: {compute_feats.shape} ({time.perf_counter()-t0:.1f}s)")

test_feats = compute_feats[compute_feats["_is_test"] == 1].copy()
test_feats = test_feats.drop(columns=["_is_test"])
print(f"  test_feats: {test_feats.shape}")

print("Reducing memory ...")
train_feats = reduce_memory(train_feats, target_col)
test_feats  = reduce_memory(test_feats, target_col)

FEATURE_COLS = select_feature_cols(
    train_feats, test_feats, timestamp_col, target_col, id_col
)

# ── 7. Validation Split ──────────────────────────────────────

from src.validation import chronological_split

X_tr, y_tr, X_val, y_val, tr_mask, val_mask = chronological_split(
    train_feats, timestamp_col, HOLDOUT_FRAC, FEATURE_COLS, target_col,
)

X_full = train_feats[FEATURE_COLS]
y_full = train_feats[target_col]
X_test = test_feats[FEATURE_COLS]

# ── 8. Model Training ────────────────────────────────────────

from src.models import train_lightgbm, train_xgboost, train_catboost

common = dict(
    train_feats=train_feats, feature_cols=FEATURE_COLS,
    target_col=target_col, timestamp_col=timestamp_col,
    X_full=X_full, y_full=y_full, X_val=X_val, y_val=y_val,
    X_test=X_test, n_folds=N_CV_FOLDS,
    n_estimators=N_ESTIMATORS, early_stop=EARLY_STOPPING, seed=SEED,
)

lgb_res = train_lightgbm(**common, lgb_device=LGB_DEVICE)
xgb_res = train_xgboost(**common)
cat_res = train_catboost(**common, categorical_cols=categorical_cols)

results = [lgb_res, xgb_res, cat_res]

print("\n" + "=" * 60)
print("OOF SUMMARY")
print("=" * 60)
for r in results:
    print(f"  {r.name:<10s}: "
          f"{['%.5f' % x for x in r.fold_rmses]} → "
          f"mean {np.mean(r.fold_rmses):.5f}")

# ── 9. Stacking ──────────────────────────────────────────────

from src.ensemble import stack_predictions

final_preds, stack_info = stack_predictions(results, y_full.values)

# ── 10. Post-Processing ──────────────────────────────────────

from src.postprocess import postprocess, write_submission

final_preds = postprocess(
    final_preds, USE_LOG_TRANSFORM, TRAIN_PATH, target_col,
)

# ── 11. Diagnostics ──────────────────────────────────────────

from src.diagnostics import run_diagnostics

run_diagnostics(
    lgb_result=lgb_res,
    results=results,
    feature_cols=FEATURE_COLS,
    X_val=X_val, y_val=y_val,
    train_feats=train_feats,
    val_mask=val_mask,
    stack_info=stack_info,
    importance_csv=FEATURE_IMPORTANCE_CSV,
    importance_png=FEATURE_IMPORTANCE_PNG,
    shap_png=SHAP_SUMMARY_PNG,
)

# ── 12. Submission ───────────────────────────────────────────

write_submission(
    final_preds, TEST_PATH, SUBMISSION_PATH, target_col, id_col,
)

print("=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
