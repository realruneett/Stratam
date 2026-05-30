"""
Global configuration: seeds, hyperparameters, and constants.
"""

import os
import random

import numpy as np

# ─── Reproducibility ────────────────────────────────────────────

SEED = 42


def seed_everything(seed: int = SEED) -> None:
    """Lock down all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ─── Test-data time-of-day window ───────────────────────────────
# Quarter-hour slots covering the test daytime window 2:15–13:45 inclusive.

TEST_TOD_SLOT_MIN: int = 9
TEST_TOD_SLOT_MAX: int = 55


# ─── Leak-free target encoding ──────────────────────────────────

# FIX 3: Reduced from 20.0 → 5.0.
# alpha=20 was pulling high-demand geohashes (mean=0.95, N=55) down to 0.72.
# alpha=5 preserves the true per-geohash signal while still smoothing sparse ones.
TE_SMOOTHING_ALPHA: float = 5.0

# FIX (interaction encoder smoothing): Reduced from 10.0 → 5.0 for same reason.
IE_SMOOTHING_ALPHA: float = 5.0

TE_N_FOLDS: int = 5


# ─── Modeling / generalization-first regime ─────────────────────

N_CV_FOLDS: int = 5
N_ESTIMATORS: int = 5000
EARLY_STOPPING: int = 100


# ─── Honest performance reporting ───────────────────────────────

RECORDED_ONLINE_SCORE: float = 90.69774
LEADERBOARD_TOP: float = 100.0


# ─── Paths ──────────────────────────────────────────────────────

TRAIN_PATH = "./data/train.csv"
TEST_PATH = "./data/test.csv"
SUBMISSION_PATH = "./submission.csv"
FEATURE_IMPORTANCE_CSV = "./feature_importance.csv"
FEATURE_IMPORTANCE_PNG = "./feature_importance.png"
SHAP_SUMMARY_PNG = "./shap_summary.png"