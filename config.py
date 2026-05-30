"""
Global configuration: seeds, hyperparameters, and constants.
"""

import os
import random

import numpy as np

# ─── Reproducibility ────────────────────────────────────────────

SEED = 42


def seed_everything(seed: int = SEED) -> None:
    """Lock down all random number generators for reproducibility.

    This is the single seeding entry point for the pipeline (Req 6.4):
    every stochastic step must be initialized via this function from the
    one configured non-negative ``SEED``.

    Args:
        seed: Integer seed value.

    Returns:
        None
    """
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
# tod_slot = (hour * 60 + minute) // 15, so 2:15 -> 9 and 13:45 -> 55.

TEST_TOD_SLOT_MIN: int = 9             # floor((2*60 + 15) / 15)
TEST_TOD_SLOT_MAX: int = 55            # floor((13*60 + 45) / 15)


# ─── Leak-free target encoding ──────────────────────────────────

TE_SMOOTHING_ALPHA: float = 20.0       # Bayesian prior weight:
#                                        (sum + prior_mean*alpha) / (count + alpha)
TE_N_FOLDS: int = 5                    # out-of-fold folds for in-training encoding


# ─── Modeling / generalization-first regime ─────────────────────

N_CV_FOLDS: int = 5                    # OOF folds (superseded by TE_N_FOLDS)
N_ESTIMATORS: int = 5000               # max trees; early stopping cuts earlier
EARLY_STOPPING: int = 100


# ─── Honest performance reporting ───────────────────────────────

RECORDED_ONLINE_SCORE: float = 83.13   # latest measured leaderboard score
LEADERBOARD_TOP: float = 93.13         # current leaderboard top (target)


# ─── Paths ──────────────────────────────────────────────────────

TRAIN_PATH = "./data/train.csv"
TEST_PATH = "./data/test.csv"
SUBMISSION_PATH = "./submission.csv"
FEATURE_IMPORTANCE_CSV = "./feature_importance.csv"
FEATURE_IMPORTANCE_PNG = "./feature_importance.png"
SHAP_SUMMARY_PNG = "./shap_summary.png"
