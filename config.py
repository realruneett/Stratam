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


# ─── Hyperparameters ────────────────────────────────────────────

MAX_LAGS: int = 168                     # 1 week back
ROLLING_WINDOWS: list = [3, 6, 12, 24, 168]
HOLDOUT_FRAC: float = 0.15             # chronological holdout
N_CV_FOLDS: int = 5                    # OOF folds
N_ESTIMATORS: int = 5000               # max trees; early stopping cuts earlier
EARLY_STOPPING: int = 100


# ─── Paths ──────────────────────────────────────────────────────

TRAIN_PATH = "./data/train.csv"
TEST_PATH = "./data/test.csv"
SUBMISSION_PATH = "./submission.csv"
FEATURE_IMPORTANCE_CSV = "./feature_importance.csv"
FEATURE_IMPORTANCE_PNG = "./feature_importance.png"
SHAP_SUMMARY_PNG = "./shap_summary.png"
