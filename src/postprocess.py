"""
Post-processing: inverse transform, clipping, rounding, sanity
checks, distribution comparison, and submission CSV output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def postprocess(
    preds: np.ndarray,
    use_log_transform: bool,
    train_path: str,
    target_col: str,
) -> np.ndarray:
    """
    Apply inverse transform, clip negatives, conditionally round,
    and validate predictions.

    Args:
        preds: Raw stacked predictions.
        use_log_transform: Whether log1p was applied to the target.
        train_path: Path to original train.csv (for integer check).
        target_col: Target column name.

    Returns:
        Cleaned prediction array.

    Raises:
        AssertionError: If NaN / Inf / negatives remain.
    """
    # Step 1 — Inverse transform
    if use_log_transform:
        preds = np.expm1(preds)
        print("✓ Inverse log1p (expm1) applied.")
    else:
        print("No inverse transform needed.")

    # Step 2 — Non-negativity
    n_neg = (preds < 0).sum()
    if n_neg > 0:
        print(f"⚠ Clipping {n_neg} negative predictions to 0.")
    preds = np.maximum(preds, 0)

    # Step 3 — Integer rounding
    orig_target = pd.read_csv(train_path)[target_col]
    if (orig_target.dropna() == orig_target.dropna().round()).all():
        preds = np.round(preds).astype(int)
        print("Integer target detected — rounding predictions.")
    else:
        print("Continuous target — no rounding.")

    # Step 4 — Sanity checks
    assert not np.any(np.isnan(preds)), "NaN in final_preds"
    assert not np.any(np.isinf(preds)), "Inf in final_preds"
    assert np.all(preds >= 0), "Negative predictions remain"
    print("✓ Sanity checks passed.")

    # Step 5 — Distribution comparison
    train_mean = orig_target.mean()
    train_std  = orig_target.std()
    pred_mean  = np.mean(preds)

    print(f"\n{'Metric':<12} {'Train':>12} {'Predictions':>12}")
    print("─" * 36)
    print(f"{'min':<12} {orig_target.min():>12.2f} {np.min(preds):>12.2f}")
    print(f"{'max':<12} {orig_target.max():>12.2f} {np.max(preds):>12.2f}")
    print(f"{'mean':<12} {train_mean:>12.2f} {pred_mean:>12.2f}")
    print(f"{'std':<12} {train_std:>12.2f} {np.std(preds):>12.2f}")

    if train_mean > 0 and abs(pred_mean - train_mean) / train_mean > 0.3:
        print("\n⚠ WARNING: Prediction mean deviates >30% from train mean.")
    else:
        print("\n✓ Prediction distribution looks reasonable.")

    return preds


def write_submission(
    preds: np.ndarray,
    test_path: str,
    submission_path: str,
    target_col: str,
    id_col: str | None,
) -> None:
    """
    Write submission.csv with full verification.

    Args:
        preds: Final predictions.
        test_path: Path to original test.csv.
        submission_path: Output CSV path.
        target_col: Target column name.
        id_col: ID column name (or None).

    Returns:
        None

    Raises:
        AssertionError: On row-count mismatch or nulls.
    """
    raw_test = pd.read_csv(test_path)

    # Duplicate check
    if id_col and id_col in raw_test.columns:
        dups = raw_test[id_col].duplicated().sum()
        if dups > 0:
            print(f"⚠ WARNING: {dups} duplicate IDs in test.csv")
            raw_test = raw_test.drop_duplicates(subset=[id_col])
            print("  Duplicates removed.")

    assert len(raw_test) == len(preds), (
        f"Row count mismatch: test={len(raw_test)}, preds={len(preds)}"
    )

    # Build DataFrame
    if id_col and id_col in raw_test.columns:
        submission = pd.DataFrame({
            id_col: raw_test[id_col].values,
            target_col: preds,
        })
    else:
        submission = pd.DataFrame({target_col: preds})

    submission.to_csv(submission_path, index=False)

    # Verify
    print(f"Shape            : {submission.shape}")
    print(f"Dtypes:\n{submission.dtypes}")
    print(f"\nNull counts:\n{submission.isnull().sum()}")
    print(f"\nHead:\n{submission.head(10)}")
    print(f"\nDescribe:\n{submission.describe()}")

    assert submission.shape[0] == len(raw_test), "Row count mismatch"
    assert submission.isnull().sum().sum() == 0, "Nulls in submission"
    print(f"\n✓ {submission_path} ready")
