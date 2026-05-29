"""
Post-processing: inverse transform, bounding to [0, train_max],
null-prediction fallback, sanity checks, and submission CSV output.

Submission contract (see design.md "src/postprocess.py — submission
contract" and requirements 5.1-5.6):
  - ``postprocess`` inverse-transforms only when the selected transform is
    ``log1p``, clips predictions to ``[0, train_max]`` (the maximum demand in
    ``train.csv``), and replaces any residual NaN/Inf with a non-negative
    fallback equal to the global train mean. The continuous target is NOT
    rounded.
  - ``write_submission`` writes exactly one data row per test row with columns
    ``Index`` and ``demand``, ``Index`` reindexed to the original test order,
    and asserts the output is null-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _is_log1p(transform: object) -> bool:
    """Resolve whether the inverse ``expm1`` step should run.

    The design specifies ``transform`` as the string ``"log1p"`` or
    ``"identity"`` (the output of ``select_transform``). For tolerance we also
    accept a legacy boolean ``use_log_transform`` flag.
    """
    if isinstance(transform, str):
        return transform.strip().lower() == "log1p"
    return bool(transform)


def postprocess(
    preds: np.ndarray,
    transform: str,
    train_path: str,
    target_col: str,
) -> np.ndarray:
    """
    Inverse-transform, bound to ``[0, train_max]``, and replace null
    predictions with a non-negative fallback.

    Args:
        preds: Raw stacked predictions.
        transform: Selected target transform, ``"log1p"`` or ``"identity"``
            (a legacy boolean ``use_log_transform`` is also accepted).
        train_path: Path to original train.csv (for ``train_max`` and the
            global-mean fallback).
        target_col: Target column name.

    Returns:
        Cleaned prediction array: finite, continuous, and within
        ``[0, train_max]``.

    Raises:
        AssertionError: If any NaN / Inf / out-of-range value remains.
    """
    preds = np.asarray(preds, dtype=float).copy()

    # Train-derived bounds and fallback (Req 5.5, 5.6).
    orig_target = pd.read_csv(train_path)[target_col]
    train_max = float(orig_target.max())
    train_mean = float(orig_target.mean())
    # Fallback must be non-negative and within the clip range (Req 5.6).
    fallback = min(max(0.0, train_mean), train_max)

    # Step 1 — Inverse transform (only for log1p).
    if _is_log1p(transform):
        preds = np.expm1(preds)
        print("✓ Inverse log1p (expm1) applied.")
    else:
        print("No inverse transform needed (identity transform).")

    # Step 2 — Replace residual NaN/Inf with the non-negative fallback
    #          (done before the sanity asserts so they hold) (Req 5.6).
    bad_mask = ~np.isfinite(preds)
    n_bad = int(bad_mask.sum())
    if n_bad > 0:
        print(f"⚠ Replacing {n_bad} non-finite predictions with fallback {fallback:.6g}.")
        preds[bad_mask] = fallback

    # Step 3 — Clip to [0, train_max] (Req 5.4, 5.5). Integer rounding is
    #          intentionally removed: the target is continuous in ~[6e-7, 1.0]
    #          and rounding would zero out almost every prediction.
    n_neg = int((preds < 0).sum())
    n_over = int((preds > train_max).sum())
    if n_neg > 0:
        print(f"⚠ Clipping {n_neg} negative predictions up to 0.")
    if n_over > 0:
        print(f"⚠ Clipping {n_over} predictions down to train_max={train_max:.6g}.")
    preds = np.clip(preds, 0.0, train_max)

    # Step 4 — Sanity checks.
    assert not np.any(np.isnan(preds)), "NaN in final_preds"
    assert not np.any(np.isinf(preds)), "Inf in final_preds"
    assert np.all(preds >= 0), "Negative predictions remain"
    assert np.all(preds <= train_max), "Predictions exceed train_max"
    print("✓ Sanity checks passed.")

    # Step 5 — Distribution comparison.
    train_std = float(orig_target.std())
    pred_mean = float(np.mean(preds)) if preds.size else 0.0

    print(f"\n{'Metric':<12} {'Train':>14} {'Predictions':>14}")
    print("─" * 42)
    print(f"{'min':<12} {orig_target.min():>14.6g} {np.min(preds) if preds.size else 0.0:>14.6g}")
    print(f"{'max':<12} {train_max:>14.6g} {np.max(preds) if preds.size else 0.0:>14.6g}")
    print(f"{'mean':<12} {train_mean:>14.6g} {pred_mean:>14.6g}")
    print(f"{'std':<12} {train_std:>14.6g} {np.std(preds) if preds.size else 0.0:>14.6g}")

    if train_mean > 0 and abs(pred_mean - train_mean) / train_mean > 0.3:
        print("\n⚠ WARNING: Prediction mean deviates >30% from train mean.")
    else:
        print("\n✓ Prediction distribution looks reasonable.")

    return preds


def write_submission(
    preds: np.ndarray,
    test_idx: np.ndarray,
    test_path: str,
    submission_path: str,
    target_col: str,
    id_col: str | None,
) -> None:
    """
    Write the submission CSV with full verification.

    Produces exactly one data row per test row with columns ``Index`` and
    ``demand``; the ``Index`` column is populated from ``test.csv`` and
    reindexed to the original test order (Req 5.1-5.3).

    Args:
        preds: Final predictions, aligned to ``test_idx``.
        test_idx: Test ``Index`` values aligned with ``preds``.
        test_path: Path to original test.csv.
        submission_path: Output CSV path.
        target_col: Target column name (``demand``).
        id_col: ID column name (``Index``) or None.

    Returns:
        None

    Raises:
        AssertionError: On row-count mismatch or nulls.
    """
    raw_test = pd.read_csv(test_path)

    # Duplicate check on the test id column.
    if id_col and id_col in raw_test.columns:
        dups = int(raw_test[id_col].duplicated().sum())
        if dups > 0:
            print(f"⚠ WARNING: {dups} duplicate IDs in test.csv")
            raw_test = raw_test.drop_duplicates(subset=[id_col])
            print("  Duplicates removed.")

    assert len(raw_test) == len(preds), (
        f"Row count mismatch: test={len(raw_test)}, preds={len(preds)}"
    )

    # Build DataFrame and reindex to the original test.csv order (Req 5.3).
    if id_col and id_col in raw_test.columns:
        submission = pd.DataFrame({
            id_col: np.asarray(test_idx),
            target_col: np.asarray(preds),
        })
        submission = (
            submission.set_index(id_col)
            .reindex(raw_test[id_col])
            .reset_index()
        )
    else:
        submission = pd.DataFrame({target_col: np.asarray(preds)})

    submission.to_csv(submission_path, index=False)

    # Verify.
    print(f"Shape            : {submission.shape}")
    print(f"Dtypes:\n{submission.dtypes}")
    print(f"\nNull counts:\n{submission.isnull().sum()}")
    print(f"\nHead:\n{submission.head(10)}")
    print(f"\nDescribe:\n{submission.describe()}")

    assert submission.shape[0] == len(raw_test), "Row count mismatch"
    assert submission.isnull().sum().sum() == 0, "Nulls in submission"
    print(f"\n✓ {submission_path} ready")
