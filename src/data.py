"""
Data loading, parsing, sorting, and target transformation.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.schema import detect_schema

# Candidate target transforms considered by ``select_transform``.
VALID_TRANSFORMS: tuple[str, ...] = ("identity", "log1p")


class TransformSelectionError(Exception):
    """Raised when no candidate target transform achieves a positive Holdout_R2.

    Per Requirement 4.4, if both the identity transform and the ``log1p``
    transform yield a Holdout_R2 less than or equal to zero, the training run
    must fail and report that no transform achieved a positive Holdout_R2.
    """


def load_data(
    train_path: str,
    test_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load train/test CSVs, detect schema, parse real time-of-day fields, and sort.

    The dataset is a two-day snapshot (``day`` ∈ {48, 49}), so the previous
    synthetic ``"2026-01-01" + day`` calendar produced constant or meaningless
    ``year/month/dayofweek/...`` fields. Instead, the raw ``timestamp`` column
    (an ``"H:M"`` clock string, NOT zero-padded — e.g. ``"0:0"``, ``"2:15"``) is
    parsed into ``hour``/``minute`` and the generalizable time-of-day fields:

      - ``abs_time = day * 1440 + hour * 60 + minute`` — strictly increasing
        absolute-minute index used only for ordering and split logic.
      - ``tod_slot = (hour * 60 + minute) // 15`` — quarter-hour slot in ``0..95``.
      - ``day`` (48 / 49) is retained as a model feature.

    This function does NOT mutate the target; transform selection is an explicit,
    reversible step handled elsewhere (see ``select_transform``).

    Args:
        train_path: Path to train.csv.
        test_path: Path to test.csv.

    Returns:
        (train_df, test_df, schema_dict)

    Raises:
        FileNotFoundError: If CSV files are missing.
        ValueError: If schema detection fails.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Detect schema on the raw frame. The timestamp column is identified by its
    # parseable "H:M" values and the name heuristic; it is left as raw strings.
    schema = detect_schema(train_df)
    ts_col = schema["timestamp_col"]
    loc_cols = schema["location_cols"]

    # ── Parse real time-of-day fields ───────────────────────────
    for df in [train_df, test_df]:
        parts = df[ts_col].astype(str).str.split(":", n=1, expand=True)
        df["hour"] = parts[0].astype(int)
        df["minute"] = parts[1].astype(int)

        minute_of_day = df["hour"] * 60 + df["minute"]
        df["abs_time"] = df["day"] * 1440 + minute_of_day
        df["tod_slot"] = minute_of_day // 15

    # ── Sort by absolute time, then geohash ─────────────────────
    loc_col = loc_cols[0] if loc_cols else None
    sort_cols = ["abs_time"] + ([loc_col] if loc_col else [])
    train_df = train_df.sort_values(sort_cols).reset_index(drop=True)
    test_df = test_df.sort_values(
        [c for c in sort_cols if c in test_df.columns]
    ).reset_index(drop=True)

    return train_df, test_df, schema


def print_diagnostics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema: dict,
) -> None:
    """
    Print dataset diagnostics to stdout.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        schema: Schema dictionary.

    Returns:
        None
    """
    ts_col = schema["timestamp_col"]
    tgt_col = schema["target_col"]
    loc_cols = schema["location_cols"]

    print("=" * 60)
    print("SCHEMA REPORT")
    print("=" * 60)
    for key, val in schema.items():
        print(f"  {key:<18s}: {val}")

    print(f"\n  Train shape     : {train_df.shape}")
    print(f"  Test  shape     : {test_df.shape}")
    print(f"  Train date range: {train_df[ts_col].min()} → {train_df[ts_col].max()}")
    print(f"  Test  date range: {test_df[ts_col].min()} → {test_df[ts_col].max()}")

    if loc_cols:
        print(f"  Unique locations: {train_df[loc_cols[0]].nunique()}")

    t = train_df[tgt_col]
    print(f"\n  Target stats:")
    print(f"    mean    : {t.mean():.4f}")
    print(f"    std     : {t.std():.4f}")
    print(f"    min     : {t.min():.4f}")
    print(f"    max     : {t.max():.4f}")
    print(f"    skew    : {t.skew():.4f}")
    print(f"    kurtosis: {t.kurtosis():.4f}")
    print(f"    nulls   : {t.isnull().sum()}")

    nulls = train_df.isnull().sum()
    print(f"\n  Null counts per column (train):")
    print(nulls[nulls > 0].to_string() if nulls.sum() > 0 else "    None")

    # Temporal overlap check
    train_ts = set(train_df[ts_col].unique())
    test_ts = set(test_df[ts_col].unique())
    overlap = train_ts & test_ts
    if overlap:
        print(f"\n  ⚠ WARNING: {len(overlap)} test timestamps overlap with train!")
    else:
        print("\n  ✓ CONFIRMED FUTURE: Test timestamps are entirely after train.")
    print("=" * 60)


def apply_target_transform(
    train_df: pd.DataFrame,
    target_col: str,
) -> bool:
    """
    Conditionally apply log1p transform to the target column in-place.

    Args:
        train_df: Training DataFrame (modified in-place).
        target_col: Name of the target column.

    Returns:
        True if log1p was applied, False otherwise.
    """
    skew = train_df[target_col].skew()
    print(f"Original target skew: {skew:.4f}")

    if abs(skew) > 1.0:
        train_df[target_col] = np.log1p(train_df[target_col].clip(lower=0))
        new_skew = train_df[target_col].skew()
        print(f"Log1p transform applied. New skew: {new_skew:.3f}")
        return True

    print(f"No transform needed. Skew: {skew:.3f}")
    return False


def apply_transform(y: np.ndarray | pd.Series, transform: str) -> np.ndarray:
    """Apply a named target transform as an explicit, reversible forward step.

    This is the forward half of the reversible transform pair used by
    ``select_transform`` and the submission pipeline. Application is explicit
    and keyed off the selected transform name rather than mutating the target
    in place.

    Args:
        y: Original-space target values.
        transform: One of ``"identity"`` or ``"log1p"``.

    Returns:
        The transformed target values as a float ``np.ndarray``. For
        ``"log1p"`` the input is clipped at a lower bound of 0 before applying
        ``np.log1p`` so the transform stays defined on the non-negative demand
        domain.

    Raises:
        ValueError: If ``transform`` is not a recognized transform name.
    """
    arr = np.asarray(y, dtype=float)
    if transform == "identity":
        return arr.copy()
    if transform == "log1p":
        return np.log1p(np.clip(arr, a_min=0.0, a_max=None))
    raise ValueError(
        f"Unknown transform {transform!r}; expected one of {VALID_TRANSFORMS}."
    )


def invert_transform(pred: np.ndarray | pd.Series, transform: str) -> np.ndarray:
    """Invert a named target transform back into the original demand space.

    This is the reverse half of the reversible transform pair. For ``"log1p"``
    the inverse is ``np.expm1``; any negative values produced by the inverse
    are clipped to 0 to respect the non-negative demand domain.

    Args:
        pred: Predictions in the transformed space.
        transform: One of ``"identity"`` or ``"log1p"``.

    Returns:
        Predictions mapped back to the original space as a float
        ``np.ndarray``.

    Raises:
        ValueError: If ``transform`` is not a recognized transform name.
    """
    arr = np.asarray(pred, dtype=float)
    if transform == "identity":
        return arr.copy()
    if transform == "log1p":
        return np.clip(np.expm1(arr), a_min=0.0, a_max=None)
    raise ValueError(
        f"Unknown transform {transform!r}; expected one of {VALID_TRANSFORMS}."
    )


def select_transform(
    y_train_part: np.ndarray | pd.Series,
    y_eval_part: np.ndarray | pd.Series,
    fit_predict_fn: Callable[[np.ndarray], np.ndarray],
) -> tuple[str, dict]:
    """Select the target transform that yields the higher Holdout_R2.

    Trains the identity and ``log1p`` candidates on the holdout training
    complement and scores them on the holdout evaluation rows, then picks the
    transform with the higher Holdout_R2 (Requirement 4.2). The choice is
    model-agnostic: ``fit_predict_fn`` is responsible for fitting a model on
    the transformed training target and returning eval-row predictions IN THE
    TRANSFORMED SPACE. ``select_transform`` then inverse-transforms those
    predictions back to the original space before scoring against the original
    ``y_eval_part`` with :func:`sklearn.metrics.r2_score`.

    The function is pure and deterministic given a deterministic
    ``fit_predict_fn``.

    Args:
        y_train_part: Original-space target values for the holdout training
            complement. Passed (after the forward transform) to
            ``fit_predict_fn``.
        y_eval_part: Original-space target values for the holdout evaluation
            rows. Used only to score predictions; never transformed for
            scoring.
        fit_predict_fn: Callback ``fit_predict_fn(y_train_transformed) ->
            y_eval_pred_transformed`` that fits a model on the transformed
            training target and returns predictions for the eval rows in the
            transformed space.

    Returns:
        A ``(chosen_transform, details)`` tuple where ``chosen_transform`` is
        ``"identity"`` or ``"log1p"`` and ``details`` is a dict containing the
        per-candidate Holdout_R2 values::

            {
                "identity_r2": float,
                "log1p_r2": float,
                "chosen": str,
                "r2_scores": {"identity": float, "log1p": float},
            }

    Raises:
        TransformSelectionError: If both candidates yield Holdout_R2 <= 0
            (Requirement 4.4).
    """
    y_eval_original = np.asarray(y_eval_part, dtype=float)

    r2_scores: dict[str, float] = {}
    for transform in VALID_TRANSFORMS:
        y_train_transformed = apply_transform(y_train_part, transform)
        eval_pred_transformed = np.asarray(
            fit_predict_fn(y_train_transformed), dtype=float
        )
        eval_pred_original = invert_transform(eval_pred_transformed, transform)
        r2_scores[transform] = float(
            r2_score(y_eval_original, eval_pred_original)
        )

    identity_r2 = r2_scores["identity"]
    log1p_r2 = r2_scores["log1p"]

    # Requirement 4.4: both transforms non-positive -> fail the training run.
    if identity_r2 <= 0.0 and log1p_r2 <= 0.0:
        raise TransformSelectionError(
            "No transform achieved a positive Holdout_R2 "
            f"(identity_r2={identity_r2:.6f}, log1p_r2={log1p_r2:.6f})."
        )

    # Requirement 4.2 / 4.3: pick the higher Holdout_R2; identity wins ties.
    chosen = "log1p" if log1p_r2 > identity_r2 else "identity"

    details = {
        "identity_r2": identity_r2,
        "log1p_r2": log1p_r2,
        "chosen": chosen,
        "r2_scores": dict(r2_scores),
    }
    return chosen, details
