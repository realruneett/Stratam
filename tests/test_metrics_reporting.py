"""Unit test for Task 11.2: honest metrics reporting fields (Req 2.6, 2.7, 6.1).

This is a plain unit test (NOT one of the numbered correctness Properties
1-14), so it intentionally carries no ``# ... Property N`` tag.

It pins down the reporting contract that ``src.diagnostics.run_diagnostics``
assembles for ``metrics_N.json`` (design "RunMetrics" / "src/diagnostics.py —
honest reporting"):

  * the returned metrics dict carries every RunMetrics field
    (``baseline_holdout_r2``, ``model_holdout_r2``, ``ensemble_holdout_r2``,
    ``local_score``, ``recorded_online_score``, ``cv_lb_gap``,
    ``leaderboard_top``, ``real_task_validation_status``,
    ``transform_selected``);
  * ``local_score == max(0, 100 * ensemble_holdout_r2)`` (Req 2.6);
  * ``cv_lb_gap == |local_score − recorded_online_score|`` (Req 2.7);
  * ``leaderboard_top == 93.13`` and the real-task validation status is passed
    through (Req 6.1);
  * the negative-R² case clamps ``local_score`` to ``0`` via ``max(0, …)``;
  * the LightGBM feature-importance CSV/PNG artifacts are written.

All inputs are tiny and deterministic: a real but minimal LightGBM is fit on a
small synthetic frame (so ``feature_importances_`` exists and aligns to
``feature_cols``), and the holdout targets / predictions are short hand-built
vectors. No GPU or network is required.
"""

from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

from config import LEADERBOARD_TOP, RECORDED_ONLINE_SCORE
from src.diagnostics import run_diagnostics
from src.models import ModelResult

#: Three model names the ensemble stacks (mirrors the Task 12 wiring).
_MODEL_NAMES = ["LightGBM", "XGBoost", "CatBoost"]

#: A tiny leak-free-style feature set for the fitted LightGBM importance source.
_FEATURE_COLS = ["geohash_mean", "tod_curve_mean", "tod_sin"]


def _fit_tiny_lightgbm() -> lgb.LGBMRegressor:
    """Fit a minimal real LightGBM so ``feature_importances_`` aligns to cols.

    Uses ~20 rows, 3 features, and 5 trees on CPU — fast and deterministic.

    Returns:
        A fitted ``LGBMRegressor`` whose ``feature_importances_`` has one entry
        per column in ``_FEATURE_COLS``.
    """
    rng = np.random.default_rng(7)
    n = 20
    X = pd.DataFrame(
        {
            "geohash_mean": rng.uniform(0.0, 1.0, n),
            "tod_curve_mean": rng.uniform(0.0, 1.0, n),
            "tod_sin": np.sin(np.linspace(0.0, 2.0 * np.pi, n)),
        },
        columns=_FEATURE_COLS,
    )
    # A target with a real dependence on the features so splits are produced.
    y = 0.5 * X["geohash_mean"] + 0.3 * X["tod_curve_mean"] + 0.2 * X["tod_sin"]
    model = lgb.LGBMRegressor(
        n_estimators=5, num_leaves=7, min_child_samples=2,
        verbose=-1, seed=42,
    )
    model.fit(X, y)
    return model


def _make_result(name: str, elapsed: float = 0.1) -> ModelResult:
    """A minimal ModelResult carrying just the fields diagnostics reads."""
    return ModelResult(
        name=name,
        oof=np.zeros(1, dtype=float),
        test_preds=np.zeros(1, dtype=float),
        elapsed=elapsed,
    )


@pytest.fixture
def lgb_result() -> ModelResult:
    """A ModelResult whose ``final_model`` is a fitted LightGBM (importance src)."""
    res = _make_result("LightGBM")
    res.final_model = _fit_tiny_lightgbm()
    return res


def test_metrics_reporting_fields_positive_r2(
    lgb_result: ModelResult, tmp_path
) -> None:
    """All RunMetrics fields are present and the score formulas hold (Req 2.6, 2.7, 6.1)."""
    # ── Holdout targets + predictions (close → positive R²) ──────
    holdout_eval_y = np.linspace(0.1, 1.0, 10)
    near = {
        "LightGBM": holdout_eval_y + 0.01,
        "XGBoost": holdout_eval_y - 0.01,
        "CatBoost": holdout_eval_y + 0.005,
    }
    ensemble_holdout_preds = holdout_eval_y + 0.008
    baseline_holdout_preds = holdout_eval_y + 0.02

    results = [_make_result(name) for name in _MODEL_NAMES]
    baseline_result = _make_result("baseline")

    importance_csv = str(tmp_path / "feature_importance_test.csv")
    importance_png = str(tmp_path / "feature_importance_test.png")
    shap_png = str(tmp_path / "shap_summary_test.png")

    metrics = run_diagnostics(
        lgb_result=lgb_result,
        results=results,
        baseline_result=baseline_result,
        feature_cols=_FEATURE_COLS,
        holdout_eval_y=holdout_eval_y,
        holdout_model_preds=near,
        ensemble_holdout_preds=ensemble_holdout_preds,
        real_task_status="FAILED_NO_MATCHING_SLOTS",
        transform_selected="identity",
        importance_csv=importance_csv,
        importance_png=importance_png,
        shap_png=shap_png,
        X_holdout=None,
        baseline_holdout_preds=baseline_holdout_preds,
    )

    # ── Every RunMetrics field is present. ───────────────────────
    expected_keys = {
        "transform_selected",
        "baseline_holdout_r2",
        "model_holdout_r2",
        "ensemble_holdout_r2",
        "local_score",
        "recorded_online_score",
        "cv_lb_gap",
        "leaderboard_top",
        "real_task_validation_status",
    }
    assert expected_keys.issubset(metrics.keys())

    # ── Per-model Holdout_R2 covers all three stacked models. ────
    assert set(metrics["model_holdout_r2"].keys()) == set(_MODEL_NAMES)

    # ── local_score == max(0, 100 * ensemble_holdout_r2) (Req 2.6). ─
    expected_r2 = r2_score(holdout_eval_y, ensemble_holdout_preds)
    assert metrics["ensemble_holdout_r2"] == pytest.approx(expected_r2)
    assert metrics["local_score"] == pytest.approx(max(0.0, 100.0 * expected_r2))
    assert metrics["local_score"] > 0.0  # close preds → genuinely positive

    # ── cv_lb_gap == |local_score − recorded_online_score| (Req 2.7). ─
    assert metrics["recorded_online_score"] == pytest.approx(RECORDED_ONLINE_SCORE)
    assert metrics["cv_lb_gap"] == pytest.approx(
        abs(metrics["local_score"] - RECORDED_ONLINE_SCORE)
    )
    assert metrics["cv_lb_gap"] == pytest.approx(
        abs(metrics["local_score"] - 83.13)
    )

    # ── Constant leaderboard top + passed-through status (Req 6.1). ─
    assert metrics["leaderboard_top"] == pytest.approx(93.13)
    assert metrics["leaderboard_top"] == pytest.approx(LEADERBOARD_TOP)
    assert metrics["real_task_validation_status"] == "FAILED_NO_MATCHING_SLOTS"
    assert metrics["transform_selected"] == "identity"

    # ── Feature-importance artifacts were written. ───────────────
    assert os.path.exists(importance_csv)
    assert os.path.exists(importance_png)
    imp_df = pd.read_csv(importance_csv)
    assert set(imp_df["feature"]) == set(_FEATURE_COLS)


def test_local_score_clamped_to_zero_on_negative_r2(
    lgb_result: ModelResult, tmp_path
) -> None:
    """A far-off ensemble (negative R²) clamps local_score to 0 via max(0, …)."""
    holdout_eval_y = np.linspace(0.1, 1.0, 10)
    # Predictions strongly anti-correlated with the target → R² well below 0.
    ensemble_holdout_preds = holdout_eval_y[::-1] * 5.0 + 3.0

    # Sanity: the crafted ensemble truly has negative R².
    assert r2_score(holdout_eval_y, ensemble_holdout_preds) < 0.0

    metrics = run_diagnostics(
        lgb_result=lgb_result,
        results=[_make_result(name) for name in _MODEL_NAMES],
        baseline_result=_make_result("baseline"),
        feature_cols=_FEATURE_COLS,
        holdout_eval_y=holdout_eval_y,
        holdout_model_preds={
            name: holdout_eval_y + 0.01 for name in _MODEL_NAMES
        },
        ensemble_holdout_preds=ensemble_holdout_preds,
        real_task_status="OK",
        transform_selected="log1p",
        importance_csv=str(tmp_path / "fi.csv"),
        importance_png=str(tmp_path / "fi.png"),
        shap_png=str(tmp_path / "shap.png"),
        X_holdout=None,
        baseline_holdout_preds=holdout_eval_y + 0.02,
    )

    # local_score floored at 0 (Req 2.6) and the gap is |0 − 83.13|.
    assert metrics["local_score"] == pytest.approx(0.0)
    assert metrics["cv_lb_gap"] == pytest.approx(83.13)
    assert metrics["leaderboard_top"] == pytest.approx(93.13)
    assert metrics["real_task_validation_status"] == "OK"
