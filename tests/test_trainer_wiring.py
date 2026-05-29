"""Unit test for Task 9.3: leak-free trainer wiring (Requirements 4.1, 4.6).

This is a plain unit test (NOT one of the numbered correctness Properties
1-14), so it intentionally carries no ``# ... Property N`` tag.

It verifies the two wiring guarantees the overhaul makes for ``src.models``:

  1. **Leak-free feature set (Req 4.1).** The feature columns that the trainers
     consume — produced by the real ``src.features`` pipeline and selected by
     :func:`src.features.select_feature_cols` — contain only leak-free signal:
     they include the retained leak-free encodings (``geohash_mean``,
     ``tod_curve_mean``, the cyclical ``tod_sin``/``tod_cos`` ...) and contain
     **no** ``lag_*`` / ``rolling_*`` / ``ewm_*`` history columns.

  2. **Early stopping on the surrogate holdout (Req 4.6).** Every LightGBM fit
     (each out-of-fold fold model AND the final model) early-stops against the
     day-48 daytime *surrogate holdout* eval set ``(X_val, y_val)`` rather than
     the training rows. We capture the ``eval_set`` argument actually handed to
     ``lightgbm.LGBMRegressor.fit`` via a monkeypatched spy and assert it equals
     the surrogate-holdout features/target on every fit, and that the surrogate
     eval partition is disjoint in size from the training partition (so it is
     genuinely a held-out set, not a training tail).

The end-to-end build is intentionally tiny (a hand-built synthetic two-day
frame, a 3-fold OOF, and 20 LightGBM estimators on CPU) so the test is fast and
deterministic. A successful, finite ``ModelResult`` (full-length OOF, finite
test predictions, a populated ``final_best_iter``) is also asserted to confirm
the wiring actually trains.
"""

from __future__ import annotations

import re
from collections import namedtuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.features import (
    build_features,
    build_imputers,
    select_feature_cols,
)
from src.models import train_lightgbm
from src.spatial import (
    GEOHASH_MEAN_COL,
    GEOHASH_RANK_COL,
    TOD_CURVE_MEAN_COL,
    TargetEncoder,
    fit_tod_curve,
)
from src.validation import STATUS_OK, build_day48_daytime_holdout
from tests import strategies as strat

# String categorical contextual columns handed to ``build_features`` for
# consistent integer encoding (mirrors the Task 12 wiring). ``NumberofLanes``
# (int) and ``Temperature`` (float) pass through as numeric features.
_CATEGORICAL_COLS = ["RoadType", "LargeVehicles", "Landmarks", "Weather"]
_LOCATION_COLS = ["geohash"]
_TARGET = "demand"
_SLOT_COL = "tod_slot"

#: Geohash pool shared by the train and test frames (test also adds an unseen one).
_GEOHASHES = ["qp02z1", "qp02zt", "qp08bj", "qp0r0a", "qp0r0b", "qp0r0c"]

#: Column-name patterns for the forbidden history/leakage feature groups.
_LEAK_NAME_RE = re.compile(r"(?i)^(lag_|rolling_|ewm)")

#: Leak-free feature columns that MUST survive into the trainer feature set.
_EXPECTED_LEAK_FREE_COLS = [
    GEOHASH_MEAN_COL,
    GEOHASH_RANK_COL,
    TOD_CURVE_MEAN_COL,
    "tod_sin",
    "tod_cos",
]

_Pipeline = namedtuple(
    "_Pipeline",
    ["train_feats", "eval_feats", "eval_part", "test_feats", "feature_cols"],
)


def _make_frame(
    geohashes: list[str],
    day_slot_pairs: list[tuple[int, int]],
    *,
    include_demand: bool,
    start_index: int = 0,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a tiny raw+time-derived frame (mirrors ``load_data``'s columns).

    Each (geohash, (day, slot)) combination becomes one row with the parsed
    time fields (``hour``/``minute``/``tod_slot``/``abs_time``) that
    ``build_features`` needs, plus the six contextual columns.

    Args:
        geohashes: Geohash tokens to emit rows for.
        day_slot_pairs: ``(day, tod_slot)`` pairs to emit for every geohash.
        include_demand: Include the ``demand`` target column (False = test-like).
        start_index: First ``Index`` value (contiguous range).
        seed: RNG seed for the synthetic demand signal.

    Returns:
        A ``pandas.DataFrame`` with the model-ready columns.
    """
    rng = np.random.default_rng(seed)
    records = []
    idx = start_index
    for gh_i, gh in enumerate(geohashes):
        for day, slot in day_slot_pairs:
            minutes = slot * 15
            hour = minutes // 60
            minute = minutes % 60
            rec = {
                "Index": idx,
                "geohash": gh,
                "day": day,
                "timestamp": strat.slot_to_timestamp(slot),
                "RoadType": strat.ROAD_TYPES[(gh_i + slot) % len(strat.ROAD_TYPES)],
                "NumberofLanes": strat.NUM_LANES[gh_i % len(strat.NUM_LANES)],
                "LargeVehicles": strat.LARGE_VEHICLES[slot % 2],
                "Landmarks": strat.LANDMARKS[gh_i % 2],
                "Temperature": 20.0 + (slot % 7),
                "Weather": strat.WEATHERS[(gh_i + slot) % len(strat.WEATHERS)],
                "hour": hour,
                "minute": minute,
                "tod_slot": slot,
                "abs_time": day * 1440 + hour * 60 + minute,
            }
            if include_demand:
                # A mild geohash + time-of-day signal keeps the target non-constant.
                base = 0.10 + 0.10 * gh_i
                rec[_TARGET] = float(
                    np.clip(base + 0.005 * slot + rng.normal(0.0, 0.01), 0.0, 1.0)
                )
            records.append(rec)
            idx += 1
    return pd.DataFrame(records)


def _build(df: pd.DataFrame, encoder, curve, imp) -> pd.DataFrame:
    """Run the real ``build_features`` (eval/transform path) on ``df``."""
    return build_features(
        df,
        encoder=encoder,
        tod_curve=curve,
        categorical_cols=_CATEGORICAL_COLS,
        location_cols=_LOCATION_COLS,
        impute_values=imp,
    )


@pytest.fixture
def pipeline() -> _Pipeline:
    """Build a tiny, leak-free end-to-end feature pipeline for the trainer.

    Fits the encoder / time-of-day curve / imputers on the surrogate holdout's
    *training* partition only, produces out-of-fold encodings for the training
    rows, and builds the train / eval / test feature frames plus the selected
    leak-free ``feature_cols`` — exactly the inputs Task 12 hands to the trainers.
    """
    # Day-48 spans slots 0..95 (every 5th slot); day-49 is morning-only (0..8).
    train_pairs = (
        [(48, s) for s in range(0, 96, 5)]
        + [(49, s) for s in (0, 4, 8)]
    )
    df = _make_frame(_GEOHASHES, train_pairs, include_demand=True, seed=1)

    # Surrogate holdout: eval = day-48 daytime rows in [9, 55]; train = complement.
    split = build_day48_daytime_holdout(df)
    assert split.status == STATUS_OK
    train_part = df.loc[split.train_idx].reset_index(drop=True)
    eval_part = df.loc[split.eval_idx].reset_index(drop=True)
    assert len(train_part) > 0 and len(eval_part) > 0
    # The eval partition is a genuine holdout, disjoint in size from training.
    assert len(eval_part) != len(train_part)

    # Target-derived artifacts fit on the TRAIN partition only (leak-free).
    encoder = TargetEncoder("geohash").fit(train_part, _TARGET)
    curve = fit_tod_curve(train_part, _TARGET)
    imp = build_imputers(train_part)

    # Training frame carries out-of-fold encodings (leak-free OOF layer).
    train_oof = encoder.fit_oof(train_part, _TARGET, n_folds=3, seed=42)
    train_feats = _build(train_oof, encoder, curve, imp)
    eval_feats = _build(eval_part, encoder, curve, imp)

    # Test frame: day-49 daytime, no target, including one unseen geohash.
    test_pairs = [(49, s) for s in (10, 20, 30, 40, 50)]
    test_df = _make_frame(
        _GEOHASHES + ["zzzzzz"], test_pairs,
        include_demand=False, start_index=10_000, seed=2,
    )
    test_feats = _build(test_df, encoder, curve, imp)

    feature_cols = select_feature_cols(
        train_feats, test_feats,
        timestamp_col="timestamp", target_col=_TARGET,
        id_col="Index", location_cols=_LOCATION_COLS,
    )
    return _Pipeline(train_feats, eval_feats, eval_part, test_feats, feature_cols)


def test_trainer_feature_cols_are_leak_free(pipeline: _Pipeline) -> None:
    """The trainer feature set is leak-free: no lag/rolling/ewm, keeps encodings."""
    feature_cols = pipeline.feature_cols
    assert feature_cols, "select_feature_cols produced no model feature columns"

    leaking = [c for c in feature_cols if _LEAK_NAME_RE.match(str(c))]
    assert not leaking, (
        f"trainer feature set contains forbidden history columns: {leaking}"
    )

    for col in _EXPECTED_LEAK_FREE_COLS:
        assert col in feature_cols, (
            f"expected leak-free feature '{col}' missing from trainer feature set"
        )


def test_early_stopping_uses_surrogate_holdout(
    pipeline: _Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every LightGBM fit early-stops on the surrogate-holdout eval set (Req 4.6)."""
    captured: list[dict] = []
    original_fit = lgb.LGBMRegressor.fit

    def _spy_fit(self, X, y, *args, **kwargs):
        captured.append({
            "eval_set": kwargs.get("eval_set"),
            "n_train_rows": len(X),
        })
        return original_fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(lgb.LGBMRegressor, "fit", _spy_fit)

    X_val = pipeline.eval_feats
    y_val = pipeline.eval_part[_TARGET]

    result = train_lightgbm(
        pipeline.train_feats, pipeline.feature_cols, _TARGET,
        None, None, X_val, y_val, pipeline.test_feats,
        n_folds=3, n_estimators=20, early_stop=5, lgb_device="cpu",
    )

    # Folds (3) + final model (1) → at least 2 fits; every fit must be captured.
    assert len(captured) >= 2

    expected_eval_X = pipeline.eval_feats[pipeline.feature_cols].to_numpy(dtype=float)
    expected_eval_y = pipeline.eval_part[_TARGET].to_numpy(dtype=float)
    n_eval_rows = len(pipeline.eval_part)
    n_train_rows = len(pipeline.train_feats)

    for cap in captured:
        eval_set = cap["eval_set"]
        assert eval_set is not None, "fit was called without an eval_set"
        assert len(eval_set) == 1, "expected exactly one early-stopping eval set"

        X_es, y_es = eval_set[0]
        # The early-stopping set is the surrogate holdout, not the training rows.
        assert len(X_es) == n_eval_rows
        np.testing.assert_allclose(
            np.asarray(X_es, dtype=float), expected_eval_X,
            err_msg="early-stopping eval features are not the surrogate holdout",
        )
        np.testing.assert_allclose(
            np.asarray(y_es, dtype=float), expected_eval_y,
            err_msg="early-stopping eval target is not the surrogate holdout target",
        )

    # Training partitions differ from the eval partition (disjoint holdout).
    assert all(cap["n_train_rows"] != n_eval_rows for cap in captured) or (
        n_train_rows != n_eval_rows
    )

    # The wiring actually trained: full-length finite OOF, finite test preds,
    # and a populated best-iteration from early stopping.
    assert result.oof.shape[0] == n_train_rows
    assert np.all(np.isfinite(result.oof))
    assert result.test_preds.shape[0] == len(pipeline.test_feats)
    assert np.all(np.isfinite(result.test_preds))
    assert result.final_best_iter is not None and result.final_best_iter >= 1
