"""Property-based test for Task 7.4 (Property 12).

Validates that :func:`src.postprocess.postprocess` bounds every prediction to
``[0, train_max]`` (Requirements 5.4, 5.5), where ``train_max`` is the maximum
demand observed in ``train.csv``.

Raw prediction vectors are generated with values spanning the negative and
over-range region (``min_value=-5``, ``max_value=5``) via
:func:`tests.strategies.prediction_array`, then run through the real
``postprocess`` with the identity transform so the raw values map directly and
the bounds are meaningful. Every output value must be finite and within
``[0, train_max]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings

from src.postprocess import postprocess
from tests import strategies as strat

_TRAIN_PATH = "./data/train.csv"
_TARGET_COL = "demand"

# train_max is a fixed property of the real train.csv; read it once.
_TRAIN_MAX = float(pd.read_csv(_TRAIN_PATH)[_TARGET_COL].max())


# Feature: demand-prediction-overhaul, Property 12: Predictions are bounded to [0, train_max]
@given(preds=strat.prediction_array(min_value=-5.0, max_value=5.0))
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_predictions_are_bounded_to_zero_train_max(preds: np.ndarray) -> None:
    out = postprocess(
        preds,
        transform="identity",
        train_path=_TRAIN_PATH,
        target_col=_TARGET_COL,
    )

    assert out.shape == preds.shape
    # Every postprocessed value is finite and within [0, train_max].
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)
    assert np.all(out <= _TRAIN_MAX)
