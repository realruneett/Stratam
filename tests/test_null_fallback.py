"""Property-based test for Task 7.5 (Property 13).

Validates that :func:`src.postprocess.postprocess` replaces null predictions
(``NaN``/``Inf``) with a defined non-negative fallback (Requirement 5.6).

Prediction vectors guaranteed to contain at least one ``NaN``/``Inf`` are
generated via :func:`tests.strategies.prediction_array_with_nans` and run
through the real ``postprocess`` with the identity transform. The fallback is
the train mean clamped into ``[0, train_max]``, so after postprocessing the
output must be entirely finite, non-negative, and within ``[0, train_max]`` —
in particular the positions that were non-finite in the input must now be
finite and in range.
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


# Feature: demand-prediction-overhaul, Property 13: Null predictions are replaced by a non-negative fallback
@given(preds=strat.prediction_array_with_nans())
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_null_predictions_replaced_by_nonnegative_fallback(preds: np.ndarray) -> None:
    # The strategy guarantees at least one non-finite value in the input.
    bad_mask = ~np.isfinite(preds)
    assert bad_mask.any(), "strategy must inject at least one NaN/Inf"

    out = postprocess(
        preds,
        transform="identity",
        train_path=_TRAIN_PATH,
        target_col=_TARGET_COL,
    )

    assert out.shape == preds.shape
    # Output is null-free (no NaN/Inf) and non-negative (Req 5.6).
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)
    # Fallback is the train mean clamped to [0, train_max], so output stays
    # within the clip range.
    assert np.all(out <= _TRAIN_MAX)

    # The positions that were NaN/Inf in the input are now finite and in range.
    replaced = out[bad_mask]
    assert np.all(np.isfinite(replaced))
    assert np.all(replaced >= 0.0)
    assert np.all(replaced <= _TRAIN_MAX)
