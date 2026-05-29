"""Property-based test for Task 7.3 (Property 11).

Validates the submission contract produced by
:func:`src.postprocess.write_submission` (Requirements 5.1, 5.2, 5.3):

  * the written submission has exactly one data row per test row,
  * it has exactly the two columns ``Index`` and ``demand``, and
  * its ``Index`` column equals the test frame's ``Index`` values in their
    original order.

The test drives the *real* ``write_submission``. A generated ``(test_df, preds)``
pair (from :func:`tests.strategies.test_frame_with_predictions`, which assigns
arbitrary unique ``Index`` values) is written to a temporary CSV that stands in
for ``test.csv``. We pass ``test_idx == test_df["Index"]`` so the function's
reindex-to-original-test-order step yields the same order as ``test_df``, and we
read the output CSV back to assert the contract on the persisted file.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings

from src.postprocess import write_submission
from tests import strategies as strat


# Feature: demand-prediction-overhaul, Property 11: Submission contract — row count, columns, and Index order
@given(pair=strat.test_frame_with_predictions())
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_submission_contract_rowcount_columns_and_index_order(pair) -> None:
    test_df, preds = pair
    test_df = test_df.reset_index(drop=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "test.csv")
        submission_path = os.path.join(tmpdir, "submission.csv")
        test_df.to_csv(test_path, index=False)

        write_submission(
            preds=preds,
            test_idx=test_df["Index"].values,
            test_path=test_path,
            submission_path=submission_path,
            target_col="demand",
            id_col="Index",
        )

        written = pd.read_csv(submission_path)

    # Row count: exactly one data row per test row (Req 5.1).
    assert len(written) == len(test_df)

    # Columns: exactly ["Index", "demand"] in that order (Req 5.2).
    assert list(written.columns) == ["Index", "demand"]

    # Index order: equals the test frame's Index in original order (Req 5.3).
    np.testing.assert_array_equal(
        written["Index"].to_numpy(),
        test_df["Index"].to_numpy(),
    )
