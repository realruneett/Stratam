"""Property-based test for the leakage-free feature set (Task 5.4, Property 1).

This file implements the single property-based test for design Property 1 of the
demand-prediction-overhaul spec:

    "No feature depends on a row's own or neighboring shifted target."

Property 1 is the central leakage invariant of the overhaul. The whole point of
the refactor is that the feature set produced by
:func:`src.features.build_features` must rely only on signal that genuinely
exists at prediction time, and must therefore contain **no** feature whose values
are a within-geohash *shift / rolling / EWM* derivation of the ``demand`` target.
Equivalently, ``build_features`` must succeed and produce the **same** feature
columns and values whether the target is present, absent, or all-NaN.

The test validates three complementary sub-claims (all in the single test below):

  (a) No leakage columns *by name*: none of the produced columns match the
      forbidden ``lag_*`` / ``rolling_*`` / ``ewm_*`` patterns, and the dropped
      synthetic-calendar names (``year``, ``month``, ``day_of_week``, ...) are
      absent.

  (b) No feature column *value-equals* a within-geohash shift / rolling / EWM of
      the target. For each geohash group (sorted by absolute time) we compute
      candidate "leak" series — ``shift(1)``, ``shift(2)``,
      ``rolling(2..3).mean().shift(1)``, and an ``ewm`` of the target — and assert
      that no produced numeric feature column is element-wise close to any of
      them over the rows where the candidate is defined. Comparisons are made
      only where the candidate is non-NaN, require more than one such row, and are
      skipped for series without enough distinct variation so a coincidentally
      low-cardinality feature cannot masquerade as a shifted target. The
      legitimately-retained leak-free *aggregate* encodings (the ``geohash_*``
      encodings and ``tod_curve_mean``) AND the contextual categorical-code
      columns are excluded from this value comparison (see
      ``_VALUE_COMPARISON_EXCLUDED_COLS`` for the full rationale). They are still
      covered by sub-claims (a) and (c).

  (c) Target-absent invariance: ``build_features`` is called again on a copy with
      ``demand`` dropped and on a copy with ``demand`` all-NaN, using the SAME
      pre-fitted encoder / curve / imputers. The produced feature column set
      (excluding the target column) and the non-target feature VALUES must be
      identical across all three frames — any shift/rolling/EWM feature would
      necessarily move with the target, so identical values is direct evidence of
      leak-freedom (Requirement 1.6 / design Property 1).

The shared strategies emit *raw-schema* frames carrying a ``day`` column and a
raw ``timestamp`` (``"H:M"``) string but no parsed ``hour`` / ``minute`` /
``tod_slot``. ``build_features`` needs those, so we reproduce ``load_data``'s
derivation (``tod_slot = (hour*60 + minute) // 15``,
``abs_time = day*1440 + hour*60 + minute``) in the test before building features.

Validates: Requirements 1.1, 1.2, 1.3, 1.6
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings

from src.features import (
    CONTEXTUAL_COLS,
    build_features,
    build_imputers,
    select_feature_cols,
)
from src.spatial import (
    ENCODING_COLS,
    TOD_CURVE_MEAN_COL,
    TargetEncoder,
    fit_tod_curve,
)
from tests import strategies as strat

_ATOL = 1e-9

#: Minimum number of distinct finite values a series must carry to be treated as
#: meaningful evidence of a *shifted/rolling/EWM target* leak. See
#: :func:`_is_informative` for why two distinct values is not enough.
_MIN_DISTINCT_VALUES = 3

#: Column-name patterns that would indicate a re-introduced history/leakage
#: feature group (lag_*, rolling_*, ewm_*). Property 1 forbids all of them.
_LEAK_NAME_RE = re.compile(r"(?i)^(lag_|rolling_|ewm)")

#: The legitimately-retained leak-free aggregate encodings (Req 1.4, 1.5): the
#: per-geohash target encodings and the day-48 time-of-day demand curve. These
#: are derived from *group aggregates* of other rows' targets — NOT from a row's
#: own or a neighboring shifted target — and are explicitly retained by the
#: requirements and validated by design Properties 2 and 3. They are therefore
#: excluded from the value-equality shift/rolling/EWM comparison in sub-claim
#: (b): a group mean can coincidentally align with a positional ``shift(k)`` in a
#: tiny degenerate frame (e.g. ``tod_curve_mean``'s per-slot repetition matching
#: ``demand.shift(2)`` when there are two day-48 slots followed by the same two
#: day-49 slots), which is a positional coincidence, not neighbor-shift leakage.
#: The name check (sub-claim a) and the target-independence check (sub-claim c)
#: still cover these columns.
_LEAK_FREE_ENCODING_COLS = set(ENCODING_COLS) | {TOD_CURVE_MEAN_COL}

#: The six contextual columns (Requirement 3.4). ``build_features`` integer-
#: encodes the *string* categoricals (``RoadType``, ``LargeVehicles``,
#: ``Landmarks``, ``Weather``) in place to category codes, and keeps the numeric
#: ``NumberofLanes`` / ``Temperature`` as-is. NONE of the six are derived from
#: the ``demand`` target: they are raw contextual attributes (or stable integer
#: encodings thereof), provably target-independent.
#:
#: They are excluded from sub-claim (b)'s value-equality comparison because the
#: low-cardinality categorical-code columns are prone to *coincidental positional
#: matches* with a shifted near-binary target. Concretely, on a tiny Hypothesis
#: frame a binary code column (e.g. ``Landmarks`` Yes/No → ``[1, 0, ...]`` or
#: ``LargeVehicles`` → ``[0, 1, ...]``) can equal ``demand.shift(1)`` when the
#: target itself collapses to two values like ``[0., 1.]`` within a geohash —
#: ``np.allclose([0., 1.], [0., 1.])`` is ``True`` even though there is no
#: dependence on the target whatsoever. That is a degenerate coincidence, not
#: neighbor-shift leakage.
#:
#: Excluding all six (rather than only the four string categoricals) is the
#: principled choice: none are target-derived, so removing them from the *value*
#: comparison cannot hide a real leak. Their leak-freedom remains rigorously
#: guaranteed by sub-claim (a) (the name check forbids ``lag_*``/``rolling_*``/
#: ``ewm_*`` columns) and especially by sub-claim (c) (target-absent invariance:
#: every non-target feature, these included, is byte-identical whether ``demand``
#: is present, dropped, or all-NaN — impossible for any shift/rolling/EWM of the
#: target).
_CONTEXTUAL_FEATURE_COLS = set(CONTEXTUAL_COLS)

#: Columns excluded from sub-claim (b)'s value-equality comparison: the leak-free
#: group-aggregate encodings plus the target-independent contextual columns.
_VALUE_COMPARISON_EXCLUDED_COLS = _LEAK_FREE_ENCODING_COLS | _CONTEXTUAL_FEATURE_COLS

#: Synthetic-calendar feature names that the overhaul removed entirely; their
#: presence would signal the deleted ``"2026-01-01" + day`` calendar block.
_REMOVED_CALENDAR_COLS = {
    "year",
    "month",
    "day_of_month",
    "day_of_week",
    "dayofweek",
    "day_of_year",
    "dayofyear",
    "week_of_year",
    "weekofyear",
    "quarter",
    "is_weekend",
}


def _derive_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add the parsed time columns ``load_data`` would produce.

    Mirrors ``src.data.load_data``: parse the raw ``"H:M"`` timestamp into
    ``hour`` / ``minute``, then derive ``tod_slot = (hour*60 + minute) // 15``
    (0..95) and the ordering-only ``abs_time = day*1440 + hour*60 + minute``.
    """
    out = df.reset_index(drop=True).copy()
    parts = out["timestamp"].astype(str).str.split(":", n=1, expand=True)
    out["hour"] = parts[0].astype(int)
    out["minute"] = parts[1].astype(int)
    minute_of_day = out["hour"] * 60 + out["minute"]
    out["tod_slot"] = minute_of_day // 15
    out["abs_time"] = out["day"] * 1440 + minute_of_day
    return out


def _string_categorical_cols(df: pd.DataFrame) -> list[str]:
    """The contextual columns that are string categoricals.

    (``RoadType``, ``LargeVehicles``, ``Landmarks``, ``Weather`` — the numeric
    ``NumberofLanes`` / ``Temperature`` are excluded from integer encoding.)
    """
    return [c for c in CONTEXTUAL_COLS if not pd.api.types.is_numeric_dtype(df[c])]


def _distinct_finite_count(values: np.ndarray, atol: float = _ATOL) -> int:
    """Number of distinct finite values, treating within-``atol`` values as one.

    Adjacent sorted values whose gap does not exceed ``atol`` are collapsed into
    a single distinct value, so floating-point jitter below tolerance is not
    counted as genuine variation.
    """
    finite_vals = np.sort(values[np.isfinite(values)])
    if finite_vals.size == 0:
        return 0
    # Count the boundaries where consecutive sorted values differ by > atol,
    # i.e. the number of distinct clusters of (near-)equal values.
    return 1 + int(np.count_nonzero(np.diff(finite_vals) > atol))


def _is_informative(values: np.ndarray) -> bool:
    """Whether a finite-valued series varies enough to evidence a *target* leak.

    A series is informative only when it carries at least ``_MIN_DISTINCT_VALUES``
    (3) distinct finite values, distinctness judged within ``_ATOL`` (see
    :func:`_distinct_finite_count`).

    This ">=3 distinct values" guard is deliberately stricter than the earlier
    ">=2 finite values with peak-to-peak range > _ATOL" check, and exists to
    eliminate *binary-coincidence* false positives. A genuine shift / rolling /
    EWM leak of the continuous ``demand`` target reproduces a near-continuous
    series, which therefore exposes many distinct values; requiring >=3 distinct
    values consequently never hides a real leak. What it does rule out is the
    degenerate Hypothesis frame where a low-cardinality column collapses to two
    values (e.g. ``[0., 1.]``) that positionally coincide with a near-binary
    ``demand.shift(k)`` — a coincidence, not neighbor-shift leakage.

    Applied to the leak *candidate* series (and, because the same guard fronts
    the per-column early-skip in sub-claim (b), to the compared column too): if
    the candidate has >=3 distinct values and a column equals it element-wise,
    that column necessarily also varies across >=3 values, so no real leak is
    skipped.
    """
    return _distinct_finite_count(values) >= _MIN_DISTINCT_VALUES


def _build(
    df: pd.DataFrame,
    encoder: TargetEncoder,
    curve: pd.Series,
    imp: dict,
    categorical_cols: list[str],
    location_cols: list[str],
) -> pd.DataFrame:
    """Run the real ``build_features`` on the eval/transform path.

    Passing a fitted ``encoder`` (and no pre-computed encoding columns) drives
    the ``encoder.transform`` path (``oof_encoded=False``), which is the path the
    target-absent invariance check needs.
    """
    return build_features(
        df,
        encoder=encoder,
        tod_curve=curve,
        categorical_cols=categorical_cols,
        location_cols=location_cols,
        impute_values=imp,
        oof_encoded=False,
    )


# Feature: demand-prediction-overhaul, Property 1: No feature depends on a row's own or neighboring shifted target
@given(df=strat.two_day_frame(max_rows_per_day=18))
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_feature_set_is_leakage_free(df: pd.DataFrame) -> None:
    df = _derive_time(df)

    categorical_cols = _string_categorical_cols(df)
    location_cols = ["geohash"]

    # Fit the target-derived artifacts ONCE on the full frame. These are passed
    # *in* to build_features (which never reads the frame's own target), so the
    # same fitted artifacts are reused for the target-present / absent / all-NaN
    # builds below — isolating the target column as the only difference.
    encoder = TargetEncoder("geohash").fit(df, target_col="demand")
    curve = fit_tod_curve(df, target_col="demand")
    imp = build_imputers(df)

    feats_with = _build(df, encoder, curve, imp, categorical_cols, location_cols)

    # The model-input feature columns (numeric; excludes demand / Index / raw
    # geohash / timestamp / abs_time).
    feat_cols = select_feature_cols(
        feats_with, feats_with, "timestamp", "demand", "Index", location_cols
    )
    assert feat_cols, "build_features produced no model feature columns"

    # ── Sub-claim (a): no leakage / removed-calendar columns by NAME ──
    leaking_names = [c for c in feats_with.columns if _LEAK_NAME_RE.match(str(c))]
    assert not leaking_names, (
        f"build_features produced forbidden history/leakage columns: {leaking_names}"
    )
    present_calendar = _REMOVED_CALENDAR_COLS.intersection(map(str, feats_with.columns))
    assert not present_calendar, (
        f"build_features produced removed synthetic-calendar columns: {present_calendar}"
    )

    # ── Sub-claim (b): no feature VALUE-equals a shifted/rolling/EWM target ──
    # For each geohash group (sorted by absolute time), compute within-group
    # shift(1)/shift(2)/rolling(2..3).mean().shift(1)/ewm of demand and assert no
    # feature column equals it over the informative (>=3 distinct finite) entries.
    #
    # Excluded from this value comparison (see _VALUE_COMPARISON_EXCLUDED_COLS):
    #   • the leak-free aggregate encodings (geohash_* and tod_curve_mean) — group
    #     aggregates of OTHER rows' targets (Req 1.4/1.5) whose per-group/per-slot
    #     repetition can positionally coincide with a shift(k) in a tiny frame; and
    #   • the contextual categorical-code columns (RoadType, LargeVehicles,
    #     Landmarks, Weather and the numeric Temperature/NumberofLanes) — integer
    #     encodings of raw context, provably target-independent, whose low
    #     cardinality makes them prone to coincidental positional matches with a
    #     shifted near-binary target (e.g. a binary code [0.,1.] equalling
    #     demand.shift(1) when demand collapses to [0.,1.]).
    # Neither group is target-derived, so excluding them cannot hide a real leak;
    # their leak-freedom is rigorously covered by sub-claim (a)'s name check and
    # sub-claim (c)'s target-absent invariance.
    comparison_cols = [
        c for c in feat_cols if c not in _VALUE_COMPARISON_EXCLUDED_COLS
    ]
    for _gh, group in feats_with.groupby("geohash", sort=False):
        group = group.sort_values("abs_time", kind="stable")
        demand = group["demand"].astype(float)

        leak_candidates = {
            "shift1": demand.shift(1).to_numpy(),
            "shift2": demand.shift(2).to_numpy(),
            "rolling2_mean_shift1": demand.rolling(window=2).mean().shift(1).to_numpy(),
            "rolling3_mean_shift1": demand.rolling(window=3).mean().shift(1).to_numpy(),
            "ewm": demand.ewm(span=2, adjust=False).mean().shift(1).to_numpy(),
        }

        for col in comparison_cols:
            col_vals = group[col].astype(float).to_numpy()
            # A feature without enough distinct variation carries no per-row
            # signal that could encode a *varying* shifted target — skip it to
            # avoid spurious low-cardinality equality matches.
            if not _is_informative(col_vals):
                continue
            col_finite = np.isfinite(col_vals)
            for kind, series in leak_candidates.items():
                defined = col_finite & np.isfinite(series)
                # Require >1 defined row and a candidate with >=3 distinct values
                # so a trivial 1-row or low-cardinality (e.g. binary) overlap
                # can't count as "leakage".
                if defined.sum() <= 1 or not _is_informative(series[defined]):
                    continue
                assert not np.allclose(
                    col_vals[defined], series[defined], atol=_ATOL
                ), (
                    f"feature '{col}' reproduces a within-geohash {kind} of the "
                    f"demand target (leakage)"
                )

    # ── Sub-claim (c): target-absent invariance ──────────────────
    # Same pre-fitted encoder / curve / imputers so the ONLY difference between
    # the three builds is the target column itself.
    df_absent = df.drop(columns=["demand"])
    df_nan = df.copy()
    df_nan["demand"] = np.nan

    feats_absent = _build(
        df_absent, encoder, curve, imp, categorical_cols, location_cols
    )
    feats_nan = _build(df_nan, encoder, curve, imp, categorical_cols, location_cols)

    cols_absent = select_feature_cols(
        feats_absent, feats_absent, "timestamp", "demand", "Index", location_cols
    )
    cols_nan = select_feature_cols(
        feats_nan, feats_nan, "timestamp", "demand", "Index", location_cols
    )

    # Identical model-feature column SET regardless of the target's presence.
    assert set(feat_cols) == set(cols_absent) == set(cols_nan), (
        "feature column set changed when the target was dropped / set to NaN: "
        f"with={sorted(feat_cols)} absent={sorted(cols_absent)} nan={sorted(cols_nan)}"
    )

    # Identical feature VALUES: engineered (non-target) features must not move
    # with the target. The encodings/curve are passed in pre-fit, so they do not
    # depend on df's target.
    np.testing.assert_allclose(
        feats_with[feat_cols].to_numpy(dtype=float),
        feats_absent[feat_cols].to_numpy(dtype=float),
        atol=_ATOL,
        err_msg="non-target feature values changed when the target column was dropped",
    )
    np.testing.assert_allclose(
        feats_with[feat_cols].to_numpy(dtype=float),
        feats_nan[feat_cols].to_numpy(dtype=float),
        atol=_ATOL,
        err_msg="non-target feature values changed when the target column was all-NaN",
    )
