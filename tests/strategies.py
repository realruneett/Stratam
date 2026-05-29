"""Reusable Hypothesis strategies for the demand-prediction-overhaul tests.

These strategies generate synthetic pandas DataFrames that mirror the *raw*
competition schema (pre-feature-engineering), so every later property test
(Properties 1-14) can build its inputs from a shared, well-understood source.

Raw schema (``train.csv`` / ``test.csv``)::

    Index          int            sequential or arbitrary unique id
    geohash        str            location identifier (base32-ish token)
    day            int            48 or 49
    timestamp      str            "H:M" (NOT zero padded, e.g. "0:0", "2:15")
    demand         float          target, continuous in ~[0, 1]   (train only)
    RoadType       str|None       {Highway, Residential, Street}   (nullable)
    NumberofLanes  int            {1, 2, 3, 4, 5}                  (complete)
    LargeVehicles  str            {Allowed, Not Allowed}
    Landmarks      str            {Yes, No}
    Temperature    float|None     ~[-20, 50]                       (nullable)
    Weather        str|None       {Sunny, Rainy, Foggy, Snowy}     (nullable)

Time-of-day slots: ``tod_slot = (hour*60 + minute) // 15`` in ``0..95``.
The real data partitions into:

    * day 48          -> all 96 slots (0..95)
    * day 49 (train)  -> morning slots 0..8 (0:00-2:00)
    * day 49 (test)   -> daytime slots 9..55 (2:15-13:45)

Convenience helpers expose the morning / daytime / full slot windows and the
common frame shapes (two-day frames, real-task frames, morning-only frames,
train/test pairs with unseen geohashes, and prediction vectors).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import strategies as st

# ─── Category vocabularies (mirror the real data) ───────────────

ROAD_TYPES = ["Highway", "Residential", "Street"]
LARGE_VEHICLES = ["Allowed", "Not Allowed"]
LANDMARKS = ["Yes", "No"]
WEATHERS = ["Sunny", "Rainy", "Foggy", "Snowy"]
NUM_LANES = [1, 2, 3, 4, 5]

#: Contextual columns into which nulls may be injected (Req 3.6).
NULLABLE_CONTEXT_COLS = ["RoadType", "Temperature", "Weather"]

#: Column order, mirroring train.csv exactly.
RAW_COLUMNS = [
    "Index", "geohash", "day", "timestamp", "demand",
    "RoadType", "NumberofLanes", "LargeVehicles", "Landmarks",
    "Temperature", "Weather",
]
#: Column order for test-like frames (no target column).
RAW_COLUMNS_NO_DEMAND = [c for c in RAW_COLUMNS if c != "demand"]

# ─── Time-of-day slot windows ───────────────────────────────────

MORNING_SLOTS = (0, 8)        # day-49 train rows (0:00-2:00)
TEST_WINDOW_SLOTS = (9, 55)   # day-49 test rows  (2:15-13:45)
FULL_SLOTS = (0, 95)          # day-48 full coverage

#: base32 geohash alphabet (no a, i, l, o).
_GEOHASH_ALPHABET = "0123456789bcdefghjkmnpqrstuvwxyz"


# ─── Pure helpers ───────────────────────────────────────────────

def slot_to_timestamp(slot: int) -> str:
    """Convert a quarter-hour slot in ``0..95`` to a raw ``"H:M"`` string.

    Mirrors the un-padded format in the CSVs (e.g. slot 0 -> ``"0:0"``,
    slot 9 -> ``"2:15"``, slot 55 -> ``"13:45"``).
    """
    minutes = slot * 15
    return f"{minutes // 60}:{minutes % 60}"


# ─── Scalar strategies ──────────────────────────────────────────

def demand_value() -> st.SearchStrategy:
    """Continuous target in ``[0, 1]`` (no NaN / Inf)."""
    return st.floats(
        min_value=0.0, max_value=1.0,
        allow_nan=False, allow_infinity=False, width=64,
    )


def temperature_value() -> st.SearchStrategy:
    """Temperature float in roughly the observed ``[-20, 50]`` range."""
    return st.floats(
        min_value=-20.0, max_value=50.0,
        allow_nan=False, allow_infinity=False, width=64,
    )


def geohash_token() -> st.SearchStrategy:
    """A single geohash-like token (6 base32 chars)."""
    return st.text(alphabet=_GEOHASH_ALPHABET, min_size=6, max_size=6)


def geohash_pool(min_size: int = 1, max_size: int = 8) -> st.SearchStrategy:
    """A list of *distinct* geohash tokens to draw rows from.

    A pool larger than the row count naturally yields single-row geohashes;
    a pool of size 1 yields an all-same-geohash frame.
    """
    return st.lists(
        geohash_token(), min_size=min_size, max_size=max_size, unique=True,
    )


def _maybe_null(strategy: st.SearchStrategy, nullable: bool) -> st.SearchStrategy:
    """Wrap a value strategy so it may emit ``None`` when ``nullable``."""
    if not nullable:
        return strategy
    return st.one_of(st.none(), strategy)


# ─── Core frame strategy ────────────────────────────────────────

@st.composite
def raw_frame(
    draw,
    *,
    min_rows: int = 1,
    max_rows: int = 40,
    days: tuple[int, ...] = (48, 49),
    slot_min: int = 0,
    slot_max: int = 95,
    include_demand: bool = True,
    nullable: bool = True,
    geohashes: list[str] | None = None,
    min_geohashes: int = 1,
    max_geohashes: int = 8,
    start_index: int = 0,
    unique_random_index: bool = False,
) -> pd.DataFrame:
    """Generate a raw-schema DataFrame.

    Args:
        min_rows / max_rows: row-count bounds.
        days: allowed ``day`` values (each row samples one).
        slot_min / slot_max: inclusive ``tod_slot`` bounds for every row.
        include_demand: include the ``demand`` target column (False = test-like).
        nullable: allow injected nulls in ``RoadType``/``Temperature``/``Weather``.
        geohashes: fixed pool to sample from; if ``None`` a pool is generated.
        min_geohashes / max_geohashes: pool-size bounds when ``geohashes`` is None.
        start_index: first ``Index`` value when ``unique_random_index`` is False.
        unique_random_index: assign arbitrary unique ``Index`` values instead of a
            contiguous range (useful for testing original-order preservation).

    Returns:
        A ``pandas.DataFrame`` with the raw column set (and order).
    """
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))

    if geohashes is None or len(geohashes) == 0:
        pool_size = draw(st.integers(
            min_value=min_geohashes,
            max_value=max(min_geohashes, min(max_geohashes, n_rows)),
        ))
        geohashes = draw(geohash_pool(min_size=pool_size, max_size=pool_size))
        if not geohashes:  # degenerate guard
            geohashes = draw(geohash_pool(min_size=1, max_size=1))

    if unique_random_index:
        index_values = draw(st.lists(
            st.integers(min_value=0, max_value=10_000_000),
            min_size=n_rows, max_size=n_rows, unique=True,
        ))
    else:
        index_values = list(range(start_index, start_index + n_rows))

    rows = []
    for i in range(n_rows):
        slot = draw(st.integers(min_value=slot_min, max_value=slot_max))
        row = {
            "Index": index_values[i],
            "geohash": draw(st.sampled_from(geohashes)),
            "day": draw(st.sampled_from(list(days))),
            "timestamp": slot_to_timestamp(slot),
            "RoadType": draw(_maybe_null(st.sampled_from(ROAD_TYPES), nullable)),
            "NumberofLanes": draw(st.sampled_from(NUM_LANES)),
            "LargeVehicles": draw(st.sampled_from(LARGE_VEHICLES)),
            "Landmarks": draw(st.sampled_from(LANDMARKS)),
            "Temperature": draw(_maybe_null(temperature_value(), nullable)),
            "Weather": draw(_maybe_null(st.sampled_from(WEATHERS), nullable)),
        }
        if include_demand:
            row["demand"] = draw(demand_value())
        rows.append(row)

    columns = RAW_COLUMNS if include_demand else RAW_COLUMNS_NO_DEMAND
    return pd.DataFrame(rows, columns=columns)


# ─── Composite / specialized frame strategies ───────────────────

@st.composite
def two_day_frame(
    draw,
    *,
    min_rows_per_day: int = 2,
    max_rows_per_day: int = 20,
    nullable: bool = True,
    include_demand: bool = True,
) -> pd.DataFrame:
    """A frame guaranteed to contain BOTH day-48 and day-49 rows.

    Day-48 rows span the full slot range; day-49 rows are morning-only
    (slots 0..8), mirroring how ``train.csv`` is actually partitioned. Both
    days share the same geohash pool so encoders see cross-day groups.
    """
    pool = draw(geohash_pool(min_size=1, max_size=6))
    if not pool:
        pool = draw(geohash_pool(min_size=1, max_size=1))

    day48 = draw(raw_frame(
        min_rows=min_rows_per_day, max_rows=max_rows_per_day,
        days=(48,), slot_min=FULL_SLOTS[0], slot_max=FULL_SLOTS[1],
        include_demand=include_demand, nullable=nullable, geohashes=pool,
    ))
    day49 = draw(raw_frame(
        min_rows=min_rows_per_day, max_rows=max_rows_per_day,
        days=(49,), slot_min=MORNING_SLOTS[0], slot_max=MORNING_SLOTS[1],
        include_demand=include_demand, nullable=nullable, geohashes=pool,
        start_index=len(day48),
    ))
    return pd.concat([day48, day49], ignore_index=True)


@st.composite
def real_task_frame(
    draw,
    *,
    nullable: bool = True,
) -> pd.DataFrame:
    """A frame for the real-task split: day-48 rows plus day-49 rows that
    include at least one slot inside the test window ``[9, 55]``.

    Used by Property 8 (real-task split structure).
    """
    pool = draw(geohash_pool(min_size=1, max_size=6))
    if not pool:
        pool = draw(geohash_pool(min_size=1, max_size=1))

    day48 = draw(raw_frame(
        min_rows=2, max_rows=20, days=(48,),
        slot_min=FULL_SLOTS[0], slot_max=FULL_SLOTS[1],
        include_demand=True, nullable=nullable, geohashes=pool,
    ))
    # day-49 rows constrained to the test daytime window so at least some
    # rows are guaranteed to match [9, 55].
    day49_daytime = draw(raw_frame(
        min_rows=1, max_rows=15, days=(49,),
        slot_min=TEST_WINDOW_SLOTS[0], slot_max=TEST_WINDOW_SLOTS[1],
        include_demand=True, nullable=nullable, geohashes=pool,
        start_index=len(day48),
    ))
    return pd.concat([day48, day49_daytime], ignore_index=True)


@st.composite
def morning_only_day49_frame(
    draw,
    *,
    nullable: bool = True,
) -> pd.DataFrame:
    """Day-48 rows plus day-49 rows confined to morning slots ``0..8``.

    No day-49 slot falls in the test window ``[9, 55]``; used to exercise the
    ``FAILED_NO_MATCHING_SLOTS`` validator path (Task 4.4).
    """
    pool = draw(geohash_pool(min_size=1, max_size=6))
    if not pool:
        pool = draw(geohash_pool(min_size=1, max_size=1))

    day48 = draw(raw_frame(
        min_rows=2, max_rows=20, days=(48,),
        slot_min=FULL_SLOTS[0], slot_max=FULL_SLOTS[1],
        include_demand=True, nullable=nullable, geohashes=pool,
    ))
    day49_morning = draw(raw_frame(
        min_rows=1, max_rows=12, days=(49,),
        slot_min=MORNING_SLOTS[0], slot_max=MORNING_SLOTS[1],
        include_demand=True, nullable=nullable, geohashes=pool,
        start_index=len(day48),
    ))
    return pd.concat([day48, day49_morning], ignore_index=True)


@st.composite
def train_test_pair(
    draw,
    *,
    with_unseen_geohash: bool = True,
    nullable: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A ``(train_df, test_df)`` pair sharing categorical vocabularies.

    The test frame is day-49 daytime (slots ``9..55``) with no ``demand``
    column. When ``with_unseen_geohash`` is True the test frame contains at
    least one geohash absent from train (Property 7), while still sharing
    geohashes for the overlap case. Categorical columns are drawn from the
    same fixed vocabularies so categories overlap (Property 5).
    """
    train_pool = draw(geohash_pool(min_size=2, max_size=6))
    if not train_pool:
        train_pool = draw(geohash_pool(min_size=1, max_size=1))

    train_df = draw(raw_frame(
        min_rows=4, max_rows=30, days=(48, 49),
        slot_min=FULL_SLOTS[0], slot_max=FULL_SLOTS[1],
        include_demand=True, nullable=nullable, geohashes=train_pool,
    ))

    test_df = draw(raw_frame(
        min_rows=4, max_rows=30, days=(49,),
        slot_min=TEST_WINDOW_SLOTS[0], slot_max=TEST_WINDOW_SLOTS[1],
        include_demand=False, nullable=nullable, geohashes=list(train_pool),
    ))

    if with_unseen_geohash:
        # Draw an unseen token and stamp it onto at least one test row so the
        # guarantee holds at the ROW level (Property 7), not just in a pool the
        # row sampler might never draw from.
        train_set = set(train_pool)
        unseen_tok = draw(geohash_token().filter(lambda g: g not in train_set))
        victim = draw(st.integers(min_value=0, max_value=len(test_df) - 1))
        test_df.loc[victim, "geohash"] = unseen_tok
    return train_df, test_df


@st.composite
def frame_with_injected_nulls(
    draw,
    *,
    min_rows: int = 3,
    max_rows: int = 30,
) -> pd.DataFrame:
    """A frame guaranteed to contain at least one null in a nullable column.

    Used by Property 6 (train-derived, null-only imputation).
    """
    df = draw(raw_frame(
        min_rows=min_rows, max_rows=max_rows, nullable=True,
    ))
    # force at least one null into a randomly chosen nullable column/row.
    col = draw(st.sampled_from(NULLABLE_CONTEXT_COLS))
    row = draw(st.integers(min_value=0, max_value=len(df) - 1))
    df.loc[row, col] = None
    return df


# ─── Prediction-vector strategies (postprocess / submission) ────

@st.composite
def prediction_array(
    draw,
    *,
    min_size: int = 1,
    max_size: int = 60,
    min_value: float = -5.0,
    max_value: float = 5.0,
    size: int | None = None,
) -> np.ndarray:
    """A raw prediction vector (finite floats, possibly negative/over-range).

    Used by Property 12 (bounds). Pass ``size`` to fix the length (e.g. to a
    test frame's row count for Property 11).
    """
    if size is None:
        size = draw(st.integers(min_value=min_size, max_value=max_size))
    vals = draw(st.lists(
        st.floats(min_value=min_value, max_value=max_value,
                  allow_nan=False, allow_infinity=False),
        min_size=size, max_size=size,
    ))
    return np.asarray(vals, dtype=float)


@st.composite
def prediction_array_with_nans(
    draw,
    *,
    min_size: int = 1,
    max_size: int = 60,
    size: int | None = None,
) -> np.ndarray:
    """A prediction vector guaranteed to contain at least one NaN or Inf.

    Used by Property 13 (null-prediction fallback).
    """
    if size is None:
        size = draw(st.integers(min_value=min_size, max_value=max_size))
    arr = draw(prediction_array(size=size))
    bad_value = draw(st.sampled_from([np.nan, np.inf, -np.inf]))
    idx = draw(st.integers(min_value=0, max_value=size - 1))
    arr = arr.copy()
    arr[idx] = bad_value
    return arr


@st.composite
def test_frame_with_predictions(
    draw,
    *,
    nullable: bool = True,
    unique_random_index: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """A ``(test_df, preds)`` pair with a prediction per test row.

    The test frame is day-49 daytime with no ``demand`` column and (by
    default) arbitrary unique ``Index`` values to exercise original-order
    preservation (Property 11).
    """
    test_df = draw(raw_frame(
        min_rows=1, max_rows=40, days=(49,),
        slot_min=TEST_WINDOW_SLOTS[0], slot_max=TEST_WINDOW_SLOTS[1],
        include_demand=False, nullable=nullable,
        unique_random_index=unique_random_index,
    ))
    preds = draw(prediction_array(size=len(test_df)))
    return test_df, preds
