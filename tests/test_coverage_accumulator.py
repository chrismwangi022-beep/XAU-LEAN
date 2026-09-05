from datetime import datetime, timezone

import pytest

from xau_lean.research.coverage import (
    CoverageAccumulator,
    CoverageInterval,
)
from xau_lean.research.timeframe import Timeframe


def ts(hour: int, minute: int = 0) -> datetime:
    return datetime(
        2026,
        1,
        5,
        hour,
        minute,
        tzinfo=timezone.utc,
    )


def test_complete_interval():
    accumulator = CoverageAccumulator(
        start=ts(0),
        end=ts(1),
        timeframe=Timeframe.H1,
    )

    for minute in range(60):
        accumulator.update(ts(0, minute))

    result = accumulator.build_summary()

    assert result.expected_intervals == 1
    assert result.complete_intervals == 1
    assert result.partial_intervals == 0
    assert result.missing_intervals == 0
    assert result.expected_m1 == 60
    assert result.observed_m1 == 60
    assert result.missing_m1 == 0
    assert result.completeness_ratio == pytest.approx(1.0)


def test_partial_interval():
    accumulator = CoverageAccumulator(
        start=ts(0),
        end=ts(1),
        timeframe=Timeframe.H1,
    )

    for minute in range(30):
        accumulator.update(ts(0, minute))

    result = accumulator.build_summary()

    assert result.expected_intervals == 1
    assert result.complete_intervals == 0
    assert result.partial_intervals == 1
    assert result.missing_intervals == 0
    assert result.expected_m1 == 60
    assert result.observed_m1 == 30
    assert result.missing_m1 == 30
    assert result.completeness_ratio == pytest.approx(0.5)


def test_completely_missing_interval():
    accumulator = CoverageAccumulator(
        start=ts(0),
        end=ts(3),
        timeframe=Timeframe.H1,
    )

    for minute in range(60):
        accumulator.update(ts(0, minute))

    for minute in range(60):
        accumulator.update(ts(2, minute))

    result = accumulator.build_summary()

    assert result.expected_intervals == 3
    assert result.complete_intervals == 2
    assert result.partial_intervals == 0
    assert result.missing_intervals == 1
    assert result.expected_m1 == 180
    assert result.observed_m1 == 120
    assert result.missing_m1 == 60
    assert result.completeness_ratio == pytest.approx(120 / 180)


def test_duplicate_timestamp_rejected():
    accumulator = CoverageAccumulator(
        start=ts(0),
        end=ts(1),
        timeframe=Timeframe.H1,
    )

    accumulator.update(ts(0))

    with pytest.raises(ValueError, match="strictly chronological"):
        accumulator.update(ts(0))


def test_out_of_range_timestamp_ignored():
    accumulator = CoverageAccumulator(
        start=ts(0),
        end=ts(1),
        timeframe=Timeframe.H1,
    )

    accumulator.update(ts(0, 30))
    accumulator.update(ts(2))

    result = accumulator.build_summary()

    assert result.observed_m1 == 1
    assert result.expected_intervals == 1
    assert result.partial_intervals == 1


def test_summary_is_consistent_with_interval_schema():
    interval = CoverageInterval(
        timestamp=ts(0),
        expected_m1=60,
        observed_m1=45,
    )

    assert interval.missing_m1 == 15
    assert interval.completeness_ratio == pytest.approx(0.75)
    assert interval.complete is False
