from datetime import datetime, timezone

import pytest

from xau_lean.research.coverage import (
    CoverageInterval,
    analyze_coverage,
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
    timestamps = [
        ts(0, minute)
        for minute in range(60)
    ]

    result = analyze_coverage(
        timestamps,
        start=ts(0),
        end=ts(1),
        timeframe=Timeframe.H1,
    )

    assert result.expected_intervals == 1
    assert result.complete_intervals == 1
    assert result.partial_intervals == 0
    assert result.missing_intervals == 0
    assert result.observed_m1 == 60
    assert result.expected_m1 == 60
    assert result.missing_m1 == 0
    assert result.completeness_ratio == pytest.approx(1.0)


def test_partial_interval():
    timestamps = [
        ts(0, minute)
        for minute in range(30)
    ]

    result = analyze_coverage(
        timestamps,
        start=ts(0),
        end=ts(1),
        timeframe=Timeframe.H1,
    )

    assert result.expected_intervals == 1
    assert result.complete_intervals == 0
    assert result.partial_intervals == 1
    assert result.missing_intervals == 0
    assert result.observed_m1 == 30
    assert result.expected_m1 == 60
    assert result.missing_m1 == 30
    assert result.completeness_ratio == pytest.approx(0.5)

def test_completely_missing_interval():
    timestamps = [
        ts(0, minute)
        for minute in range(60)
    ] + [
        ts(2, minute)
        for minute in range(60)
    ]

    result = analyze_coverage(
        timestamps,
        start=ts(0),
        end=ts(3),
        timeframe=Timeframe.H1,
    )

    assert result.expected_intervals == 3
    assert result.complete_intervals == 2
    assert result.partial_intervals == 0
    assert result.missing_intervals == 1
    assert result.observed_m1 == 120
    assert result.expected_m1 == 180
    assert result.missing_m1 == 60
    assert result.completeness_ratio == pytest.approx(120 / 180)

def test_duplicate_timestamp_rejected():
    timestamps = [
        ts(0),
        ts(0),
    ]

    with pytest.raises(ValueError, match="strictly chronological"):
        analyze_coverage(
            timestamps,
            start=ts(0),
            end=ts(1),
            timeframe=Timeframe.H1,
        )


def test_timezone_required():
    timestamps = [
        datetime(2026, 1, 5, 0, tzinfo=timezone.utc),
    ]

    with pytest.raises(ValueError, match="timezone-aware"):
        analyze_coverage(
            timestamps,
            start=datetime(2026, 1, 5, 0),
            end=ts(1),
            timeframe=Timeframe.H1,
        )


def test_interval_result_is_immutable():
    interval = CoverageInterval(
        timestamp=ts(0),
        expected_m1=60,
        observed_m1=60,
    )

    with pytest.raises(AttributeError):
        interval.observed_m1 = 30
