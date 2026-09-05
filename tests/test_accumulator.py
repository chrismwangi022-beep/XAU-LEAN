from datetime import datetime, timezone

import pytest

from xau_lean.research.accumulator import ResearchAccumulator
from xau_lean.research.report import ResearchMetadata
from xau_lean.research.timeframe import Timeframe, TimeframeCandle


def make_candle(
    timestamp,
    open_,
    high,
    low,
    close,
    spread=None,
    complete=True,
):
    return TimeframeCandle(
        timestamp=timestamp,
        bid_open=open_,
        bid_high=high,
        bid_low=low,
        bid_close=close,
        ask_open=(
            open_ + spread
            if spread is not None
            else None
        ),
        ask_high=(
            high + spread
            if spread is not None
            else None
        ),
        ask_low=(
            low + spread
            if spread is not None
            else None
        ),
        ask_close=(
            close + spread
            if spread is not None
            else None
        ),
        expected_m1=5,
        observed_m1=5 if complete else 4,
    )


def metadata():
    return ResearchMetadata(
        source="Dukascopy",
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2020, 1, 2, tzinfo=timezone.utc),
        timeframe=Timeframe.M5,
        timezone="UTC",
        aggregation="fixed UTC calendar intervals",
        weekend_policy="preserve",
        spread_warning=1.0,
        generated_at=datetime(2026, 9, 4, tzinfo=timezone.utc),
    )


def test_accumulator_counts_and_direction():
    candles = [
        make_candle(
            datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc),
            100, 105, 95, 103, spread=0.5,
        ),
        make_candle(
            datetime(2020, 1, 1, 0, 5, tzinfo=timezone.utc),
            103, 108, 100, 106, spread=1.0,
        ),
        make_candle(
            datetime(2020, 1, 1, 0, 10, tzinfo=timezone.utc),
            106, 109, 101, 104, spread=2.0,
            complete=False,
        ),
    ]

    accumulator = ResearchAccumulator(metadata=metadata())
    accumulator.update_many(candles)

    report = accumulator.build_report()

    assert report.candle_count == 3
    assert report.complete_candle_count == 2
    assert report.incomplete_candle_count == 1

    assert report.bullish_count == 2
    assert report.bearish_count == 1
    assert report.flat_count == 0

    assert report.persistence_rate == pytest.approx(0.5)
    assert report.reversal_rate == pytest.approx(0.5)

    assert report.mean_range == pytest.approx((10 + 8 + 8) / 3)
    assert report.median_range == pytest.approx(8.0)
    assert report.mean_body == pytest.approx((3 + 3 + 2) / 3)

    assert report.mean_spread == pytest.approx(1.1666666667)
    assert report.median_spread == pytest.approx(1.0)
    assert report.spread_ge_050_pct == pytest.approx(1.0)
    assert report.spread_ge_100_pct == pytest.approx(2 / 3)
    assert report.spread_ge_200_pct == pytest.approx(1 / 3)
    assert report.spread_ge_500_pct == pytest.approx(0.0)

    assert report.candles_per_day == pytest.approx(3.0)
    assert report.candles_per_week == pytest.approx(3.0)
    assert report.candles_per_month == pytest.approx(3.0)


def test_accumulator_atr():
    candles = [
        make_candle(
            datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc),
            100, 105, 95, 103,
        ),
        make_candle(
            datetime(2020, 1, 1, 0, 5, tzinfo=timezone.utc),
            103, 108, 100, 106,
        ),
        make_candle(
            datetime(2020, 1, 1, 0, 10, tzinfo=timezone.utc),
            106, 109, 101, 104,
        ),
    ]

    accumulator = ResearchAccumulator(
        metadata=metadata(),
        atr_period=2,
    )
    accumulator.update_many(candles)

    report = accumulator.build_report()

    # TR values are 10, 8, 8.
    # First ATR after two candles = 9.
    # Second ATR = 8.
    assert report.mean_atr == pytest.approx(8.5)


def test_accumulator_requires_chronological_input():
    first = make_candle(
        datetime(2020, 1, 1, 0, 5, tzinfo=timezone.utc),
        100, 105, 95, 103,
    )
    second = make_candle(
        datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc),
        103, 108, 100, 106,
    )

    accumulator = ResearchAccumulator(metadata=metadata())
    accumulator.update(first)

    with pytest.raises(ValueError, match="strictly chronological"):
        accumulator.update(second)


def test_accumulator_requires_timezone():
    candle = make_candle(
        datetime(2020, 1, 1, 0, 0),
        100, 105, 95, 103,
    )

    accumulator = ResearchAccumulator(metadata=metadata())

    with pytest.raises(ValueError, match="timezone-aware"):
        accumulator.update(candle)


def test_online_median_empty_and_small_samples():
    from xau_lean.research.accumulator import _OnlineMedian

    median = _OnlineMedian()
    assert median.median is None

    for value in [10.0, 2.0, 7.0, 4.0]:
        median.update(value)

    assert median.median == pytest.approx(5.5)


def test_online_median_exact_five_sample_initialization():
    from xau_lean.research.accumulator import _OnlineMedian

    median = _OnlineMedian()

    for value in [9.0, 1.0, 7.0, 3.0, 5.0]:
        median.update(value)

    assert median.median == pytest.approx(5.0)


def test_online_median_tracks_large_ordered_sample():
    from xau_lean.research.accumulator import _OnlineMedian

    median = _OnlineMedian()

    for value in range(1, 1001):
        median.update(float(value))

    assert median.median == pytest.approx(500.5, abs=1.0)


def test_online_median_memory_is_bounded():
    from xau_lean.research.accumulator import _OnlineMedian

    median = _OnlineMedian()

    for value in range(1, 10001):
        median.update(float(value))

    assert median._heights is not None
    assert len(median._heights) == 5
    assert len(median._initial) == 0
