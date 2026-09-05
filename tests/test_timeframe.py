from datetime import datetime, timezone

import pytest

from xau_lean.data.dukascopy import DukascopyBar
from xau_lean.research.timeframe import (
    Timeframe,
    aggregate_bars,
    aggregate_timeframes,
    find_gaps,
    interval_start,
)


def make_bar(
    minute: int,
    *,
    base: float = 100.0,
) -> DukascopyBar:
    timestamp = datetime(
        2020,
        1,
        2,
        10,
        minute,
        tzinfo=timezone.utc,
    )

    value = base + minute

    return DukascopyBar(
        timestamp=timestamp,
        bid_open=value,
        bid_high=value + 0.5,
        bid_low=value - 0.5,
        bid_close=value + 0.25,
        ask_open=value + 0.2,
        ask_high=value + 0.7,
        ask_low=value - 0.3,
        ask_close=value + 0.45,
    )


def test_interval_boundaries():
    timestamp = datetime(
        2020,
        1,
        2,
        10,
        37,
        42,
        tzinfo=timezone.utc,
    )

    assert interval_start(timestamp, Timeframe.M5).minute == 35
    assert interval_start(timestamp, Timeframe.M15).minute == 30
    assert interval_start(timestamp, Timeframe.H1).hour == 10
    assert interval_start(timestamp, Timeframe.H4).hour == 8
    assert interval_start(timestamp, Timeframe.H6).hour == 6


def test_interval_requires_timezone():
    timestamp = datetime(2020, 1, 2, 10, 37)

    with pytest.raises(ValueError):
        interval_start(timestamp, Timeframe.M5)


def test_m5_aggregation_uses_correct_ohlc():
    bars = [make_bar(i) for i in range(5)]

    candles = list(
        aggregate_bars(
            bars,
            Timeframe.M5,
        )
    )

    assert len(candles) == 1

    candle = candles[0]

    assert candle.timestamp == datetime(
        2020,
        1,
        2,
        10,
        0,
        tzinfo=timezone.utc,
    )

    assert candle.bid_open == 100.0
    assert candle.bid_close == 104.25
    assert candle.bid_high == 104.5
    assert candle.bid_low == 99.5

    assert candle.ask_open == 100.2
    assert candle.ask_close == 104.45
    assert candle.ask_high == 104.7
    assert candle.ask_low == 99.7

    assert candle.expected_m1 == 5
    assert candle.observed_m1 == 5
    assert candle.missing_m1 == 0
    assert candle.complete is True
    assert candle.completeness_ratio == 1.0


def test_missing_m1_is_not_fabricated():
    bars = [
        make_bar(0),
        make_bar(1),
        make_bar(2),
        make_bar(4),
    ]

    candles = list(
        aggregate_bars(
            bars,
            Timeframe.M5,
        )
    )

    assert len(candles) == 1

    candle = candles[0]

    assert candle.expected_m1 == 5
    assert candle.observed_m1 == 4
    assert candle.missing_m1 == 1
    assert candle.complete is False
    assert candle.completeness_ratio == 0.8


def test_chronological_order_is_required():
    bars = [
        make_bar(1),
        make_bar(0),
    ]

    with pytest.raises(ValueError):
        list(
            aggregate_bars(
                bars,
                Timeframe.M5,
            )
        )


def test_missing_ask_is_explicit():
    bars = [
        make_bar(0),
        make_bar(1),
        make_bar(2),
        make_bar(3),
        make_bar(4),
    ]

    bars[3] = DukascopyBar(
        timestamp=bars[3].timestamp,
        bid_open=bars[3].bid_open,
        bid_high=bars[3].bid_high,
        bid_low=bars[3].bid_low,
        bid_close=bars[3].bid_close,
        ask_open=None,
        ask_high=None,
        ask_low=None,
        ask_close=None,
    )

    candle = next(
        aggregate_bars(
            bars,
            Timeframe.M5,
        )
    )

    assert candle.ask_open is None
    assert candle.ask_high is None
    assert candle.ask_low is None
    assert candle.ask_close is None


def test_all_timeframes_have_expected_candle_duration():
    expected = {
        Timeframe.M5: 5,
        Timeframe.M15: 15,
        Timeframe.H1: 60,
        Timeframe.H4: 240,
        Timeframe.H6: 360,
    }

    for timeframe, minutes in expected.items():
        assert timeframe.minutes == minutes


def test_cross_boundary_aggregation_is_deterministic():
    bars = []

    for minute in range(58, 60):
        timestamp = datetime(
            2020,
            1,
            2,
            10,
            minute,
            tzinfo=timezone.utc,
        )

        bars.append(
            DukascopyBar(
                timestamp=timestamp,
                bid_open=100.0,
                bid_high=101.0,
                bid_low=99.0,
                bid_close=100.5,
                ask_open=100.2,
                ask_high=101.2,
                ask_low=99.2,
                ask_close=100.7,
            )
        )

    for minute in range(4):
        timestamp = datetime(
            2020,
            1,
            2,
            11,
            minute,
            tzinfo=timezone.utc,
        )

        bars.append(
            DukascopyBar(
                timestamp=timestamp,
                bid_open=101.0,
                bid_high=102.0,
                bid_low=100.0,
                bid_close=101.5,
                ask_open=101.2,
                ask_high=102.2,
                ask_low=100.2,
                ask_close=101.7,
            )
        )

    candles = list(
        aggregate_bars(
            bars,
            Timeframe.M5,
        )
    )

    assert len(candles) == 2

    assert candles[0].timestamp == datetime(
        2020,
        1,
        2,
        10,
        55,
        tzinfo=timezone.utc,
    )

    assert candles[0].observed_m1 == 2
    assert candles[0].complete is False

    assert candles[1].timestamp == datetime(
        2020,
        1,
        2,
        11,
        0,
        tzinfo=timezone.utc,
    )

    assert candles[1].observed_m1 == 4
    assert candles[1].complete is False


def test_find_gaps_detects_missing_minutes():
    bars = [
        make_bar(0),
        make_bar(1),
        make_bar(4),
    ]

    gaps = list(find_gaps(bars))

    assert len(gaps) == 1
    assert gaps[0].missing_minutes == 2
    assert gaps[0].previous_timestamp == bars[1].timestamp
    assert gaps[0].current_timestamp == bars[2].timestamp


def test_find_gaps_does_not_report_normal_one_minute_spacing():
    bars = [make_bar(i) for i in range(5)]

    assert list(find_gaps(bars)) == []


def test_find_gaps_requires_chronological_input():
    bars = [
        make_bar(2),
        make_bar(1),
    ]

    with pytest.raises(ValueError):
        list(find_gaps(bars))


def test_streaming_aggregation_accepts_generators():
    def bar_generator():
        for minute in range(5):
            yield make_bar(minute)

    candles = list(
        aggregate_bars(
            bar_generator(),
            Timeframe.M5,
        )
    )

    assert len(candles) == 1
    assert candles[0].complete is True


def test_multi_timeframe_aggregation_uses_one_stream():
    bars = [
        make_bar(minute)
        for minute in range(15)
    ]

    results = list(
        aggregate_timeframes(
            bars,
            [
                Timeframe.M5,
                Timeframe.M15,
            ],
        )
    )

    m5 = [
        candle
        for timeframe, candle in results
        if timeframe is Timeframe.M5
    ]

    m15 = [
        candle
        for timeframe, candle in results
        if timeframe is Timeframe.M15
    ]

    assert len(m5) == 3
    assert len(m15) == 1

    assert [candle.observed_m1 for candle in m5] == [
        5,
        5,
        5,
    ]

    assert m15[0].observed_m1 == 15

    assert all(candle.complete for candle in m5)
    assert m15[0].complete is True


def test_multi_timeframe_preserves_partial_candles():
    bars = [
        make_bar(minute)
        for minute in range(7)
    ]

    results = list(
        aggregate_timeframes(
            bars,
            [Timeframe.M5],
        )
    )

    candles = [
        candle
        for timeframe, candle in results
        if timeframe is Timeframe.M5
    ]

    assert len(candles) == 2
    assert candles[0].observed_m1 == 5
    assert candles[0].complete is True
    assert candles[1].observed_m1 == 2
    assert candles[1].complete is False


def test_multi_timeframe_accepts_generator():
    def bar_generator():
        for minute in range(15):
            yield make_bar(minute)

    results = list(
        aggregate_timeframes(
            bar_generator(),
            [
                Timeframe.M5,
                Timeframe.M15,
            ],
        )
    )

    assert sum(
        timeframe is Timeframe.M5
        for timeframe, _ in results
    ) == 3

    assert sum(
        timeframe is Timeframe.M15
        for timeframe, _ in results
    ) == 1


def test_multi_timeframe_requires_timeframe():
    with pytest.raises(
        ValueError,
        match="At least one timeframe",
    ):
        list(
            aggregate_timeframes(
                [make_bar(0)],
                [],
            )
        )


def test_multi_timeframe_requires_chronological_input():
    bars = [
        make_bar(1),
        make_bar(0),
    ]

    with pytest.raises(
        ValueError,
        match="strictly chronological",
    ):
        list(
            aggregate_timeframes(
                bars,
                [
                    Timeframe.M5,
                    Timeframe.H1,
                ],
            )
        )
