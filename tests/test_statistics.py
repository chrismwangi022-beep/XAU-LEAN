from datetime import datetime, timezone

import pytest

from xau_lean.research.statistics import (
    candle_statistics,
    count_candles_by_period,
    direction_counts,
    directional_persistence,
    directional_run_lengths,
    iter_candle_statistics,
    realized_volatility,
    spread_threshold_percentages,
    spread_to_range_values,
    summarize_distribution,
    rolling_atr,
    rolling_atr,
    true_range,
)
from xau_lean.research.timeframe import TimeframeCandle


def make_candle(
    timestamp: datetime,
    open_price: float,
    high: float,
    low: float,
    close: float,
    spread: float = 0.5,
) -> TimeframeCandle:
    return TimeframeCandle(
        timestamp=timestamp,
        bid_open=open_price,
        bid_high=high,
        bid_low=low,
        bid_close=close,
        ask_open=open_price + spread,
        ask_high=high + spread,
        ask_low=low + spread,
        ask_close=close + spread,
        expected_m1=5,
        observed_m1=5,
    )


def test_bullish_candle_structure():
    candle = make_candle(
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        100.0,
        110.0,
        95.0,
        108.0,
    )

    stats = candle_statistics(candle)

    assert stats.bid_range == 15.0
    assert stats.bid_body == 8.0
    assert stats.body_ratio == pytest.approx(8.0 / 15.0)

    assert stats.upper_wick == 2.0
    assert stats.lower_wick == 5.0

    assert stats.direction == 1
    assert stats.simple_return is None
    assert stats.log_return is None

    assert stats.close_spread == pytest.approx(0.5)
    assert stats.spread_to_range == pytest.approx(0.5 / 15.0)


def test_bearish_candle_direction():
    candle = make_candle(
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        108.0,
        110.0,
        95.0,
        100.0,
    )

    stats = candle_statistics(candle)

    assert stats.direction == -1


def test_flat_candle_direction():
    candle = make_candle(
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        100.0,
        105.0,
        95.0,
        100.0,
    )

    stats = candle_statistics(candle)

    assert stats.direction == 0


def test_return_calculation_uses_previous_close():
    candle = make_candle(
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        105.0,
        110.0,
        100.0,
        110.0,
    )

    stats = candle_statistics(
        candle,
        previous_close=100.0,
    )

    assert stats.simple_return == pytest.approx(0.10)
    assert stats.log_return == pytest.approx(__import__("math").log(1.10))


def test_invalid_previous_close_rejected():
    candle = make_candle(
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        100.0,
        105.0,
        95.0,
        102.0,
    )

    with pytest.raises(ValueError):
        candle_statistics(candle, previous_close=0.0)


def test_statistics_stream_is_chronological_and_non_anticipating():
    candles = [
        make_candle(
            datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc),
            100.0,
            105.0,
            95.0,
            102.0,
        ),
        make_candle(
            datetime(2020, 1, 1, 0, 5, tzinfo=timezone.utc),
            102.0,
            108.0,
            101.0,
            106.0,
        ),
        make_candle(
            datetime(2020, 1, 1, 0, 10, tzinfo=timezone.utc),
            106.0,
            109.0,
            104.0,
            105.0,
        ),
    ]

    results = list(iter_candle_statistics(candles))

    assert len(results) == 3

    assert results[0].simple_return is None
    assert results[0].log_return is None

    assert results[1].simple_return == pytest.approx(106.0 / 102.0 - 1.0)
    assert results[2].simple_return == pytest.approx(105.0 / 106.0 - 1.0)

    assert results[0].timestamp < results[1].timestamp
    assert results[1].timestamp < results[2].timestamp


def test_true_range_without_previous_close():
    assert true_range(110.0, 100.0) == pytest.approx(10.0)


def test_true_range_uses_previous_close_gap():
    # High-low = 10
    # High-prev_close = 15
    # Low-prev_close = 5
    # Therefore TR = 15.
    assert true_range(
        115.0,
        105.0,
        previous_close=100.0,
    ) == pytest.approx(15.0)


def test_realized_volatility():
    values = [0.01, 0.02, 0.03]

    result = realized_volatility(values)

    expected_mean = 0.02
    expected_variance = (
        ((0.01 - expected_mean) ** 2)
        + ((0.02 - expected_mean) ** 2)
        + ((0.03 - expected_mean) ** 2)
    ) / 2

    assert result == pytest.approx(expected_variance ** 0.5)


def test_realized_volatility_requires_two_observations():
    assert realized_volatility([]) is None
    assert realized_volatility([0.01]) is None


def test_rolling_atr():
    candles = [
        make_candle(
            datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc),
            100.0,
            105.0,
            95.0,
            102.0,
        ),
        make_candle(
            datetime(2020, 1, 1, 0, 5, tzinfo=timezone.utc),
            102.0,
            110.0,
            100.0,
            108.0,
        ),
        make_candle(
            datetime(2020, 1, 1, 0, 10, tzinfo=timezone.utc),
            108.0,
            115.0,
            105.0,
            110.0,
        ),
    ]

    results = list(
        __import__(
            "xau_lean.research.statistics",
            fromlist=["rolling_atr"],
        ).rolling_atr(
            candles,
            period=2,
        )
    )

    assert results[0] is None

    # TR1 = 10
    # TR2 = max(10, 8, 2) = 10
    # ATR2 = 10
    assert results[1] == pytest.approx(10.0)

    # TR3 = max(10, 7, 3) = 10
    # ATR3 = (10 + 10) / 2 = 10
    assert results[2] == pytest.approx(10.0)


def test_direction_counts():
    result = direction_counts(
        [1, 1, -1, 0, -1, 1, 0]
    )

    assert result == {
        "bullish": 3,
        "bearish": 2,
        "flat": 2,
    }


def test_directional_persistence():
    result = directional_persistence(
        [1, 1, -1, -1, 1]
    )

    assert result["directional_observations"] == 4
    assert result["same_direction"] == 2
    assert result["reversals"] == 2
    assert result["persistence_rate"] == pytest.approx(0.5)
    assert result["reversal_rate"] == pytest.approx(0.5)


def test_directional_persistence_ignores_flat_candles():
    result = directional_persistence(
        [1, 0, 1, -1]
    )

    assert result["directional_observations"] == 2
    assert result["same_direction"] == 1
    assert result["reversals"] == 1


def test_directional_run_lengths():
    result = directional_run_lengths(
        [1, 1, 1, -1, -1, 0, 1]
    )

    assert result == [3, 2, 1]


def test_directional_run_lengths_all_flat():
    assert directional_run_lengths([0, 0, 0]) == []


def test_summarize_distribution():
    result = summarize_distribution(
        [1.0, 2.0, 3.0, 4.0]
    )

    assert result.count == 4
    assert result.mean == pytest.approx(2.5)
    assert result.median == pytest.approx(2.5)
    assert result.minimum == 1.0
    assert result.maximum == 4.0


def test_summarize_empty_distribution():
    result = summarize_distribution([])

    assert result.count == 0
    assert result.mean is None
    assert result.median is None
    assert result.minimum is None
    assert result.maximum is None


def test_spread_to_range_values():
    candles = [
        make_candle(
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            100.0,
            110.0,
            95.0,
            105.0,
            spread=1.0,
        ),
        make_candle(
            datetime(2020, 1, 1, 0, 5, tzinfo=timezone.utc),
            105.0,
            107.0,
            100.0,
            102.0,
            spread=0.5,
        ),
    ]

    statistics = list(
        iter_candle_statistics(candles)
    )

    ratios = list(
        spread_to_range_values(statistics)
    )

    assert ratios == pytest.approx(
        [1.0 / 15.0, 0.5 / 7.0]
    )


def test_spread_threshold_percentages():
    candles = [
        make_candle(
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            100.0,
            110.0,
            90.0,
            105.0,
            spread=0.5,
        ),
        make_candle(
            datetime(2020, 1, 1, 0, 5, tzinfo=timezone.utc),
            105.0,
            115.0,
            95.0,
            110.0,
            spread=1.0,
        ),
        make_candle(
            datetime(2020, 1, 1, 0, 10, tzinfo=timezone.utc),
            110.0,
            120.0,
            100.0,
            115.0,
            spread=2.0,
        ),
    ]

    statistics = list(
        iter_candle_statistics(candles)
    )

    result = spread_threshold_percentages(
        statistics,
        thresholds=[0.50, 1.0, 2.0],
    )

    assert result[0.50] == pytest.approx(1.0)
    assert result[1.0] == pytest.approx(2.0 / 3.0)
    assert result[2.0] == pytest.approx(1.0 / 3.0)


def test_count_candles_by_period():
    candles = [
        make_candle(
            datetime(2020, 1, 2, 10, 0, tzinfo=timezone.utc),
            100.0,
            105.0,
            95.0,
            102.0,
        ),
        make_candle(
            datetime(2020, 1, 2, 11, 0, tzinfo=timezone.utc),
            102.0,
            106.0,
            101.0,
            105.0,
        ),
        make_candle(
            datetime(2020, 1, 6, 10, 0, tzinfo=timezone.utc),
            105.0,
            108.0,
            103.0,
            107.0,
        ),
    ]

    result = count_candles_by_period(candles)

    assert result["day:2020-01-02"] == 2
    assert result["day:2020-01-06"] == 1

    assert result["week:2020-W01"] == 2

    assert result["week:2020-W02"] == 1
    assert result["month:2020-01"] == 3
