from __future__ import annotations

from datetime import datetime, timezone
from math import log
from types import SimpleNamespace

import pytest

from xau_lean.research.regime_accumulator import (
    RegimeDistributionAccumulator,
)
from xau_lean.research.regimes import REGIMES


def make_timestamp(
    minute: int = 0,
    *,
    hour: int = 0,
    day: int = 4,
    month: int = 1,
    year: int = 2010,
) -> datetime:
    return datetime(
        year,
        month,
        day,
        hour,
        minute,
        tzinfo=timezone.utc,
    )


def make_candle(
    timestamp: datetime,
    *,
    open_price: float = 100.0,
    high: float = 102.0,
    low: float = 99.0,
    close: float = 101.0,
    spread: float | None = 0.50,
    complete: bool = True,
    include_ask: bool = True,
) -> SimpleNamespace:
    candle = SimpleNamespace(
        timestamp=timestamp,
        bid_open=open_price,
        bid_high=high,
        bid_low=low,
        bid_close=close,
        complete=complete,
    )

    if include_ask:
        if spread is None:
            spread = 0.50

        candle.ask_open = open_price + spread
        candle.ask_high = high + spread
        candle.ask_low = low + spread
        candle.ask_close = close + spread

    return candle


def make_accumulator(
    *,
    timeframe: str = "M5",
    atr_period: int = 14,
    acf_lags: tuple[int, ...] = (1, 5),
) -> RegimeDistributionAccumulator:
    return RegimeDistributionAccumulator(
        REGIMES[0],
        timeframe,
        atr_period=atr_period,
        acf_lags=acf_lags,
    )


def test_empty_accumulator_builds_valid_report() -> None:
    accumulator = make_accumulator()

    report = accumulator.build_report()

    assert report["timeframe"] == "M5"
    assert report["regime"]["name"] == REGIMES[0].name

    assert report["counts"]["candles"] == 0
    assert report["counts"]["complete"] == 0
    assert report["counts"]["incomplete"] == 0
    assert report["counts"]["completeness_ratio"] is None

    assert report["range"]["mean"] is None
    assert report["body"]["mean"] is None
    assert report["body_ratio"]["mean"] is None

    assert report["volatility"]["log_return_std"] is None
    assert report["volatility"]["mean_abs_log_return"] is None
    assert report["volatility"]["mean_squared_log_return"] is None
    assert report["volatility"]["atr_mean"] is None

    assert report["spread"]["mean"] is None
    assert report["spread_to_range"]["mean"] is None

    assert report["direction"]["bullish"] == 0
    assert report["direction"]["bearish"] == 0
    assert report["direction"]["flat"] == 0


def test_accumulator_tracks_basic_distribution_statistics() -> None:
    accumulator = make_accumulator()

    accumulator.update(
        make_candle(
            make_timestamp(0),
            open_price=100.0,
            high=103.0,
            low=99.0,
            close=102.0,
        )
    )

    accumulator.update(
        make_candle(
            make_timestamp(1),
            open_price=102.0,
            high=106.0,
            low=101.0,
            close=105.0,
        )
    )

    report = accumulator.build_report()

    assert report["counts"]["candles"] == 2
    assert report["counts"]["complete"] == 2
    assert report["counts"]["incomplete"] == 0
    assert report["counts"]["completeness_ratio"] == 1.0

    assert report["range"]["mean"] == pytest.approx(4.5)
    assert report["body"]["mean"] == pytest.approx(2.5)

    assert report["body_ratio"]["mean"] == pytest.approx(
        (2.0 / 4.0 + 3.0 / 5.0) / 2.0
    )

    assert report["spread"]["mean"] == pytest.approx(0.50)

    assert report["spread_to_range"]["mean"] == pytest.approx(
        ((0.50 / 4.0) + (0.50 / 5.0)) / 2.0
    )


def test_accumulator_tracks_directional_persistence_and_reversal() -> None:
    accumulator = make_accumulator()

    candles = [
        (100.0, 101.0),
        (101.0, 103.0),
        (103.0, 105.0),
        (105.0, 103.0),
        (103.0, 101.0),
    ]

    for minute, (open_price, close_price) in enumerate(candles):
        accumulator.update(
            make_candle(
                make_timestamp(minute),
                open_price=open_price,
                high=max(open_price, close_price) + 1.0,
                low=min(open_price, close_price) - 1.0,
                close=close_price,
            )
        )

    report = accumulator.build_report()

    assert report["direction"]["bullish"] == 3
    assert report["direction"]["bearish"] == 2
    assert report["direction"]["flat"] == 0

    behavior = report["directional_behavior"]

    assert behavior["persistence"] == 3
    assert behavior["reversals"] == 1

    # Two completed runs:
    # bullish = 3
    # bearish = 2
    # average = 2.5
    assert behavior["average_run_length"] == pytest.approx(2.5)
    assert behavior["max_run_length"] == 3


def test_flat_candles_are_counted_but_do_not_create_directional_runs() -> None:
    accumulator = make_accumulator()

    candles = [
        (100.0, 101.0),
        (101.0, 101.0),
        (101.0, 101.0),
        (101.0, 100.0),
    ]

    for minute, (open_price, close_price) in enumerate(candles):
        accumulator.update(
            make_candle(
                make_timestamp(minute),
                open_price=open_price,
                high=max(open_price, close_price) + 1.0,
                low=min(open_price, close_price) - 1.0,
                close=close_price,
            )
        )

    report = accumulator.build_report()

    assert report["direction"]["bullish"] == 1
    assert report["direction"]["bearish"] == 1
    assert report["direction"]["flat"] == 2

    behavior = report["directional_behavior"]

    assert behavior["persistence"] == 0
    assert behavior["reversals"] == 1
    assert behavior["average_run_length"] == pytest.approx(1.0)
    assert behavior["max_run_length"] == 1


def test_accumulator_tracks_hour_and_weekday() -> None:
    accumulator = make_accumulator()

    monday = make_timestamp(
        hour=9,
        minute=0,
    )

    monday_later = make_timestamp(
        hour=14,
        minute=0,
    )

    tuesday = make_timestamp(
        day=5,
        hour=9,
        minute=0,
    )

    for timestamp in (
        monday,
        monday_later,
        tuesday,
    ):
        accumulator.update(
            make_candle(timestamp)
        )

    report = accumulator.build_report()

    hours = report["hour_of_day"]
    weekdays = report["weekday"]

    assert hours["9"] == 2
    assert hours["14"] == 1

    assert weekdays[str(monday.weekday())] == 2
    assert weekdays[str(tuesday.weekday())] == 1

    assert sum(hours.values()) == 3
    assert sum(weekdays.values()) == 3


def test_accumulator_rejects_naive_timestamp() -> None:
    accumulator = make_accumulator()

    timestamp = datetime(
        2010,
        1,
        4,
        0,
        0,
    )

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        accumulator.update(
            make_candle(timestamp)
        )


def test_accumulator_rejects_outside_regime() -> None:
    accumulator = make_accumulator()

    timestamp = REGIMES[0].end

    with pytest.raises(
        ValueError,
        match="outside regime",
    ):
        accumulator.update(
            make_candle(timestamp)
        )


def test_regime_start_is_inclusive_and_regime_end_is_exclusive() -> None:
    accumulator = make_accumulator()

    accumulator.update(
        make_candle(REGIMES[0].start)
    )

    assert accumulator.candle_count == 1

    with pytest.raises(
        ValueError,
        match="outside regime",
    ):
        accumulator.update(
            make_candle(REGIMES[0].end)
        )


def test_accumulator_rejects_non_chronological_input() -> None:
    accumulator = make_accumulator()

    first = make_timestamp(5)
    second = make_timestamp(4)

    accumulator.update(
        make_candle(first)
    )

    with pytest.raises(
        ValueError,
        match="strictly chronological",
    ):
        accumulator.update(
            make_candle(second)
        )


def test_accumulator_rejects_duplicate_timestamp() -> None:
    accumulator = make_accumulator()

    timestamp = make_timestamp(5)

    accumulator.update(
        make_candle(timestamp)
    )

    with pytest.raises(
        ValueError,
        match="strictly chronological",
    ):
        accumulator.update(
            make_candle(timestamp)
        )


def test_quantile_output_contains_default_percentiles() -> None:
    accumulator = make_accumulator()

    for minute in range(10):
        accumulator.update(
            make_candle(
                make_timestamp(minute),
                open_price=100.0 + minute,
                high=103.0 + minute,
                low=99.0 + minute,
                close=102.0 + minute,
            )
        )

    report = accumulator.build_report()

    expected_percentiles = {
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
    }

    assert set(
        report["range"]["quantiles"]
    ) == expected_percentiles

    assert set(
        report["body"]["quantiles"]
    ) == expected_percentiles

    assert set(
        report["body_ratio"]["quantiles"]
    ) == expected_percentiles

    assert set(
        report["spread"]["quantiles"]
    ) == expected_percentiles

    assert set(
        report["spread_to_range"]["quantiles"]
    ) == expected_percentiles


def test_zero_range_has_zero_body_ratio() -> None:
    accumulator = make_accumulator()

    accumulator.update(
        make_candle(
            make_timestamp(0),
            open_price=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
        )
    )

    report = accumulator.build_report()

    assert report["range"]["mean"] == pytest.approx(0.0)
    assert report["body"]["mean"] == pytest.approx(0.0)
    assert report["body_ratio"]["mean"] == pytest.approx(0.0)


def test_incomplete_candles_are_tracked() -> None:
    accumulator = make_accumulator()

    accumulator.update(
        make_candle(
            make_timestamp(0),
            complete=True,
        )
    )

    accumulator.update(
        make_candle(
            make_timestamp(1),
            complete=False,
        )
    )

    report = accumulator.build_report()

    assert report["counts"]["candles"] == 2
    assert report["counts"]["complete"] == 1
    assert report["counts"]["incomplete"] == 1
    assert report["counts"]["completeness_ratio"] == pytest.approx(0.5)


def test_missing_ask_data_is_tracked_without_breaking_bid_statistics() -> None:
    accumulator = make_accumulator()

    accumulator.update(
        make_candle(
            make_timestamp(0),
            open_price=100.0,
            high=103.0,
            low=99.0,
            close=102.0,
            include_ask=False,
        )
    )

    report = accumulator.build_report()

    assert report["counts"]["candles"] == 1
    assert report["counts"]["missing_ask"] == 1

    assert report["range"]["mean"] == pytest.approx(4.0)
    assert report["body"]["mean"] == pytest.approx(2.0)

    assert report["spread"]["mean"] is None
    assert report["spread_to_range"]["mean"] is None


def test_abnormal_spread_is_counted() -> None:
    accumulator = make_accumulator()

    accumulator.update(
        make_candle(
            make_timestamp(0),
            spread=1.50,
        )
    )

    accumulator.update(
        make_candle(
            make_timestamp(1),
            spread=0.50,
        )
    )

    report = accumulator.build_report()

    assert report["counts"]["abnormal_spread"] == 1
    assert report["spread"]["mean"] == pytest.approx(1.0)


def test_spread_exactly_at_threshold_is_not_abnormal() -> None:
    accumulator = make_accumulator()

    accumulator.update(
        make_candle(
            make_timestamp(0),
            spread=1.0,
        )
    )

    report = accumulator.build_report()

    assert report["counts"]["abnormal_spread"] == 0


def test_atr_is_not_available_until_period_is_reached() -> None:
    accumulator = make_accumulator(
        atr_period=3
    )

    for minute in range(2):
        accumulator.update(
            make_candle(
                make_timestamp(minute),
                high=102.0,
                low=99.0,
                close=101.0,
            )
        )

    report = accumulator.build_report()

    assert report["volatility"]["atr_mean"] is None

    expected_percentiles = {
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
    }

    assert set(
        report["volatility"]["atr_quantiles"]
    ) == expected_percentiles

    assert all(
        value is None
        for value in report["volatility"]["atr_quantiles"].values()
    )


def test_atr_is_calculated_after_period_is_reached() -> None:
    accumulator = make_accumulator(
        atr_period=2
    )

    first = make_candle(
        make_timestamp(0),
        open_price=99.0,
        high=102.0,
        low=99.0,
        close=100.0,
    )

    second = make_candle(
        make_timestamp(1),
        open_price=104.0,
        high=106.0,
        low=104.0,
        close=105.0,
    )

    accumulator.update(first)
    accumulator.update(second)

    report = accumulator.build_report()

    # First true range = 3.
    #
    # Second true range:
    # max(
    #     high - low = 2,
    #     abs(high - previous_close) = 6,
    #     abs(low - previous_close) = 4,
    # ) = 6.
    #
    # ATR = (3 + 6) / 2 = 4.5.
    assert report["volatility"]["atr_mean"] == pytest.approx(
        4.5
    )


def test_true_range_uses_previous_close_for_gap() -> None:
    accumulator = make_accumulator(
        atr_period=2
    )

    accumulator.update(
        make_candle(
            make_timestamp(0),
            open_price=99.0,
            high=102.0,
            low=99.0,
            close=100.0,
        )
    )

    accumulator.update(
        make_candle(
            make_timestamp(1),
            open_price=105.0,
            high=106.0,
            low=104.0,
            close=105.0,
        )
    )

    report = accumulator.build_report()

    assert report["volatility"]["atr_mean"] == pytest.approx(
        4.5
    )


def test_density_counts_unique_days_weeks_and_months() -> None:
    accumulator = make_accumulator()

    timestamps = [
        make_timestamp(
            day=4,
            month=1,
            minute=0,
        ),
        make_timestamp(
            day=4,
            month=1,
            minute=1,
        ),
        make_timestamp(
            day=5,
            month=1,
            minute=0,
        ),
        make_timestamp(
            day=1,
            month=2,
            minute=0,
        ),
    ]

    for timestamp in timestamps:
        accumulator.update(
            make_candle(timestamp)
        )

    report = accumulator.build_report()
    density = report["density"]

    assert density["candles_per_day"] == pytest.approx(
        4.0 / 3.0
    )

    assert density["candles_per_week"] == pytest.approx(
        2.0
    )

    assert density["candles_per_month"] == pytest.approx(
        2.0
    )


def test_log_return_statistics_are_calculated() -> None:
    accumulator = make_accumulator()

    accumulator.update(
        make_candle(
            make_timestamp(0),
            close=100.0,
        )
    )

    accumulator.update(
        make_candle(
            make_timestamp(1),
            close=110.0,
        )
    )

    expected_return = log(
        110.0 / 100.0
    )

    report = accumulator.build_report()
    volatility = report["volatility"]

    assert volatility["mean_abs_log_return"] == pytest.approx(
        abs(expected_return)
    )

    assert volatility["mean_squared_log_return"] == pytest.approx(
        expected_return**2
    )

    assert volatility["log_return_std"] is None


def test_log_return_std_is_available_with_multiple_returns() -> None:
    accumulator = make_accumulator()

    closes = [
        100.0,
        110.0,
        99.0,
        108.0,
        102.0,
    ]

    for minute, close in enumerate(closes):
        accumulator.update(
            make_candle(
                make_timestamp(minute),
                close=close,
            )
        )

    report = accumulator.build_report()

    assert report["volatility"]["log_return_std"] is not None
    assert report["volatility"]["log_return_std"] >= 0.0


def test_spread_to_range_is_calculated_from_close_spread() -> None:
    accumulator = make_accumulator()

    accumulator.update(
        make_candle(
            make_timestamp(0),
            open_price=100.0,
            high=104.0,
            low=99.0,
            close=102.0,
            spread=0.50,
        )
    )

    report = accumulator.build_report()

    assert report["spread_to_range"]["mean"] == pytest.approx(
        0.50 / 5.0
    )


def test_acf_fields_are_present() -> None:
    accumulator = make_accumulator()

    closes = [
        100.0,
        101.0,
        103.0,
        102.0,
        105.0,
        107.0,
        104.0,
        108.0,
        110.0,
        109.0,
        113.0,
        115.0,
    ]

    for minute, close in enumerate(closes):
        accumulator.update(
            make_candle(
                make_timestamp(minute),
                close=close,
            )
        )

    report = accumulator.build_report()
    clustering = report["volatility_clustering"]

    assert set(
        clustering["abs_log_return_acf"]
    ) == {
        "lag_1",
        "lag_5",
    }

    assert set(
        clustering["squared_return_acf"]
    ) == {
        "lag_1",
        "lag_5",
    }


def test_acf_constant_series_returns_zero() -> None:
    accumulator = make_accumulator()

    for minute in range(8):
        accumulator.update(
            make_candle(
                make_timestamp(minute),
                close=100.0,
            )
        )

    report = accumulator.build_report()

    assert (
        report["volatility_clustering"]
        ["abs_log_return_acf"]["lag_1"]
        == 0.0
    )

    assert (
        report["volatility_clustering"]
        ["abs_log_return_acf"]["lag_5"]
        == 0.0
    )

    assert (
        report["volatility_clustering"]
        ["squared_return_acf"]["lag_1"]
        == 0.0
    )

    assert (
        report["volatility_clustering"]
        ["squared_return_acf"]["lag_5"]
        == 0.0
    )


def test_acf_is_none_when_insufficient_return_observations() -> None:
    accumulator = make_accumulator()

    accumulator.update(
        make_candle(
            make_timestamp(0),
            close=100.0,
        )
    )

    accumulator.update(
        make_candle(
            make_timestamp(1),
            close=101.0,
        )
    )

    report = accumulator.build_report()

    assert (
        report["volatility_clustering"]
        ["abs_log_return_acf"]["lag_1"]
        is None
    )

    assert (
        report["volatility_clustering"]
        ["squared_return_acf"]["lag_1"]
        is None
    )


def test_custom_acf_lags_are_respected() -> None:
    accumulator = make_accumulator(
        acf_lags=(2, 3)
    )

    closes = [
        100.0,
        101.0,
        103.0,
        102.0,
        105.0,
        107.0,
        104.0,
        108.0,
        110.0,
        109.0,
    ]

    for minute, close in enumerate(closes):
        accumulator.update(
            make_candle(
                make_timestamp(minute),
                close=close,
            )
        )

    report = accumulator.build_report()

    assert set(
        report["volatility_clustering"]
        ["abs_log_return_acf"]
    ) == {
        "lag_2",
        "lag_3",
    }

    assert set(
        report["volatility_clustering"]
        ["squared_return_acf"]
    ) == {
        "lag_2",
        "lag_3",
    }


def test_invalid_constructor_arguments_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="atr_period must be positive",
    ):
        make_accumulator(
            atr_period=0
        )

    with pytest.raises(
        ValueError,
        match="timeframe must not be empty",
    ):
        RegimeDistributionAccumulator(
            REGIMES[0],
            "",
        )

    with pytest.raises(
        ValueError,
        match="acf_lags must not be empty",
    ):
        make_accumulator(
            acf_lags=()
        )

    with pytest.raises(
        ValueError,
        match="positive integers",
    ):
        make_accumulator(
            acf_lags=(1, 0)
        )

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        make_accumulator(
            acf_lags=(1, 1)
        )


def test_build_report_does_not_mutate_current_run() -> None:
    accumulator = make_accumulator()

    accumulator.update(
        make_candle(
            make_timestamp(0),
            open_price=100.0,
            close=101.0,
        )
    )

    accumulator.update(
        make_candle(
            make_timestamp(1),
            open_price=101.0,
            close=102.0,
        )
    )

    accumulator.update(
        make_candle(
            make_timestamp(2),
            open_price=102.0,
            close=103.0,
        )
    )

    first_report = accumulator.build_report()
    second_report = accumulator.build_report()

    assert first_report == second_report

    assert (
        first_report["directional_behavior"]
        ["average_run_length"]
        == 3.0
    )

    assert (
        first_report["directional_behavior"]
        ["max_run_length"]
        == 3
    )

    assert accumulator._run_count == 0
    assert accumulator._current_run == 3


def test_current_run_is_included_in_average_and_max_run_length() -> None:
    accumulator = make_accumulator()

    for minute in range(4):
        accumulator.update(
            make_candle(
                make_timestamp(minute),
                open_price=100.0 + minute,
                close=101.0 + minute,
            )
        )

    report = accumulator.build_report()
    behavior = report["directional_behavior"]

    assert behavior["persistence"] == 3
    assert behavior["reversals"] == 0
    assert behavior["average_run_length"] == pytest.approx(
        4.0
    )
    assert behavior["max_run_length"] == 4


def test_report_contains_all_24_hours_and_7_weekdays() -> None:
    accumulator = make_accumulator()

    accumulator.update(
        make_candle(
            make_timestamp(0)
        )
    )

    report = accumulator.build_report()

    assert set(
        report["hour_of_day"]
    ) == {
        str(hour)
        for hour in range(24)
    }

    assert set(
        report["weekday"]
    ) == {
        str(day)
        for day in range(7)
    }


def test_last_timestamp_is_updated_after_valid_candle() -> None:
    accumulator = make_accumulator()

    timestamp = make_timestamp(7)

    accumulator.update(
        make_candle(timestamp)
    )

    assert accumulator.last_timestamp == timestamp


def test_complete_defaults_to_true_when_attribute_is_missing() -> None:
    accumulator = make_accumulator()

    candle = SimpleNamespace(
        timestamp=make_timestamp(0),
        bid_open=100.0,
        bid_high=102.0,
        bid_low=99.0,
        bid_close=101.0,
        ask_open=100.5,
        ask_high=102.5,
        ask_low=99.5,
        ask_close=101.5,
    )

    accumulator.update(candle)

    report = accumulator.build_report()

    assert report["counts"]["candles"] == 1
    assert report["counts"]["complete"] == 1
    assert report["counts"]["incomplete"] == 0