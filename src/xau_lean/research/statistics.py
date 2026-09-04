from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Iterable

from .timeframe import TimeframeCandle


@dataclass(frozen=True, slots=True)
class CandleStatistics:
    """Descriptive statistics for a single timeframe candle."""

    timestamp: object

    bid_range: float
    bid_body: float
    body_ratio: float | None

    upper_wick: float
    lower_wick: float

    direction: int

    simple_return: float | None
    log_return: float | None

    close_spread: float | None
    spread_to_range: float | None


def candle_statistics(
    candle: TimeframeCandle,
    previous_close: float | None = None,
) -> CandleStatistics:
    """
    Calculate descriptive statistics for one timeframe candle.

    Direction:
        +1 = bullish
         0 = flat
        -1 = bearish

    Returns are calculated only when a valid previous close exists.
    No future information is used.
    """

    bid_range = candle.bid_high - candle.bid_low
    bid_body = abs(candle.bid_close - candle.bid_open)

    if bid_range > 0:
        body_ratio = bid_body / bid_range
    else:
        body_ratio = None

    upper_wick = candle.bid_high - max(
        candle.bid_open,
        candle.bid_close,
    )

    lower_wick = min(
        candle.bid_open,
        candle.bid_close,
    ) - candle.bid_low

    if candle.bid_close > candle.bid_open:
        direction = 1
    elif candle.bid_close < candle.bid_open:
        direction = -1
    else:
        direction = 0

    simple_return = None
    log_return = None

    if previous_close is not None:
        if previous_close <= 0:
            raise ValueError("previous_close must be positive")

        if candle.bid_close <= 0:
            raise ValueError("bid_close must be positive")

        simple_return = candle.bid_close / previous_close - 1.0
        log_return = log(candle.bid_close / previous_close)

    close_spread = candle.close_spread

    if close_spread is not None and bid_range > 0:
        spread_to_range = close_spread / bid_range
    else:
        spread_to_range = None

    return CandleStatistics(
        timestamp=candle.timestamp,
        bid_range=bid_range,
        bid_body=bid_body,
        body_ratio=body_ratio,
        upper_wick=upper_wick,
        lower_wick=lower_wick,
        direction=direction,
        simple_return=simple_return,
        log_return=log_return,
        close_spread=close_spread,
        spread_to_range=spread_to_range,
    )


def iter_candle_statistics(
    candles: Iterable[TimeframeCandle],
) -> Iterable[CandleStatistics]:
    """
    Stream candle statistics chronologically.

    The first candle has no return because there is no previous
    close available. Subsequent returns use only the immediately
    preceding candle.
    """

    previous_close: float | None = None

    for candle in candles:
        statistics = candle_statistics(
            candle,
            previous_close=previous_close,
        )

        yield statistics

        previous_close = candle.bid_close


def true_range(
    high: float,
    low: float,
    previous_close: float | None = None,
) -> float:
    """
    Calculate True Range for one candle.

    TR = max(
        high - low,
        abs(high - previous_close),
        abs(low - previous_close),
    )

    When previous_close is unavailable, high-low is used.
    """

    if high <= 0:
        raise ValueError("high must be positive")

    if low <= 0:
        raise ValueError("low must be positive")

    if high < low:
        raise ValueError("high must be greater than or equal to low")

    if previous_close is None:
        return high - low

    if previous_close <= 0:
        raise ValueError("previous_close must be positive")

    return max(
        high - low,
        abs(high - previous_close),
        abs(low - previous_close),
    )


def realized_volatility(
    log_returns: Iterable[float],
) -> float | None:
    """
    Calculate standard deviation of chronological log returns.

    Returns None when fewer than two observations are available.

    This is a descriptive volatility measure and does not annualize
    the result because the caller may be working with different
    timeframe frequencies.
    """

    values = list(log_returns)

    if len(values) < 2:
        return None

    mean = sum(values) / len(values)

    variance = sum(
        (value - mean) ** 2
        for value in values
    ) / (len(values) - 1)

    return variance ** 0.5


def rolling_atr(
    candles: Iterable[TimeframeCandle],
    period: int = 14,
) -> Iterable[float | None]:
    """
    Stream a simple rolling ATR using True Range.

    The period is intentionally fixed and is not an optimization
    parameter.

    The first period-1 observations return None.
    """

    if period <= 0:
        raise ValueError("period must be positive")

    true_ranges: list[float] = []
    previous_close: float | None = None

    for candle in candles:
        current_true_range = true_range(
            candle.bid_high,
            candle.bid_low,
            previous_close,
        )

        true_ranges.append(current_true_range)

        if len(true_ranges) < period:
            yield None
        else:
            window = true_ranges[-period:]
            yield sum(window) / period

        previous_close = candle.bid_close


def direction_counts(
    directions: Iterable[int],
) -> dict[str, int]:
    """
    Count bullish, bearish, and flat candles.

    Direction convention:
        +1 = bullish
         0 = flat
        -1 = bearish
    """

    counts = {
        "bullish": 0,
        "bearish": 0,
        "flat": 0,
    }

    for direction in directions:
        if direction == 1:
            counts["bullish"] += 1
        elif direction == -1:
            counts["bearish"] += 1
        elif direction == 0:
            counts["flat"] += 1
        else:
            raise ValueError(
                "direction must be -1, 0, or 1"
            )

    return counts


def directional_persistence(
    directions: Iterable[int],
) -> dict[str, float | int | None]:
    """
    Calculate descriptive directional persistence.

    A persistence observation occurs when two consecutive
    non-flat candles have the same direction.

    Flat candles do not create a bullish or bearish continuation
    and are excluded from the directional comparison.

    Returns:
        directional_observations:
            Number of comparable non-flat transitions.

        same_direction:
            Number of transitions continuing in the same direction.

        reversals:
            Number of transitions changing direction.

        persistence_rate:
            same_direction / directional_observations.

        reversal_rate:
            reversals / directional_observations.
    """

    previous_direction: int | None = None
    directional_observations = 0
    same_direction = 0
    reversals = 0

    for direction in directions:
        if direction not in (-1, 0, 1):
            raise ValueError(
                "direction must be -1, 0, or 1"
            )

        if direction == 0:
            continue

        if previous_direction is not None:
            directional_observations += 1

            if direction == previous_direction:
                same_direction += 1
            else:
                reversals += 1

        previous_direction = direction

    if directional_observations == 0:
        persistence_rate = None
        reversal_rate = None
    else:
        persistence_rate = (
            same_direction / directional_observations
        )
        reversal_rate = (
            reversals / directional_observations
        )

    return {
        "directional_observations": directional_observations,
        "same_direction": same_direction,
        "reversals": reversals,
        "persistence_rate": persistence_rate,
        "reversal_rate": reversal_rate,
    }


def directional_run_lengths(
    directions: Iterable[int],
) -> list[int]:
    """
    Return lengths of consecutive runs of the same non-flat direction.

    Flat candles terminate the current directional run.

    Example:
        [1, 1, 1, -1, -1, 0, 1]
        -> [3, 2, 1]
    """

    runs: list[int] = []

    previous_direction: int | None = None
    current_length = 0

    for direction in directions:
        if direction not in (-1, 0, 1):
            raise ValueError(
                "direction must be -1, 0, or 1"
            )

        if direction == 0:
            if current_length > 0:
                runs.append(current_length)

            previous_direction = None
            current_length = 0
            continue

        if direction == previous_direction:
            current_length += 1
        else:
            if current_length > 0:
                runs.append(current_length)

            previous_direction = direction
            current_length = 1

    if current_length > 0:
        runs.append(current_length)

    return runs


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    """Basic descriptive summary for a numeric distribution."""

    count: int
    mean: float | None
    median: float | None
    minimum: float | None
    maximum: float | None


def summarize_distribution(
    values: Iterable[float],
) -> DistributionSummary:
    """Calculate deterministic descriptive statistics."""

    data = list(values)

    if not data:
        return DistributionSummary(
            count=0,
            mean=None,
            median=None,
            minimum=None,
            maximum=None,
        )

    ordered = sorted(data)
    count = len(ordered)

    if count % 2 == 1:
        median = ordered[count // 2]
    else:
        middle = count // 2
        median = (
            ordered[middle - 1] + ordered[middle]
        ) / 2.0

    return DistributionSummary(
        count=count,
        mean=sum(ordered) / count,
        median=median,
        minimum=ordered[0],
        maximum=ordered[-1],
    )


def spread_to_range_values(
    statistics: Iterable[CandleStatistics],
) -> Iterable[float]:
    """Yield spread/range ratios where both values are valid."""

    for value in statistics:
        if (
            value.close_spread is not None
            and value.bid_range > 0
        ):
            yield value.close_spread / value.bid_range


def spread_threshold_percentages(
    statistics: Iterable[CandleStatistics],
    thresholds: Iterable[float] = (0.50, 1.0, 2.0, 5.0),
) -> dict[float, float]:
    """
    Calculate the percentage of candles whose close spread
    is greater than or equal to each descriptive threshold.

    Thresholds are descriptive reporting bins, not trading rules.
    """

    threshold_values = sorted(set(thresholds))

    if any(value < 0 for value in threshold_values):
        raise ValueError("thresholds must be non-negative")

    spreads = [
        value.close_spread
        for value in statistics
        if value.close_spread is not None
    ]

    if not spreads:
        return {
            threshold: 0.0
            for threshold in threshold_values
        }

    count = len(spreads)

    return {
        threshold: (
            sum(
                spread >= threshold
                for spread in spreads
            )
            / count
        )
        for threshold in threshold_values
    }


def count_candles_by_period(
    candles: Iterable[TimeframeCandle],
) -> dict[str, int]:
    """
    Count candles by UTC calendar day, ISO week, and month.

    Keys are:
        day: YYYY-MM-DD
        week: YYYY-Www
        month: YYYY-MM
    """

    result: dict[str, int] = {}

    for candle in candles:
        timestamp = candle.timestamp

        if timestamp.tzinfo is None:
            raise ValueError(
                "candle timestamp must be timezone-aware"
            )

        day_key = timestamp.strftime("%Y-%m-%d")
        week_key = timestamp.strftime("%G-W%V")
        month_key = timestamp.strftime("%Y-%m")

        for key in (
            f"day:{day_key}",
            f"week:{week_key}",
            f"month:{month_key}",
        ):
            result[key] = result.get(key, 0) + 1

    return result
