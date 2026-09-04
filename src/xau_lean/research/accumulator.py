from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from heapq import heappop, heappush
from math import log, sqrt
from typing import Iterable

from .report import ResearchMetadata, ResearchReport
from .timeframe import Timeframe, TimeframeCandle


@dataclass
class _OnlineMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def update(self, value: float) -> None:
        self.count += 1

        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

        if self.minimum is None or value < self.minimum:
            self.minimum = value
        if self.maximum is None or value > self.maximum:
            self.maximum = value

    @property
    def variance(self) -> float | None:
        if self.count < 2:
            return None
        return self.m2 / (self.count - 1)

    @property
    def standard_deviation(self) -> float | None:
        variance = self.variance
        if variance is None:
            return None
        return sqrt(max(variance, 0.0))


@dataclass
class _OnlineMedian:
    lower: list[float] = field(default_factory=list)
    upper: list[float] = field(default_factory=list)

    def update(self, value: float) -> None:
        if not self.lower or value <= -self.lower[0]:
            heappush(self.lower, -value)
        else:
            heappush(self.upper, value)

        if len(self.lower) > len(self.upper) + 1:
            heappush(self.upper, -heappop(self.lower))

        if len(self.upper) > len(self.lower):
            heappush(self.lower, -heappop(self.upper))

    @property
    def median(self) -> float | None:
        total = len(self.lower) + len(self.upper)
        if total == 0:
            return None

        if len(self.lower) > len(self.upper):
            return -self.lower[0]

        return (-self.lower[0] + self.upper[0]) / 2.0


class ResearchAccumulator:
    """
    Streaming accumulator for one timeframe.

    The accumulator never requires the complete candle history in memory.
    Only bounded rolling state and small period counters are retained.
    """

    def __init__(
        self,
        *,
        metadata: ResearchMetadata,
        atr_period: int = 14,
        spread_thresholds: Iterable[float] = (0.50, 1.0, 2.0, 5.0),
    ) -> None:
        if atr_period <= 0:
            raise ValueError("atr_period must be positive")

        self.metadata = metadata
        self.atr_period = atr_period

        thresholds = sorted(set(spread_thresholds))
        if any(value < 0 for value in thresholds):
            raise ValueError("spread thresholds must be non-negative")

        self._spread_thresholds = tuple(thresholds)

        self.candle_count = 0
        self.complete_candle_count = 0
        self.incomplete_candle_count = 0

        self.direction_counts = Counter(
            {
                1: 0,
                0: 0,
                -1: 0,
            }
        )

        self._previous_direction: int | None = None
        self._same_direction = 0
        self._reversals = 0
        self._direction_comparisons = 0

        self._previous_close: float | None = None

        self._range = _OnlineMoments()
        self._body = _OnlineMoments()
        self._body_ratio = _OnlineMoments()

        self._spread = _OnlineMoments()
        self._spread_to_range = _OnlineMoments()

        self._log_returns = _OnlineMoments()

        self._atr_values = _OnlineMoments()
        self._tr_window: deque[float] = deque(maxlen=atr_period)

        self._spread_threshold_counts = Counter()
        self._spread_observation_count = 0

        self._day_counts: Counter[str] = Counter()
        self._week_counts: Counter[str] = Counter()
        self._month_counts: Counter[str] = Counter()

        self._last_timestamp: datetime | None = None

    def update(self, candle: TimeframeCandle) -> None:
        timestamp = candle.timestamp

        if timestamp.tzinfo is None:
            raise ValueError("candle timestamp must be timezone-aware")

        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("candles must be strictly chronological")

        self._last_timestamp = timestamp

        self.candle_count += 1

        if candle.complete:
            self.complete_candle_count += 1
        else:
            self.incomplete_candle_count += 1

        self._day_counts[timestamp.strftime("%Y-%m-%d")] += 1
        self._week_counts[timestamp.strftime("%G-W%V")] += 1
        self._month_counts[timestamp.strftime("%Y-%m")] += 1

        candle_range = candle.bid_range
        body = candle.bid_body

        self._range.update(candle_range)
        self._body.update(body)

        if candle_range > 0:
            self._body_ratio.update(body / candle_range)

        if candle.bid_close > candle.bid_open:
            direction = 1
        elif candle.bid_close < candle.bid_open:
            direction = -1
        else:
            direction = 0

        self.direction_counts[direction] += 1

        if direction != 0 and self._previous_direction is not None:
            self._direction_comparisons += 1

            if direction == self._previous_direction:
                self._same_direction += 1
            else:
                self._reversals += 1

        if direction != 0:
            self._previous_direction = direction

        if self._previous_close is not None:
            if self._previous_close <= 0 or candle.bid_close <= 0:
                raise ValueError("prices must be positive")

            simple_return = candle.bid_close / self._previous_close - 1.0
            log_return = log(candle.bid_close / self._previous_close)

            # simple_return is intentionally computed for validation and
            # future extensions; realized volatility uses log returns.
            _ = simple_return
            self._log_returns.update(log_return)

        true_range = max(
            candle.bid_high - candle.bid_low,
            abs(candle.bid_high - self._previous_close)
            if self._previous_close is not None
            else 0.0,
            abs(candle.bid_low - self._previous_close)
            if self._previous_close is not None
            else 0.0,
        )

        self._tr_window.append(true_range)

        if len(self._tr_window) == self.atr_period:
            atr = sum(self._tr_window) / self.atr_period
            self._atr_values.update(atr)

        spread = candle.close_spread

        if spread is not None:
            self._spread.update(spread)
            self._spread_observation_count += 1

            for threshold in self._spread_thresholds:
                if spread >= threshold:
                    self._spread_threshold_counts[threshold] += 1

            if candle_range > 0:
                self._spread_to_range.update(spread / candle_range)

        self._previous_close = candle.bid_close

    def update_many(self, candles: Iterable[TimeframeCandle]) -> None:
        for candle in candles:
            self.update(candle)

    @staticmethod
    def _average_period_count(counter: Counter[str]) -> float | None:
        if not counter:
            return None
        return sum(counter.values()) / len(counter)

    def build_report(self) -> ResearchReport:
        if self._direction_comparisons:
            persistence_rate = (
                self._same_direction / self._direction_comparisons
            )
            reversal_rate = (
                self._reversals / self._direction_comparisons
            )
        else:
            persistence_rate = None
            reversal_rate = None

        threshold_percentages: dict[float, float] = {}

        for threshold in self._spread_thresholds:
            if self._spread_observation_count:
                threshold_percentages[threshold] = (
                    self._spread_threshold_counts[threshold]
                    / self._spread_observation_count
                )
            else:
                threshold_percentages[threshold] = 0.0

        return ResearchReport(
            metadata=self.metadata,
            candle_count=self.candle_count,
            complete_candle_count=self.complete_candle_count,
            incomplete_candle_count=self.incomplete_candle_count,
            bullish_count=self.direction_counts[1],
            bearish_count=self.direction_counts[-1],
            flat_count=self.direction_counts[0],
            persistence_rate=persistence_rate,
            reversal_rate=reversal_rate,
            mean_range=(
                self._range.mean if self._range.count else None
            ),
            median_range=None,
            mean_body=(
                self._body.mean if self._body.count else None
            ),
            median_body=None,
            mean_body_ratio=(
                self._body_ratio.mean
                if self._body_ratio.count
                else None
            ),
            realized_volatility=self._log_returns.standard_deviation,
            mean_atr=(
                self._atr_values.mean
                if self._atr_values.count
                else None
            ),
            mean_spread=(
                self._spread.mean
                if self._spread.count
                else None
            ),
            median_spread=None,
            mean_spread_to_range=(
                self._spread_to_range.mean
                if self._spread_to_range.count
                else None
            ),
            spread_ge_050_pct=threshold_percentages.get(0.50, 0.0),
            spread_ge_100_pct=threshold_percentages.get(1.0, 0.0),
            spread_ge_200_pct=threshold_percentages.get(2.0, 0.0),
            spread_ge_500_pct=threshold_percentages.get(5.0, 0.0),
            candles_per_day=self._average_period_count(self._day_counts),
            candles_per_week=self._average_period_count(self._week_counts),
            candles_per_month=self._average_period_count(self._month_counts),
        )
