"""Streaming distributional research accumulator for Phase 6.1.

This module collects regime-specific descriptive statistics for one
timeframe/regime cell without retaining the full observation history.

The accumulator is intentionally separate from the original Phase 6
ResearchAccumulator. Phase 6 remains unchanged and serves as the
baseline descriptive research layer.

Phase 6.1 adds:

- distribution quantiles
- volatility diagnostics
- spread diagnostics
- directional behavior
- run-length behavior
- hour/weekday distributions
- volatility clustering diagnostics

All calculations are performed online with bounded memory.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date, datetime
from math import log, sqrt
from typing import Any

from .quantiles import StreamingQuantiles
from .regimes import ResearchRegime


DEFAULT_ATR_PERIOD = 14
DEFAULT_ACF_LAGS = (1, 5)


class _OnlineMoments:
    """Bounded-memory online mean/variance accumulator."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum: float | None = None
        self.maximum: float | None = None

    def update(self, value: float) -> None:
        value = float(value)

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
    def std(self) -> float | None:
        variance = self.variance

        if variance is None:
            return None

        return sqrt(variance)


@dataclass(slots=True)
class _LagProduct:
    """Streaming lag-product accumulator.

    For a series x_t and lag k, stores:

        sum(x_t * x_(t-k))

    together with the corresponding current and lagged sums.

    This allows the lag-k sample autocorrelation numerator to be
    reconstructed without retaining the full observation history.
    """

    lag: int
    count: int = 0
    product_sum: float = 0.0
    current_sum: float = 0.0
    lagged_sum: float = 0.0

    def update(self, current: float, lagged: float) -> None:
        self.count += 1
        self.product_sum += current * lagged
        self.current_sum += current
        self.lagged_sum += lagged


class RegimeDistributionAccumulator:
    """Streaming descriptive accumulator for one timeframe/regime cell.

    Parameters
    ----------
    regime:
        The canonical ResearchRegime assigned to this accumulator.

    timeframe:
        Timeframe name, e.g. "M5", "M15", "H1", "H4", or "H6".

    atr_period:
        ATR period. Defaults to 14.

    acf_lags:
        Volatility-clustering autocorrelation lags.
        Defaults to lag 1 and lag 5.
    """

    def __init__(
        self,
        regime: ResearchRegime,
        timeframe: str,
        *,
        atr_period: int = DEFAULT_ATR_PERIOD,
        acf_lags: tuple[int, ...] = DEFAULT_ACF_LAGS,
    ) -> None:
        if atr_period <= 0:
            raise ValueError("atr_period must be positive")

        if not timeframe:
            raise ValueError("timeframe must not be empty")

        if not acf_lags:
            raise ValueError("acf_lags must not be empty")

        if any(lag <= 0 for lag in acf_lags):
            raise ValueError(
                "acf_lags must contain positive integers"
            )

        if len(set(acf_lags)) != len(acf_lags):
            raise ValueError("acf_lags must be unique")

        self.regime = regime
        self.timeframe = timeframe
        self.atr_period = atr_period
        self.acf_lags = tuple(acf_lags)

        # ----------------------------------------------------------
        # Distribution quantiles
        # ----------------------------------------------------------

        self._range_quantiles = StreamingQuantiles()
        self._body_quantiles = StreamingQuantiles()
        self._body_ratio_quantiles = StreamingQuantiles()
        self._spread_quantiles = StreamingQuantiles()
        self._spread_range_quantiles = StreamingQuantiles()
        self._atr_quantiles = StreamingQuantiles()

        # ----------------------------------------------------------
        # First/second moments
        # ----------------------------------------------------------

        self._range = _OnlineMoments()
        self._body = _OnlineMoments()
        self._body_ratio = _OnlineMoments()
        self._spread = _OnlineMoments()
        self._spread_range = _OnlineMoments()
        self._log_returns = _OnlineMoments()
        self._abs_log_returns = _OnlineMoments()
        self._squared_log_returns = _OnlineMoments()
        self._atr = _OnlineMoments()

        # ----------------------------------------------------------
        # Directional behavior
        # ----------------------------------------------------------

        self.bullish = 0
        self.bearish = 0
        self.flat = 0

        self.persistence = 0
        self.reversals = 0

        self._previous_direction: int | None = None
        self._current_run = 0
        self._run_count = 0
        self._run_total = 0
        self._max_run = 0

        # ----------------------------------------------------------
        # Temporal density
        # ----------------------------------------------------------

        self._days: set[date] = set()
        self._weeks: set[tuple[int, int]] = set()
        self._months: set[tuple[int, int]] = set()

        self._hour_counts = [0] * 24
        self._weekday_counts = [0] * 7

        # ----------------------------------------------------------
        # Volatility / ATR state
        # ----------------------------------------------------------

        self._previous_close: float | None = None
        self._tr_window: deque[float] = deque(maxlen=atr_period)

        # ----------------------------------------------------------
        # Volatility clustering state
        #
        # Absolute returns and squared returns deliberately use
        # separate bounded histories because they are different
        # time series.
        # ----------------------------------------------------------

        self._acf_abs_history: dict[int, deque[float]] = {
            lag: deque(maxlen=lag)
            for lag in self.acf_lags
        }

        self._acf_squared_history: dict[int, deque[float]] = {
            lag: deque(maxlen=lag)
            for lag in self.acf_lags
        }

        self._acf_abs_products = {
            lag: _LagProduct(lag)
            for lag in self.acf_lags
        }

        self._acf_squared_products = {
            lag: _LagProduct(lag)
            for lag in self.acf_lags
        }

        # ----------------------------------------------------------
        # Coverage / quality
        # ----------------------------------------------------------

        self.candle_count = 0
        self.complete_candles = 0
        self.incomplete_candles = 0
        self.abnormal_spread_candles = 0
        self.missing_ask_candles = 0

        self.last_timestamp: datetime | None = None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _direction(
        open_price: float,
        close_price: float,
    ) -> int:
        if close_price > open_price:
            return 1

        if close_price < open_price:
            return -1

        return 0

    def _update_run(self, direction: int) -> None:
        # Flat candles do not start, continue, or terminate
        # directional runs. This keeps run statistics focused
        # on actual direction.
        if direction == 0:
            return

        if self._previous_direction is None:
            self._current_run = 1

        elif direction == self._previous_direction:
            self.persistence += 1
            self._current_run += 1

        else:
            self.reversals += 1

            if self._current_run > 0:
                self._run_count += 1
                self._run_total += self._current_run
                self._max_run = max(
                    self._max_run,
                    self._current_run,
                )

            self._current_run = 1

        self._previous_direction = direction

    def _update_clustering_pair(
        self,
        value: float,
        products: dict[int, _LagProduct],
        history: dict[int, deque[float]],
    ) -> None:
        """Update one bounded-memory lag-product series."""

        for lag in self.acf_lags:
            lag_history = history[lag]

            if len(lag_history) == lag:
                lagged = lag_history[0]
                products[lag].update(value, lagged)

            lag_history.append(value)

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, candle: Any) -> None:
        """Add one timeframe candle.

        The candle is expected to expose:

            timestamp
            bid_open
            bid_high
            bid_low
            bid_close
            ask_open
            ask_high
            ask_low
            ask_close

        If a candle exposes ``complete``, it is used for completeness
        accounting. Otherwise the candle is treated as complete.
        """

        timestamp = candle.timestamp

        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError(
                "candle timestamp must be timezone-aware"
            )

        if self.last_timestamp is not None:
            if timestamp <= self.last_timestamp:
                raise ValueError(
                    "candles must be strictly chronological"
                )

        if not (
            self.regime.start
            <= timestamp
            < self.regime.end
        ):
            raise ValueError(
                f"candle timestamp {timestamp.isoformat()} "
                f"is outside regime {self.regime.name}"
            )

        self.last_timestamp = timestamp
        self.candle_count += 1

        complete = getattr(candle, "complete", True)

        if complete:
            self.complete_candles += 1
        else:
            self.incomplete_candles += 1

        bid_open = float(candle.bid_open)
        bid_high = float(candle.bid_high)
        bid_low = float(candle.bid_low)
        bid_close = float(candle.bid_close)

        ask_open = getattr(candle, "ask_open", None)
        ask_high = getattr(candle, "ask_high", None)
        ask_low = getattr(candle, "ask_low", None)
        ask_close = getattr(candle, "ask_close", None)

        # ----------------------------------------------------------
        # Range / body
        # ----------------------------------------------------------

        candle_range = bid_high - bid_low
        body = abs(bid_close - bid_open)

        body_ratio = (
            body / candle_range
            if candle_range > 0.0
            else 0.0
        )

        self._range.update(candle_range)
        self._range_quantiles.update(candle_range)

        self._body.update(body)
        self._body_quantiles.update(body)

        self._body_ratio.update(body_ratio)
        self._body_ratio_quantiles.update(body_ratio)

        # ----------------------------------------------------------
        # Direction
        # ----------------------------------------------------------

        direction = self._direction(
            bid_open,
            bid_close,
        )

        if direction > 0:
            self.bullish += 1
        elif direction < 0:
            self.bearish += 1
        else:
            self.flat += 1

        self._update_run(direction)

        # ----------------------------------------------------------
        # Spread
        # ----------------------------------------------------------

        if (
            ask_open is None
            or ask_high is None
            or ask_low is None
            or ask_close is None
        ):
            self.missing_ask_candles += 1
            spread = None

        else:
            ask_open = float(ask_open)
            ask_high = float(ask_high)
            ask_low = float(ask_low)
            ask_close = float(ask_close)

            spread = ask_close - bid_close

        if spread is not None:
            self._spread.update(spread)
            self._spread_quantiles.update(spread)

            if candle_range > 0.0:
                spread_range = spread / candle_range

                self._spread_range.update(spread_range)
                self._spread_range_quantiles.update(
                    spread_range
                )

            if spread > 1.0:
                self.abnormal_spread_candles += 1

        # ----------------------------------------------------------
        # Log return / true range / ATR
        # ----------------------------------------------------------

        if (
            self._previous_close is not None
            and self._previous_close > 0.0
        ):
            log_return = self._safe_log_return(
                bid_close,
                self._previous_close,
            )

            self._log_returns.update(log_return)

            abs_return = abs(log_return)
            squared_return = log_return * log_return

            self._abs_log_returns.update(abs_return)
            self._squared_log_returns.update(squared_return)

            self._update_clustering_pair(
                abs_return,
                self._acf_abs_products,
                self._acf_abs_history,
            )

            self._update_clustering_pair(
                squared_return,
                self._acf_squared_products,
                self._acf_squared_history,
            )

        true_range = self._true_range(
            bid_high,
            bid_low,
            self._previous_close,
        )

        self._tr_window.append(true_range)

        if len(self._tr_window) == self.atr_period:
            atr = sum(self._tr_window) / self.atr_period

            self._atr.update(atr)
            self._atr_quantiles.update(atr)

        self._previous_close = bid_close

        # ----------------------------------------------------------
        # Time distributions
        # ----------------------------------------------------------

        self._days.add(timestamp.date())

        iso = timestamp.isocalendar()

        self._weeks.add(
            (iso.year, iso.week)
        )

        self._months.add(
            (timestamp.year, timestamp.month)
        )

        self._hour_counts[timestamp.hour] += 1
        self._weekday_counts[timestamp.weekday()] += 1

    @staticmethod
    def _safe_log_return(
        current: float,
        previous: float,
    ) -> float:
        """Calculate a log return from two positive prices."""

        if current <= 0.0 or previous <= 0.0:
            raise ValueError(
                "prices must be positive for log returns"
            )

        return log(current / previous)

    @staticmethod
    def _true_range(
        high: float,
        low: float,
        previous_close: float | None,
    ) -> float:
        if previous_close is None:
            return high - low

        return max(
            high - low,
            abs(high - previous_close),
            abs(low - previous_close),
        )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_or_none(
        moments: _OnlineMoments,
    ) -> float | None:
        if moments.count == 0:
            return None

        return moments.mean

    @staticmethod
    def _acf(
        moments: _OnlineMoments,
        pair: _LagProduct,
    ) -> float | None:
        """Calculate sample autocorrelation for a lagged series.

        Denominator:

            sum((x_t - mean)^2)

        Numerator:

            sum((x_t - mean)(x_(t-k) - mean))
        """

        if pair.count == 0 or moments.count < 2:
            return None

        denominator = moments.m2

        if denominator <= 0.0:
            return 0.0

        mean = moments.mean

        numerator = (
            pair.product_sum
            - mean * pair.current_sum
            - mean * pair.lagged_sum
            + pair.count * mean * mean
        )

        return numerator / denominator

    @property
    def average_run_length(self) -> float | None:
        """Average completed directional run length."""

        if self._run_count == 0:
            return None

        return self._run_total / self._run_count

    @property
    def completeness_ratio(self) -> float | None:
        if self.candle_count == 0:
            return None

        return self.complete_candles / self.candle_count

    def build_report(self) -> dict[str, Any]:
        """Build a serializable regime distribution report.

        The current directional run is included in run-length
        statistics without mutating the accumulator.
        """

        completed_runs = self._run_count
        completed_total = self._run_total
        completed_max = self._max_run

        if self._current_run > 0:
            completed_runs += 1
            completed_total += self._current_run
            completed_max = max(
                completed_max,
                self._current_run,
            )

        average_run = (
            completed_total / completed_runs
            if completed_runs > 0
            else None
        )

        volatility_acf = {}
        squared_return_acf = {}

        for lag in self.acf_lags:
            volatility_acf[f"lag_{lag}"] = self._acf(
                self._abs_log_returns,
                self._acf_abs_products[lag],
            )

            squared_return_acf[f"lag_{lag}"] = self._acf(
                self._squared_log_returns,
                self._acf_squared_products[lag],
            )

        return {
            "timeframe": self.timeframe,
            "regime": {
                "name": self.regime.name,
                "start": self.regime.start.isoformat(),
                "end": self.regime.end.isoformat(),
            },
            "counts": {
                "candles": self.candle_count,
                "complete": self.complete_candles,
                "incomplete": self.incomplete_candles,
                "completeness_ratio": self.completeness_ratio,
                "missing_ask": self.missing_ask_candles,
                "abnormal_spread": self.abnormal_spread_candles,
            },
            "range": {
                "mean": self._mean_or_none(self._range),
                "quantiles": self._range_quantiles.to_dict(),
            },
            "body": {
                "mean": self._mean_or_none(self._body),
                "quantiles": self._body_quantiles.to_dict(),
            },
            "body_ratio": {
                "mean": self._mean_or_none(self._body_ratio),
                "quantiles": self._body_ratio_quantiles.to_dict(),
            },
            "volatility": {
                "log_return_std": self._log_returns.std,
                "mean_abs_log_return": (
                    self._abs_log_returns.mean
                    if self._abs_log_returns.count > 0
                    else None
                ),
                "mean_squared_log_return": (
                    self._squared_log_returns.mean
                    if self._squared_log_returns.count > 0
                    else None
                ),
                "atr_mean": (
                    self._atr.mean
                    if self._atr.count > 0
                    else None
                ),
                "atr_quantiles": self._atr_quantiles.to_dict(),
            },
            "spread": {
                "mean": (
                    self._spread.mean
                    if self._spread.count > 0
                    else None
                ),
                "quantiles": self._spread_quantiles.to_dict(),
            },
            "spread_to_range": {
                "mean": (
                    self._spread_range.mean
                    if self._spread_range.count > 0
                    else None
                ),
                "quantiles": (
                    self._spread_range_quantiles.to_dict()
                ),
            },
            "direction": {
                "bullish": self.bullish,
                "bearish": self.bearish,
                "flat": self.flat,
            },
            "directional_behavior": {
                "persistence": self.persistence,
                "reversals": self.reversals,
                "average_run_length": average_run,
                "max_run_length": completed_max,
            },
            "density": {
                "candles_per_day": (
                    self.candle_count / len(self._days)
                    if self._days
                    else None
                ),
                "candles_per_week": (
                    self.candle_count / len(self._weeks)
                    if self._weeks
                    else None
                ),
                "candles_per_month": (
                    self.candle_count / len(self._months)
                    if self._months
                    else None
                ),
            },
            "hour_of_day": {
                str(hour): count
                for hour, count in enumerate(
                    self._hour_counts
                )
            },
            "weekday": {
                str(day): count
                for day, count in enumerate(
                    self._weekday_counts
                )
            },
            "volatility_clustering": {
                "abs_log_return_acf": volatility_acf,
                "squared_return_acf": squared_return_acf,
            },
        }


__all__ = [
    "RegimeDistributionAccumulator",
]