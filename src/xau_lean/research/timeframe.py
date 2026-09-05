"""
Deterministic XAUUSD timeframe aggregation.

Transforms canonical aligned M1 BID/ASK observations into research
timeframes without modifying the underlying data.

Supported timeframes:
    M5
    M15
    H1
    H4
    H6

Design principles:
- UTC only
- fixed calendar boundaries
- candle timestamp represents interval OPEN
- BID and ASK aggregated independently
- no interpolation
- no fill-forward
- explicit completeness metadata
- chronological, deterministic processing
- streaming operation
- missing intervals are reported, never fabricated
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Iterable, Iterator

from xau_lean.data.dukascopy import DukascopyBar


class Timeframe(str, Enum):
    """Supported research timeframes."""

    M5 = "M5"
    M15 = "M15"
    H1 = "H1"
    H4 = "H4"
    H6 = "H6"

    @property
    def minutes(self) -> int:
        """Return timeframe duration in minutes."""
        return {
            Timeframe.M5: 5,
            Timeframe.M15: 15,
            Timeframe.H1: 60,
            Timeframe.H4: 240,
            Timeframe.H6: 360,
        }[self]


@dataclass(frozen=True, slots=True)
class TimeframeCandle:
    """Aggregated BID/ASK research candle."""

    timestamp: datetime

    bid_open: float
    bid_high: float
    bid_low: float
    bid_close: float

    ask_open: float | None
    ask_high: float | None
    ask_low: float | None
    ask_close: float | None

    expected_m1: int
    observed_m1: int

    @property
    def missing_m1(self) -> int:
        """Number of expected M1 observations not observed."""
        return max(0, self.expected_m1 - self.observed_m1)

    @property
    def completeness_ratio(self) -> float:
        """Fraction of expected M1 observations that were observed."""
        if self.expected_m1 <= 0:
            return 0.0

        return self.observed_m1 / self.expected_m1

    @property
    def complete(self) -> bool:
        """Whether all expected M1 observations are present."""
        return self.observed_m1 == self.expected_m1

    @property
    def bid_range(self) -> float:
        """BID candle range."""
        return self.bid_high - self.bid_low

    @property
    def bid_body(self) -> float:
        """Absolute BID candle body."""
        return abs(self.bid_close - self.bid_open)

    @property
    def ask_range(self) -> float | None:
        """ASK candle range when ASK exists."""
        if self.ask_high is None or self.ask_low is None:
            return None

        return self.ask_high - self.ask_low

    @property
    def mid_open(self) -> float | None:
        """Derived midpoint at candle open."""
        if self.ask_open is None:
            return None

        return (self.bid_open + self.ask_open) / 2.0

    @property
    def mid_high(self) -> float | None:
        """Derived midpoint high."""
        if self.ask_high is None:
            return None

        return (self.bid_high + self.ask_high) / 2.0

    @property
    def mid_low(self) -> float | None:
        """Derived midpoint low."""
        if self.ask_low is None:
            return None

        return (self.bid_low + self.ask_low) / 2.0

    @property
    def mid_close(self) -> float | None:
        """Derived midpoint at candle close."""
        if self.ask_close is None:
            return None

        return (self.bid_close + self.ask_close) / 2.0

    @property
    def close_spread(self) -> float | None:
        """ASK-BID spread at candle close."""
        if self.ask_close is None:
            return None

        return self.ask_close - self.bid_close


@dataclass(frozen=True, slots=True)
class TimeframeGap:
    """
    Gap between consecutive observed M1 timestamps.

    This is separate from candle completeness so that a missing observation
    is never confused with a fabricated candle.
    """

    previous_timestamp: datetime
    current_timestamp: datetime
    missing_minutes: int

    @property
    def duration(self) -> timedelta:
        """Elapsed time between the two observed timestamps."""
        return self.current_timestamp - self.previous_timestamp


def _ensure_utc(timestamp: datetime) -> datetime:
    """Validate and normalize a timestamp to UTC."""
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")

    return timestamp.astimezone(timezone.utc)


def interval_start(
    timestamp: datetime,
    timeframe: Timeframe,
) -> datetime:
    """
    Return the UTC opening timestamp of the timeframe containing timestamp.

    Boundaries are anchored to midnight UTC.
    """
    timestamp = _ensure_utc(timestamp)

    total_minutes = timestamp.hour * 60 + timestamp.minute
    bucket_minutes = timeframe.minutes

    bucket_start = (total_minutes // bucket_minutes) * bucket_minutes

    hour, minute = divmod(bucket_start, 60)

    return timestamp.replace(
        hour=hour,
        minute=minute,
        second=0,
        microsecond=0,
    )


def _aggregate_group(
    group: list[DukascopyBar],
    timeframe: Timeframe,
) -> TimeframeCandle:
    """Aggregate one chronological M1 group."""
    if not group:
        raise ValueError("Cannot aggregate an empty group")

    expected_m1 = timeframe.minutes

    bid_open = group[0].bid_open
    bid_high = max(bar.bid_high for bar in group)
    bid_low = min(bar.bid_low for bar in group)
    bid_close = group[-1].bid_close

    ask_complete = all(
        bar.ask_open is not None
        and bar.ask_high is not None
        and bar.ask_low is not None
        and bar.ask_close is not None
        for bar in group
    )

    if ask_complete:
        ask_open = group[0].ask_open
        ask_high = max(bar.ask_high for bar in group)
        ask_low = min(bar.ask_low for bar in group)
        ask_close = group[-1].ask_close
    else:
        ask_open = None
        ask_high = None
        ask_low = None
        ask_close = None

    return TimeframeCandle(
        timestamp=interval_start(group[0].timestamp, timeframe),
        bid_open=bid_open,
        bid_high=bid_high,
        bid_low=bid_low,
        bid_close=bid_close,
        ask_open=ask_open,
        ask_high=ask_high,
        ask_low=ask_low,
        ask_close=ask_close,
        expected_m1=expected_m1,
        observed_m1=len(group),
    )


def aggregate_bars(
    bars: Iterable[DukascopyBar],
    timeframe: Timeframe,
) -> Iterator[TimeframeCandle]:
    """
    Aggregate chronological M1 BID/ASK bars into one timeframe.

    The function is fully streaming and does not materialize the complete
    input dataset.
    """
    current_interval: datetime | None = None
    group: list[DukascopyBar] = []
    previous_timestamp: datetime | None = None

    for raw_bar in bars:
        timestamp = _ensure_utc(raw_bar.timestamp)

        if (
            previous_timestamp is not None
            and timestamp <= previous_timestamp
        ):
            raise ValueError(
                "bars must be strictly chronological"
            )

        previous_timestamp = timestamp

        bucket = interval_start(timestamp, timeframe)

        if current_interval is None:
            current_interval = bucket

        if bucket != current_interval:
            if group:
                yield _aggregate_group(
                    group,
                    timeframe,
                )

            current_interval = bucket
            group = []

        group.append(raw_bar)

    if group:
        yield _aggregate_group(
            group,
            timeframe,
        )


def aggregate_timeframe(
    bars: Iterable[DukascopyBar],
    timeframe: Timeframe,
) -> Iterator[TimeframeCandle]:
    """Streaming alias for aggregate_bars()."""
    yield from aggregate_bars(
        bars,
        timeframe,
    )


def aggregate_timeframes(
    bars: Iterable[DukascopyBar],
    timeframes: Iterable[Timeframe],
) -> Iterator[tuple[Timeframe, TimeframeCandle]]:
    """
    Stream one chronological M1 source into multiple timeframes.

    Each M1 observation is consumed exactly once. A small active aggregation
    group is maintained for each requested timeframe.

    Results are yielded as (timeframe, candle) pairs whenever an aggregation
    interval closes.

    Completely absent intervals are not fabricated.
    """
    selected_timeframes = tuple(dict.fromkeys(timeframes))

    if not selected_timeframes:
        raise ValueError("At least one timeframe is required")

    groups: dict[Timeframe, list[DukascopyBar]] = {}
    current_intervals: dict[Timeframe, datetime] = {}

    previous_timestamp: datetime | None = None

    for raw_bar in bars:
        timestamp = _ensure_utc(raw_bar.timestamp)

        if (
            previous_timestamp is not None
            and timestamp <= previous_timestamp
        ):
            raise ValueError(
                "bars must be strictly chronological"
            )

        previous_timestamp = timestamp

        for timeframe in selected_timeframes:
            bucket = interval_start(
                timestamp,
                timeframe,
            )

            if timeframe not in current_intervals:
                current_intervals[timeframe] = bucket
                groups[timeframe] = [raw_bar]
                continue

            if bucket != current_intervals[timeframe]:
                yield (
                    timeframe,
                    _aggregate_group(
                        groups[timeframe],
                        timeframe,
                    ),
                )

                current_intervals[timeframe] = bucket
                groups[timeframe] = [raw_bar]
            else:
                groups[timeframe].append(raw_bar)

    for timeframe in selected_timeframes:
        group = groups.get(timeframe)

        if group:
            yield (
                timeframe,
                _aggregate_group(
                    group,
                    timeframe,
                ),
            )


def find_gaps(
    bars: Iterable[DukascopyBar],
    *,
    minimum_gap_minutes: int = 2,
) -> Iterator[TimeframeGap]:
    """
    Detect gaps between consecutive observed M1 timestamps.

    Session closures and weekends are intentionally not classified here.
    This function reports temporal gaps only.
    """
    if minimum_gap_minutes < 1:
        raise ValueError("minimum_gap_minutes must be at least 1")

    previous_timestamp: datetime | None = None

    for raw_bar in bars:
        timestamp = _ensure_utc(raw_bar.timestamp)

        if (
            previous_timestamp is not None
            and timestamp <= previous_timestamp
        ):
            raise ValueError(
                "bars must be strictly chronological"
            )

        if previous_timestamp is not None:
            elapsed_minutes = int(
                (
                    timestamp - previous_timestamp
                ).total_seconds()
                // 60
            )

            missing_minutes = elapsed_minutes - 1

            if missing_minutes >= minimum_gap_minutes:
                yield TimeframeGap(
                    previous_timestamp=previous_timestamp,
                    current_timestamp=timestamp,
                    missing_minutes=missing_minutes,
                )

        previous_timestamp = timestamp


__all__ = [
    "Timeframe",
    "TimeframeCandle",
    "TimeframeGap",
    "interval_start",
    "aggregate_bars",
    "aggregate_timeframe",
    "aggregate_timeframes",
    "find_gaps",
]
