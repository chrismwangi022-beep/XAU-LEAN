"""XAU-LEAN research interfaces."""

from .timeframe import (
    Timeframe,
    TimeframeCandle,
    TimeframeGap,
    aggregate_bars,
    aggregate_timeframe,
    find_gaps,
    interval_start,
)

__all__ = [
    "Timeframe",
    "TimeframeCandle",
    "TimeframeGap",
    "aggregate_bars",
    "aggregate_timeframe",
    "find_gaps",
    "interval_start",
]
