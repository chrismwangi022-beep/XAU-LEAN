"""
XAUUSD session and tradability layer.

This module does NOT alter raw market data.

Its purpose is to decide whether an observation is suitable for
research/execution logic.

Session hours remain configurable because broker/venue schedules,
DST conventions, and execution environments can differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum


class SessionStatus(str, Enum):
    TRADABLE = "TRADABLE"
    WEEKEND = "WEEKEND"
    STALE = "STALE"
    ABNORMAL_SPREAD = "ABNORMAL_SPREAD"
    MISSING_ASK = "MISSING_ASK"


@dataclass(frozen=True, slots=True)
class TradabilityConfig:
    """
    Configuration for downstream tradability decisions.

    spread_warning:
        Spread at or above this value is classified as abnormal.

    stale_after:
        Maximum age of a quote before it is considered stale.

    weekend_filter:
        Prevent Saturday/Sunday observations from being treated as
        tradable by default.

    Notes:
        The default spread threshold is intentionally configurable and
        is not a claim about broker execution conditions.
    """

    spread_warning: float = 1.0
    stale_after: timedelta = timedelta(minutes=2)
    weekend_filter: bool = True


def evaluate_tradability(
    timestamp: datetime,
    *,
    spread: float | None,
    last_timestamp: datetime | None = None,
    config: TradabilityConfig | None = None,
) -> SessionStatus:
    """
    Evaluate whether a quote should be considered tradable.

    This is deliberately conservative:
    - missing ASK is not tradable
    - weekend observations are not tradable by default
    - stale observations are not tradable
    - abnormal spreads are flagged
    """

    config = config or TradabilityConfig()

    if timestamp.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")

    timestamp = timestamp.astimezone(timezone.utc)

    if spread is None:
        return SessionStatus.MISSING_ASK

    if spread < 0:
        return SessionStatus.ABNORMAL_SPREAD

    if config.weekend_filter and timestamp.weekday() >= 5:
        return SessionStatus.WEEKEND

    if last_timestamp is not None:
        if last_timestamp.tzinfo is None:
            raise ValueError("last_timestamp must be timezone-aware")

        last_timestamp = last_timestamp.astimezone(timezone.utc)

        if timestamp - last_timestamp > config.stale_after:
            return SessionStatus.STALE

    if spread >= config.spread_warning:
        return SessionStatus.ABNORMAL_SPREAD

    return SessionStatus.TRADABLE
