"""
Canonical market-regime definitions for XAUUSD research.

Phase 6.1 uses fixed, non-overlapping UTC calendar periods.
All intervals are half-open: [start, end).

These definitions are descriptive research partitions only.
They are not strategy-selection parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class ResearchRegime:
    """One fixed UTC research regime."""

    name: str
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        """Validate regime boundaries."""

        if self.start.tzinfo is None:
            raise ValueError("regime start must be timezone-aware")

        if self.end.tzinfo is None:
            raise ValueError("regime end must be timezone-aware")

        start_utc = self.start.astimezone(timezone.utc)
        end_utc = self.end.astimezone(timezone.utc)

        if end_utc <= start_utc:
            raise ValueError(
                "regime end must be after regime start"
            )

    def contains(self, timestamp: datetime) -> bool:
        """Return True when timestamp belongs to this regime."""

        if timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")

        timestamp_utc = timestamp.astimezone(timezone.utc)

        return (
            self.start.astimezone(timezone.utc)
            <= timestamp_utc
            < self.end.astimezone(timezone.utc)
        )


REGIMES: tuple[ResearchRegime, ...] = (
    ResearchRegime(
        name="2010-2014",
        start=datetime(
            2010, 1, 1, tzinfo=timezone.utc
        ),
        end=datetime(
            2015, 1, 1, tzinfo=timezone.utc
        ),
    ),
    ResearchRegime(
        name="2015-2019",
        start=datetime(
            2015, 1, 1, tzinfo=timezone.utc
        ),
        end=datetime(
            2020, 1, 1, tzinfo=timezone.utc
        ),
    ),
    ResearchRegime(
        name="2020-2021",
        start=datetime(
            2020, 1, 1, tzinfo=timezone.utc
        ),
        end=datetime(
            2022, 1, 1, tzinfo=timezone.utc
        ),
    ),
    ResearchRegime(
        name="2022-2023",
        start=datetime(
            2022, 1, 1, tzinfo=timezone.utc
        ),
        end=datetime(
            2024, 1, 1, tzinfo=timezone.utc
        ),
    ),
    ResearchRegime(
        name="2024-2026",
        start=datetime(
            2024, 1, 1, tzinfo=timezone.utc
        ),
        end=datetime(
            2026, 8, 21, tzinfo=timezone.utc
        ),
    ),
)


def get_regime(
    timestamp: datetime,
    regimes: tuple[ResearchRegime, ...] = REGIMES,
) -> ResearchRegime | None:
    """
    Return the regime containing timestamp.

    Returns None when timestamp falls outside the configured
    research window.
    """

    if timestamp.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")

    timestamp_utc = timestamp.astimezone(timezone.utc)

    for regime in regimes:
        if (
            regime.start <= timestamp_utc < regime.end
        ):
            return regime

    return None


def validate_regimes(
    regimes: tuple[ResearchRegime, ...] = REGIMES,
) -> None:
    """
    Validate that regimes are chronological and non-overlapping.
    """

    if not regimes:
        raise ValueError("at least one regime is required")

    normalized = tuple(
        sorted(
            regimes,
            key=lambda regime: (
                regime.start.astimezone(timezone.utc)
            ),
        )
    )

    for previous, current in zip(
        normalized,
        normalized[1:],
    ):
        previous_end = previous.end.astimezone(
            timezone.utc
        )
        current_start = current.start.astimezone(
            timezone.utc
        )

        if current_start < previous_end:
            raise ValueError(
                "research regimes must not overlap: "
                f"{previous.name} and {current.name}"
            )


validate_regimes()


__all__ = [
    "ResearchRegime",
    "REGIMES",
    "get_regime",
    "validate_regimes",
]
