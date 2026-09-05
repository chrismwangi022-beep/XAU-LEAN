from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Iterator

from .timeframe import Timeframe, interval_start


@dataclass(frozen=True, slots=True)
class CoverageInterval:
    """Observed M1 coverage for one expected timeframe interval."""

    timestamp: datetime
    expected_m1: int
    observed_m1: int

    @property
    def missing_m1(self) -> int:
        """Number of M1 observations missing from the interval."""
        return max(0, self.expected_m1 - self.observed_m1)

    @property
    def completeness_ratio(self) -> float:
        """Fraction of expected M1 observations observed."""
        if self.expected_m1 <= 0:
            return 0.0

        return self.observed_m1 / self.expected_m1

    @property
    def complete(self) -> bool:
        """Whether all expected M1 observations are present."""
        return self.observed_m1 == self.expected_m1


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    """Streaming coverage summary for a requested research interval."""

    expected_intervals: int
    complete_intervals: int
    partial_intervals: int
    missing_intervals: int
    expected_m1_per_interval: int
    observed_m1: int

    @property
    def incomplete_intervals(self) -> int:
        """Intervals that exist but are not fully covered."""
        return self.partial_intervals

    @property
    def expected_m1(self) -> int:
        """Total M1 observations expected across all intervals."""
        return self.expected_intervals * self.expected_m1_per_interval

    @property
    def missing_m1(self) -> int:
        """Total M1 observations missing across all expected intervals."""
        return max(0, self.expected_m1 - self.observed_m1)

    @property
    def completeness_ratio(self) -> float:
        """Fraction of expected M1 observations observed."""
        if self.expected_m1 <= 0:
            return 0.0

        return self.observed_m1 / self.expected_m1


def _ensure_utc(timestamp: datetime) -> datetime:
    """Validate and normalize a timestamp to UTC."""
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")

    return timestamp.astimezone(timezone.utc)


def iter_coverage_intervals(
    timestamps: Iterable[datetime],
    *,
    start: datetime,
    end: datetime,
    timeframe: Timeframe,
) -> Iterator[CoverageInterval]:
    """
    Stream expected timeframe intervals and observed M1 counts.

    Completely missing intervals are explicitly emitted with zero
    observations. No data is fabricated; this is coverage metadata only.
    """
    start = _ensure_utc(start)
    end = _ensure_utc(end)

    if end <= start:
        raise ValueError("end must be after start")

    current_interval = interval_start(start, timeframe)
    interval_delta = timedelta(minutes=timeframe.minutes)

    observed_count = 0
    previous_timestamp: datetime | None = None

    for raw_timestamp in timestamps:
        timestamp = _ensure_utc(raw_timestamp)

        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError(
                "timestamps must be strictly chronological"
            )

        previous_timestamp = timestamp

        if timestamp < start:
            continue

        if timestamp >= end:
            break

        timestamp_interval = interval_start(timestamp, timeframe)

        while current_interval < timestamp_interval:
            if current_interval >= start:
                yield CoverageInterval(
                    timestamp=current_interval,
                    expected_m1=timeframe.minutes,
                    observed_m1=observed_count,
                )

            current_interval += interval_delta
            observed_count = 0

        if current_interval >= start and current_interval < end:
            observed_count += 1

    while current_interval < end:
        if current_interval >= start:
            yield CoverageInterval(
                timestamp=current_interval,
                expected_m1=timeframe.minutes,
                observed_m1=observed_count,
            )

        current_interval += interval_delta
        observed_count = 0


class CoverageAccumulator:
    """
    Streaming coverage tracker for M1 timestamps.

    The accumulator does not retain the individual timestamps or intervals.
    It keeps only the state required to produce the final CoverageSummary.
    """

    def __init__(
        self,
        *,
        start: datetime,
        end: datetime,
        timeframe: Timeframe,
    ) -> None:
        self.start = _ensure_utc(start)
        self.end = _ensure_utc(end)

        if self.end <= self.start:
            raise ValueError("end must be after start")

        self.timeframe = timeframe
        self.expected_m1_per_interval = timeframe.minutes
        self.interval_delta = timedelta(minutes=timeframe.minutes)

        self.current_interval = interval_start(
            self.start,
            timeframe,
        )

        self.observed_count = 0
        self.previous_timestamp: datetime | None = None

        self.expected_intervals = 0
        self.complete_intervals = 0
        self.partial_intervals = 0
        self.missing_intervals = 0
        self.observed_m1 = 0

        self._finalized = False

    def _record_current_interval(self) -> None:
        """Finalize the current interval and update summary counters."""
        self.expected_intervals += 1
        self.observed_m1 += self.observed_count

        if self.observed_count == 0:
            self.missing_intervals += 1
        elif self.observed_count == self.expected_m1_per_interval:
            self.complete_intervals += 1
        else:
            self.partial_intervals += 1

    def update(self, timestamp: datetime) -> None:
        """Consume one M1 timestamp."""
        if self._finalized:
            raise RuntimeError(
                "cannot update CoverageAccumulator after build_summary()"
            )

        timestamp = _ensure_utc(timestamp)

        if (
            self.previous_timestamp is not None
            and timestamp <= self.previous_timestamp
        ):
            raise ValueError(
                "timestamps must be strictly chronological"
            )

        self.previous_timestamp = timestamp

        if timestamp < self.start:
            return

        if timestamp >= self.end:
            return

        timestamp_interval = interval_start(
            timestamp,
            self.timeframe,
        )

        while self.current_interval < timestamp_interval:
            if self.current_interval >= self.start:
                self._record_current_interval()

            self.current_interval += self.interval_delta
            self.observed_count = 0

        if (
            self.current_interval >= self.start
            and self.current_interval < self.end
        ):
            self.observed_count += 1

    def update_many(self, timestamps: Iterable[datetime]) -> None:
        """Consume timestamps from any iterable without materializing them."""
        for timestamp in timestamps:
            self.update(timestamp)

    def build_summary(self) -> CoverageSummary:
        """Finalize remaining intervals and return the coverage summary."""
        if not self._finalized:
            while self.current_interval < self.end:
                if self.current_interval >= self.start:
                    self._record_current_interval()

                self.current_interval += self.interval_delta
                self.observed_count = 0

            self._finalized = True

        return CoverageSummary(
            expected_intervals=self.expected_intervals,
            complete_intervals=self.complete_intervals,
            partial_intervals=self.partial_intervals,
            missing_intervals=self.missing_intervals,
            expected_m1_per_interval=self.expected_m1_per_interval,
            observed_m1=self.observed_m1,
        )


def analyze_coverage(
    timestamps: Iterable[datetime],
    *,
    start: datetime,
    end: datetime,
    timeframe: Timeframe,
) -> CoverageSummary:
    """Return streaming coverage statistics for the requested interval."""
    accumulator = CoverageAccumulator(
        start=start,
        end=end,
        timeframe=timeframe,
    )

    accumulator.update_many(timestamps)

    return accumulator.build_summary()


__all__ = [
    "CoverageInterval",
    "CoverageSummary",
    "CoverageAccumulator",
    "iter_coverage_intervals",
    "analyze_coverage",
]
