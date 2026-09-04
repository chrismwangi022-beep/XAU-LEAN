"""
Production Dukascopy XAUUSD data adapter.

Canonical source:
    data/external/dukascopy/Market-Data-Lab-main/xauusd/

The raw Dukascopy files are immutable. This module only reads them.

Design principles:
- UTC timestamps internally
- Monthly BID/ASK files
- No interpolation
- No fill-forward
- No deletion of raw observations
- Explicit BID/ASK alignment
- Streaming rather than loading the full 2010-present history
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True, slots=True)
class DukascopyBar:
    """One aligned XAUUSD minute observation."""

    timestamp: datetime
    bid_open: float
    bid_high: float
    bid_low: float
    bid_close: float
    ask_open: float | None
    ask_high: float | None
    ask_low: float | None
    ask_close: float | None

    @property
    def spread(self) -> float | None:
        """Close-price ASK-BID spread, if both sides exist."""
        if self.ask_close is None:
            return None
        return self.ask_close - self.bid_close

    @property
    def mid_close(self) -> float | None:
        """Midpoint of BID/ASK close prices."""
        if self.ask_close is None:
            return None
        return (self.bid_close + self.ask_close) / 2.0


class DukascopyDataError(RuntimeError):
    """Raised when the canonical Dukascopy structure is invalid."""


class DukascopyAdapter:
    """
    Read canonical Dukascopy XAUUSD monthly BID/ASK data.

    The adapter deliberately operates month-by-month so that a research
    job can request only the period it actually needs.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

        self.bid_dir = self.root / "bid" / "m1"
        self.ask_dir = self.root / "ask" / "m1"

        if not self.bid_dir.is_dir():
            raise DukascopyDataError(
                f"BID directory does not exist: {self.bid_dir}"
            )

        if not self.ask_dir.is_dir():
            raise DukascopyDataError(
                f"ASK directory does not exist: {self.ask_dir}"
            )

    @staticmethod
    def _month_range(
        start: datetime,
        end: datetime,
    ) -> Iterator[tuple[int, int]]:
        """Yield YYYY, MM pairs between start and end inclusive."""

        year = start.year
        month = start.month

        while (year, month) <= (end.year, end.month):
            yield year, month

            if month == 12:
                year += 1
                month = 1
            else:
                month += 1

    def _file_for(
        self,
        side: str,
        year: int,
        month: int,
    ) -> Path:
        filename = f"xauusd_{side}_m1_{year:04d}_{month:02d}.csv"

        if side == "bid":
            path = self.bid_dir / filename
        elif side == "ask":
            path = self.ask_dir / filename
        else:
            raise ValueError(f"Unknown side: {side}")

        return path

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        """Convert Unix milliseconds to timezone-aware UTC datetime."""

        try:
            milliseconds = int(value)
        except ValueError as exc:
            raise DukascopyDataError(
                f"Invalid timestamp: {value!r}"
            ) from exc

        return datetime.fromtimestamp(
            milliseconds / 1000.0,
            tz=timezone.utc,
        )

    @staticmethod
    def _parse_ohlc(parts: list[str], line_number: int) -> tuple[float, ...]:
        if len(parts) != 5:
            raise DukascopyDataError(
                f"Expected 5 columns at line {line_number}, got {len(parts)}"
            )

        try:
            values = tuple(float(value) for value in parts[1:5])
        except ValueError as exc:
            raise DukascopyDataError(
                f"Invalid OHLC at line {line_number}: {parts!r}"
            ) from exc

        if any(value <= 0 for value in values):
            raise DukascopyDataError(
                f"Non-positive OHLC at line {line_number}: {values}"
            )

        return values

    def _read_side(
        self,
        path: Path,
        start: datetime,
        end: datetime,
    ) -> Iterator[tuple[datetime, tuple[float, ...]]]:
        """
        Stream one monthly file.

        Rows outside [start, end) are ignored by the adapter but never
        modified in the source file.
        """

        if not path.is_file():
            return

        previous: datetime | None = None

        with path.open("r", encoding="utf-8", newline="") as handle:
            header = handle.readline().strip()

            if header != "timestamp,open,high,low,close":
                raise DukascopyDataError(
                    f"Unexpected header in {path}: {header!r}"
                )

            for line_number, raw_line in enumerate(handle, start=2):
                line = raw_line.strip()

                if not line:
                    continue

                parts = line.split(",")
                timestamp = self._parse_timestamp(parts[0])
                values = self._parse_ohlc(parts, line_number)

                if previous is not None and timestamp <= previous:
                    raise DukascopyDataError(
                        f"Non-increasing timestamp in {path} "
                        f"at line {line_number}: {timestamp.isoformat()}"
                    )

                previous = timestamp

                if timestamp < start:
                    continue

                if timestamp >= end:
                    continue

                yield timestamp, values

    def iter_bars(
        self,
        start: datetime,
        end: datetime,
        *,
        require_ask: bool = True,
    ) -> Iterator[DukascopyBar]:
        """
        Stream aligned BID/ASK minute observations.

        Parameters
        ----------
        start:
            Inclusive UTC start.
        end:
            Exclusive UTC end.
        require_ask:
            If True, observations without ASK are omitted from the
            processed stream. The raw ASK-only/BID-only anomaly remains
            untouched in the source files.

        Notes
        -----
        The adapter uses a timestamp merge rather than assuming that
        every BID timestamp exists in ASK.
        """

        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware")

        start = start.astimezone(timezone.utc)
        end = end.astimezone(timezone.utc)

        if end <= start:
            raise ValueError("end must be after start")

        for year, month in self._month_range(start, end):
            bid_path = self._file_for("bid", year, month)
            ask_path = self._file_for("ask", year, month)

            bid_iter = self._read_side(bid_path, start, end)
            ask_iter = self._read_side(ask_path, start, end)

            bid_row = next(bid_iter, None)
            ask_row = next(ask_iter, None)

            while bid_row is not None or ask_row is not None:
                if bid_row is None:
                    # ASK-only source observation.
                    ask_row = next(ask_iter, None)
                    continue

                if ask_row is None:
                    # BID-only source observation.
                    bid_row = next(bid_iter, None)
                    continue

                bid_time, bid_values = bid_row
                ask_time, ask_values = ask_row

                if bid_time < ask_time:
                    bid_row = next(bid_iter, None)
                    continue

                if ask_time < bid_time:
                    ask_row = next(ask_iter, None)
                    continue

                # Exact timestamp alignment.
                yield DukascopyBar(
                    timestamp=bid_time,
                    bid_open=bid_values[0],
                    bid_high=bid_values[1],
                    bid_low=bid_values[2],
                    bid_close=bid_values[3],
                    ask_open=ask_values[0],
                    ask_high=ask_values[1],
                    ask_low=ask_values[2],
                    ask_close=ask_values[3],
                )

                bid_row = next(bid_iter, None)
                ask_row = next(ask_iter, None)

    def count_bars(
        self,
        start: datetime,
        end: datetime,
    ) -> int:
        """Count aligned BID/ASK observations without storing them."""

        return sum(1 for _ in self.iter_bars(start, end))

    def months_available(self) -> list[tuple[int, int]]:
        """Return months for which both BID and ASK files exist."""

        months: set[tuple[int, int]] = set()

        for path in self.bid_dir.glob("xauusd_bid_m1_*.csv"):
            stem = path.stem
            suffix = stem.removeprefix("xauusd_bid_m1_")

            try:
                year, month = map(int, suffix.split("_"))
            except ValueError:
                continue

            months.add((year, month))

        result = []

        for year, month in sorted(months):
            if self._file_for("ask", year, month).is_file():
                result.append((year, month))

        return result
