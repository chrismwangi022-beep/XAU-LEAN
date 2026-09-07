"""
Phase 6.3A — XAUUSD Momentum Direction Research.

Hypothesis:
    When volatility expands, does the direction of the expansion
    candle predict the direction of subsequent XAUUSD movement?

Methodology:
    1. Build validated timeframe candles from canonical M1 BID/ASK data.
    2. Calculate ATR(14) from completed candles.
    3. Compare current ATR against the mean of the PREVIOUS 20 ATR values.
    4. Classify volatility expansion using frozen Phase 6.2 thresholds.
    5. Classify the expansion candle direction:
           bullish = close > open
           bearish = close < open
           flat    = close == open
    6. Signal is known only after candle t closes.
    7. Entry occurs at candle t+1 OPEN.
    8. Measure signed forward return at 1, 3 and 5 subsequent candles.
    9. Report results by:
           regime × volatility bucket × candle direction × horizon.

This is RESEARCH, not a trading strategy.
No optimization is performed.
No future information is used to create signals.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from xau_lean.data.dukascopy import DukascopyAdapter
from xau_lean.research.regimes import (
    REGIMES,
    get_regime,
    validate_regimes,
)
from xau_lean.research.timeframe import (
    Timeframe,
    aggregate_timeframe,
    aggregate_timeframes,
)


# ---------------------------------------------------------------------------
# Frozen research parameters
# ---------------------------------------------------------------------------

ATR_PERIOD = 14
BASELINE_PERIOD = 20
HORIZONS = (1, 3, 5)

BUCKETS = (
    ("normal", 0.0, 1.25),
    ("elevated", 1.25, 1.50),
    ("high", 1.50, 2.00),
    ("extreme", 2.00, float("inf")),
)

SOURCE_NAME = "Dukascopy XAUUSD BID/ASK M1"
TIMEZONE_NAME = "UTC"

DEFAULT_DATA_ROOT = Path(
    "data/external/dukascopy/Market-Data-Lab-main/xauusd"
)

DEFAULT_OUTPUT_ROOT = Path(
    "research_results/direction"
)

ALL_TIMEFRAMES = tuple(Timeframe)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_datetime(value: str) -> datetime:
    """Parse ISO-8601 datetime and normalize it to UTC."""
    text = value.strip()

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    parsed = datetime.fromisoformat(text)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def format_timestamp(value: datetime) -> str:
    """Return stable UTC timestamp for filenames."""
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .strftime("%Y%m%dT%H%M%SZ")
    )


def selected_timeframes(name: str) -> tuple[Timeframe, ...]:
    """Resolve CLI timeframe selection."""
    if name == "ALL":
        return ALL_TIMEFRAMES

    return (Timeframe[name],)


def classify_bucket(ratio: float) -> str:
    """Classify ATR expansion ratio using frozen thresholds."""
    for name, lower, upper in BUCKETS:
        if lower <= ratio < upper:
            return name

    raise ValueError(f"Unable to classify expansion ratio: {ratio}")


def candle_direction(candle: Any) -> str:
    """Return bullish, bearish or flat direction."""
    if candle.bid_close > candle.bid_open:
        return "bullish"

    if candle.bid_close < candle.bid_open:
        return "bearish"

    return "flat"


def true_range(
    high: float,
    low: float,
    previous_close: float | None,
) -> float:
    """Calculate Wilder-style true range."""
    if previous_close is None:
        return high - low

    return max(
        high - low,
        abs(high - previous_close),
        abs(low - previous_close),
    )


# ---------------------------------------------------------------------------
# Pending signal
# ---------------------------------------------------------------------------

@dataclass
class PendingSignal:
    """
    Signal generated after candle t closes.

    Entry is taken at candle t+1 OPEN.
    """

    regime: str
    bucket: str
    direction: str
    signal_timestamp: datetime
    signal_close: float

    entry_price: float | None = None
    entry_timestamp: datetime | None = None
    candles_since_entry: int = 0


# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------

@dataclass
class ResultStats:
    """Collect directional research observations."""

    observations: int = 0
    positive: int = 0
    negative: int = 0
    flat: int = 0

    signed_returns: list[float] | None = None
    abs_returns: list[float] | None = None
    directional_efficiency: list[float] | None = None

    def __post_init__(self) -> None:
        if self.signed_returns is None:
            self.signed_returns = []

        if self.abs_returns is None:
            self.abs_returns = []

        if self.directional_efficiency is None:
            self.directional_efficiency = []

    def update(
        self,
        signed_return: float,
        abs_return: float,
        efficiency: float,
    ) -> None:
        self.observations += 1

        if signed_return > 0:
            self.positive += 1
        elif signed_return < 0:
            self.negative += 1
        else:
            self.flat += 1

        self.signed_returns.append(signed_return)
        self.abs_returns.append(abs_return)
        self.directional_efficiency.append(efficiency)

    def build(self) -> dict[str, Any]:
        if self.observations == 0:
            return {
                "observations": 0,
                "positive": 0,
                "negative": 0,
                "flat": 0,
                "directional_accuracy": None,
                "mean_signed_return_pct": None,
                "median_signed_return_pct": None,
                "mean_abs_return_pct": None,
                "mean_directional_efficiency": None,
            }

        assert self.signed_returns is not None
        assert self.abs_returns is not None
        assert self.directional_efficiency is not None

        return {
            "observations": self.observations,
            "positive": self.positive,
            "negative": self.negative,
            "flat": self.flat,
            "directional_accuracy": (
                self.positive / self.observations
            ),
            "mean_signed_return_pct": (
                sum(self.signed_returns)
                / self.observations
                * 100.0
            ),
            "median_signed_return_pct": (
                median(self.signed_returns) * 100.0
            ),
            "mean_abs_return_pct": (
                sum(self.abs_returns)
                / self.observations
                * 100.0
            ),
            "mean_directional_efficiency": (
                sum(self.directional_efficiency)
                / self.observations
            ),
        }


# ---------------------------------------------------------------------------
# Research engine
# ---------------------------------------------------------------------------

class MomentumResearch:
    """Streaming Phase 6.3A momentum researcher for one timeframe."""

    def __init__(self, timeframe: Timeframe) -> None:
        self.timeframe = timeframe

        self.previous_close: float | None = None

        self.tr_window: deque[float] = deque(
            maxlen=ATR_PERIOD
        )

        # Contains COMPLETED ATR values only.
        # The current ATR is deliberately excluded from the baseline.
        self.atr_history: deque[float] = deque(
            maxlen=BASELINE_PERIOD
        )

        self.pending: dict[int, list[PendingSignal]] = {
            horizon: []
            for horizon in HORIZONS
        }

        self.results: dict[
            tuple[str, str, str, int],
            ResultStats,
        ] = {}

        self.candles = 0
        self.complete_candles = 0
        self.atr_ready = 0
        self.signals_created = 0
        self.signals_resolved = 0

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def stats(
        self,
        regime: str,
        bucket: str,
        direction: str,
        horizon: int,
    ) -> ResultStats:
        key = (
            regime,
            bucket,
            direction,
            horizon,
        )

        if key not in self.results:
            self.results[key] = ResultStats()

        return self.results[key]

    # ------------------------------------------------------------------
    # Candle processing
    # ------------------------------------------------------------------

    def process(self, candle: Any) -> None:
        self.candles += 1

        if not getattr(candle, "complete", True):
            return

        self.complete_candles += 1

        # --------------------------------------------------------------
        # Resolve signals from earlier candles.
        #
        # The current candle can serve as:
        #   - entry candle for a signal from t-1
        #   - horizon candle for older signals
        #
        # The entry price is therefore the CURRENT OPEN.
        # --------------------------------------------------------------

        self._process_pending(candle)

        # --------------------------------------------------------------
        # Current candle measurements
        # --------------------------------------------------------------

        bid_open = float(candle.bid_open)
        bid_high = float(candle.bid_high)
        bid_low = float(candle.bid_low)
        bid_close = float(candle.bid_close)

        tr = true_range(
            bid_high,
            bid_low,
            self.previous_close,
        )

        self.tr_window.append(tr)

        self.previous_close = bid_close

        if len(self.tr_window) < ATR_PERIOD:
            return

        atr = sum(self.tr_window) / ATR_PERIOD

        # We need 20 PRIOR ATR values to create a valid baseline.
        if len(self.atr_history) < BASELINE_PERIOD:
            self.atr_history.append(atr)
            return

        baseline = (
            sum(self.atr_history)
            / len(self.atr_history)
        )

        if baseline <= 0.0:
            self.atr_history.append(atr)
            return

        ratio = atr / baseline

        self.atr_ready += 1

        regime = get_regime(candle.timestamp)

        if regime is None:
            self.atr_history.append(atr)
            return

        bucket = classify_bucket(ratio)
        direction = candle_direction(candle)

        # --------------------------------------------------------------
        # Create signal ONLY AFTER candle close.
        #
        # It cannot use the current candle as entry.
        # Entry happens on the next completed candle.
        # --------------------------------------------------------------

        if direction != "flat":
            signal = PendingSignal(
                regime=regime.name,
                bucket=bucket,
                direction=direction,
                signal_timestamp=candle.timestamp,
                signal_close=bid_close,
            )

            for horizon in HORIZONS:
                self.pending[horizon].append(
                    PendingSignal(
                        regime=signal.regime,
                        bucket=signal.bucket,
                        direction=signal.direction,
                        signal_timestamp=signal.signal_timestamp,
                        signal_close=signal.signal_close,
                    )
                )

                self.signals_created += 1

        # Current ATR becomes available to future baselines only now.
        self.atr_history.append(atr)

    # ------------------------------------------------------------------
    # Pending signal handling
    # ------------------------------------------------------------------

    def _process_pending(self, candle: Any) -> None:
        current_open = float(candle.bid_open)
        current_close = float(candle.bid_close)

        for horizon in HORIZONS:
            queue = self.pending[horizon]

            if not queue:
                continue

            remaining: list[PendingSignal] = []

            for signal in queue:
                # First candle after signal:
                # enter at this candle's OPEN.
                if signal.entry_price is None:
                    signal.entry_price = current_open
                    signal.entry_timestamp = candle.timestamp
                    signal.candles_since_entry = 1
                else:
                    signal.candles_since_entry += 1

                if signal.candles_since_entry < horizon:
                    remaining.append(signal)
                    continue

                if signal.entry_price <= 0.0:
                    continue

                raw_return = (
                    current_close - signal.entry_price
                ) / signal.entry_price

                # Test whether the signal's candle direction
                # correctly predicted the forward movement.
                if signal.direction == "bullish":
                    signed_return = raw_return
                else:
                    signed_return = -raw_return

                abs_return = abs(raw_return)

                # Directional efficiency:
                # absolute directional movement divided by
                # absolute movement from entry to exit.
                #
                # For this research horizon the value is 1 or 0
                # at the endpoint, so we retain the same metric
                # definition only as a simple directional outcome.
                efficiency = (
                    1.0 if signed_return > 0.0 else 0.0
                )

                stats = self.stats(
                    signal.regime,
                    signal.bucket,
                    signal.direction,
                    horizon,
                )

                stats.update(
                    signed_return=signed_return,
                    abs_return=abs_return,
                    efficiency=efficiency,
                )

                self.signals_resolved += 1

            self.pending[horizon] = remaining

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def build_report(self) -> dict[str, Any]:
        cells: dict[str, Any] = {}

        for (
            regime,
            bucket,
            direction,
            horizon,
        ), stats in sorted(self.results.items()):
            key = (
                f"{regime}|{bucket}|"
                f"{direction}|H{horizon}"
            )

            cells[key] = {
                "regime": regime,
                "bucket": bucket,
                "direction": direction,
                "horizon_candles": horizon,
                **stats.build(),
            }

        return {
            "timeframe": self.timeframe.name,
            "parameters": {
                "atr_period": ATR_PERIOD,
                "baseline_period": BASELINE_PERIOD,
                "horizons": list(HORIZONS),
                "buckets": [
                    {
                        "name": name,
                        "lower": lower,
                        "upper": (
                            None
                            if upper == float("inf")
                            else upper
                        ),
                    }
                    for name, lower, upper in BUCKETS
                ],
            },
            "counts": {
                "candles": self.candles,
                "complete_candles": self.complete_candles,
                "atr_ready_candles": self.atr_ready,
                "signals_created": self.signals_created,
                "signals_resolved": self.signals_resolved,
            },
            "cells": cells,
        }


# ---------------------------------------------------------------------------
# Metadata / output
# ---------------------------------------------------------------------------

def build_metadata(
    start: datetime,
    end: datetime,
    generated_at: datetime,
    timeframe: Timeframe,
) -> dict[str, Any]:
    return {
        "phase": "6.3A",
        "research": "momentum_direction",
        "hypothesis": (
            "When volatility expands, does the direction "
            "of the expansion candle predict subsequent "
            "XAUUSD movement?"
        ),
        "instrument": "XAUUSD",
        "source": SOURCE_NAME,
        "timezone": TIMEZONE_NAME,
        "timeframe": timeframe.name,
        "requested_window": {
            "start": start.isoformat(),
            "end": end.isoformat(),
        },
        "generated_at": generated_at.isoformat(),
        "methodology": {
            "input_frequency": "M1",
            "aggregation": "calendar-aligned UTC",
            "interpolation": False,
            "fill_forward": False,
            "raw_data_modified": False,
            "streaming": True,
            "signal_known": "after candle close",
            "entry": "next candle open",
            "forward_measurement": (
                "direction-adjusted return from entry "
                "to horizon close"
            ),
        },
    }


def write_report(
    output_root: Path,
    start: datetime,
    end: datetime,
    generated_at: datetime,
    engine: MomentumResearch,
) -> Path:
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        "xauusd_momentum_direction_"
        f"{engine.timeframe.name.lower()}_"
        f"{format_timestamp(start)}_"
        f"{format_timestamp(end)}.json"
    )

    output_path = output_root / filename

    payload = {
        "metadata": build_metadata(
            start=start,
            end=end,
            generated_at=generated_at,
            timeframe=engine.timeframe,
        ),
        "report": engine.build_report(),
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 6.3A XAUUSD momentum-direction research."
        )
    )

    parser.add_argument(
        "--start",
        required=True,
        help="UTC research start, e.g. 2014-05-01T00:00:00Z",
    )

    parser.add_argument(
        "--end",
        required=True,
        help="UTC research end, e.g. 2014-06-01T00:00:00Z",
    )

    parser.add_argument(
        "--timeframe",
        choices=["ALL", *[tf.name for tf in ALL_TIMEFRAMES]],
        default="ALL",
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--generated-at",
        default=None,
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    start = parse_datetime(args.start)
    end = parse_datetime(args.end)

    if end <= start:
        parser.error("--end must be after --start")

    generated_at = (
        parse_datetime(args.generated_at)
        if args.generated_at
        else datetime.now(timezone.utc)
    )

    timeframes = selected_timeframes(args.timeframe)

    validate_regimes(REGIMES)

    adapter = DukascopyAdapter(args.data_root)

    engines = {
        timeframe: MomentumResearch(timeframe)
        for timeframe in timeframes
    }

    print()
    print("=" * 78)
    print("XAUUSD — PHASE 6.3A MOMENTUM DIRECTION RESEARCH")
    print("=" * 78)
    print(f"Source:      {SOURCE_NAME}")
    print(f"Window:      {start.isoformat()} -> {end.isoformat()}")
    print(
        "Timeframes:  "
        + ", ".join(tf.name for tf in timeframes)
    )
    print(
        "Parameters:  "
        f"ATR={ATR_PERIOD}, "
        f"Baseline={BASELINE_PERIOD}, "
        f"Horizons={HORIZONS}"
    )
    print(f"Data root:   {args.data_root}")
    print(f"Output root: {args.output_root}")
    print()

    bars = adapter.iter_bars(
        start,
        end,
        require_ask=True,
    )

    if args.timeframe == "ALL":
        for timeframe, candle in aggregate_timeframes(
            bars,
            timeframes,
        ):
            engines[timeframe].process(candle)
    else:
        timeframe = timeframes[0]

        for candle in aggregate_timeframe(
            bars,
            timeframe,
        ):
            engines[timeframe].process(candle)

    print("RESULTS")
    print("-" * 78)

    for timeframe in timeframes:
        engine = engines[timeframe]

        print(
            f"{timeframe.name}: "
            f"candles={engine.candles:,} | "
            f"complete={engine.complete_candles:,} | "
            f"ATR-ready={engine.atr_ready:,} | "
            f"signals={engine.signals_created:,} | "
            f"resolved={engine.signals_resolved:,}"
        )

        output_path = write_report(
            output_root=(
                args.output_root / timeframe.name
                if args.timeframe == "ALL"
                else args.output_root
            ),
            start=start,
            end=end,
            generated_at=generated_at,
            engine=engine,
        )

        print(f"Output: {output_path}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
