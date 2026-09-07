#!/usr/bin/env python3
"""
Phase 6.2A — XAUUSD volatility-expansion hypothesis research.

Research question
-----------------
Does unusually high current volatility predict unusually high
future price movement?

Method
------
Canonical Dukascopy M1 BID/ASK
        |
        v
multi-timeframe aggregation
        |
        v
completed timeframe candles
        |
        v
TR -> ATR(14)
        |
        v
current ATR / prior 20-candle ATR baseline
        |
        v
volatility-expansion bucket
        |
        v
NEXT candle entry
        |
        v
forward movement over 1 / 3 / 5 candles

Important:
- Streaming M1 input.
- No interpolation.
- No fill-forward.
- No raw-data modification.
- No future information is used to classify a signal.
- Signal classification occurs only after the signal candle closes.
- Entry is the NEXT candle open.
- Forward outcomes use subsequent candles only.
- Incomplete timeframe candles are excluded.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from xau_lean.data.dukascopy import DukascopyAdapter
from xau_lean.research.regimes import (
    REGIMES,
    get_regime,
    validate_regimes,
)
from xau_lean.research.timeframe import (
    Timeframe,
    TimeframeCandle,
    aggregate_timeframes,
)


DEFAULT_DATA_ROOT = Path(
    "data/external/dukascopy/Market-Data-Lab-main/xauusd"
)

DEFAULT_OUTPUT_ROOT = Path(
    "research_results/signals"
)

SOURCE_NAME = "Dukascopy XAUUSD BID/ASK M1"
TIMEZONE_NAME = "UTC"

ALL_TIMEFRAMES = tuple(Timeframe)

ATR_PERIOD = 14
BASELINE_PERIOD = 20
HORIZONS = (1, 3, 5)

EXPANSION_BUCKETS = (
    ("normal", 0.0, 1.25),
    ("elevated", 1.25, 1.50),
    ("high", 1.50, 2.00),
    ("extreme", 2.00, float("inf")),
)


@dataclass
class BucketStats:
    """Aggregate statistics for one regime/bucket/horizon."""

    observations: int = 0
    sum_abs_return: float = 0.0
    sum_abs_return_pct: float = 0.0
    sum_directional_efficiency: float = 0.0
    sum_forward_range: float = 0.0

    def update(
        self,
        *,
        abs_return: float,
        abs_return_pct: float,
        directional_efficiency: float,
        forward_range: float,
    ) -> None:
        self.observations += 1
        self.sum_abs_return += abs_return
        self.sum_abs_return_pct += abs_return_pct
        self.sum_directional_efficiency += directional_efficiency
        self.sum_forward_range += forward_range

    def to_dict(self) -> dict[str, Any]:
        if self.observations == 0:
            return {
                "observations": 0,
                "mean_abs_return": None,
                "mean_abs_return_pct": None,
                "mean_directional_efficiency": None,
                "mean_forward_range": None,
            }

        n = self.observations

        return {
            "observations": n,
            "mean_abs_return": self.sum_abs_return / n,
            "mean_abs_return_pct": self.sum_abs_return_pct / n,
            "mean_directional_efficiency": (
                self.sum_directional_efficiency / n
            ),
            "mean_forward_range": self.sum_forward_range / n,
        }


@dataclass
class PendingSignal:
    """Signal waiting for future candles to resolve."""

    regime: str
    bucket: str
    horizon: int
    entry_price: float | None = None
    candles_seen: int = 0
    high: float | None = None
    low: float | None = None


class SignalAccumulator:
    """Collect volatility-expansion results."""

    def __init__(self) -> None:
        self.stats: dict[
            tuple[str, str, int],
            BucketStats,
        ] = {}

        for regime in REGIMES:
            for bucket_name, _, _ in EXPANSION_BUCKETS:
                for horizon in HORIZONS:
                    self.stats[
                        (regime.name, bucket_name, horizon)
                    ] = BucketStats()

    def update(
        self,
        *,
        regime: str,
        bucket: str,
        horizon: int,
        abs_return: float,
        abs_return_pct: float,
        directional_efficiency: float,
        forward_range: float,
    ) -> None:
        self.stats[
            (regime, bucket, horizon)
        ].update(
            abs_return=abs_return,
            abs_return_pct=abs_return_pct,
            directional_efficiency=directional_efficiency,
            forward_range=forward_range,
        )

    def build_report(self) -> dict[str, Any]:
        report: dict[str, Any] = {}

        for regime in REGIMES:
            regime_data: dict[str, Any] = {}

            for bucket_name, lower, upper in EXPANSION_BUCKETS:
                bucket_data: dict[str, Any] = {
                    "ratio_range": {
                        "lower": lower,
                        "upper": (
                            None
                            if upper == float("inf")
                            else upper
                        ),
                    },
                    "horizons": {},
                }

                for horizon in HORIZONS:
                    bucket_data["horizons"][
                        str(horizon)
                    ] = self.stats[
                        (
                            regime.name,
                            bucket_name,
                            horizon,
                        )
                    ].to_dict()

                regime_data[bucket_name] = bucket_data

            report[regime.name] = regime_data

        return report


def parse_datetime(value: str) -> datetime:
    """Parse ISO-8601 datetime and normalize to UTC."""

    text = value.strip()

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime: {value!r}"
        ) from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 6.2A XAUUSD volatility-expansion "
            "hypothesis research."
        )
    )

    parser.add_argument(
        "--start",
        required=True,
        type=parse_datetime,
    )

    parser.add_argument(
        "--end",
        required=True,
        type=parse_datetime,
    )

    parser.add_argument(
        "--timeframe",
        choices=[
            "ALL",
            *[tf.name for tf in ALL_TIMEFRAMES],
        ],
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

    return parser


def selected_timeframes(
    value: str,
) -> tuple[Timeframe, ...]:
    if value == "ALL":
        return ALL_TIMEFRAMES

    return (Timeframe[value],)


def true_range(
    candle: TimeframeCandle,
    previous_close: float | None,
) -> float:
    """Calculate BID true range."""

    if previous_close is None:
        return candle.bid_range

    return max(
        candle.bid_high - candle.bid_low,
        abs(candle.bid_high - previous_close),
        abs(candle.bid_low - previous_close),
    )


def expansion_bucket(ratio: float) -> str:
    """Return the fixed volatility-expansion bucket."""

    for name, lower, upper in EXPANSION_BUCKETS:
        if lower <= ratio < upper:
            return name

    return EXPANSION_BUCKETS[-1][0]


def resolve_signal(
    *,
    signal: PendingSignal,
    candle: TimeframeCandle,
    accumulator: SignalAccumulator,
) -> bool:
    """
    Advance a signal using one FUTURE candle.

    The first candle processed here is the candle immediately
    after the signal candle.
    """

    mid_open = candle.mid_open
    mid_close = candle.mid_close
    mid_high = candle.mid_high
    mid_low = candle.mid_low

    if (
        mid_open is None
        or mid_close is None
        or mid_high is None
        or mid_low is None
    ):
        return False

    # Establish entry on the first candle after the signal.
    if signal.entry_price is None:
        signal.entry_price = mid_open
        signal.high = mid_open
        signal.low = mid_open

    signal.candles_seen += 1

    signal.high = max(
        signal.high if signal.high is not None else mid_high,
        mid_high,
    )

    signal.low = min(
        signal.low if signal.low is not None else mid_low,
        mid_low,
    )

    if signal.candles_seen < signal.horizon:
        return False

    entry_price = signal.entry_price

    if entry_price <= 0:
        return True

    move = mid_close - entry_price
    abs_move = abs(move)

    abs_return_pct = abs_move / entry_price

    forward_range = (
        (signal.high or entry_price)
        - (signal.low or entry_price)
    )

    if forward_range > 0:
        directional_efficiency = (
            abs_move / forward_range
        )
    else:
        directional_efficiency = 0.0

    accumulator.update(
        regime=signal.regime,
        bucket=signal.bucket,
        horizon=signal.horizon,
        abs_return=abs_move,
        abs_return_pct=abs_return_pct,
        directional_efficiency=directional_efficiency,
        forward_range=forward_range,
    )

    return True


def run(
    *,
    start: datetime,
    end: datetime,
    timeframes: tuple[Timeframe, ...],
    data_root: Path,
) -> tuple[SignalAccumulator, dict[str, int]]:
    """Run the research in one physical M1 stream."""

    validate_regimes(REGIMES)

    adapter = DukascopyAdapter(data_root)

    accumulator = SignalAccumulator()

    # Previous BID close for true-range calculation.
    previous_close: dict[
        Timeframe,
        float | None,
    ] = {
        timeframe: None
        for timeframe in timeframes
    }

    # Completed ATR values.
    atr_values: dict[
        Timeframe,
        deque[float],
    ] = {
        timeframe: deque(
            maxlen=BASELINE_PERIOD
        )
        for timeframe in timeframes
    }

    # Pending signals.
    pending: dict[
        Timeframe,
        list[PendingSignal],
    ] = {
        timeframe: []
        for timeframe in timeframes
    }

    counts = {
        "aggregated_candles": 0,
        "complete_candles": 0,
        "atr_ready_candles": 0,
        "signals_created": 0,
        "signals_resolved": 0,
    }

    bars = adapter.iter_bars(
        start,
        end,
        require_ask=True,
    )

    for timeframe, candle in aggregate_timeframes(
        bars,
        timeframes,
    ):
        counts["aggregated_candles"] += 1

        if not candle.complete:
            continue

        mid_open = candle.mid_open
        mid_close = candle.mid_close

        if mid_open is None or mid_close is None:
            continue

        counts["complete_candles"] += 1

        # ---------------------------------------------------------
        # STEP 1
        # Resolve signals generated by PREVIOUS candles.
        # ---------------------------------------------------------

        still_pending: list[PendingSignal] = []

        for signal in pending[timeframe]:
            finished = resolve_signal(
                signal=signal,
                candle=candle,
                accumulator=accumulator,
            )

            if finished:
                counts["signals_resolved"] += 1
            else:
                still_pending.append(signal)

        pending[timeframe] = still_pending

        # ---------------------------------------------------------
        # STEP 2
        # Calculate current TR.
        # ---------------------------------------------------------

        tr = true_range(
            candle,
            previous_close[timeframe],
        )

        previous_close[timeframe] = candle.bid_close

        # ---------------------------------------------------------
        # STEP 3
        # Calculate CURRENT ATR(14).
        #
        # Current ATR uses the current candle.
        # Baseline uses ONLY ATR values from earlier candles.
        # ---------------------------------------------------------

        atr_history = atr_values[timeframe]

        if len(atr_history) >= ATR_PERIOD:
            prior_atr_values = list(atr_history)[
                -ATR_PERIOD:
            ]

            current_atr = (
                sum(
                    prior_atr_values
                ) + tr
            ) / ATR_PERIOD

            if len(atr_history) >= BASELINE_PERIOD:
                baseline_values = list(
                    atr_history
                )[-BASELINE_PERIOD:]

                baseline_atr = (
                    sum(baseline_values)
                    / len(baseline_values)
                )

                if baseline_atr > 0:
                    ratio = (
                        current_atr
                        / baseline_atr
                    )

                    bucket = expansion_bucket(
                        ratio
                    )

                    regime = get_regime(
                        candle.timestamp
                    )

                    if regime is not None:
                        # Signal is created AFTER this candle closes.
                        # It will be entered on the NEXT candle.
                        for horizon in HORIZONS:
                            pending[timeframe].append(
                                PendingSignal(
                                    regime=regime.name,
                                    bucket=bucket,
                                    horizon=horizon,
                                )
                            )

                        counts["signals_created"] += len(
                            HORIZONS
                        )

                    counts["atr_ready_candles"] += 1

        # ---------------------------------------------------------
        # STEP 4
        # Store current ATR only AFTER classification.
        #
        # Therefore it cannot contaminate the historical baseline.
        # ---------------------------------------------------------

        if len(atr_history) >= ATR_PERIOD:
            current_atr_for_history = (
                sum(
                    list(atr_history)[-(
                        ATR_PERIOD - 1
                    ):]
                ) + tr
            ) / ATR_PERIOD
        else:
            current_atr_for_history = tr

        atr_history.append(
            current_atr_for_history
        )

    return accumulator, counts


def build_metadata(
    *,
    start: datetime,
    end: datetime,
    timeframes: tuple[Timeframe, ...],
) -> dict[str, Any]:
    return {
        "phase": "6.2A",
        "research": "volatility_expansion",
        "instrument": "XAUUSD",
        "source": SOURCE_NAME,
        "timezone": TIMEZONE_NAME,
        "requested_window": {
            "start": start.isoformat(),
            "end": end.isoformat(),
        },
        "timeframes": [
            timeframe.name
            for timeframe in timeframes
        ],
        "methodology": {
            "input_frequency": "M1",
            "aggregation": "calendar-aligned UTC",
            "interpolation": False,
            "fill_forward": False,
            "raw_data_modified": False,
            "streaming": True,
            "atr_period": ATR_PERIOD,
            "baseline_period": BASELINE_PERIOD,
            "horizons": list(HORIZONS),
            "expansion_buckets": [
                {
                    "name": name,
                    "lower": lower,
                    "upper": (
                        None
                        if upper == float("inf")
                        else upper
                    ),
                }
                for name, lower, upper
                in EXPANSION_BUCKETS
            ],
            "signal_timing": (
                "Current ATR is classified at candle close "
                "against a baseline of prior ATR values; "
                "entry begins at the next candle open."
            ),
        },
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.end <= args.start:
        parser.error("--end must be after --start")

    timeframes = selected_timeframes(
        args.timeframe
    )

    print()
    print("=" * 78)
    print(
        "XAUUSD — PHASE 6.2A "
        "VOLATILITY EXPANSION RESEARCH"
    )
    print("=" * 78)
    print(
        f"Window:      "
        f"{args.start.isoformat()} -> "
        f"{args.end.isoformat()}"
    )
    print(
        "Timeframes:  "
        + ", ".join(
            timeframe.name
            for timeframe in timeframes
        )
    )
    print(f"ATR period:  {ATR_PERIOD}")
    print(f"Baseline:    {BASELINE_PERIOD}")
    print(
        "Horizons:    "
        + ", ".join(
            str(horizon)
            for horizon in HORIZONS
        )
    )
    print(f"Data root:   {args.data_root}")
    print()

    accumulator, counts = run(
        start=args.start,
        end=args.end,
        timeframes=timeframes,
        data_root=args.data_root,
    )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "metadata": build_metadata(
            start=args.start,
            end=args.end,
            timeframes=timeframes,
        ),
        "counts": counts,
        "report": accumulator.build_report(),
    }

    filename = (
        "xauusd_volatility_expansion_"
        f"{args.start.strftime('%Y%m%dT%H%M%SZ')}_"
        f"{args.end.strftime('%Y%m%dT%H%M%SZ')}.json"
    )

    output_path = args.output_root / filename

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

    print("=" * 78)
    print("RESEARCH COMPLETE")
    print("=" * 78)
    print(
        f"Aggregated candles: "
        f"{counts['aggregated_candles']:,}"
    )
    print(
        f"Complete candles:   "
        f"{counts['complete_candles']:,}"
    )
    print(
        f"ATR-ready candles:  "
        f"{counts['atr_ready_candles']:,}"
    )
    print(
        f"Signals created:    "
        f"{counts['signals_created']:,}"
    )
    print(
        f"Signals resolved:   "
        f"{counts['signals_resolved']:,}"
    )
    print(f"Output:             {output_path}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
