"""
Phase 6.1 XAUUSD regime-distribution research runner.

Streams canonical Dukascopy XAUUSD M1 BID/ASK data once, aggregates
validated research timeframes, assigns each completed timeframe candle
to a fixed UTC research regime, and writes regime × timeframe reports.

Architecture:

    Dukascopy M1
        |
        v
    multi-timeframe aggregation
        |
        v
    regime assignment
        |
        v
    RegimeDistributionAccumulator
        |
        v
    JSON reports

Important:
- The existing Phase 6 timeframe runner is intentionally untouched.
- No interpolation or fill-forward is performed.
- No raw data is modified.
- No future information is used for regime assignment.
- The full M1 dataset is never loaded into memory.
- ALL timeframes are processed from one physical M1 stream.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from xau_lean.data.dukascopy import DukascopyAdapter
from xau_lean.research.regime_accumulator import (
    RegimeDistributionAccumulator,
)
from xau_lean.research.regimes import (
    REGIMES,
    ResearchRegime,
    get_regime,
    validate_regimes,
)
from xau_lean.research.timeframe import (
    Timeframe,
    aggregate_timeframe,
    aggregate_timeframes,
)


DEFAULT_DATA_ROOT = Path(
    "data/external/dukascopy/Market-Data-Lab-main/xauusd"
)

DEFAULT_OUTPUT_ROOT = Path(
    "research_results/regime"
)

SOURCE_NAME = "Dukascopy XAUUSD BID/ASK M1"
TIMEZONE_NAME = "UTC"

ALL_TIMEFRAMES = tuple(Timeframe)


def parse_datetime(value: str) -> datetime:
    """Parse an ISO-8601 datetime and normalize it to UTC."""

    text = value.strip()

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    parsed = datetime.fromisoformat(text)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def format_timestamp(value: datetime) -> str:
    """Return a stable UTC timestamp for filenames."""

    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .strftime("%Y%m%dT%H%M%SZ")
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 6.1 XAUUSD regime-distribution research "
            "using canonical Dukascopy M1 data."
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
        choices=[
            "ALL",
            *[tf.name for tf in ALL_TIMEFRAMES],
        ],
        default="ALL",
        help="Timeframe to research. Default: ALL.",
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Canonical Dukascopy XAUUSD data root.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for regime research reports.",
    )

    parser.add_argument(
        "--generated-at",
        default=None,
        help=(
            "Optional report generation timestamp. "
            "If omitted, current UTC time is used."
        ),
    )

    return parser


def selected_timeframes(
    name: str,
) -> tuple[Timeframe, ...]:
    """Resolve CLI timeframe selection."""

    if name == "ALL":
        return ALL_TIMEFRAMES

    return (Timeframe[name],)


def build_accumulators(
    timeframes: tuple[Timeframe, ...],
) -> dict[
    tuple[Timeframe, str],
    RegimeDistributionAccumulator,
]:
    """
    Build one bounded-memory accumulator for every
    timeframe × regime cell.
    """

    validate_regimes(REGIMES)

    accumulators: dict[
        tuple[Timeframe, str],
        RegimeDistributionAccumulator,
    ] = {}

    for timeframe in timeframes:
        for regime in REGIMES:
            accumulators[
                (timeframe, regime.name)
            ] = RegimeDistributionAccumulator(
                regime=regime,
                timeframe=timeframe.name,
            )

    return accumulators


def build_metadata(
    start: datetime,
    end: datetime,
    generated_at: datetime,
    timeframe: Timeframe,
    regime: ResearchRegime,
) -> dict[str, Any]:
    """Build stable report metadata."""

    return {
        "phase": "6.1",
        "research": "regime_distribution",
        "instrument": "XAUUSD",
        "source": SOURCE_NAME,
        "timezone": TIMEZONE_NAME,
        "timeframe": timeframe.name,
        "regime": {
            "name": regime.name,
            "start": regime.start.astimezone(
                timezone.utc
            ).isoformat(),
            "end": regime.end.astimezone(
                timezone.utc
            ).isoformat(),
        },
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
        },
    }


def write_report(
    output_root: Path,
    start: datetime,
    end: datetime,
    generated_at: datetime,
    timeframe: Timeframe,
    regime: ResearchRegime,
    accumulator: RegimeDistributionAccumulator,
) -> Path:
    """Write one regime × timeframe JSON report."""

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"xauusd_"
        f"{timeframe.name.lower()}_"
        f"{regime.name.replace('-', '_')}_"
        f"{format_timestamp(start)}_"
        f"{format_timestamp(end)}.json"
    )

    output_path = output_root / filename

    payload = {
        "metadata": build_metadata(
            start=start,
            end=end,
            generated_at=generated_at,
            timeframe=timeframe,
            regime=regime,
        ),
        "report": accumulator.build_report(),
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


def assign_candle(
    timeframe: Timeframe,
    candle: Any,
    accumulators: dict[
        tuple[Timeframe, str],
        RegimeDistributionAccumulator,
    ],
) -> None:
    """
    Assign one completed timeframe candle to its fixed regime.

    The regime is determined solely from the candle timestamp.
    """

    regime = get_regime(candle.timestamp)

    if regime is None:
        return

    accumulator = accumulators[
        (timeframe, regime.name)
    ]

    accumulator.update(candle)


def run_single_timeframe(
    adapter: DukascopyAdapter,
    start: datetime,
    end: datetime,
    timeframe: Timeframe,
    accumulators: dict[
        tuple[Timeframe, str],
        RegimeDistributionAccumulator,
    ],
) -> int:
    """Run one timeframe from the canonical M1 stream."""

    candle_count = 0

    bars = adapter.iter_bars(
        start,
        end,
        require_ask=True,
    )

    for candle in aggregate_timeframe(
        bars,
        timeframe,
    ):
        assign_candle(
            timeframe,
            candle,
            accumulators,
        )

        candle_count += 1

    return candle_count


def run_all_timeframes(
    adapter: DukascopyAdapter,
    start: datetime,
    end: datetime,
    timeframes: tuple[Timeframe, ...],
    accumulators: dict[
        tuple[Timeframe, str],
        RegimeDistributionAccumulator,
    ],
) -> int:
    """
    Run every requested timeframe from one physical M1 stream.
    """

    candle_count = 0

    bars = adapter.iter_bars(
        start,
        end,
        require_ask=True,
    )

    for timeframe, candle in aggregate_timeframes(
        bars,
        timeframes,
    ):
        assign_candle(
            timeframe,
            candle,
            accumulators,
        )

        candle_count += 1

    return candle_count


def write_all_reports(
    output_root: Path,
    start: datetime,
    end: datetime,
    generated_at: datetime,
    timeframes: tuple[Timeframe, ...],
    accumulators: dict[
        tuple[Timeframe, str],
        RegimeDistributionAccumulator,
    ],
) -> list[Path]:
    """Write all regime × timeframe reports."""

    paths: list[Path] = []

    for timeframe in timeframes:
        for regime in REGIMES:
            accumulator = accumulators[
                (timeframe, regime.name)
            ]

            path = write_report(
                output_root=output_root,
                start=start,
                end=end,
                generated_at=generated_at,
                timeframe=timeframe,
                regime=regime,
                accumulator=accumulator,
            )

            paths.append(path)

    return paths


def print_summary(
    timeframes: tuple[Timeframe, ...],
    accumulators: dict[
        tuple[Timeframe, str],
        RegimeDistributionAccumulator,
    ],
) -> None:
    """Print a concise regime coverage summary."""

    print()
    print("REGIME COVERAGE SUMMARY")
    print("-" * 78)

    for timeframe in timeframes:
        print(f"\n{timeframe.name}")

        for regime in REGIMES:
            accumulator = accumulators[
                (timeframe, regime.name)
            ]

            print(
                f"  {regime.name:<10} "
                f"{accumulator.candle_count:>10,} candles"
            )


def main() -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args()

    start = parse_datetime(args.start)
    end = parse_datetime(args.end)

    if end <= start:
        parser.error("--end must be after --start")

    if args.generated_at is None:
        generated_at = datetime.now(timezone.utc)
    else:
        generated_at = parse_datetime(
            args.generated_at
        )

    timeframes = selected_timeframes(
        args.timeframe
    )

    validate_regimes(REGIMES)

    adapter = DukascopyAdapter(
        args.data_root
    )

    accumulators = build_accumulators(
        timeframes
    )

    print()
    print("=" * 78)
    print("XAUUSD — PHASE 6.1 REGIME DISTRIBUTION RESEARCH")
    print("=" * 78)
    print(f"Source:      {SOURCE_NAME}")
    print(
        f"Window:      {start.isoformat()} -> "
        f"{end.isoformat()}"
    )
    print(
        "Timeframes:  "
        + ", ".join(
            tf.name for tf in timeframes
        )
    )
    print(
        "Regimes:     "
        + ", ".join(
            regime.name for regime in REGIMES
        )
    )
    print(f"Data root:   {args.data_root}")
    print(f"Output root: {args.output_root}")
    print()

    if args.timeframe == "ALL":
        candle_count = run_all_timeframes(
            adapter=adapter,
            start=start,
            end=end,
            timeframes=timeframes,
            accumulators=accumulators,
        )
    else:
        candle_count = run_single_timeframe(
            adapter=adapter,
            start=start,
            end=end,
            timeframe=timeframes[0],
            accumulators=accumulators,
        )

    paths = write_all_reports(
        output_root=args.output_root,
        start=start,
        end=end,
        generated_at=generated_at,
        timeframes=timeframes,
        accumulators=accumulators,
    )

    print_summary(
        timeframes=timeframes,
        accumulators=accumulators,
    )

    print()
    print(
        f"Aggregated timeframe candles: {candle_count:,}"
    )
    print(
        f"Reports written:              {len(paths):,}"
    )
    print()
    print("=" * 78)
    print("PHASE 6.1 REGIME RESEARCH COMPLETE")
    print("=" * 78)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
