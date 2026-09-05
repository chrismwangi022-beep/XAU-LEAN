#!/usr/bin/env python3
"""
Phase 6 — XAUUSD timeframe research runner.

Streams canonical Dukascopy M1 BID/ASK data through:

    M1 source
        ↓
    coverage tracking
        ↓
    multi-timeframe aggregation
        ↓
    research accumulators
        ↓
    standardized JSON reports

For --timeframe ALL, the canonical M1 dataset is read exactly once.

No full dataset is loaded into memory.
No interpolation or fill-forward is performed.
No raw data is modified.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from xau_lean.data.dukascopy import DukascopyAdapter
from xau_lean.research.accumulator import ResearchAccumulator
from xau_lean.research.coverage import CoverageAccumulator
from xau_lean.research.report import ResearchMetadata
from xau_lean.research.timeframe import (
    Timeframe,
    aggregate_timeframe,
    aggregate_timeframes,
)


DEFAULT_DATA_ROOT = Path(
    "data/external/dukascopy/Market-Data-Lab-main/xauusd"
)

DEFAULT_OUTPUT_ROOT = Path("research_results/timeframe")

SOURCE_NAME = "Dukascopy XAUUSD BID/ASK M1"

TIMEZONE_NAME = "UTC"

AGGREGATION_DESCRIPTION = (
    "Fixed UTC calendar intervals anchored to midnight UTC; "
    "BID and ASK aggregated independently; timestamp represents "
    "interval OPEN; no interpolation or fill-forward."
)

WEEKEND_POLICY = (
    "Raw observations preserved; weekend observations included in "
    "structural statistics and not silently deleted."
)

SPREAD_WARNING = 1.0

ATR_PERIOD = 14

SPREAD_THRESHOLDS = (0.50, 1.0, 2.0, 5.0)

ALL_TIMEFRAMES = tuple(Timeframe)


def parse_datetime(value: str) -> datetime:
    """Parse an ISO date or datetime and normalize it to UTC."""

    try:
        if len(value) == 10:
            parsed = datetime.fromisoformat(value).replace(
                tzinfo=timezone.utc
            )
        else:
            parsed = datetime.fromisoformat(value)

            if parsed.tzinfo is None:
                parsed = parsed.replace(
                    tzinfo=timezone.utc
                )

            parsed = parsed.astimezone(timezone.utc)

    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date/datetime: {value!r}. "
            "Use YYYY-MM-DD or ISO-8601 datetime."
        ) from exc

    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description="Run streaming XAUUSD timeframe research."
    )

    parser.add_argument(
        "--start",
        required=True,
        type=parse_datetime,
        help="Inclusive UTC start date/datetime.",
    )

    parser.add_argument(
        "--end",
        required=True,
        type=parse_datetime,
        help="Exclusive UTC end date/datetime.",
    )

    parser.add_argument(
        "--timeframe",
        default="ALL",
        choices=[
            "ALL",
            *[timeframe.value for timeframe in Timeframe],
        ],
        help="Timeframe to research, or ALL. Default: ALL.",
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Canonical Dukascopy XAUUSD root.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for JSON research reports.",
    )

    parser.add_argument(
        "--generated-at",
        type=parse_datetime,
        default=None,
        help=(
            "UTC execution timestamp for reproducible output. "
            "Defaults to the current UTC time."
        ),
    )

    return parser


def build_metadata(
    *,
    start: datetime,
    end: datetime,
    timeframe: Timeframe,
    generated_at: datetime,
) -> ResearchMetadata:
    """Build standardized research metadata."""

    return ResearchMetadata(
        source=SOURCE_NAME,
        start=start,
        end=end,
        timeframe=timeframe,
        timezone=TIMEZONE_NAME,
        aggregation=AGGREGATION_DESCRIPTION,
        weekend_policy=WEEKEND_POLICY,
        spread_warning=SPREAD_WARNING,
        generated_at=generated_at,
    )


def build_accumulators(
    *,
    start: datetime,
    end: datetime,
    timeframes: tuple[Timeframe, ...],
    generated_at: datetime,
) -> tuple[
    dict[Timeframe, ResearchAccumulator],
    dict[Timeframe, CoverageAccumulator],
]:
    """Create research and coverage accumulators."""

    research_accumulators: dict[
        Timeframe,
        ResearchAccumulator,
    ] = {}

    coverage_accumulators: dict[
        Timeframe,
        CoverageAccumulator,
    ] = {}

    for timeframe in timeframes:
        metadata = build_metadata(
            start=start,
            end=end,
            timeframe=timeframe,
            generated_at=generated_at,
        )

        research_accumulators[timeframe] = ResearchAccumulator(
            metadata=metadata,
            atr_period=ATR_PERIOD,
            spread_thresholds=SPREAD_THRESHOLDS,
        )

        coverage_accumulators[timeframe] = CoverageAccumulator(
            start=start,
            end=end,
            timeframe=timeframe,
        )

    return (
        research_accumulators,
        coverage_accumulators,
    )


def report_filename(
    *,
    timeframe: Timeframe,
    start: datetime,
    end: datetime,
) -> str:
    """Build deterministic report filename."""

    return (
        f"xauusd_{timeframe.value.lower()}_"
        f"{start.strftime('%Y%m%dT%H%M%SZ')}_"
        f"{end.strftime('%Y%m%dT%H%M%SZ')}.json"
    )


def coverage_to_dict(coverage) -> dict[str, object]:
    """Serialize coverage summary."""

    return {
        "expected_intervals": coverage.expected_intervals,
        "complete_intervals": coverage.complete_intervals,
        "partial_intervals": coverage.partial_intervals,
        "missing_intervals": coverage.missing_intervals,
        "expected_m1_per_interval": (
            coverage.expected_m1_per_interval
        ),
        "expected_m1": coverage.expected_m1,
        "observed_m1": coverage.observed_m1,
        "missing_m1": coverage.missing_m1,
        "completeness_ratio": coverage.completeness_ratio,
    }


def write_report(
    *,
    report,
    coverage,
    output_root: Path,
    timeframe: Timeframe,
    start: datetime,
    end: datetime,
) -> Path:
    """Write standardized research report with coverage."""

    payload = report.to_dict()

    payload["coverage"] = coverage_to_dict(
        coverage
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_root / report_filename(
        timeframe=timeframe,
        start=start,
        end=end,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    return output_path


def print_report_summary(
    *,
    timeframe: Timeframe,
    report,
    coverage,
    output_path: Path,
) -> None:
    """Print concise research and coverage results."""

    print(
        f"{timeframe.value}: "
        f"{report.candle_count:,} candles | "
        f"{report.complete_candle_count:,} complete | "
        f"{report.incomplete_candle_count:,} incomplete"
    )

    print(
        f"  coverage: "
        f"{coverage.complete_intervals:,} complete | "
        f"{coverage.partial_intervals:,} partial | "
        f"{coverage.missing_intervals:,} missing"
    )

    print(
        f"  M1 coverage: "
        f"{coverage.observed_m1:,}/"
        f"{coverage.expected_m1:,} "
        f"({coverage.completeness_ratio:.6%})"
    )

    print(
        f"  mean range: "
        f"{report.mean_range}"
    )

    print(
        f"  median range: "
        f"{report.median_range}"
    )

    print(
        f"  realized volatility: "
        f"{report.realized_volatility}"
    )

    print(
        f"  mean ATR: "
        f"{report.mean_atr}"
    )

    print(
        f"  mean spread: "
        f"{report.mean_spread}"
    )

    print(
        f"  spread/range: "
        f"{report.mean_spread_to_range}"
    )

    print(
        f"  output: "
        f"{output_path}"
    )


def tracked_m1_stream(
    *,
    bars: Iterable,
    coverage_accumulators: dict[
        Timeframe,
        CoverageAccumulator,
    ],
) -> Iterable:
    """
    Wrap the canonical M1 stream.

    Every M1 bar updates coverage for every requested timeframe,
    then the exact same bar continues downstream to aggregation.

    This guarantees one physical M1 read for ALL timeframes.
    """

    for bar in bars:
        for coverage in coverage_accumulators.values():
            coverage.update(
                bar.timestamp
            )

        yield bar


def run_single_timeframe(
    *,
    adapter: DukascopyAdapter,
    start: datetime,
    end: datetime,
    timeframe: Timeframe,
    output_root: Path,
    generated_at: datetime,
) -> Path:
    """Run one timeframe using one streaming M1 pass."""

    (
        research_accumulators,
        coverage_accumulators,
    ) = build_accumulators(
        start=start,
        end=end,
        timeframes=(timeframe,),
        generated_at=generated_at,
    )

    bars = adapter.iter_bars(
        start,
        end,
        require_ask=True,
    )

    tracked_bars = tracked_m1_stream(
        bars=bars,
        coverage_accumulators=coverage_accumulators,
    )

    for candle in aggregate_timeframe(
        tracked_bars,
        timeframe,
    ):
        research_accumulators[
            timeframe
        ].update(candle)

    report = research_accumulators[
        timeframe
    ].build_report()

    coverage = coverage_accumulators[
        timeframe
    ].build_summary()

    output_path = write_report(
        report=report,
        coverage=coverage,
        output_root=output_root,
        timeframe=timeframe,
        start=start,
        end=end,
    )

    print_report_summary(
        timeframe=timeframe,
        report=report,
        coverage=coverage,
        output_path=output_path,
    )

    return output_path


def run_all_timeframes(
    *,
    adapter: DukascopyAdapter,
    start: datetime,
    end: datetime,
    output_root: Path,
    generated_at: datetime,
) -> list[Path]:
    """
    Run all five timeframes using exactly ONE canonical M1 stream.
    """

    timeframes = ALL_TIMEFRAMES

    (
        research_accumulators,
        coverage_accumulators,
    ) = build_accumulators(
        start=start,
        end=end,
        timeframes=timeframes,
        generated_at=generated_at,
    )

    source_bars = adapter.iter_bars(
        start,
        end,
        require_ask=True,
    )

    tracked_bars = tracked_m1_stream(
        bars=source_bars,
        coverage_accumulators=coverage_accumulators,
    )

    for timeframe, candle in aggregate_timeframes(
        tracked_bars,
        timeframes,
    ):
        research_accumulators[
            timeframe
        ].update(candle)

    outputs: list[Path] = []

    for timeframe in timeframes:
        report = research_accumulators[
            timeframe
        ].build_report()

        coverage = coverage_accumulators[
            timeframe
        ].build_summary()

        output_path = write_report(
            report=report,
            coverage=coverage,
            output_root=output_root,
            timeframe=timeframe,
            start=start,
            end=end,
        )

        print_report_summary(
            timeframe=timeframe,
            report=report,
            coverage=coverage,
            output_path=output_path,
        )

        outputs.append(output_path)

    return outputs


def main() -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args()

    if args.end <= args.start:
        parser.error(
            "--end must be after --start"
        )

    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    generated_at = (
        args.generated_at
        or datetime.now(timezone.utc)
    )

    print("=" * 78)
    print("XAUUSD — PHASE 6 TIMEFRAME RESEARCH")
    print("=" * 78)
    print(
        f"Source:     {SOURCE_NAME}"
    )
    print(
        f"Data root:  {data_root}"
    )
    print(
        f"Start:      {args.start.isoformat()}"
    )
    print(
        f"End:        {args.end.isoformat()}"
    )
    print(
        f"Timezone:   {TIMEZONE_NAME}"
    )
    print(
        f"Output:     {output_root}"
    )
    print()

    if not data_root.is_dir():
        parser.error(
            "Dukascopy data root does not exist: "
            f"{data_root}"
        )

    adapter = DukascopyAdapter(
        data_root
    )

    if args.timeframe == "ALL":
        print(
            "Timeframes: "
            + ", ".join(
                timeframe.value
                for timeframe in ALL_TIMEFRAMES
            )
        )

        print(
            "Mode: ONE streaming M1 pass "
            "for coverage + all timeframe aggregation"
        )

        print()

        outputs = run_all_timeframes(
            adapter=adapter,
            start=args.start,
            end=args.end,
            output_root=output_root,
            generated_at=generated_at,
        )

    else:
        timeframe = Timeframe(
            args.timeframe
        )

        print(
            f"Timeframes: {timeframe.value}"
        )

        print(
            "Mode: one streaming M1 pass"
        )

        print()

        output = run_single_timeframe(
            adapter=adapter,
            start=args.start,
            end=args.end,
            timeframe=timeframe,
            output_root=output_root,
            generated_at=generated_at,
        )

        outputs = [output]

    print("=" * 78)
    print("RESEARCH RUN COMPLETE")
    print("=" * 78)
    print(
        f"Reports generated: {len(outputs)}"
    )

    for output in outputs:
        print(
            f"  {output}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
