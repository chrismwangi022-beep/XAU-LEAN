#!/usr/bin/env python3
"""
Phase 6 — XAUUSD timeframe research runner.

Streams canonical Dukascopy M1 BID/ASK data through the existing
timeframe aggregation and research accumulator layers.

No full dataset is loaded into memory.

Example:
    python scripts/run_timeframe_research.py \
        --start 2014-05-01 \
        --end 2014-06-01

Run one timeframe:
    python scripts/run_timeframe_research.py \
        --start 2014-05-01 \
        --end 2014-06-01 \
        --timeframe H1

Run all supported timeframes:
    python scripts/run_timeframe_research.py \
        --start 2014-05-01 \
        --end 2014-06-01 \
        --timeframe ALL
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from xau_lean.data.dukascopy import DukascopyAdapter
from xau_lean.research.accumulator import ResearchAccumulator
from xau_lean.research.report import ResearchMetadata
from xau_lean.research.timeframe import (
    Timeframe,
    aggregate_timeframe,
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
                parsed = parsed.replace(tzinfo=timezone.utc)

            parsed = parsed.astimezone(timezone.utc)

    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date/datetime: {value!r}. "
            "Use YYYY-MM-DD or ISO-8601 datetime."
        ) from exc

    return parsed


def build_parser() -> argparse.ArgumentParser:
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
        choices=["ALL", *[timeframe.value for timeframe in Timeframe]],
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


def run_timeframe(
    *,
    adapter: DukascopyAdapter,
    start: datetime,
    end: datetime,
    timeframe: Timeframe,
    output_root: Path,
    generated_at: datetime,
) -> Path:
    """Run one timeframe in a fully streaming manner."""

    metadata = ResearchMetadata(
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

    accumulator = ResearchAccumulator(
        metadata=metadata,
        atr_period=ATR_PERIOD,
        spread_thresholds=SPREAD_THRESHOLDS,
    )

    m1_bars = adapter.iter_bars(
        start,
        end,
        require_ask=True,
    )

    candles = aggregate_timeframe(
        m1_bars,
        timeframe,
    )

    accumulator.update_many(candles)

    report = accumulator.build_report()

    output_root.mkdir(parents=True, exist_ok=True)

    filename = (
        f"xauusd_{timeframe.value.lower()}_"
        f"{start.strftime('%Y%m%dT%H%M%SZ')}_"
        f"{end.strftime('%Y%m%dT%H%M%SZ')}.json"
    )

    output_path = output_root / filename

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            report.to_dict(),
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print(
        f"{timeframe.value}: "
        f"{report.candle_count:,} candles | "
        f"{report.complete_candle_count:,} complete | "
        f"{report.incomplete_candle_count:,} incomplete"
    )

    print(f"  mean range: {report.mean_range}")
    print(f"  median range: {report.median_range}")
    print(f"  realized volatility: {report.realized_volatility}")
    print(f"  mean ATR: {report.mean_atr}")
    print(f"  mean spread: {report.mean_spread}")
    print(f"  spread/range: {report.mean_spread_to_range}")
    print(f"  output: {output_path}")

    return output_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.end <= args.start:
        parser.error("--end must be after --start")

    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    print("=" * 78)
    print("XAUUSD — PHASE 6 TIMEFRAME RESEARCH")
    print("=" * 78)
    print(f"Source:     {SOURCE_NAME}")
    print(f"Data root:  {data_root}")
    print(f"Start:      {args.start.isoformat()}")
    print(f"End:        {args.end.isoformat()}")
    print(f"Timezone:   {TIMEZONE_NAME}")
    print(f"Output:     {output_root}")
    print()

    if not data_root.is_dir():
        parser.error(
            f"Dukascopy data root does not exist: {data_root}"
        )

    adapter = DukascopyAdapter(data_root)

    generated_at = args.generated_at or datetime.now(timezone.utc)

    if args.timeframe == "ALL":
        timeframes = list(Timeframe)
    else:
        timeframes = [Timeframe(args.timeframe)]

    print(
        "Timeframes: "
        + ", ".join(timeframe.value for timeframe in timeframes)
    )
    print()

    outputs: list[Path] = []

    for index, timeframe in enumerate(timeframes, start=1):
        print(
            f"[{index}/{len(timeframes)}] "
            f"Running {timeframe.value}..."
        )

        output_path = run_timeframe(
            adapter=adapter,
            start=args.start,
            end=args.end,
            timeframe=timeframe,
            output_root=output_root,
            generated_at=generated_at,
        )

        outputs.append(output_path)
        print()

    print("=" * 78)
    print("RESEARCH RUN COMPLETE")
    print("=" * 78)
    print(f"Reports generated: {len(outputs)}")

    for output in outputs:
        print(f"  {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
