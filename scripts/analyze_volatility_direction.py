#!/usr/bin/env python3
"""
Phase 6.2B — Volatility Expansion × Direction Interaction

Research question
-----------------
Does volatility expansion change the predictive value of the signal
candle's direction?

This is a diagnostic research study only.

It does NOT:
- select a trading strategy
- optimize thresholds
- optimize horizons
- invert signals
- modify the canonical signal engine
- use future information for volatility classification

Canonical mechanics
-------------------
- Dukascopy XAUUSD BID/ASK M1
- calendar-aligned UTC aggregation
- ATR(14)
- baseline = previous 20 ATR values
- existing canonical volatility buckets
- signal direction from the signal candle
- signal created after candle close
- entry at next candle OPEN
- horizons = 1, 3, 5 candles

Primary diagnostics
-------------------
For each period × timeframe × volatility bucket × horizon:

- bullish accuracy
- bearish accuracy
- bullish mean signed return
- bearish mean signed return
- bullish mean absolute return
- bearish mean absolute return
- directional spread
- sample size

Primary interaction statistic
-----------------------------
For each volatility bucket:

    direction_accuracy_spread =
        bullish_accuracy - bearish_accuracy

and

    direction_return_spread =
        bullish_mean_return - bearish_mean_return

We then compare these against the normal-volatility bucket:

    volatility_direction_interaction =
        bucket_direction_spread - normal_direction_spread

This asks whether volatility changes directional information,
rather than merely asking whether volatility increases movement size.

No candidate selection is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from xau_lean.data.dukascopy import DukascopyAdapter
from xau_lean.research.regimes import REGIMES, get_regime, validate_regimes
from xau_lean.research.timeframe import Timeframe, aggregate_timeframes

from scripts.run_direction_research import (
    ATR_PERIOD,
    BASELINE_PERIOD,
    HORIZONS,
    BUCKETS,
    candle_direction,
    classify_bucket,
    true_range,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "dukascopy"
    / "Market-Data-Lab-main"
    / "xauusd"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "research_results"
    / "volatility_direction"
)

SOURCE_NAME = "Dukascopy XAUUSD BID/ASK M1"
TIMEZONE_NAME = "UTC"

TIMEFRAMES = tuple(Timeframe)

PERIODS = (
    ("2010-2014", "2010-01-01T00:00:00Z", "2015-01-01T00:00:00Z"),
    ("2015-2019", "2015-01-01T00:00:00Z", "2020-01-01T00:00:00Z"),
    ("2020-2021", "2020-01-01T00:00:00Z", "2022-01-01T00:00:00Z"),
    ("2022-2023", "2022-01-01T00:00:00Z", "2024-01-01T00:00:00Z"),
    ("2024-2026", "2024-01-01T00:00:00Z", "2026-08-22T00:00:00Z"),
)


@dataclass
class PendingSignal:
    timeframe: str
    regime: str
    bucket: str
    direction: str
    horizon: int
    signal_timestamp: datetime

    entry_timestamp: datetime | None = None
    bid_entry: float | None = None
    ask_entry: float | None = None
    candles_since_entry: int = 0


@dataclass(frozen=True)
class Observation:
    timeframe: str
    regime: str
    bucket: str
    direction: str
    horizon: int
    signal_timestamp: datetime
    entry_timestamp: datetime
    bid_entry: float
    ask_entry: float
    bid_exit: float
    ask_exit: float
    signed_return: float

    @property
    def abs_return(self) -> float:
        return abs(self.signed_return)


def parse_datetime(value: str) -> datetime:
    text = value.strip()

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    parsed = datetime.fromisoformat(text)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def period_name(timestamp: datetime) -> str | None:
    for name, start_text, end_text in PERIODS:
        start = parse_datetime(start_text)
        end = parse_datetime(end_text)

        if start <= timestamp < end:
            return name

    return None


class ObservationEngine:
    """
    Reconstruct canonical direction observations while preserving
    the volatility bucket.
    """

    def __init__(self, timeframe: Timeframe) -> None:
        self.timeframe = timeframe

        self.previous_close: float | None = None

        self.tr_window: deque[float] = deque(maxlen=ATR_PERIOD)
        self.atr_history: deque[float] = deque(maxlen=BASELINE_PERIOD)

        self.pending: dict[int, list[PendingSignal]] = {
            horizon: []
            for horizon in HORIZONS
        }

        self.observations: list[Observation] = []

        self.candles = 0
        self.complete_candles = 0
        self.atr_ready = 0
        self.signals_created = 0
        self.signals_resolved = 0

    def process(self, candle: Any) -> None:
        self.candles += 1

        if not getattr(candle, "complete", True):
            return

        self.complete_candles += 1

        # Resolve only signals created by previous candles.
        self._process_pending(candle)

        bid_open = float(candle.bid_open)
        bid_high = float(candle.bid_high)
        bid_low = float(candle.bid_low)
        bid_close = float(candle.bid_close)

        ask_open = (
            float(candle.ask_open)
            if candle.ask_open is not None
            else None
        )

        ask_close = (
            float(candle.ask_close)
            if candle.ask_close is not None
            else None
        )

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

        if len(self.atr_history) < BASELINE_PERIOD:
            self.atr_history.append(atr)
            return

        baseline = sum(self.atr_history) / len(self.atr_history)

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

        if direction != "flat":
            for horizon in HORIZONS:
                self.pending[horizon].append(
                    PendingSignal(
                        timeframe=self.timeframe.name,
                        regime=regime.name,
                        bucket=bucket,
                        direction=direction,
                        horizon=horizon,
                        signal_timestamp=candle.timestamp,
                    )
                )

                self.signals_created += 1

        # IMPORTANT:
        # Current ATR becomes eligible for future baselines only after
        # current classification is complete.
        self.atr_history.append(atr)

    def _process_pending(self, candle: Any) -> None:
        bid_open = float(candle.bid_open)
        bid_close = float(candle.bid_close)

        ask_open = (
            float(candle.ask_open)
            if candle.ask_open is not None
            else None
        )

        ask_close = (
            float(candle.ask_close)
            if candle.ask_close is not None
            else None
        )

        for horizon in HORIZONS:
            queue = self.pending[horizon]

            if not queue:
                continue

            remaining: list[PendingSignal] = []

            for signal in queue:
                if signal.entry_timestamp is None:
                    if ask_open is None:
                        continue

                    signal.entry_timestamp = candle.timestamp
                    signal.bid_entry = bid_open
                    signal.ask_entry = ask_open
                    signal.candles_since_entry = 1
                else:
                    signal.candles_since_entry += 1

                if signal.candles_since_entry < horizon:
                    remaining.append(signal)
                    continue

                if (
                    signal.bid_entry is None
                    or signal.ask_entry is None
                    or ask_close is None
                    or bid_close <= 0.0
                ):
                    continue

                if signal.direction == "bullish":
                    reference_return = (
                        bid_close - signal.bid_entry
                    ) / signal.bid_entry
                else:
                    reference_return = (
                        signal.bid_entry - bid_close
                    ) / signal.bid_entry

                self.observations.append(
                    Observation(
                        timeframe=signal.timeframe,
                        regime=signal.regime,
                        bucket=signal.bucket,
                        direction=signal.direction,
                        horizon=signal.horizon,
                        signal_timestamp=signal.signal_timestamp,
                        entry_timestamp=signal.entry_timestamp,
                        bid_entry=signal.bid_entry,
                        ask_entry=signal.ask_entry,
                        bid_exit=bid_close,
                        ask_exit=ask_close,
                        signed_return=reference_return,
                    )
                )

                self.signals_resolved += 1

            self.pending[horizon] = remaining


def summarize(
    observations: list[Observation],
) -> dict[str, Any]:
    """
    Calculate diagnostics for one bucket × direction × horizon cell.
    """

    n = len(observations)

    if n == 0:
        return {
            "observations": 0,
            "accuracy": None,
            "mean_signed_return": None,
            "mean_abs_return": None,
        }

    correct = sum(
        observation.signed_return > 0.0
        for observation in observations
    )

    signed = sum(
        observation.signed_return
        for observation in observations
    )

    absolute = sum(
        observation.abs_return
        for observation in observations
    )

    return {
        "observations": n,
        "accuracy": correct / n,
        "mean_signed_return": signed / n,
        "mean_abs_return": absolute / n,
    }


def interaction(
    bullish: dict[str, Any],
    bearish: dict[str, Any],
    normal_bullish: dict[str, Any],
    normal_bearish: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare a volatility bucket's directional effect with normal volatility.

    Primary:
        accuracy interaction

    Secondary:
        signed-return interaction
    """

    required = (
        bullish,
        bearish,
        normal_bullish,
        normal_bearish,
    )

    if any(item["observations"] == 0 for item in required):
        return {
            "accuracy_interaction": None,
            "signed_return_interaction": None,
        }

    bucket_accuracy_spread = (
        bullish["accuracy"] - bearish["accuracy"]
    )

    normal_accuracy_spread = (
        normal_bullish["accuracy"]
        - normal_bearish["accuracy"]
    )

    bucket_return_spread = (
        bullish["mean_signed_return"]
        - bearish["mean_signed_return"]
    )

    normal_return_spread = (
        normal_bullish["mean_signed_return"]
        - normal_bearish["mean_signed_return"]
    )

    return {
        "accuracy_interaction": (
            bucket_accuracy_spread
            - normal_accuracy_spread
        ),
        "signed_return_interaction": (
            bucket_return_spread
            - normal_return_spread
        ),
        "direction_accuracy_spread": bucket_accuracy_spread,
        "normal_direction_accuracy_spread": normal_accuracy_spread,
        "direction_return_spread": bucket_return_spread,
        "normal_direction_return_spread": normal_return_spread,
    }


def analyze_group(
    observations: list[Observation],
) -> dict[str, Any]:
    bullish = summarize(
        [
            item
            for item in observations
            if item.direction == "bullish"
        ]
    )

    bearish = summarize(
        [
            item
            for item in observations
            if item.direction == "bearish"
        ]
    )

    return {
        "bullish": bullish,
        "bearish": bearish,
    }


def build_results(
    observations: list[Observation],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []

    for timeframe in [item.name for item in TIMEFRAMES]:
        results[timeframe] = {}

        timeframe_obs = [
            item
            for item in observations
            if item.timeframe == timeframe
        ]

        for period, _, _ in PERIODS:
            period_obs = [
                item
                for item in timeframe_obs
                if period_name(item.signal_timestamp) == period
            ]

            results[timeframe][period] = {}

            for horizon in HORIZONS:
                horizon_obs = [
                    item
                    for item in period_obs
                    if item.horizon == horizon
                ]

                bucket_results: dict[str, Any] = {}

                normal_groups = analyze_group(
                    [
                        item
                        for item in horizon_obs
                        if item.bucket == "normal"
                    ]
                )

                for bucket_name, _, _ in BUCKETS:
                    bucket_obs = [
                        item
                        for item in horizon_obs
                        if item.bucket == bucket_name
                    ]

                    groups = analyze_group(bucket_obs)

                    if bucket_name == "normal":
                        interaction_result = {
                            "accuracy_interaction": 0.0,
                            "signed_return_interaction": 0.0,
                            "direction_accuracy_spread": (
                                groups["bullish"]["accuracy"]
                                - groups["bearish"]["accuracy"]
                                if (
                                    groups["bullish"]["accuracy"] is not None
                                    and groups["bearish"]["accuracy"] is not None
                                )
                                else None
                            ),
                            "normal_direction_accuracy_spread": (
                                groups["bullish"]["accuracy"]
                                - groups["bearish"]["accuracy"]
                                if (
                                    groups["bullish"]["accuracy"] is not None
                                    and groups["bearish"]["accuracy"] is not None
                                )
                                else None
                            ),
                            "direction_return_spread": (
                                groups["bullish"]["mean_signed_return"]
                                - groups["bearish"]["mean_signed_return"]
                                if (
                                    groups["bullish"]["mean_signed_return"] is not None
                                    and groups["bearish"]["mean_signed_return"] is not None
                                )
                                else None
                            ),
                            "normal_direction_return_spread": (
                                groups["bullish"]["mean_signed_return"]
                                - groups["bearish"]["mean_signed_return"]
                                if (
                                    groups["bullish"]["mean_signed_return"] is not None
                                    and groups["bearish"]["mean_signed_return"] is not None
                                )
                                else None
                            ),
                        }
                    else:
                        interaction_result = interaction(
                            groups["bullish"],
                            groups["bearish"],
                            normal_groups["bullish"],
                            normal_groups["bearish"],
                        )

                    bucket_results[bucket_name] = {
                        **groups,
                        "interaction_vs_normal": interaction_result,
                    }

                    for direction in ("bullish", "bearish"):
                        summary = groups[direction]

                        csv_rows.append(
                            {
                                "timeframe": timeframe,
                                "period": period,
                                "horizon": horizon,
                                "bucket": bucket_name,
                                "direction": direction,
                                "observations": summary["observations"],
                                "accuracy": summary["accuracy"],
                                "mean_signed_return": summary[
                                    "mean_signed_return"
                                ],
                                "mean_abs_return": summary[
                                    "mean_abs_return"
                                ],
                                "accuracy_interaction": interaction_result[
                                    "accuracy_interaction"
                                ],
                                "signed_return_interaction": interaction_result[
                                    "signed_return_interaction"
                                ],
                            }
                        )

                results[timeframe][period][f"H{horizon}"] = bucket_results

    return results, csv_rows


def sanitize(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, dict):
        return {
            key: sanitize(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [sanitize(item) for item in value]

    return value


def format_timestamp(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .strftime("%Y%m%dT%H%M%SZ")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 6.2B XAUUSD volatility expansion × direction analysis."
        )
    )

    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)

    parser.add_argument(
        "--timeframe",
        choices=["ALL", *[item.name for item in TIMEFRAMES]],
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


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    start = parse_datetime(args.start)
    end = parse_datetime(args.end)

    if end <= start:
        parser.error("--end must be after --start")

    validate_regimes(REGIMES)

    selected_timeframes = (
        TIMEFRAMES
        if args.timeframe == "ALL"
        else (Timeframe[args.timeframe],)
    )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 110)
    print("XAUUSD — PHASE 6.2B VOLATILITY EXPANSION × DIRECTION")
    print("=" * 110)
    print()
    print("Research question:")
    print("Does volatility expansion change directional information?")
    print()
    print(
        f"Window:      {start.isoformat()} -> {end.isoformat()}"
    )
    print(
        "Timeframes:  "
        + ", ".join(item.name for item in selected_timeframes)
    )
    print(f"Horizons:    {HORIZONS}")
    print(f"ATR period:  {ATR_PERIOD}")
    print(f"Baseline:    {BASELINE_PERIOD}")
    print()

    adapter = DukascopyAdapter(args.data_root)

    engines = {
        timeframe: ObservationEngine(timeframe)
        for timeframe in selected_timeframes
    }

    bars = adapter.iter_bars(
        start,
        end,
        require_ask=True,
    )

    total_candles = 0

    for timeframe, candle in aggregate_timeframes(
        bars,
        selected_timeframes,
    ):
        engines[timeframe].process(candle)
        total_candles += 1

    observations: list[Observation] = []

    for engine in engines.values():
        observations.extend(engine.observations)

    observations.sort(
        key=lambda item: (
            item.timeframe,
            item.signal_timestamp,
            item.horizon,
        )
    )

    results, csv_rows = build_results(observations)

    filename = (
        "xauusd_volatility_direction_"
        f"{format_timestamp(start)}_"
        f"{format_timestamp(end)}"
    )

    json_path = args.output_root / f"{filename}.json"
    csv_path = args.output_root / f"{filename}.csv"

    metadata = {
        "research": {
            "phase": "6.2B",
            "name": "Volatility Expansion × Direction",
            "research_question": (
                "Does volatility expansion change the predictive "
                "value of signal-candle direction?"
            ),
            "source": SOURCE_NAME,
            "timezone": TIMEZONE_NAME,
            "input_frequency": "M1",
            "aggregation": "calendar-aligned UTC",
            "atr_period": ATR_PERIOD,
            "baseline_period": BASELINE_PERIOD,
            "horizons": list(HORIZONS),
            "timeframes": [
                item.name
                for item in selected_timeframes
            ],
            "interpolation": False,
            "fill_forward": False,
            "raw_data_modified": False,
            "signal_timing": (
                "Current ATR is classified at candle close against "
                "prior ATR values; signal entry begins at the next "
                "candle open."
            ),
            "primary_statistic": (
                "bucket direction accuracy spread minus normal "
                "direction accuracy spread"
            ),
            "secondary_statistic": (
                "bucket direction signed-return spread minus normal "
                "direction signed-return spread"
            ),
            "strategy_modification": False,
            "candidate_selection": False,
        },
        "counts": {
            "aggregated_candles": total_candles,
            "observations": len(observations),
            "signals_created": sum(
                engine.signals_created
                for engine in engines.values()
            ),
            "signals_resolved": sum(
                engine.signals_resolved
                for engine in engines.values()
            ),
        },
        "engine_counts": {
            timeframe.name: {
                "candles": engine.candles,
                "complete_candles": engine.complete_candles,
                "atr_ready": engine.atr_ready,
                "signals_created": engine.signals_created,
                "signals_resolved": engine.signals_resolved,
            }
            for timeframe, engine in engines.items()
        },
        "results": results,
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            sanitize(metadata),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        if csv_rows:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(csv_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(csv_rows)

    print("RESEARCH COMPLETE")
    print(f"Aggregated candles: {total_candles:,}")
    print(f"Observations:       {len(observations):,}")
    print(
        "Signals created:    "
        f"{sum(engine.signals_created for engine in engines.values()):,}"
    )
    print(
        "Signals resolved:   "
        f"{sum(engine.signals_resolved for engine in engines.values()):,}"
    )
    print()
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
