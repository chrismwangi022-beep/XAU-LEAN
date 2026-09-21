#!/usr/bin/env python3
"""
Phase 6.2B Formal Validation — Volatility Expansion × Direction

This module formally validates the descriptive Phase 6.2B
Volatility Expansion × Direction interaction using a
dependence-aware moving-block bootstrap.

Research philosophy
-------------------
- Research first, strategy second.
- No strategy selection.
- No parameter optimization.
- No threshold optimization.
- No signal inversion.
- No modification of the canonical 6.2B observation engine.
- No future information.
- No interpolation.
- No fill-forward.
- No null-hypothesis p-value is claimed from bootstrap sign
  probabilities.

Primary statistic
-----------------
For a volatility bucket:

    bucket_accuracy_spread =
        bucket_bullish_accuracy - bucket_bearish_accuracy

    normal_accuracy_spread =
        normal_bullish_accuracy - normal_bearish_accuracy

    accuracy_interaction =
        bucket_accuracy_spread - normal_accuracy_spread

Secondary statistic:

    bucket_return_spread =
        bucket_bullish_signed_return - bucket_bearish_signed_return

    normal_return_spread =
        normal_bullish_signed_return - normal_bearish_signed_return

    signed_return_interaction =
        bucket_return_spread - normal_return_spread

Bootstrap
---------
A moving-block bootstrap is applied to the chronological observations
within each timeframe × period × horizon × bucket comparison.

The block length follows the canonical Phase 6.3B rule:

    min(50, max(5, horizon, round(sqrt(n))))

where n is the number of observations in the comparison sample.

The bootstrap estimates uncertainty around the observed interaction.
It does NOT provide a p-value.

Outputs
-------
JSON and CSV are written under:

    research_results/volatility_direction_formal/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.analyze_volatility_direction import (
    BUCKETS,
    HORIZONS,
    PERIODS,
    TIMEFRAMES,
    Observation,
    ObservationEngine,
    format_timestamp,
    parse_datetime,
    sanitize,
    validate_regimes,
)

from xau_lean.data.dukascopy import DukascopyAdapter
from xau_lean.research.regimes import REGIMES
from xau_lean.research.timeframe import Timeframe, aggregate_timeframes


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
    / "volatility_direction_formal"
)

SOURCE_NAME = "Dukascopy XAUUSD BID/ASK M1"
TIMEZONE_NAME = "UTC"

MIN_BLOCK_LENGTH = 5
MAX_BLOCK_LENGTH = 50


@dataclass(frozen=True)
class BootstrapResult:
    statistic: float | None
    lower_95: float | None
    upper_95: float | None
    probability_positive: float | None
    probability_negative: float | None
    replicates: int
    valid_replicates: int
    block_length: int


def choose_block_length(n: int, horizon: int) -> int:
    """
    Canonical dependence-aware moving-block bootstrap length.

    Matches the Phase 6.3B rule:
        min(50, max(5, horizon, round(sqrt(n))))
    """
    if n <= 0:
        return MIN_BLOCK_LENGTH

    return min(
        MAX_BLOCK_LENGTH,
        max(
            MIN_BLOCK_LENGTH,
            horizon,
            round(math.sqrt(n)),
        ),
    )


def group_accuracy(
    observations: list[Observation],
) -> float:
    """
    Directional accuracy.

    A bullish signal is correct when the forward signed return
    is positive.

    A bearish signal is correct when the forward signed return
    is negative.

    This matches the directional-accuracy definition used by
    Phase 6.2B.
    """
    if not observations:
        return float("nan")

    correct = 0

    for item in observations:
        if item.direction == "bullish":
            if item.signed_return > 0:
                correct += 1
        elif item.direction == "bearish":
            if item.signed_return < 0:
                correct += 1

    return correct / len(observations)


def group_signed_return(
    observations: list[Observation],
) -> float:
    if not observations:
        return float("nan")

    return sum(
        item.signed_return
        for item in observations
    ) / len(observations)


def interaction_from_observations(
    observations: list[Observation],
    bucket: str,
) -> tuple[float, float]:
    """
    Calculate the exact Phase 6.2B interaction statistics.

    The comparison is:

        bucket bullish
        bucket bearish
        normal bullish
        normal bearish
    """

    bucket_bullish = [
        item
        for item in observations
        if item.bucket == bucket
        and item.direction == "bullish"
    ]

    bucket_bearish = [
        item
        for item in observations
        if item.bucket == bucket
        and item.direction == "bearish"
    ]

    normal_bullish = [
        item
        for item in observations
        if item.bucket == "normal"
        and item.direction == "bullish"
    ]

    normal_bearish = [
        item
        for item in observations
        if item.bucket == "normal"
        and item.direction == "bearish"
    ]

    groups = (
        bucket_bullish,
        bucket_bearish,
        normal_bullish,
        normal_bearish,
    )

    if any(not group for group in groups):
        return float("nan"), float("nan")

    bucket_accuracy_spread = (
        group_accuracy(bucket_bullish)
        - group_accuracy(bucket_bearish)
    )

    normal_accuracy_spread = (
        group_accuracy(normal_bullish)
        - group_accuracy(normal_bearish)
    )

    accuracy_interaction = (
        bucket_accuracy_spread
        - normal_accuracy_spread
    )

    bucket_return_spread = (
        group_signed_return(bucket_bullish)
        - group_signed_return(bucket_bearish)
    )

    normal_return_spread = (
        group_signed_return(normal_bullish)
        - group_signed_return(normal_bearish)
    )

    signed_return_interaction = (
        bucket_return_spread
        - normal_return_spread
    )

    return (
        accuracy_interaction,
        signed_return_interaction,
    )


def percentile(
    values: list[float],
    probability: float,
) -> float:
    """
    Linear-interpolation percentile.

    The implementation is deterministic and avoids an external
    numerical dependency for this small statistic.
    """
    if not values:
        return float("nan")

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower

    return (
        ordered[lower] * (1.0 - weight)
        + ordered[upper] * weight
    )


def moving_block_sample(
    observations: list[Observation],
    block_length: int,
    rng: random.Random,
) -> list[Observation]:
    """
    Moving-block bootstrap sample.

    Blocks are contiguous in the chronological observation sequence.
    Sampling continues with replacement until n observations are
    obtained, then the result is truncated to exactly n.
    """
    n = len(observations)

    if n == 0:
        return []

    if n <= block_length:
        return list(observations)

    sampled: list[Observation] = []

    max_start = n - block_length

    while len(sampled) < n:
        start = rng.randint(0, max_start)
        sampled.extend(
            observations[
                start:start + block_length
            ]
        )

    return sampled[:n]


def bootstrap_interaction(
    observations: list[Observation],
    bucket: str,
    horizon: int,
    replicates: int,
    seed: int,
) -> BootstrapResult:
    """
    Bootstrap both Phase 6.2B interaction statistics.

    This function is deliberately separate from the canonical
    6.2B engine. It operates only on already-created observations.
    """
    ordered = sorted(
        observations,
        key=lambda item: (
            item.signal_timestamp,
            item.entry_timestamp,
        ),
    )

    accuracy_statistic, _ = interaction_from_observations(
        ordered,
        bucket,
    )

    if math.isnan(accuracy_statistic):
        return BootstrapResult(
            statistic=None,
            lower_95=None,
            upper_95=None,
            probability_positive=None,
            probability_negative=None,
            replicates=replicates,
            valid_replicates=0,
            block_length=choose_block_length(
                len(ordered),
                horizon,
            ),
        )

    block_length = choose_block_length(
        len(ordered),
        horizon,
    )

    rng = random.Random(seed)

    accuracy_values: list[float] = []

    for _ in range(replicates):
        sample = moving_block_sample(
            ordered,
            block_length,
            rng,
        )

        accuracy_value, _ = interaction_from_observations(
            sample,
            bucket,
        )

        if not math.isnan(accuracy_value):
            accuracy_values.append(accuracy_value)

    if not accuracy_values:
        return BootstrapResult(
            statistic=accuracy_statistic,
            lower_95=None,
            upper_95=None,
            probability_positive=None,
            probability_negative=None,
            replicates=replicates,
            valid_replicates=0,
            block_length=block_length,
        )

    positive = sum(
        value > 0
        for value in accuracy_values
    )

    negative = sum(
        value < 0
        for value in accuracy_values
    )

    return BootstrapResult(
        statistic=accuracy_statistic,
        lower_95=percentile(
            accuracy_values,
            0.025,
        ),
        upper_95=percentile(
            accuracy_values,
            0.975,
        ),
        probability_positive=(
            positive / len(accuracy_values)
        ),
        probability_negative=(
            negative / len(accuracy_values)
        ),
        replicates=replicates,
        valid_replicates=len(accuracy_values),
        block_length=block_length,
    )


def bootstrap_signed_return(
    observations: list[Observation],
    bucket: str,
    horizon: int,
    replicates: int,
    seed: int,
) -> BootstrapResult:
    """
    Bootstrap the Phase 6.2B signed-return interaction.
    """
    ordered = sorted(
        observations,
        key=lambda item: (
            item.signal_timestamp,
            item.entry_timestamp,
        ),
    )

    _, signed_statistic = interaction_from_observations(
        ordered,
        bucket,
    )

    if math.isnan(signed_statistic):
        return BootstrapResult(
            statistic=None,
            lower_95=None,
            upper_95=None,
            probability_positive=None,
            probability_negative=None,
            replicates=replicates,
            valid_replicates=0,
            block_length=choose_block_length(
                len(ordered),
                horizon,
            ),
        )

    block_length = choose_block_length(
        len(ordered),
        horizon,
    )

    rng = random.Random(seed)

    values: list[float] = []

    for _ in range(replicates):
        sample = moving_block_sample(
            ordered,
            block_length,
            rng,
        )

        _, value = interaction_from_observations(
            sample,
            bucket,
        )

        if not math.isnan(value):
            values.append(value)

    if not values:
        return BootstrapResult(
            statistic=signed_statistic,
            lower_95=None,
            upper_95=None,
            probability_positive=None,
            probability_negative=None,
            replicates=replicates,
            valid_replicates=0,
            block_length=block_length,
        )

    positive = sum(
        value > 0
        for value in values
    )

    negative = sum(
        value < 0
        for value in values
    )

    return BootstrapResult(
        statistic=signed_statistic,
        lower_95=percentile(values, 0.025),
        upper_95=percentile(values, 0.975),
        probability_positive=(
            positive / len(values)
        ),
        probability_negative=(
            negative / len(values)
        ),
        replicates=replicates,
        valid_replicates=len(values),
        block_length=block_length,
    )


def group_counts(
    observations: list[Observation],
    bucket: str,
) -> dict[str, int]:
    return {
        "bucket_bullish": sum(
            item.bucket == bucket
            and item.direction == "bullish"
            for item in observations
        ),
        "bucket_bearish": sum(
            item.bucket == bucket
            and item.direction == "bearish"
            for item in observations
        ),
        "normal_bullish": sum(
            item.bucket == "normal"
            and item.direction == "bullish"
            for item in observations
        ),
        "normal_bearish": sum(
            item.bucket == "normal"
            and item.direction == "bearish"
            for item in observations
        ),
    }


def build_formal_results(
    observations: list[Observation],
    bootstrap_replicates: int,
    seed: int,
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
            results[timeframe][period] = {}

            period_obs = [
                item
                for item in timeframe_obs
                if period_name(item.signal_timestamp) == period
            ]

            for horizon in HORIZONS:
                horizon_obs = [
                    item
                    for item in period_obs
                    if item.horizon == horizon
                ]

                horizon_results: dict[str, Any] = {}

                for bucket_name, _, _ in BUCKETS:
                    if bucket_name == "normal":
                        continue

                    comparison_obs = [
                        item
                        for item in horizon_obs
                        if item.bucket in {
                            "normal",
                            bucket_name,
                        }
                    ]

                    counts = group_counts(
                        comparison_obs,
                        bucket_name,
                    )

                    accuracy_bootstrap = bootstrap_interaction(
                        horizon_obs,
                        bucket=bucket_name,
                        horizon=horizon,
                        replicates=bootstrap_replicates,
                        seed=seed,
                    )

                    signed_bootstrap = bootstrap_signed_return(
                        horizon_obs,
                        bucket=bucket_name,
                        horizon=horizon,
                        replicates=bootstrap_replicates,
                        seed=seed + 1,
                    )

                    accuracy = (
                        interaction_from_observations(
                            comparison_obs,
                            bucket_name,
                        )[0]
                    )

                    signed_return = (
                        interaction_from_observations(
                            comparison_obs,
                            bucket_name,
                        )[1]
                    )

                    result = {
                        "observations": len(comparison_obs),
                        "group_counts": counts,
                        "accuracy_interaction": {
                            "statistic": (
                                None
                                if math.isnan(accuracy)
                                else accuracy
                            ),
                            "lower_95": (
                                accuracy_bootstrap.lower_95
                            ),
                            "upper_95": (
                                accuracy_bootstrap.upper_95
                            ),
                            "probability_positive": (
                                accuracy_bootstrap.probability_positive
                            ),
                            "probability_negative": (
                                accuracy_bootstrap.probability_negative
                            ),
                            "replicates": (
                                accuracy_bootstrap.replicates
                            ),
                            "valid_replicates": (
                                accuracy_bootstrap.valid_replicates
                            ),
                            "block_length": (
                                accuracy_bootstrap.block_length
                            ),
                        },
                        "signed_return_interaction": {
                            "statistic": (
                                None
                                if math.isnan(signed_return)
                                else signed_return
                            ),
                            "lower_95": (
                                signed_bootstrap.lower_95
                            ),
                            "upper_95": (
                                signed_bootstrap.upper_95
                            ),
                            "probability_positive": (
                                signed_bootstrap.probability_positive
                            ),
                            "probability_negative": (
                                signed_bootstrap.probability_negative
                            ),
                            "replicates": (
                                signed_bootstrap.replicates
                            ),
                            "valid_replicates": (
                                signed_bootstrap.valid_replicates
                            ),
                            "block_length": (
                                signed_bootstrap.block_length
                            ),
                        },
                        "null_p_value": None,
                        "null_test_status": (
                            "not_performed"
                        ),
                    }

                    horizon_results[bucket_name] = result

                    csv_rows.append(
                        {
                            "timeframe": timeframe,
                            "period": period,
                            "horizon": horizon,
                            "bucket": bucket_name,
                            "observations": len(comparison_obs),
                            **counts,
                            "accuracy_interaction": accuracy_bootstrap.statistic,
                            "accuracy_lower_95": (
                                accuracy_bootstrap.lower_95
                            ),
                            "accuracy_upper_95": (
                                accuracy_bootstrap.upper_95
                            ),
                            "accuracy_probability_positive": (
                                accuracy_bootstrap.probability_positive
                            ),
                            "accuracy_probability_negative": (
                                accuracy_bootstrap.probability_negative
                            ),
                            "accuracy_replicates": (
                                accuracy_bootstrap.replicates
                            ),
                            "accuracy_valid_replicates": (
                                accuracy_bootstrap.valid_replicates
                            ),
                            "accuracy_block_length": (
                                accuracy_bootstrap.block_length
                            ),
                            "signed_return_interaction": (
                                signed_bootstrap.statistic
                            ),
                            "signed_return_lower_95": (
                                signed_bootstrap.lower_95
                            ),
                            "signed_return_upper_95": (
                                signed_bootstrap.upper_95
                            ),
                            "signed_return_probability_positive": (
                                signed_bootstrap.probability_positive
                            ),
                            "signed_return_probability_negative": (
                                signed_bootstrap.probability_negative
                            ),
                            "signed_return_replicates": (
                                signed_bootstrap.replicates
                            ),
                            "signed_return_valid_replicates": (
                                signed_bootstrap.valid_replicates
                            ),
                            "signed_return_block_length": (
                                signed_bootstrap.block_length
                            ),
                            "null_p_value": None,
                        }
                    )

                results[timeframe][period][
                    f"H{horizon}"
                ] = horizon_results

    return results, csv_rows


def period_name(timestamp: datetime) -> str:
    year = timestamp.year

    if 2010 <= year <= 2014:
        return "2010-2014"
    if 2015 <= year <= 2019:
        return "2015-2019"
    if 2020 <= year <= 2021:
        return "2020-2021"
    if 2022 <= year <= 2023:
        return "2022-2023"
    if 2024 <= year <= 2026:
        return "2024-2026"

    raise ValueError(
        f"Timestamp outside defined research periods: {timestamp}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 6.2B formal volatility × direction validation."
        )
    )

    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)

    parser.add_argument(
        "--timeframe",
        choices=[
            "ALL",
            *[item.name for item in TIMEFRAMES],
        ],
        default="ALL",
    )

    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260911,
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

    if args.bootstrap_replicates <= 0:
        parser.error(
            "--bootstrap-replicates must be positive"
        )

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
    print(
        "XAUUSD — PHASE 6.2B FORMAL "
        "VOLATILITY × DIRECTION VALIDATION"
    )
    print("=" * 110)
    print()
    print("Method:        Moving-block bootstrap")
    print("Confidence:    95% percentile interval")
    print(
        "P-value:       NOT calculated / NOT claimed"
    )
    print(
        f"Replicates:    {args.bootstrap_replicates:,}"
    )
    print(
        f"Seed:          {args.seed}"
    )
    print(
        f"Window:        {start.isoformat()} -> {end.isoformat()}"
    )
    print(
        "Timeframes:    "
        + ", ".join(
            item.name
            for item in selected_timeframes
        )
    )
    print(
        f"Horizons:      {HORIZONS}"
    )
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

    results, csv_rows = build_formal_results(
        observations,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )

    filename = (
        "xauusd_volatility_direction_formal_"
        f"{format_timestamp(start)}_"
        f"{format_timestamp(end)}"
    )

    json_path = args.output_root / f"{filename}.json"
    csv_path = args.output_root / f"{filename}.csv"

    metadata = {
        "research": {
            "phase": "6.2B-formal",
            "name": (
                "Volatility Expansion × Direction "
                "Formal Validation"
            ),
            "research_question": (
                "Does volatility expansion change "
                "directional information?"
            ),
            "source": SOURCE_NAME,
            "timezone": TIMEZONE_NAME,
            "input_frequency": "M1",
            "aggregation": "calendar-aligned UTC",
            "atr_period": 14,
            "baseline_period": 20,
            "horizons": list(HORIZONS),
            "timeframes": [
                item.name
                for item in selected_timeframes
            ],
            "bootstrap": {
                "method": "moving-block bootstrap",
                "confidence_level": 0.95,
                "percentile_interval": True,
                "min_block_length": MIN_BLOCK_LENGTH,
                "max_block_length": MAX_BLOCK_LENGTH,
                "block_length_rule": (
                    "min(50, max(5, horizon, round(sqrt(n))))"
                ),
                "replicates": args.bootstrap_replicates,
                "seed": args.seed,
            },
            "uncertainty_interpretation": {
                "probability_positive": (
                    "Fraction of bootstrap statistics "
                    "greater than zero."
                ),
                "probability_negative": (
                    "Fraction of bootstrap statistics "
                    "less than zero."
                ),
                "p_value": (
                    "Not calculated. Bootstrap sign "
                    "probabilities are not treated as "
                    "null-hypothesis p-values."
                ),
            },
            "interpolation": False,
            "fill_forward": False,
            "raw_data_modified": False,
            "strategy_modification": False,
            "candidate_selection": False,
            "parameter_optimization": False,
        },
        "counts": {
            "aggregated_candles": total_candles,
            "observations": len(observations),
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

    print("FORMAL VALIDATION COMPLETE")
    print(
        f"Aggregated candles: {total_candles:,}"
    )
    print(
        f"Observations:       {len(observations):,}"
    )
    print()
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
