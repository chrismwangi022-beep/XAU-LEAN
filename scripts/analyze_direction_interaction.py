from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from xau_lean.data.dukascopy import DukascopyAdapter
from xau_lean.research.regimes import REGIMES, validate_regimes
from xau_lean.research.timeframe import (
    Timeframe,
    aggregate_timeframes,
)

from scripts.analyze_direction_walkforward import (
    DEFAULT_DATA_ROOT,
    HORIZONS,
    Observation,
    ObservationEngine,
    PERIODS,
)


# =============================================================================
# RESEARCH SPECIFICATION
# =============================================================================

SOURCE_NAME = "Dukascopy XAUUSD BID/ASK M1"
TIMEZONE_NAME = "UTC"

TIMEFRAME = Timeframe.M15

ATR_PERIOD = 14
BASELINE_PERIOD = 20

BUCKETS = (
    ("normal", 0.0, 1.25),
    ("elevated", 1.25, 1.50),
    ("high", 1.50, 2.00),
    ("extreme", 2.00, float("inf")),
)

BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 63042

MIN_BLOCK_LENGTH = 5
MAX_BLOCK_LENGTH = 60

DEFAULT_OUTPUT_ROOT = Path(
    "research_results/direction_interaction"
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class CompactObservation:
    """
    Minimal representation required by the interaction test.

    group:
        0 = extreme bullish
        1 = extreme bearish
        2 = non-extreme bullish
        3 = non-extreme bearish

    correct:
        1 if signed_return > 0
        0 otherwise

    signed_return:
        Direction-normalized forward return.
    """

    group: int
    correct: int
    signed_return: float


@dataclass(frozen=True)
class BootstrapResult:
    statistic: float
    lower: float
    upper: float
    probability_positive: float
    probability_negative: float
    replicates: int
    block_length: int


# =============================================================================
# TIME / PERIOD HELPERS
# =============================================================================

def parse_datetime(value: str) -> datetime:
    text = value.strip()

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    parsed = datetime.fromisoformat(text)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def period_bounds(period) -> tuple[datetime, datetime]:
    """
    Supports the existing PERIODS representation used by the canonical
    walk-forward research module.

    Normalizes period boundaries to timezone-aware UTC datetimes.
    """

    def normalize(value) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)

        if isinstance(value, str):
            return parse_datetime(value)

        raise TypeError(
            f"Unsupported period boundary type: {type(value).__name__}: {value!r}"
        )

    if hasattr(period, "start") and hasattr(period, "end"):
        return (
            normalize(period.start),
            normalize(period.end),
        )

    if isinstance(period, dict):
        return (
            normalize(period["start"]),
            normalize(period["end"]),
        )

    if isinstance(period, (tuple, list)):
        if len(period) >= 3:
            return (
                normalize(period[1]),
                normalize(period[2]),
            )

        if len(period) == 2:
            return (
                normalize(period[0]),
                normalize(period[1]),
            )

    raise TypeError(
        f"Unsupported PERIODS entry: {period!r}"
    )


def period_name(period) -> str:
    if hasattr(period, "name"):
        return str(period.name)

    if isinstance(period, dict):
        return str(
            period.get(
                "name",
                period.get("label", "unknown"),
            )
        )

    if isinstance(period, (tuple, list)):
        if len(period) >= 3:
            return str(period[0])

    return "unknown"


# =============================================================================
# BUCKET HELPERS
# =============================================================================

def is_extreme(observation: Observation) -> bool:
    return observation.bucket == "extreme"


def direction_group(observation: Observation) -> int:
    """
    Convert the structural state into one compact group.

    0 = extreme bullish
    1 = extreme bearish
    2 = non-extreme bullish
    3 = non-extreme bearish
    """

    if observation.direction == "bullish":
        return 0 if is_extreme(observation) else 2

    if observation.direction == "bearish":
        return 1 if is_extreme(observation) else 3

    raise ValueError(
        f"Unexpected direction: {observation.direction!r}"
    )


def compact_observations(
    observations: Iterable[Observation],
) -> list[CompactObservation]:
    """
    Convert canonical observations into the minimal representation required
    by the interaction test.

    The observations are sorted chronologically by signal_timestamp before
    conversion. This preserves the time ordering required by the
    moving-block bootstrap.
    """

    ordered = sorted(
        observations,
        key=lambda item: item.signal_timestamp,
    )

    compact: list[CompactObservation] = []

    for observation in ordered:
        signed_return = observation.signed_return

        compact.append(
            CompactObservation(
                group=direction_group(observation),
                correct=1 if signed_return > 0.0 else 0,
                signed_return=float(signed_return),
            )
        )

    return compact


# =============================================================================
# RAW METRICS
# =============================================================================

def group_accuracy(
    counts: tuple[int, int],
) -> float:
    n, correct = counts

    if n <= 0:
        return float("nan")

    return correct / n


def group_signed_return(
    total: float,
    n: int,
) -> float:
    if n <= 0:
        return float("nan")

    return total / n


def direction_effect_accuracy(
    counts: list[tuple[int, int]],
) -> float:
    """
    Bullish accuracy - bearish accuracy.

    counts:
        index 0 = extreme bullish
        index 1 = extreme bearish
        index 2 = non-extreme bullish
        index 3 = non-extreme bearish
    """

    extreme_bullish = group_accuracy(counts[0])
    extreme_bearish = group_accuracy(counts[1])

    non_extreme_bullish = group_accuracy(counts[2])
    non_extreme_bearish = group_accuracy(counts[3])

    if any(
        math.isnan(value)
        for value in (
            extreme_bullish,
            extreme_bearish,
            non_extreme_bullish,
            non_extreme_bearish,
        )
    ):
        return float("nan")

    extreme_effect = (
        extreme_bullish - extreme_bearish
    )

    non_extreme_effect = (
        non_extreme_bullish - non_extreme_bearish
    )

    return extreme_effect - non_extreme_effect


def direction_effect_signed_return(
    counts: list[tuple[int, float]],
) -> float:
    """
    Extreme bullish-vs-bearish signed-return difference minus the
    corresponding non-extreme difference.

    This is a directional asymmetry diagnostic, not a regression coefficient.
    """

    extreme_bullish = group_signed_return(
        counts[0][1],
        counts[0][0],
    )

    extreme_bearish = group_signed_return(
        counts[1][1],
        counts[1][0],
    )

    non_extreme_bullish = group_signed_return(
        counts[2][1],
        counts[2][0],
    )

    non_extreme_bearish = group_signed_return(
        counts[3][1],
        counts[3][0],
    )

    if any(
        math.isnan(value)
        for value in (
            extreme_bullish,
            extreme_bearish,
            non_extreme_bullish,
            non_extreme_bearish,
        )
    ):
        return float("nan")

    extreme_effect = (
        extreme_bullish - extreme_bearish
    )

    non_extreme_effect = (
        non_extreme_bullish - non_extreme_bearish
    )

    return extreme_effect - non_extreme_effect


# =============================================================================
# OBSERVATION SUMMARIES
# =============================================================================

def summarize_observations(
    observations: list[CompactObservation],
) -> dict:
    counts = [
        [0, 0]
        for _ in range(4)
    ]

    signed_totals = [
        0.0
        for _ in range(4)
    ]

    for observation in observations:
        group = observation.group

        counts[group][0] += 1
        counts[group][1] += observation.correct
        signed_totals[group] += observation.signed_return

    total_n = len(observations)

    total_correct = sum(
        counts[group][1]
        for group in range(4)
    )

    total_signed = sum(signed_totals)

    extreme_n = (
        counts[0][0] +
        counts[1][0]
    )

    non_extreme_n = (
        counts[2][0] +
        counts[3][0]
    )

    extreme_correct = (
        counts[0][1] +
        counts[1][1]
    )

    non_extreme_correct = (
        counts[2][1] +
        counts[3][1]
    )

    result = {
        "n": total_n,
        "accuracy": (
            total_correct / total_n
            if total_n
            else float("nan")
        ),
        "mean_signed_return": (
            total_signed / total_n
            if total_n
            else float("nan")
        ),
        "groups": {},
    }

    names = (
        "extreme_bullish",
        "extreme_bearish",
        "non_extreme_bullish",
        "non_extreme_bearish",
    )

    for index, name in enumerate(names):
        n, correct = counts[index]

        result["groups"][name] = {
            "n": n,
            "accuracy": (
                correct / n
                if n
                else float("nan")
            ),
            "mean_signed_return": (
                signed_totals[index] / n
                if n
                else float("nan")
            ),
        }

    result["extreme"] = {
        "n": extreme_n,
        "accuracy": (
            extreme_correct / extreme_n
            if extreme_n
            else float("nan")
        ),
    }

    result["non_extreme"] = {
        "n": non_extreme_n,
        "accuracy": (
            non_extreme_correct / non_extreme_n
            if non_extreme_n
            else float("nan")
        ),
    }

    result["interaction_accuracy"] = (
        direction_effect_accuracy(
            [
                (
                    counts[index][0],
                    counts[index][1],
                )
                for index in range(4)
            ]
        )
    )

    result["interaction_signed_return"] = (
        direction_effect_signed_return(
            [
                (
                    counts[index][0],
                    signed_totals[index],
                )
                for index in range(4)
            ]
        )
    )

    return result


# =============================================================================
# FAST MOVING-BLOCK BOOTSTRAP
# =============================================================================

def choose_block_length(
    n: int,
    horizon: int,
) -> int:
    return min(
        MAX_BLOCK_LENGTH,
        max(
            MIN_BLOCK_LENGTH,
            horizon,
            round(math.sqrt(n)),
        ),
    )


@dataclass
class BlockStatistics:
    """
    Prefix-sum representation of the four interaction groups.

    For each group we store:
        count
        correct
        signed-return sum

    Prefix arrays make every block statistic O(1).
    """

    counts: list[list[int]]
    correct: list[list[int]]
    signed: list[list[float]]


def build_prefix_statistics(
    observations: list[CompactObservation],
) -> BlockStatistics:
    n = len(observations)

    counts = [
        [0] * (n + 1)
        for _ in range(4)
    ]

    correct = [
        [0] * (n + 1)
        for _ in range(4)
    ]

    signed = [
        [0.0] * (n + 1)
        for _ in range(4)
    ]

    for index, observation in enumerate(observations, start=1):
        previous = index - 1

        for group in range(4):
            counts[group][index] = counts[group][previous]
            correct[group][index] = correct[group][previous]
            signed[group][index] = signed[group][previous]

        group = observation.group

        counts[group][index] += 1
        correct[group][index] += observation.correct
        signed[group][index] += observation.signed_return

    return BlockStatistics(
        counts=counts,
        correct=correct,
        signed=signed,
    )


def block_statistics(
    prefix: BlockStatistics,
    start: int,
    end: int,
) -> tuple[
    tuple[int, int, float],
    tuple[int, int, float],
    tuple[int, int, float],
    tuple[int, int, float],
]:
    """
    Return statistics for observations [start:end].
    """

    result = []

    for group in range(4):
        n = (
            prefix.counts[group][end]
            - prefix.counts[group][start]
        )

        correct = (
            prefix.correct[group][end]
            - prefix.correct[group][start]
        )

        signed = (
            prefix.signed[group][end]
            - prefix.signed[group][start]
        )

        result.append(
            (
                n,
                correct,
                signed,
            )
        )

    return tuple(result)  # type: ignore[return-value]


def bootstrap_interaction_fast(
    observations: list[CompactObservation],
    horizon: int,
    replicates: int,
    seed: int,
    metric: str,
) -> BootstrapResult:
    """
    Dependence-aware moving-block bootstrap.

    Important implementation detail:

    The original slow implementation rebuilt a full list of Observation
    objects for every bootstrap replicate and then scanned the entire
    sample repeatedly.

    This implementation preserves the same moving-block bootstrap idea but
    uses prefix sums. Each sampled block contributes its aggregate counts,
    correct outcomes, and signed-return sum in O(1).

    Therefore computational complexity is approximately:

        O(replicates * number_of_blocks)

    rather than:

        O(replicates * number_of_observations)
    """

    n = len(observations)

    if n < 2:
        raise ValueError(
            "At least two observations are required "
            "for bootstrap analysis."
        )

    block_length = choose_block_length(
        n,
        horizon,
    )

    if block_length > n:
        block_length = n

    prefix = build_prefix_statistics(
        observations
    )

    max_start = n - block_length

    rng = random.Random(seed)

    bootstrap_values: list[float] = []

    for _ in range(replicates):
        sampled_n = 0

        counts = [
            0,
            0,
            0,
            0,
        ]

        correct = [
            0,
            0,
            0,
            0,
        ]

        signed = [
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        while sampled_n < n:
            start = rng.randint(
                0,
                max_start,
            )

            length = min(
                block_length,
                n - sampled_n,
            )

            end = start + length

            block = block_statistics(
                prefix,
                start,
                end,
            )

            for group in range(4):
                counts[group] += block[group][0]
                correct[group] += block[group][1]
                signed[group] += block[group][2]

            sampled_n += length

        if metric == "accuracy":
            value = direction_effect_accuracy(
                [
                    (
                        counts[group],
                        correct[group],
                    )
                    for group in range(4)
                ]
            )

        elif metric == "signed_return":
            value = direction_effect_signed_return(
                [
                    (
                        counts[group],
                        signed[group],
                    )
                    for group in range(4)
                ]
            )

        else:
            raise ValueError(
                f"Unknown bootstrap metric: {metric}"
            )

        if not math.isnan(value):
            bootstrap_values.append(value)

    if not bootstrap_values:
        raise RuntimeError(
            "Bootstrap produced no valid replicates."
        )

    bootstrap_values.sort()

    actual_counts = [
        [0, 0]
        for _ in range(4)
    ]

    actual_signed = [
        0.0
        for _ in range(4)
    ]

    for observation in observations:
        group = observation.group

        actual_counts[group][0] += 1
        actual_counts[group][1] += observation.correct
        actual_signed[group] += observation.signed_return

    if metric == "accuracy":
        statistic = direction_effect_accuracy(
            [
                (
                    actual_counts[group][0],
                    actual_counts[group][1],
                )
                for group in range(4)
            ]
        )
    else:
        statistic = direction_effect_signed_return(
            [
                (
                    actual_counts[group][0],
                    actual_signed[group],
                )
                for group in range(4)
            ]
        )

    lower_index = max(
        0,
        int(
            math.floor(
                0.025 *
                (len(bootstrap_values) - 1)
            )
        ),
    )

    upper_index = min(
        len(bootstrap_values) - 1,
        int(
            math.ceil(
                0.975 *
                (len(bootstrap_values) - 1)
            )
        ),
    )

    lower = bootstrap_values[lower_index]
    upper = bootstrap_values[upper_index]

    probability_positive = (
        sum(
            value > 0.0
            for value in bootstrap_values
        )
        / len(bootstrap_values)
    )

    probability_negative = (
        sum(
            value < 0.0
            for value in bootstrap_values
        )
        / len(bootstrap_values)
    )

    return BootstrapResult(
        statistic=statistic,
        lower=lower,
        upper=upper,
        probability_positive=probability_positive,
        probability_negative=probability_negative,
        replicates=len(bootstrap_values),
        block_length=block_length,
    )


# =============================================================================
# PERIOD FILTERING
# =============================================================================

def filter_period(
    observations: list[Observation],
    start: datetime,
    end: datetime,
) -> list[Observation]:
    return [
        observation
        for observation in observations
        if (
            start
            <= observation.signal_timestamp
            < end
        )
    ]


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_subset(
    observations: list[Observation],
    horizon: int,
    seed: int,
    bootstrap_replicates: int,
) -> dict:
    compact = compact_observations(
        observations
    )

    if not compact:
        return {
            "n": 0,
            "status": "insufficient_data",
        }

    summary = summarize_observations(
        compact
    )

    accuracy_bootstrap = (
        bootstrap_interaction_fast(
            compact,
            horizon=horizon,
            replicates=bootstrap_replicates,
            seed=seed,
            metric="accuracy",
        )
    )

    signed_bootstrap = (
        bootstrap_interaction_fast(
            compact,
            horizon=horizon,
            replicates=bootstrap_replicates,
            seed=seed + 1,
            metric="signed_return",
        )
    )

    return {
        **summary,
        "status": "ok",
        "bootstrap": {
            "accuracy": {
                "statistic": accuracy_bootstrap.statistic,
                "lower_95": accuracy_bootstrap.lower,
                "upper_95": accuracy_bootstrap.upper,
                "probability_positive": (
                    accuracy_bootstrap.probability_positive
                ),
                "probability_negative": (
                    accuracy_bootstrap.probability_negative
                ),
                "replicates": (
                    accuracy_bootstrap.replicates
                ),
                "block_length": (
                    accuracy_bootstrap.block_length
                ),
            },
            "signed_return": {
                "statistic": signed_bootstrap.statistic,
                "lower_95": signed_bootstrap.lower,
                "upper_95": signed_bootstrap.upper,
                "probability_positive": (
                    signed_bootstrap.probability_positive
                ),
                "probability_negative": (
                    signed_bootstrap.probability_negative
                ),
                "replicates": (
                    signed_bootstrap.replicates
                ),
                "block_length": (
                    signed_bootstrap.block_length
                ),
            },
        },
    }


# =============================================================================
# FORMATTING
# =============================================================================

def fmt_percent(value: float) -> str:
    if math.isnan(value):
        return "nan"

    return f"{value * 100:.4f}%"


def fmt_pp(value: float) -> str:
    if math.isnan(value):
        return "nan"

    return f"{value * 100:+.4f}pp"


def fmt_return(value: float) -> str:
    if math.isnan(value):
        return "nan"

    return f"{value * 100:+.6f}%"


def print_group(
    name: str,
    group: dict,
) -> None:
    print(
        f"    {name:<24}"
        f"n={group['n']:>8} | "
        f"accuracy={fmt_percent(group['accuracy']):>10} | "
        f"mean_signed={fmt_return(group['mean_signed_return']):>12}"
    )


def print_analysis(
    label: str,
    result: dict,
) -> None:
    print()
    print(
        f"{label}"
    )
    print(
        "-" * 110
    )

    if result["status"] != "ok":
        print(
            f"    Status: {result['status']}"
        )
        return

    print(
        f"    n={result['n']:,} | "
        f"accuracy={fmt_percent(result['accuracy'])} | "
        f"mean_signed_return="
        f"{fmt_return(result['mean_signed_return'])}"
    )

    groups = result["groups"]

    print_group(
        "Extreme bullish",
        groups["extreme_bullish"],
    )

    print_group(
        "Extreme bearish",
        groups["extreme_bearish"],
    )

    print_group(
        "Non-extreme bullish",
        groups["non_extreme_bullish"],
    )

    print_group(
        "Non-extreme bearish",
        groups["non_extreme_bearish"],
    )

    accuracy = result["bootstrap"]["accuracy"]

    signed = result["bootstrap"]["signed_return"]

    print()
    print(
        "    PRIMARY — accuracy interaction:"
    )
    print(
        f"        statistic = "
        f"{fmt_pp(accuracy['statistic'])}"
    )
    print(
        f"        95% CI   = "
        f"[{fmt_pp(accuracy['lower_95'])}, "
        f"{fmt_pp(accuracy['upper_95'])}]"
    )
    print(
        f"        P(>0)    = "
        f"{accuracy['probability_positive']:.4f}"
    )
    print(
        f"        P(<0)    = "
        f"{accuracy['probability_negative']:.4f}"
    )
    print(
        f"        blocks   = "
        f"{accuracy['block_length']}"
    )

    print()
    print(
        "    SECONDARY — signed-return interaction:"
    )
    print(
        f"        statistic = "
        f"{fmt_return(signed['statistic'])}"
    )
    print(
        f"        95% CI   = "
        f"[{fmt_return(signed['lower_95'])}, "
        f"{fmt_return(signed['upper_95'])}]"
    )
    print(
        f"        P(>0)    = "
        f"{signed['probability_positive']:.4f}"
    )
    print(
        f"        P(<0)    = "
        f"{signed['probability_negative']:.4f}"
    )
    print(
        f"        blocks   = "
        f"{signed['block_length']}"
    )


# =============================================================================
# CSV
# =============================================================================

def flatten_result(
    horizon: int,
    scope: str,
    period: str,
    result: dict,
) -> dict:
    row = {
        "timeframe": "M15",
        "horizon": horizon,
        "scope": scope,
        "period": period,
        "n": result.get("n", 0),
        "accuracy": result.get("accuracy"),
        "mean_signed_return": result.get(
            "mean_signed_return"
        ),
        "interaction_accuracy": None,
        "interaction_accuracy_ci_lower": None,
        "interaction_accuracy_ci_upper": None,
        "interaction_accuracy_probability_positive": None,
        "interaction_accuracy_probability_negative": None,
        "interaction_signed_return": None,
        "interaction_signed_return_ci_lower": None,
        "interaction_signed_return_ci_upper": None,
        "interaction_signed_return_probability_positive": None,
        "interaction_signed_return_probability_negative": None,
    }

    if result.get("status") != "ok":
        return row

    accuracy = result["bootstrap"]["accuracy"]
    signed = result["bootstrap"]["signed_return"]

    row.update(
        {
            "interaction_accuracy": (
                accuracy["statistic"]
            ),
            "interaction_accuracy_ci_lower": (
                accuracy["lower_95"]
            ),
            "interaction_accuracy_ci_upper": (
                accuracy["upper_95"]
            ),
            "interaction_accuracy_probability_positive": (
                accuracy["probability_positive"]
            ),
            "interaction_accuracy_probability_negative": (
                accuracy["probability_negative"]
            ),
            "interaction_signed_return": (
                signed["statistic"]
            ),
            "interaction_signed_return_ci_lower": (
                signed["lower_95"]
            ),
            "interaction_signed_return_ci_upper": (
                signed["upper_95"]
            ),
            "interaction_signed_return_probability_positive": (
                signed["probability_positive"]
            ),
            "interaction_signed_return_probability_negative": (
                signed["probability_negative"]
            ),
        }
    )

    return row


def write_csv(
    rows: list[dict],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# JSON SERIALIZATION
# =============================================================================

def sanitize_for_json(value):
    """
    Convert NaN/inf values to None so the output remains strict JSON.
    """

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

        return value

    if isinstance(value, dict):
        return {
            key: sanitize_for_json(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            sanitize_for_json(item)
            for item in value
        ]

    return value


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 6.3B formal M15 direction × "
            "volatility interaction test."
        )
    )

    parser.add_argument(
        "--start",
        required=True,
        help="UTC start datetime, e.g. 2010-01-01",
    )

    parser.add_argument(
        "--end",
        required=True,
        help="UTC end datetime, e.g. 2026-08-21",
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
        "--bootstrap-replicates",
        type=int,
        default=BOOTSTRAP_REPLICATES,
    )

    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=BOOTSTRAP_SEED,
    )

    return parser


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    start = parse_datetime(args.start)
    end = parse_datetime(args.end)

    if end <= start:
        parser.error(
            "--end must be after --start"
        )

    if args.bootstrap_replicates < 100:
        parser.error(
            "--bootstrap-replicates must be >= 100"
        )

    validate_regimes(REGIMES)

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 110)
    print(
        "XAUUSD — PHASE 6.3B FORMAL "
        "DIRECTION × VOLATILITY INTERACTION TEST"
    )
    print("=" * 110)
    print()
    print(
        "Primary interaction:"
    )
    print(
        "    Extreme direction effect - "
        "non-extreme direction effect"
    )
    print()
    print(
        f"Source:       {SOURCE_NAME}"
    )
    print(
        f"Window:       {start.isoformat()} -> "
        f"{end.isoformat()}"
    )
    print(
        "Timeframe:    M15"
    )
    print(
        f"Horizons:     {HORIZONS}"
    )
    print(
        f"Bootstrap:    {args.bootstrap_replicates}"
    )
    print(
        "Block rule:   "
        "max(horizon, round(sqrt(n)), 5), capped at 60"
    )
    print()

    print(
        "RECONSTRUCTION"
    )
    print(
        "-" * 110
    )

    adapter = DukascopyAdapter(
        args.data_root
    )

    engine = ObservationEngine(
        TIMEFRAME
    )

    bars = adapter.iter_bars(
        start,
        end,
        require_ask=True,
    )

    for timeframe, candle in aggregate_timeframes(
        bars,
        (TIMEFRAME,),
    ):
        engine.process(candle)

    observations = engine.observations

    print(
        f"M15: observations reconstructed="
        f"{len(observations):,}"
    )

    # -------------------------------------------------------------------------
    # Restrict observations to the requested analysis window.
    # -------------------------------------------------------------------------

    observations = [
        observation
        for observation in observations
        if (
            start
            <= observation.signal_timestamp
            < end
        )
    ]

    # Ensure chronological order once, before all horizon analysis.
    observations.sort(
        key=lambda item: item.signal_timestamp
    )

    all_results = {}
    csv_rows = []

    print()
    print(
        "ANALYSIS"
    )
    print(
        "-" * 110
    )

    for horizon_index, horizon in enumerate(
        HORIZONS
    ):
        print()
        print(
            f"▶ H{horizon}"
        )

        horizon_observations = [
            observation
            for observation in observations
            if observation.horizon_candles == horizon
        ]

        print(
            f"    observations={len(horizon_observations):,}"
        )

        if not horizon_observations:
            continue

        seed = (
            args.bootstrap_seed
            + horizon_index * 1000
        )

        full_result = analyze_subset(
            horizon_observations,
            horizon=horizon,
            seed=seed,
            bootstrap_replicates=args.bootstrap_replicates,
        )

        all_results.setdefault(
            str(horizon),
            {}
        )["full_window"] = full_result

        print_analysis(
            f"FULL WINDOW — H{horizon}",
            full_result,
        )

        csv_rows.append(
            flatten_result(
                horizon=horizon,
                scope="full_window",
                period="full",
                result=full_result,
            )
        )

        # ---------------------------------------------------------------------
        # Chronological research partitions.
        #
        # These are evaluation partitions only. They are NOT treated as
        # market-state regimes and are NOT candidate-defining features.
        # ---------------------------------------------------------------------

        all_results[str(horizon)]["periods"] = {}

        for period in PERIODS:
            name = period_name(period)
            period_start, period_end = period_bounds(
                period
            )

            scoped_start = max(
                start,
                period_start,
            )

            scoped_end = min(
                end,
                period_end,
            )

            if scoped_end <= scoped_start:
                continue

            period_observations = filter_period(
                horizon_observations,
                scoped_start,
                scoped_end,
            )

            if not period_observations:
                continue

            period_seed = (
                seed
                + abs(hash(name)) % 100000
            )

            period_result = analyze_subset(
                period_observations,
                horizon=horizon,
                seed=period_seed,
                bootstrap_replicates=args.bootstrap_replicates,
            )

            all_results[str(horizon)]["periods"][
                name
            ] = period_result

            print_analysis(
                f"PERIOD — {name} — H{horizon}",
                period_result,
            )

            csv_rows.append(
                flatten_result(
                    horizon=horizon,
                    scope="period",
                    period=name,
                    result=period_result,
                )
            )

    # -------------------------------------------------------------------------
    # Save results.
    # -------------------------------------------------------------------------

    metadata = {
        "research": {
            "name": (
                "Phase 6.3B Formal Direction × "
                "Volatility Interaction Test"
            ),
            "hypothesis_null": (
                "The relationship between current M15 "
                "direction and future directional outcome "
                "does not depend on volatility bucket."
            ),
            "hypothesis_alternative": (
                "The relationship between current M15 "
                "direction and future directional outcome "
                "changes with volatility bucket."
            ),
            "primary_interaction": (
                "Extreme direction effect - "
                "non-extreme direction effect"
            ),
            "primary_outcome": (
                "Directional correctness: signed_return > 0"
            ),
            "secondary_outcome": (
                "Direction-normalized signed return"
            ),
            "focal_bucket": "extreme",
            "reference_bucket": "non-extreme",
            "timeframe": "M15",
            "horizons": list(HORIZONS),
            "source": SOURCE_NAME,
            "timezone": TIMEZONE_NAME,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "block_rule": (
                "max(horizon, round(sqrt(n)), 5), capped at 60"
            ),
            "selection_decision": False,
            "strategy_modification": False,
        },
        "reconstruction": {
            "observations": len(observations),
        },
        "results": all_results,
    }

    json_path = (
        args.output_root
        / "direction_interaction.json"
    )

    csv_path = (
        args.output_root
        / "direction_interaction.csv"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            sanitize_for_json(metadata),
            handle,
            indent=2,
            sort_keys=True,
        )

    write_csv(
        [
            sanitize_for_json(row)
            for row in csv_rows
        ],
        csv_path,
    )

    # -------------------------------------------------------------------------
    # Final compact summary.
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print(
        "SUMMARY"
    )
    print("=" * 110)

    for horizon in HORIZONS:
        result = all_results.get(
            str(horizon),
            {},
        ).get("full_window")

        if not result or result.get("status") != "ok":
            continue

        accuracy = result["bootstrap"]["accuracy"]
        signed = result["bootstrap"]["signed_return"]

        print()
        print(
            f"H{horizon}"
        )
        print(
            f"    n={result['n']:,}"
        )
        print(
            f"    Accuracy interaction: "
            f"{fmt_pp(accuracy['statistic'])}"
        )
        print(
            f"    95% CI: "
            f"[{fmt_pp(accuracy['lower_95'])}, "
            f"{fmt_pp(accuracy['upper_95'])}]"
        )
        print(
            f"    P(interaction > 0): "
            f"{accuracy['probability_positive']:.4f}"
        )
        print(
            f"    Signed-return interaction: "
            f"{fmt_return(signed['statistic'])}"
        )
        print(
            f"    Signed-return 95% CI: "
            f"[{fmt_return(signed['lower_95'])}, "
            f"{fmt_return(signed['upper_95'])}]"
        )

    print()
    print(
        "OUTPUT"
    )
    print(
        "-" * 110
    )
    print(
        f"JSON: {json_path}"
    )
    print(
        f"CSV:  {csv_path}"
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "    This is a research diagnostic only."
    )
    print(
        "    It does not select candidates, change thresholds,"
    )
    print(
        "    invert signals, or modify strategy logic."
    )
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
