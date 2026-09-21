#!/usr/bin/env python3
"""
Phase 6.3D — Temporal Robustness of Direction × Extreme-Volatility Interaction

Purpose
-------
Test whether the fixed Phase 6.3C persistence/reversal decomposition of the
Phase 6.3B interaction survives chronological out-of-sample evaluation.

CRITICAL STATISTICAL CONTRACT
-----------------------------
The primary and secondary statistics MUST be identical to Phase 6.3B:

    accuracy interaction =
        (extreme bullish accuracy - extreme bearish accuracy)
        - (non-extreme bullish accuracy - non-extreme bearish accuracy)

    signed-return interaction =
        (extreme bullish mean return - extreme bearish mean return)
        - (non-extreme bullish mean return - non-extreme bearish mean return)

This file deliberately does NOT use:

    extreme accuracy - non-extreme accuracy

because that is NOT the Phase 6.3B interaction statistic.

Persistence classification
--------------------------
For each signal candle, compare its direction with the previous non-flat
signal candle direction:

    persistent  = same direction as previous non-flat candle
    reversal    = different direction from previous non-flat candle
    unclassified = no previous non-flat candle exists

Flat candles do not create or terminate directional comparisons, matching the
canonical directional persistence definition.

Reconciliation contract
-----------------------
For every chronological period and horizon:

    pool(persistent + reversal) == original Phase 6.3B subset

and therefore recomputing the four-group interaction after pooling MUST match
the canonical Phase 6.3B statistic to floating-point tolerance.

No strategy selection, threshold optimization, inversion, or strategy change
is performed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from xau_lean.data.dukascopy import DukascopyAdapter
from xau_lean.research.timeframe import Timeframe, aggregate_timeframes

from scripts.analyze_direction_interaction import (
    BootstrapResult,
    bootstrap_interaction_fast,
    direction_effect_accuracy,
    direction_effect_signed_return,
    direction_group,
)
from scripts.analyze_direction_walkforward import (
    DEFAULT_DATA_ROOT,
    Observation,
    ObservationEngine,
    WALKFORWARD_WINDOWS,
)
from scripts.run_direction_research import HORIZONS, candle_direction


SOURCE_NAME = "Dukascopy XAUUSD BID/ASK M1"
TIMEZONE_NAME = "UTC"
TIMEFRAME = Timeframe.M15

BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 63063

DEFAULT_OUTPUT_ROOT = Path(
    "research_results/direction_temporal_oos"
)

# Tight enough to detect a real implementation mismatch while allowing normal
# IEEE-754 summation differences.
RECONCILIATION_TOLERANCE = 1e-12

PERSISTENCE_STATES = ("persistent", "reversal")


@dataclass(frozen=True)
class DecomposedObservation:
    observation: Observation
    persistence: str


@dataclass(frozen=True)
class CompactObservation:
    """Minimal representation consumed by the canonical 6.3B bootstrap."""

    group: int
    correct: int
    signed_return: float


def parse_datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def period_bounds(period: Any) -> tuple[datetime, datetime]:
    """Normalize supported canonical PERIOD/REGIME representations to UTC."""

    def normalize(value: Any) -> datetime:
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
        return normalize(period.start), normalize(period.end)

    if isinstance(period, dict):
        return normalize(period["start"]), normalize(period["end"])

    if isinstance(period, (tuple, list)):
        if len(period) >= 3:
            return normalize(period[1]), normalize(period[2])
        if len(period) == 2:
            return normalize(period[0]), normalize(period[1])

    raise TypeError(f"Unsupported period entry: {period!r}")


def period_name(period: Any) -> str:
    if hasattr(period, "name"):
        return str(period.name)
    if isinstance(period, dict):
        return str(period.get("name", period.get("label", "unknown")))
    if isinstance(period, (tuple, list)) and len(period) >= 3:
        return str(period[0])
    return "unknown"


def stable_seed(text: str) -> int:
    """Deterministic replacement for Python's process-randomized hash()."""
    value = 0
    for char in text.encode("utf-8"):
        value = (value * 131 + char) % 1_000_000
    return value


def filter_period(
    observations: Iterable[Observation],
    start: datetime,
    end: datetime,
) -> list[Observation]:
    return [
        observation
        for observation in observations
        if start <= observation.signal_timestamp < end
    ]


def build_persistence_map(
    candles: Iterable[Any],
) -> dict[datetime, str]:
    """
    Classify signal candles using only information available at signal close.

    Flat candles are ignored. The first non-flat direction is unclassified.
    """

    previous_direction: str | None = None
    result: dict[datetime, str] = {}

    for candle in candles:
        if not getattr(candle, "complete", True):
            continue

        direction = candle_direction(candle)
        if direction == "flat":
            continue

        timestamp = candle.timestamp

        if previous_direction is None:
            result[timestamp] = "unclassified"
        elif direction == previous_direction:
            result[timestamp] = "persistent"
        else:
            result[timestamp] = "reversal"

        previous_direction = direction

    return result


def compact_decomposed(
    observations: Iterable[DecomposedObservation],
) -> list[CompactObservation]:
    """Convert decomposed observations to the exact 6.3B compact statistic."""

    ordered = sorted(
        observations,
        key=lambda item: item.observation.signal_timestamp,
    )

    return [
        CompactObservation(
            group=direction_group(item.observation),
            correct=1 if item.observation.signed_return > 0.0 else 0,
            signed_return=float(item.observation.signed_return),
        )
        for item in ordered
    ]


def interaction_from_observations(
    observations: Iterable[Observation],
) -> tuple[float, float]:
    """Compute the exact Phase 6.3B accuracy and signed-return interactions."""

    counts = [[0, 0] for _ in range(4)]
    signed_totals = [0.0 for _ in range(4)]

    for observation in observations:
        group = direction_group(observation)
        counts[group][0] += 1
        counts[group][1] += observation.signed_return > 0.0
        signed_totals[group] += observation.signed_return

    accuracy = direction_effect_accuracy(
        [(n, correct) for n, correct in counts]
    )
    signed_return = direction_effect_signed_return(
        [(counts[i][0], signed_totals[i]) for i in range(4)]
    )
    return accuracy, signed_return


def summarize_groups(
    observations: Iterable[Observation],
) -> dict[str, Any]:
    """Return the four canonical 6.3B groups for diagnostics/reconciliation."""

    counts = [[0, 0] for _ in range(4)]
    signed_totals = [0.0 for _ in range(4)]
    names = (
        "extreme_bullish",
        "extreme_bearish",
        "non_extreme_bullish",
        "non_extreme_bearish",
    )

    total = 0
    for observation in observations:
        group = direction_group(observation)
        counts[group][0] += 1
        counts[group][1] += observation.signed_return > 0.0
        signed_totals[group] += observation.signed_return
        total += 1

    groups: dict[str, Any] = {}
    for index, name in enumerate(names):
        n, correct = counts[index]
        groups[name] = {
            "n": n,
            "accuracy": correct / n if n else None,
            "mean_signed_return": signed_totals[index] / n if n else None,
        }

    return {
        "n": total,
        "groups": groups,
        "interaction_accuracy": direction_effect_accuracy(
            [(n, correct) for n, correct in counts]
        ),
        "interaction_signed_return": direction_effect_signed_return(
            [(counts[i][0], signed_totals[i]) for i in range(4)]
        ),
    }


def bootstrap_result_dict(result: BootstrapResult) -> dict[str, Any]:
    return {
        "statistic": result.statistic,
        "lower_95": result.lower,
        "upper_95": result.upper,
        "probability_positive": result.probability_positive,
        "probability_negative": result.probability_negative,
        "replicates": result.replicates,
        "block_length": result.block_length,
    }


def analyze_state(
    observations: list[Observation],
    horizon: int,
    seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    """Analyze one persistence/reversal subset with canonical 6.3B statistics."""

    compact = compact_decomposed(
        DecomposedObservation(observation, persistence="subset")
        for observation in observations
    )

    summary = summarize_groups(observations)

    if len(compact) < 2:
        return {
            "status": "insufficient_sample",
            **summary,
        }

    accuracy = bootstrap_interaction_fast(
        compact,
        horizon=horizon,
        replicates=bootstrap_replicates,
        seed=seed,
        metric="accuracy",
    )
    signed = bootstrap_interaction_fast(
        compact,
        horizon=horizon,
        replicates=bootstrap_replicates,
        seed=seed + 1,
        metric="signed_return",
    )

    return {
        "status": "ok",
        **summary,
        "bootstrap": {
            "accuracy": bootstrap_result_dict(accuracy),
            "signed_return": bootstrap_result_dict(signed),
        },
    }


def reconcile(
    all_observations: list[Observation],
    persistent_observations: list[Observation],
    reversal_observations: list[Observation],
) -> dict[str, Any]:
    """
    Prove that persistent + reversal exactly reconstructs the original sample.

    Because the interaction is a nonlinear function of group-specific
    denominators, reconciliation is performed by pooling raw observations and
    recomputing the four-group statistic—not by averaging subgroup effects.
    """

    pooled = sorted(
        [*persistent_observations, *reversal_observations],
        key=lambda item: item.signal_timestamp,
    )
    original = sorted(
        all_observations,
        key=lambda item: item.signal_timestamp,
    )

    original_accuracy, original_signed = interaction_from_observations(original)
    pooled_accuracy, pooled_signed = interaction_from_observations(pooled)

    timestamps_match = [
        item.signal_timestamp for item in original
    ] == [
        item.signal_timestamp for item in pooled
    ]

    accuracy_diff = pooled_accuracy - original_accuracy
    signed_diff = pooled_signed - original_signed

    passed = (
        len(original) == len(pooled)
        and timestamps_match
        and abs(accuracy_diff) <= RECONCILIATION_TOLERANCE
        and abs(signed_diff) <= RECONCILIATION_TOLERANCE
    )

    return {
        "passed": passed,
        "original_n": len(original),
        "persistent_n": len(persistent_observations),
        "reversal_n": len(reversal_observations),
        "pooled_n": len(pooled),
        "timestamps_match": timestamps_match,
        "original_interaction_accuracy": original_accuracy,
        "pooled_interaction_accuracy": pooled_accuracy,
        "accuracy_difference": accuracy_diff,
        "original_interaction_signed_return": original_signed,
        "pooled_interaction_signed_return": pooled_signed,
        "signed_return_difference": signed_diff,
        "tolerance": RECONCILIATION_TOLERANCE,
    }


def fmt_pp(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value * 100:+.4f}pp"


def fmt_return(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value * 100:+.6f}%"


def print_state(
    label: str,
    result: dict[str, Any],
) -> None:
    print(f"    {label}")
    print(f"        n={result['n']:,}")
    print(
        "        interaction accuracy: "
        f"{fmt_pp(result['interaction_accuracy'])}"
    )
    print(
        "        interaction signed return: "
        f"{fmt_return(result['interaction_signed_return'])}"
    )
    if result.get("status") == "ok":
        accuracy = result["bootstrap"]["accuracy"]
        signed = result["bootstrap"]["signed_return"]
        print(
            f"        accuracy CI: [{fmt_pp(accuracy['lower_95'])}, "
            f"{fmt_pp(accuracy['upper_95'])}] "
            f"P(>0)={accuracy['probability_positive']:.4f}"
        )
        print(
            f"        signed CI:   [{fmt_return(signed['lower_95'])}, "
            f"{fmt_return(signed['upper_95'])}] "
            f"P(>0)={signed['probability_positive']:.4f}"
        )


def flatten_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        result = row["result"]
        bootstrap = result.get("bootstrap", {})
        accuracy = bootstrap.get("accuracy", {})
        signed = bootstrap.get("signed_return", {})
        flat.append(
            {
                "timeframe": "M15",
                "horizon": row["horizon"],
                "period": row["period"],
                "persistence": row["persistence"],
                "n": result.get("n", 0),
                "interaction_accuracy": result.get("interaction_accuracy"),
                "interaction_accuracy_ci_lower": accuracy.get("lower_95"),
                "interaction_accuracy_ci_upper": accuracy.get("upper_95"),
                "interaction_accuracy_probability_positive": accuracy.get("probability_positive"),
                "interaction_accuracy_probability_negative": accuracy.get("probability_negative"),
                "interaction_signed_return": result.get("interaction_signed_return"),
                "interaction_signed_return_ci_lower": signed.get("lower_95"),
                "interaction_signed_return_ci_upper": signed.get("upper_95"),
                "interaction_signed_return_probability_positive": signed.get("probability_positive"),
                "interaction_signed_return_probability_negative": signed.get("probability_negative"),
                "bootstrap_replicates": accuracy.get("replicates"),
                "block_length": accuracy.get("block_length"),
            }
        )
    return flat


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    return value


def stable_window_seed(
    base_seed: int,
    window_id: str,
    split: str,
    horizon: int,
    state: str,
    metric_offset: int = 0,
) -> int:
    """Deterministic seed for a window/split/horizon/state/metric cell."""
    return (
        base_seed
        + stable_seed(window_id) * 1000
        + stable_seed(split) * 100
        + horizon * 10
        + stable_seed(state)
        + metric_offset
    )


def window_bounds(window: dict[str, str], split: str) -> tuple[datetime, datetime]:
    if split == "train":
        return parse_datetime(window["train_start"]), parse_datetime(window["train_end"])
    if split == "oos":
        return parse_datetime(window["oos_start"]), parse_datetime(window["oos_end"])
    raise ValueError(f"Unknown split: {split}")


def analyze_split(
    items: list[DecomposedObservation],
    horizon: int,
    seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    """Analyze one fixed temporal split, separately by persistence state."""
    observations = [item.observation for item in items]
    original_accuracy, original_signed = interaction_from_observations(observations)

    result: dict[str, Any] = {
        "n": len(observations),
        "original_interaction_accuracy": original_accuracy,
        "original_interaction_signed_return": original_signed,
        "states": {},
    }

    for state in PERSISTENCE_STATES:
        state_observations = [
            item.observation
            for item in items
            if item.persistence == state
        ]
        result["states"][state] = analyze_state(
            state_observations,
            horizon=horizon,
            seed=seed + stable_seed(state),
            bootstrap_replicates=bootstrap_replicates,
        )

    return result


def attach_persistence(
    observations: list[Observation],
    persistence_map: dict[datetime, str],
) -> tuple[list[DecomposedObservation], int]:
    decomposed: list[DecomposedObservation] = []
    missing = 0
    for observation in observations:
        state = persistence_map.get(observation.signal_timestamp)
        if state is None:
            missing += 1
            continue
        decomposed.append(DecomposedObservation(observation, state))
    return decomposed, missing


def observations_for_split(
    decomposed: list[DecomposedObservation],
    start: datetime,
    end: datetime,
    horizon: int,
) -> list[DecomposedObservation]:
    return [
        item
        for item in decomposed
        if (
            start <= item.observation.signal_timestamp < end
            and item.observation.horizon_candles == horizon
        )
    ]


def print_split(
    window_id: str,
    split: str,
    horizon: int,
    result: dict[str, Any],
) -> None:
    print()
    print(f"{window_id} — {split.upper()} — H{horizon}")
    print("-" * 110)
    print(
        f"    all observations: n={result['n']:,} | "
        f"canonical interaction accuracy={fmt_pp(result['original_interaction_accuracy'])} | "
        f"signed={fmt_return(result['original_interaction_signed_return'])}"
    )
    for state in PERSISTENCE_STATES:
        print_state(f"    {state}", result["states"][state])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 6.3D temporal OOS validation of the fixed 6.3C interaction."
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=BOOTSTRAP_REPLICATES,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    start = parse_datetime(args.start)
    end = parse_datetime(args.end)
    if end <= start:
        parser.error("--end must be after --start")
    if args.bootstrap_replicates < 100:
        parser.error("--bootstrap-replicates must be >= 100")

    print()
    print("=" * 110)
    print("XAUUSD — PHASE 6.3D TEMPORAL OOS VALIDATION")
    print("=" * 110)
    print()
    print("Research question:")
    print("Does the fixed Phase 6.3C persistence/reversal interaction survive chronological OOS testing?")
    print()
    print(f"Requested window: {start.isoformat()} -> {end.isoformat()}")
    print("Timeframe:        M15")
    print(f"Horizons:         {HORIZONS}")
    print(f"Bootstrap:        {args.bootstrap_replicates}")
    print("Statistic:        EXACT Phase 6.3B direction × volatility interaction")
    print("Selection:        NONE")
    print("Optimization:     NONE")
    print("Strategy change:  NONE")
    print()

    adapter = DukascopyAdapter(args.data_root)
    engine = ObservationEngine(TIMEFRAME)

    bars = adapter.iter_bars(start, end, require_ask=True)
    candles = []
    for timeframe, candle in aggregate_timeframes(bars, (TIMEFRAME,)):
        candles.append(candle)
        engine.process(candle)

    persistence_map = build_persistence_map(candles)
    scoped_engine_observations = sorted(
        [
            observation
            for observation in engine.observations
            if start <= observation.signal_timestamp < end
        ],
        key=lambda item: item.signal_timestamp,
    )
    observations, missing = attach_persistence(
        scoped_engine_observations,
        persistence_map,
    )

    state_counts = {
        state: sum(item.persistence == state for item in observations)
        for state in (*PERSISTENCE_STATES, "unclassified")
    }

    print("RECONSTRUCTION")
    print("-" * 110)
    print(f"M15 candles:             {len(candles):,}")
    print(f"Canonical observations:  {len(engine.observations):,}")
    print(f"Scoped observations:     {len(observations):,}")
    print(f"Missing persistence:     {missing:,}")
    print()
    print("PERSISTENCE DISTRIBUTION")
    print("-" * 110)
    print(f"persistent              {state_counts['persistent']:>8,}")
    print(f"reversal                {state_counts['reversal']:>8,}")
    print(f"unclassified            {state_counts['unclassified']:>8,}")

    if missing:
        print()
        print("FATAL: canonical observations are missing persistence classification.")
        return 2

    if state_counts["unclassified"]:
        print()
        print("FATAL: unclassified observations reached temporal analysis.")
        print("       They must be excluded from the decomposed sample.")
        return 2

    window_results: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    reconciliation_rows: list[dict[str, Any]] = []

    for window in WALKFORWARD_WINDOWS:
        window_id = str(window["id"])
        train_start, train_end = window_bounds(window, "train")
        oos_start, oos_end = window_bounds(window, "oos")

        train_start_scoped = max(start, train_start)
        train_end_scoped = min(end, train_end)
        oos_start_scoped = max(start, oos_start)
        oos_end_scoped = min(end, oos_end)

        has_train = train_end_scoped > train_start_scoped
        has_oos = oos_end_scoped > oos_start_scoped

        if not has_train and not has_oos:
            continue

        print()
        print("=" * 110)
        print(f"{window_id} — CHRONOLOGICAL WINDOW")
        print("=" * 110)
        if has_train:
            print(f"TRAIN: {train_start_scoped.isoformat()} -> {train_end_scoped.isoformat()}")
        else:
            print("TRAIN: outside requested window")
        if has_oos:
            print(f"OOS:   {oos_start_scoped.isoformat()} -> {oos_end_scoped.isoformat()}")
        else:
            print("OOS:   outside requested window")

        window_payload: dict[str, Any] = {
            "id": window_id,
            "canonical": window,
            "requested_scope": {
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
            "splits": {},
        }

        for horizon in HORIZONS:
            for split, split_start, split_end, present in (
                ("train", train_start_scoped, train_end_scoped, has_train),
                ("oos", oos_start_scoped, oos_end_scoped, has_oos),
            ):
                if not present:
                    continue

                items = observations_for_split(
                    observations,
                    split_start,
                    split_end,
                    horizon,
                )
                split_observations = [item.observation for item in items]

                result = analyze_split(
                    items,
                    horizon=horizon,
                    seed=stable_window_seed(
                        args.bootstrap_seed,
                        window_id,
                        split,
                        horizon,
                        "base",
                    ),
                    bootstrap_replicates=args.bootstrap_replicates,
                )
                window_payload["splits"][f"{split}_H{horizon}"] = result
                print_split(window_id, split, horizon, result)

                persistent = [
                    item.observation
                    for item in items
                    if item.persistence == "persistent"
                ]
                reversal = [
                    item.observation
                    for item in items
                    if item.persistence == "reversal"
                ]

                recon = reconcile(
                    split_observations,
                    persistent,
                    reversal,
                )
                reconciliation_rows.append(
                    {
                        "window": window_id,
                        "split": split,
                        "horizon": horizon,
                        **recon,
                    }
                )

                for state in PERSISTENCE_STATES:
                    state_result = result["states"][state]
                    accuracy_bootstrap = state_result.get("bootstrap", {}).get("accuracy", {})
                    signed_bootstrap = state_result.get("bootstrap", {}).get("signed_return", {})
                    csv_rows.append(
                        {
                            "window": window_id,
                            "split": split,
                            "horizon": horizon,
                            "persistence": state,
                            "n": state_result.get("n", 0),
                            "interaction_accuracy": state_result.get("interaction_accuracy"),
                            "interaction_accuracy_ci_lower": accuracy_bootstrap.get("lower_95"),
                            "interaction_accuracy_ci_upper": accuracy_bootstrap.get("upper_95"),
                            "interaction_accuracy_probability_positive": accuracy_bootstrap.get("probability_positive"),
                            "interaction_accuracy_probability_negative": accuracy_bootstrap.get("probability_negative"),
                            "interaction_signed_return": state_result.get("interaction_signed_return"),
                            "interaction_signed_return_ci_lower": signed_bootstrap.get("lower_95"),
                            "interaction_signed_return_ci_upper": signed_bootstrap.get("upper_95"),
                            "interaction_signed_return_probability_positive": signed_bootstrap.get("probability_positive"),
                            "interaction_signed_return_probability_negative": signed_bootstrap.get("probability_negative"),
                            "bootstrap_replicates": accuracy_bootstrap.get("replicates"),
                            "block_length": accuracy_bootstrap.get("block_length"),
                        }
                    )

        window_results.append(window_payload)

    if not window_results:
        parser.error(
            "Requested window does not overlap any established WF1-WF3 temporal window."
        )

    all_reconciliations_pass = all(row["passed"] for row in reconciliation_rows)

    metadata = {
        "phase": "6.3D",
        "title": "Temporal Robustness of Direction × Extreme-Volatility Interaction",
        "research_question": (
            "Does the fixed Phase 6.3C persistence/reversal interaction "
            "survive chronological out-of-sample testing?"
        ),
        "source": SOURCE_NAME,
        "timezone": TIMEZONE_NAME,
        "timeframe": "M15",
        "horizons": list(HORIZONS),
        "requested_window": {
            "start": start.isoformat(),
            "end": end.isoformat(),
        },
        "methodology": {
            "signal_definition": "Phase 6.3A mechanics reproduced through the established ObservationEngine.",
            "persistence_definition": (
                "Current non-flat signal direction equals previous non-flat signal direction; "
                "flat candles are skipped."
            ),
            "reversal_definition": (
                "Current non-flat signal direction differs from previous non-flat signal direction."
            ),
            "unclassified_definition": "First non-flat signal direction; excluded from decomposition.",
            "primary_statistic": (
                "(extreme bullish accuracy - extreme bearish accuracy) - "
                "(non-extreme bullish accuracy - non-extreme bearish accuracy)"
            ),
            "secondary_statistic": (
                "(extreme bullish mean signed return - extreme bearish mean signed return) - "
                "(non-extreme bullish mean signed return - non-extreme bearish mean signed return)"
            ),
            "training_role": "Historical measurement only; no parameter fitting or candidate selection.",
            "oos_role": "Same fixed statistic measured on the subsequent chronological unseen window.",
            "random_split": False,
            "candidate_selection": False,
            "optimization": False,
            "strategy_modification": False,
            "bootstrap_method": "canonical Phase 6.3B moving-block bootstrap",
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "block_rule": "min(50, max(5, horizon, round(sqrt(n))))",
        },
        "walk_forward_windows": list(WALKFORWARD_WINDOWS),
        "reconstruction": {
            "m15_candles": len(candles),
            "engine_observations": len(engine.observations),
            "scoped_decomposed_observations": len(observations),
            "missing_persistence": missing,
            "persistence_distribution": state_counts,
        },
        "reconciliation": {
            "all_passed": all_reconciliations_pass,
            "tolerance": RECONCILIATION_TOLERANCE,
            "cells": reconciliation_rows,
        },
        "windows": window_results,
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    json_path = args.output_root / "direction_temporal_oos.json"
    csv_path = args.output_root / "direction_temporal_oos.csv"
    reconciliation_csv_path = args.output_root / "reconciliation.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_for_json(metadata), handle, indent=2, sort_keys=True)

    write_csv(csv_rows, csv_path)
    write_csv(reconciliation_rows, reconciliation_csv_path)

    print()
    print("=" * 110)
    print("RECONCILIATION")
    print("=" * 110)
    print(f"Cells checked: {len(reconciliation_rows)}")
    print(f"All cells pass: {all_reconciliations_pass}")
    for row in reconciliation_rows:
        status = "PASS" if row["passed"] else "FAIL"
        print(
            f"{status:<5} {row['window']:<4} {row['split']:<5} H{row['horizon']} "
            f"n={row['original_n']:,} "
            f"Δaccuracy={row['accuracy_difference']:+.3e} "
            f"Δsigned={row['signed_return_difference']:+.3e}"
        )

    print()
    print("OUTPUT")
    print("-" * 110)
    print(f"JSON:           {json_path}")
    print(f"CSV:            {csv_path}")
    print(f"Reconciliation: {reconciliation_csv_path}")

    print()
    if not all_reconciliations_pass:
        print("FATAL: reconciliation failed. Do NOT interpret temporal OOS results.")
        return 3

    print("IMPORTANT:")
    print("    Reconciliation passed against the canonical Phase 6.3B statistic.")
    print("    Training windows were measurement-only; no candidates were selected.")
    print("    OOS windows use the same fixed statistic with no strategy modification.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
