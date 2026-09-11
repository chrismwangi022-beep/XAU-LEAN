#!/usr/bin/env python3
"""
Phase 6.3D — Chronological Persistence × Extreme-Volatility Robustness

Purpose
-------
Test whether the Phase 6.3C persistence/reversal decomposition survives
chronological out-of-sample separation.

This is a diagnostic robustness test.

It does NOT:
    - select a trading strategy
    - optimize thresholds
    - invert negative relationships
    - select H1/H3/H5
    - change entries or exits
    - use OOS information for any training decision

Primary statistic
-----------------
The exact Phase 6.3B interaction:

    accuracy interaction =
        (extreme bullish accuracy - extreme bearish accuracy)
        - (non-extreme bullish accuracy - non-extreme bearish accuracy)

    signed-return interaction =
        (extreme bullish mean return - extreme bearish mean return)
        - (non-extreme bullish mean return - non-extreme bearish mean return)

Persistence definition
----------------------
For each signal candle:

    persistent = same direction as previous non-flat signal candle
    reversal   = different direction from previous non-flat signal candle

Flat candles are ignored.

Chronological windows
---------------------
WF1:
    TRAIN 2010-01-01 -> 2019-01-01
    OOS   2019-01-01 -> 2022-01-01

WF2:
    TRAIN 2010-01-01 -> 2022-01-01
    OOS   2022-01-01 -> 2024-01-01

WF3:
    TRAIN 2010-01-01 -> 2024-01-01
    OOS   2024-01-01 -> 2026-08-22

Important
---------
Persistence for the first OOS signal is allowed to depend on the immediately
preceding TRAIN-period signal. This is causal information available at the
time of the OOS signal and therefore is not future leakage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from xau_lean.data.dukascopy import DukascopyAdapter
from xau_lean.research.regimes import REGIMES, validate_regimes
from xau_lean.research.timeframe import Timeframe, aggregate_timeframes

from scripts.analyze_direction_regime_decomposition import (
    DecomposedObservation,
    RECONCILIATION_TOLERANCE,
    analyze_state,
    build_persistence_map,
    period_bounds,
    period_name,
    reconcile,
)
from scripts.analyze_direction_walkforward import (
    DEFAULT_DATA_ROOT,
    ObservationEngine,
)
from scripts.run_direction_research import HORIZONS


SOURCE_NAME = "Dukascopy XAUUSD BID/ASK M1"
TIMEZONE_NAME = "UTC"
TIMEFRAME = Timeframe.M15

BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 63063

DEFAULT_OUTPUT_ROOT = Path(
    "research_results/direction_persistence_walkforward"
)

PERSISTENCE_STATES = ("persistent", "reversal")


WALKFORWARD_WINDOWS = (
    {
        "id": "WF1",
        "train_start": "2010-01-01T00:00:00Z",
        "train_end": "2019-01-01T00:00:00Z",
        "oos_start": "2019-01-01T00:00:00Z",
        "oos_end": "2022-01-01T00:00:00Z",
    },
    {
        "id": "WF2",
        "train_start": "2010-01-01T00:00:00Z",
        "train_end": "2022-01-01T00:00:00Z",
        "oos_start": "2022-01-01T00:00:00Z",
        "oos_end": "2024-01-01T00:00:00Z",
    },
    {
        "id": "WF3",
        "train_start": "2010-01-01T00:00:00Z",
        "train_end": "2024-01-01T00:00:00Z",
        "oos_start": "2024-01-01T00:00:00Z",
        "oos_end": "2026-08-22T00:00:00Z",
    },
)


def parse_datetime(value: str):
    from datetime import datetime, timezone

    text = value.strip()

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    parsed = datetime.fromisoformat(text)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def stable_seed(text: str) -> int:
    """
    Deterministic seed.

    Do not use Python's hash(), because hash randomization makes results
    process-dependent.
    """
    value = 0

    for char in text.encode("utf-8"):
        value = (value * 131 + char) % 1_000_000

    return value


def scoped_items(
    decomposed: list[DecomposedObservation],
    start,
    end,
) -> list[DecomposedObservation]:
    return [
        item
        for item in decomposed
        if start <= item.observation.signal_timestamp < end
    ]


def analyze_scope(
    items: list[DecomposedObservation],
    horizon: int,
    seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    horizon_items = [
        item
        for item in items
        if item.observation.horizon_candles == horizon
    ]

    all_observations = [
        item.observation
        for item in horizon_items
    ]

    state_results: dict[str, Any] = {}

    for state_index, state in enumerate(PERSISTENCE_STATES):
        observations = [
            item.observation
            for item in horizon_items
            if item.persistence == state
        ]

        state_results[state] = analyze_state(
            observations,
            horizon=horizon,
            seed=seed + state_index * 100,
            bootstrap_replicates=bootstrap_replicates,
        )

    persistent = [
        item.observation
        for item in horizon_items
        if item.persistence == "persistent"
    ]

    reversal = [
        item.observation
        for item in horizon_items
        if item.persistence == "reversal"
    ]

    reconciliation = reconcile(
        all_observations,
        persistent,
        reversal,
    )

    return {
        "n": len(all_observations),
        "persistent_n": len(persistent),
        "reversal_n": len(reversal),
        "states": state_results,
        "reconciliation": reconciliation,
    }


def interaction_values(
    result: dict[str, Any],
) -> tuple[float | None, float | None]:
    persistent = result["states"]["persistent"]
    reversal = result["states"]["reversal"]

    return (
        persistent.get("interaction_accuracy"),
        reversal.get("interaction_accuracy"),
    )


def sign(value: float | None) -> int | None:
    if value is None:
        return None

    if math.isnan(value):
        return None

    if value > 0:
        return 1

    if value < 0:
        return -1

    return 0


def compare_train_oos(
    train: dict[str, Any],
    oos: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for state in PERSISTENCE_STATES:
        train_result = train["states"][state]
        oos_result = oos["states"][state]

        train_accuracy = train_result.get(
            "interaction_accuracy"
        )
        oos_accuracy = oos_result.get(
            "interaction_accuracy"
        )

        train_signed = train_result.get(
            "interaction_signed_return"
        )
        oos_signed = oos_result.get(
            "interaction_signed_return"
        )

        result[state] = {
            "train_interaction_accuracy": train_accuracy,
            "oos_interaction_accuracy": oos_accuracy,
            "train_oos_accuracy_difference": (
                oos_accuracy - train_accuracy
                if train_accuracy is not None
                and oos_accuracy is not None
                else None
            ),
            "train_interaction_signed_return": train_signed,
            "oos_interaction_signed_return": oos_signed,
            "train_oos_signed_return_difference": (
                oos_signed - train_signed
                if train_signed is not None
                and oos_signed is not None
                else None
            ),
            "accuracy_sign_agreement": (
                sign(train_accuracy) == sign(oos_accuracy)
                if train_accuracy is not None
                and oos_accuracy is not None
                else None
            ),
            "signed_return_sign_agreement": (
                sign(train_signed) == sign(oos_signed)
                if train_signed is not None
                and oos_signed is not None
                else None
            ),
        }

    return result


def flatten_rows(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []

    for window_result in results:
        window = window_result["window"]

        for horizon in HORIZONS:
            train = window_result["train"][str(horizon)]
            oos = window_result["oos"][str(horizon)]

            for state in PERSISTENCE_STATES:
                train_state = train["states"][state]
                oos_state = oos["states"][state]
                comparison = window_result[
                    "comparison"
                ][str(horizon)][state]

                train_bootstrap = train_state.get(
                    "bootstrap", {}
                )
                oos_bootstrap = oos_state.get(
                    "bootstrap", {}
                )

                train_accuracy_bootstrap = (
                    train_bootstrap.get("accuracy", {})
                )
                oos_accuracy_bootstrap = (
                    oos_bootstrap.get("accuracy", {})
                )

                train_signed_bootstrap = (
                    train_bootstrap.get("signed_return", {})
                )
                oos_signed_bootstrap = (
                    oos_bootstrap.get("signed_return", {})
                )

                rows.append(
                    {
                        "walkforward_id": window["id"],
                        "scope": "train",
                        "horizon": horizon,
                        "persistence": state,
                        "n": train_state.get("n", 0),
                        "interaction_accuracy": train_state.get(
                            "interaction_accuracy"
                        ),
                        "interaction_signed_return": train_state.get(
                            "interaction_signed_return"
                        ),
                        "accuracy_ci_lower": train_accuracy_bootstrap.get(
                            "lower_95"
                        ),
                        "accuracy_ci_upper": train_accuracy_bootstrap.get(
                            "upper_95"
                        ),
                        "accuracy_probability_positive": (
                            train_accuracy_bootstrap.get(
                                "probability_positive"
                            )
                        ),
                        "signed_return_ci_lower": (
                            train_signed_bootstrap.get(
                                "lower_95"
                            )
                        ),
                        "signed_return_ci_upper": (
                            train_signed_bootstrap.get(
                                "upper_95"
                            )
                        ),
                        "signed_return_probability_positive": (
                            train_signed_bootstrap.get(
                                "probability_positive"
                            )
                        ),
                    }
                )

                rows.append(
                    {
                        "walkforward_id": window["id"],
                        "scope": "oos",
                        "horizon": horizon,
                        "persistence": state,
                        "n": oos_state.get("n", 0),
                        "interaction_accuracy": oos_state.get(
                            "interaction_accuracy"
                        ),
                        "interaction_signed_return": oos_state.get(
                            "interaction_signed_return"
                        ),
                        "accuracy_ci_lower": oos_accuracy_bootstrap.get(
                            "lower_95"
                        ),
                        "accuracy_ci_upper": oos_accuracy_bootstrap.get(
                            "upper_95"
                        ),
                        "accuracy_probability_positive": (
                            oos_accuracy_bootstrap.get(
                                "probability_positive"
                            )
                        ),
                        "signed_return_ci_lower": (
                            oos_signed_bootstrap.get(
                                "lower_95"
                            )
                        ),
                        "signed_return_ci_upper": (
                            oos_signed_bootstrap.get(
                                "upper_95"
                            )
                        ),
                        "signed_return_probability_positive": (
                            oos_signed_bootstrap.get(
                                "probability_positive"
                            )
                        ),
                        "train_oos_accuracy_difference": comparison[
                            "train_oos_accuracy_difference"
                        ],
                        "train_oos_signed_return_difference": comparison[
                            "train_oos_signed_return_difference"
                        ],
                        "accuracy_sign_agreement": comparison[
                            "accuracy_sign_agreement"
                        ],
                        "signed_return_sign_agreement": comparison[
                            "signed_return_sign_agreement"
                        ],
                    }
                )

    return rows


def sanitize_for_json(value: Any) -> Any:
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


def write_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = []
        seen = set()

        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def fmt_pp(value: float | None) -> str:
    if value is None:
        return "NA"

    if isinstance(value, float) and math.isnan(value):
        return "NA"

    return f"{value * 100:+.4f}pp"


def fmt_return(value: float | None) -> str:
    if value is None:
        return "NA"

    if isinstance(value, float) and math.isnan(value):
        return "NA"

    return f"{value * 100:+.6f}%"


def print_state(
    label: str,
    result: dict[str, Any],
) -> None:
    print(
        f"    {label:<18} "
        f"n={result['n']:,} "
        f"accuracy={fmt_pp(result.get('interaction_accuracy'))} "
        f"signed={fmt_return(result.get('interaction_signed_return'))}"
    )

    if result.get("status") == "ok":
        accuracy = result["bootstrap"]["accuracy"]
        signed = result["bootstrap"]["signed_return"]

        print(
            f"        accuracy CI "
            f"[{fmt_pp(accuracy['lower_95'])}, "
            f"{fmt_pp(accuracy['upper_95'])}] "
            f"P(>0)={accuracy['probability_positive']:.4f}"
        )

        print(
            f"        signed CI   "
            f"[{fmt_return(signed['lower_95'])}, "
            f"{fmt_return(signed['upper_95'])}] "
            f"P(>0)={signed['probability_positive']:.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 6.3D chronological persistence × "
            "extreme-volatility robustness."
        )
    )

    parser.add_argument(
        "--start",
        default="2010-01-01T00:00:00Z",
    )

    parser.add_argument(
        "--end",
        default="2026-08-21T00:00:00Z",
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


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    start = parse_datetime(args.start)
    end = parse_datetime(args.end)

    if end <= start:
        parser.error("--end must be after --start")

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
        "XAUUSD — PHASE 6.3D CHRONOLOGICAL "
        "PERSISTENCE × EXTREME-VOLATILITY ROBUSTNESS"
    )
    print("=" * 110)
    print()
    print("Research question:")
    print(
        "Does the Phase 6.3C persistence/reversal "
        "effect survive chronological OOS separation?"
    )
    print()
    print(f"Source:       {SOURCE_NAME}")
    print(f"Window:       {start.isoformat()} -> {end.isoformat()}")
    print("Timeframe:    M15")
    print(f"Horizons:     {HORIZONS}")
    print(f"Bootstrap:    {args.bootstrap_replicates}")
    print(
        "Statistic:    EXACT Phase 6.3B direction × "
        "volatility interaction"
    )
    print("Strategy modification: FALSE")
    print()

    adapter = DukascopyAdapter(args.data_root)
    engine = ObservationEngine(TIMEFRAME)

    bars = adapter.iter_bars(
        start,
        end,
        require_ask=True,
    )

    candles = []

    for timeframe, candle in aggregate_timeframes(
        bars,
        (TIMEFRAME,),
    ):
        candles.append(candle)
        engine.process(candle)

    observations = [
        observation
        for observation in engine.observations
        if start <= observation.signal_timestamp < end
    ]

    observations.sort(
        key=lambda item: item.signal_timestamp
    )

    persistence_map = build_persistence_map(
        candles
    )

    decomposed = []

    missing = 0
    unclassified = 0

    for observation in observations:
        state = persistence_map.get(
            observation.signal_timestamp
        )

        if state is None:
            missing += 1
            continue

        if state == "unclassified":
            unclassified += 1
            continue

        decomposed.append(
            DecomposedObservation(
                observation=observation,
                persistence=state,
            )
        )

    print("RECONSTRUCTION")
    print("-" * 110)
    print(
        f"M15 candles:             {len(candles):,}"
    )
    print(
        f"Canonical observations:  {len(observations):,}"
    )
    print(
        f"Decomposed observations: {len(decomposed):,}"
    )
    print(
        f"Missing persistence:      {missing:,}"
    )
    print(
        f"Unclassified observations:{unclassified:,}"
    )
    print()

    if missing or unclassified:
        print(
            "FATAL: persistence decomposition is incomplete."
        )
        return 2

    window_results = []
    all_reconciliations_pass = True

    for window in WALKFORWARD_WINDOWS:
        train_start = parse_datetime(
            window["train_start"]
        )
        train_end = parse_datetime(
            window["train_end"]
        )
        oos_start = parse_datetime(
            window["oos_start"]
        )
        oos_end = parse_datetime(
            window["oos_end"]
        )

        if train_start < start:
            train_start = start

        if oos_end > end:
            oos_end = end

        train_items = scoped_items(
            decomposed,
            train_start,
            train_end,
        )

        oos_items = scoped_items(
            decomposed,
            oos_start,
            oos_end,
        )

        print()
        print("=" * 110)
        print(
            f"{window['id']} | "
            f"TRAIN {train_start.isoformat()} -> "
            f"{train_end.isoformat()} | "
            f"OOS {oos_start.isoformat()} -> "
            f"{oos_end.isoformat()}"
        )
        print("=" * 110)

        train_results: dict[str, Any] = {}
        oos_results: dict[str, Any] = {}
        comparisons: dict[str, Any] = {}

        for horizon_index, horizon in enumerate(
            HORIZONS
        ):
            seed_base = (
                args.bootstrap_seed
                + horizon_index * 10000
                + stable_seed(window["id"])
            )

            train_result = analyze_scope(
                train_items,
                horizon,
                seed_base,
                args.bootstrap_replicates,
            )

            oos_result = analyze_scope(
                oos_items,
                horizon,
                seed_base + 5000,
                args.bootstrap_replicates,
            )

            comparison = compare_train_oos(
                train_result,
                oos_result,
            )

            train_results[str(horizon)] = train_result
            oos_results[str(horizon)] = oos_result
            comparisons[str(horizon)] = comparison

            all_reconciliations_pass = (
                all_reconciliations_pass
                and train_result["reconciliation"]["passed"]
                and oos_result["reconciliation"]["passed"]
            )

            print()
            print(
                f"H{horizon} — TRAIN"
            )
            print("-" * 110)

            for state in PERSISTENCE_STATES:
                print_state(
                    state,
                    train_result["states"][state],
                )

            print(
                f"    reconciliation: "
                f"{train_result['reconciliation']['passed']}"
            )

            print()
            print(
                f"H{horizon} — OOS"
            )
            print("-" * 110)

            for state in PERSISTENCE_STATES:
                print_state(
                    state,
                    oos_result["states"][state],
                )

            print(
                f"    reconciliation: "
                f"{oos_result['reconciliation']['passed']}"
            )

            print()
            print(
                f"H{horizon} — TRAIN → OOS COMPARISON"
            )
            print("-" * 110)

            for state in PERSISTENCE_STATES:
                comparison_state = comparison[state]

                print(
                    f"    {state:<18} "
                    f"accuracy "
                    f"{fmt_pp(comparison_state['train_interaction_accuracy'])}"
                    f" -> "
                    f"{fmt_pp(comparison_state['oos_interaction_accuracy'])} "
                    f"| sign agreement="
                    f"{comparison_state['accuracy_sign_agreement']}"
                )

                print(
                    f"    {'':18} "
                    f"signed "
                    f"{fmt_return(comparison_state['train_interaction_signed_return'])}"
                    f" -> "
                    f"{fmt_return(comparison_state['oos_interaction_signed_return'])} "
                    f"| sign agreement="
                    f"{comparison_state['signed_return_sign_agreement']}"
                )

        window_results.append(
            {
                "window": {
                    **window,
                    "train_start": train_start.isoformat(),
                    "train_end": train_end.isoformat(),
                    "oos_start": oos_start.isoformat(),
                    "oos_end": oos_end.isoformat(),
                },
                "train": train_results,
                "oos": oos_results,
                "comparison": comparisons,
            }
        )

    total_state_comparisons = (
        len(WALKFORWARD_WINDOWS)
        * len(HORIZONS)
        * len(PERSISTENCE_STATES)
    )

    accuracy_sign_agreements = 0
    signed_sign_agreements = 0
    valid_accuracy_comparisons = 0
    valid_signed_comparisons = 0

    for window_result in window_results:
        for horizon in HORIZONS:
            for state in PERSISTENCE_STATES:
                comparison = window_result[
                    "comparison"
                ][str(horizon)][state]

                if comparison[
                    "accuracy_sign_agreement"
                ] is not None:
                    valid_accuracy_comparisons += 1
                    accuracy_sign_agreements += int(
                        comparison[
                            "accuracy_sign_agreement"
                        ]
                    )

                if comparison[
                    "signed_return_sign_agreement"
                ] is not None:
                    valid_signed_comparisons += 1
                    signed_sign_agreements += int(
                        comparison[
                            "signed_return_sign_agreement"
                        ]
                    )

    summary = {
        "walkforward_windows": len(window_results),
        "horizons": list(HORIZONS),
        "persistence_states": list(PERSISTENCE_STATES),
        "total_state_comparisons": total_state_comparisons,
        "valid_accuracy_sign_comparisons": valid_accuracy_comparisons,
        "accuracy_sign_agreements": accuracy_sign_agreements,
        "accuracy_sign_agreement_rate": (
            accuracy_sign_agreements
            / valid_accuracy_comparisons
            if valid_accuracy_comparisons
            else None
        ),
        "valid_signed_return_sign_comparisons": (
            valid_signed_comparisons
        ),
        "signed_return_sign_agreements": (
            signed_sign_agreements
        ),
        "signed_return_sign_agreement_rate": (
            signed_sign_agreements
            / valid_signed_comparisons
            if valid_signed_comparisons
            else None
        ),
        "all_reconciliations_pass": (
            all_reconciliations_pass
        ),
        "strategy_modification": False,
        "future_information_used_for_selection": False,
    }

    payload = {
        "phase": "6.3D",
        "title": (
            "Chronological Persistence × "
            "Extreme-Volatility Robustness"
        ),
        "source": SOURCE_NAME,
        "timeframe": "M15",
        "horizons": list(HORIZONS),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "methodology": {
            "primary_statistic": (
                "(extreme bullish accuracy - extreme bearish accuracy) "
                "- (non-extreme bullish accuracy - "
                "non-extreme bearish accuracy)"
            ),
            "secondary_statistic": (
                "(extreme bullish mean signed return - "
                "extreme bearish mean signed return) "
                "- (non-extreme bullish mean signed return - "
                "non-extreme bearish mean signed return)"
            ),
            "persistence_definition": (
                "Current non-flat signal direction equals previous "
                "non-flat signal direction."
            ),
            "reversal_definition": (
                "Current non-flat signal direction differs from "
                "previous non-flat signal direction."
            ),
            "flat_candles": "Skipped",
            "chronological_split": True,
            "random_split": False,
            "candidate_selection": False,
            "strategy_modification": False,
            "future_information_used_for_selection": False,
            "oos_persistence_uses_prior_train_signal": True,
            "reconciliation_tolerance": (
                RECONCILIATION_TOLERANCE
            ),
        },
        "reconstruction": {
            "m15_candles": len(candles),
            "canonical_observations": len(observations),
            "decomposed_observations": len(decomposed),
            "missing_persistence": missing,
            "unclassified_observations": unclassified,
        },
        "windows": window_results,
        "summary": summary,
    }

    json_path = (
        args.output_root
        / "direction_persistence_walkforward.json"
    )

    csv_path = (
        args.output_root
        / "direction_persistence_walkforward.csv"
    )

    reconciliation_path = (
        args.output_root
        / "reconciliation.csv"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            sanitize_for_json(payload),
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    write_csv(
        flatten_rows(window_results),
        csv_path,
    )

    reconciliation_rows = []

    for window_result in window_results:
        window_id = window_result["window"]["id"]

        for scope_name in ("train", "oos"):
            for horizon in HORIZONS:
                recon = window_result[
                    scope_name
                ][str(horizon)]["reconciliation"]

                reconciliation_rows.append(
                    {
                        "walkforward_id": window_id,
                        "scope": scope_name,
                        "horizon": horizon,
                        **recon,
                    }
                )

    write_csv(
        reconciliation_rows,
        reconciliation_path,
    )

    print()
    print("=" * 110)
    print("PHASE 6.3D SUMMARY")
    print("=" * 110)
    print(
        f"Accuracy sign agreements: "
        f"{accuracy_sign_agreements}/"
        f"{valid_accuracy_comparisons}"
    )
    print(
        f"Signed-return sign agreements: "
        f"{signed_sign_agreements}/"
        f"{valid_signed_comparisons}"
    )
    print(
        f"All reconciliations pass: "
        f"{all_reconciliations_pass}"
    )
    print()
    print("OUTPUT")
    print("-" * 110)
    print(f"JSON:            {json_path}")
    print(f"CSV:             {csv_path}")
    print(f"Reconciliation:  {reconciliation_path}")
    print()

    if not all_reconciliations_pass:
        print(
            "FATAL: reconciliation failure. "
            "Do NOT interpret the robustness results."
        )
        return 3

    print(
        "IMPORTANT: This is chronological diagnostic evidence only."
    )
    print(
        "No strategy thresholds, entries, exits, or "
        "candidate-selection rules were changed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
