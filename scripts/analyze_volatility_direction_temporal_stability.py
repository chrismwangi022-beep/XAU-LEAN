#!/usr/bin/env python3
"""
Phase 6.2B — Training-Period Temporal Stability Screen

Screens the existing Phase 6.2B cross-period volatility × direction
results for temporal sign consistency using training periods only.

Training periods:
    2010-2014
    2015-2019
    2020-2021
    2022-2023

Held-out period:
    2024-2026

This module:
- consumes the existing Phase 6.2B cross-period artifact,
- does not rerun the canonical observation engine,
- does not modify thresholds, horizons, or research periods,
- does not use the held-out period for candidate selection,
- reports descriptive effect-size and temporal-stability statistics.

A candidate passes the temporal stability screen only when all four
training-period estimates have the same non-zero sign.

This is a historical stability screen. It does NOT establish
tradability, profitability, or statistical significance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "research_results"
    / "volatility_direction_cross_period"
    / "xauusd_volatility_direction_cross_period_M15.json"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "research_results"
    / "volatility_direction_temporal_stability"
)

TRAINING_PERIODS = (
    "2010-2014",
    "2015-2019",
    "2020-2021",
    "2022-2023",
)

HELD_OUT_PERIOD = "2024-2026"

REQUIRED_PERIODS = TRAINING_PERIODS + (HELD_OUT_PERIOD,)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def validate_input(payload: dict[str, Any], path: Path) -> None:
    if "metadata" not in payload:
        raise ValueError(f"{path}: missing metadata")

    if "results" not in payload:
        raise ValueError(f"{path}: missing results")

    results = payload["results"]

    if results.get("timeframe") != "M15":
        raise ValueError(
            f"{path}: expected M15 timeframe"
        )

    periods = tuple(results.get("periods", []))

    if periods != REQUIRED_PERIODS:
        raise ValueError(
            f"{path}: unexpected periods: {periods}"
        )

    required_horizons = {"H1", "H3", "H5"}
    required_buckets = {"elevated", "high", "extreme"}
    required_statistics = {
        "accuracy_interaction",
        "signed_return_interaction",
    }

    if set(results.get("horizons", [])) != required_horizons:
        raise ValueError(
            f"{path}: unexpected horizons"
        )

    if set(results.get("buckets", [])) != required_buckets:
        raise ValueError(
            f"{path}: unexpected buckets"
        )

    if set(results.get("statistics", [])) != required_statistics:
        raise ValueError(
            f"{path}: unexpected statistics"
        )


def validate_period_rows(
    period_estimates: list[dict[str, Any]],
    *,
    candidate: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    rows = {
        row["period"]: row
        for row in period_estimates
    }

    missing = set(REQUIRED_PERIODS) - set(rows)

    if missing:
        raise ValueError(
            f"Candidate {candidate} missing periods: "
            f"{sorted(missing)}"
        )

    return rows


def screen_candidate(
    *,
    horizon: str,
    bucket: str,
    statistic: str,
    period_estimates: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate = {
        "timeframe": "M15",
        "horizon": horizon,
        "bucket": bucket,
        "statistic": statistic,
    }

    rows = validate_period_rows(
        period_estimates,
        candidate=candidate,
    )

    training_estimates = [
        float(rows[period]["estimate"])
        for period in TRAINING_PERIODS
    ]

    held_out_estimate = float(
        rows[HELD_OUT_PERIOD]["estimate"]
    )

    training_signs = [
        sign(value)
        for value in training_estimates
    ]

    positive_training_periods = sum(
        value > 0.0
        for value in training_estimates
    )

    negative_training_periods = sum(
        value < 0.0
        for value in training_estimates
    )

    zero_training_periods = sum(
        math.isclose(value, 0.0, abs_tol=1e-15)
        for value in training_estimates
    )

    training_sign_consistent = (
        positive_training_periods == len(TRAINING_PERIODS)
        or
        negative_training_periods == len(TRAINING_PERIODS)
    )

    if positive_training_periods == len(TRAINING_PERIODS):
        training_sign = "positive"
    elif negative_training_periods == len(TRAINING_PERIODS):
        training_sign = "negative"
    else:
        training_sign = "mixed_or_zero"

    held_out_sign = (
        "positive"
        if held_out_estimate > 0.0
        else "negative"
        if held_out_estimate < 0.0
        else "zero"
    )

    held_out_agrees = (
        training_sign in {"positive", "negative"}
        and held_out_sign == training_sign
    )

    return {
        **candidate,
        "training_periods": list(TRAINING_PERIODS),
        "held_out_period": HELD_OUT_PERIOD,
        "training_estimates": [
            {
                "period": period,
                "estimate": float(rows[period]["estimate"]),
                "lower_95": float(rows[period]["lower_95"]),
                "upper_95": float(rows[period]["upper_95"]),
                "observations": int(rows[period]["observations"]),
            }
            for period in TRAINING_PERIODS
        ],
        "held_out_estimate": held_out_estimate,
        "held_out_lower_95": float(
            rows[HELD_OUT_PERIOD]["lower_95"]
        ),
        "held_out_upper_95": float(
            rows[HELD_OUT_PERIOD]["upper_95"]
        ),
        "held_out_observations": int(
            rows[HELD_OUT_PERIOD]["observations"]
        ),
        "positive_training_periods": positive_training_periods,
        "negative_training_periods": negative_training_periods,
        "zero_training_periods": zero_training_periods,
        "training_sign_consistent": training_sign_consistent,
        "training_sign": training_sign,
        "training_mean": mean(training_estimates),
        "training_median": median(training_estimates),
        "training_min": min(training_estimates),
        "training_max": max(training_estimates),
        "training_range": (
            max(training_estimates)
            - min(training_estimates)
        ),
        "training_mean_absolute": mean(
            abs(value) for value in training_estimates
        ),
        "training_between_period_stddev": stdev(
            training_estimates
        ),
        "held_out_sign": held_out_sign,
        "held_out_agrees_with_training": held_out_agrees,
        "temporal_stability_pass": training_sign_consistent,
    }


def build_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload["results"]

    analyses: list[dict[str, Any]] = []

    for item in results["analyses"]:
        analyses.append(
            screen_candidate(
                horizon=item["horizon"],
                bucket=item["bucket"],
                statistic=item["statistic"],
                period_estimates=item["period_estimates"],
            )
        )

    return {
        "timeframe": "M15",
        "training_periods": list(TRAINING_PERIODS),
        "held_out_period": HELD_OUT_PERIOD,
        "candidate_count": len(analyses),
        "passing_candidates": sum(
            item["temporal_stability_pass"]
            for item in analyses
        ),
        "analyses": analyses,
    }


def write_json(
    path: Path,
    analysis: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "phase": "6.2B",
            "analysis": (
                "Training-period temporal stability screen "
                "for volatility × direction"
            ),
            "timeframe": "M15",
            "training_periods": list(TRAINING_PERIODS),
            "held_out_period": HELD_OUT_PERIOD,
            "method": (
                "Candidates are screened using training periods "
                "only. A candidate passes only when all four "
                "training estimates have the same non-zero sign."
            ),
            "held_out_usage": (
                "2024-2026 is evaluated only after training "
                "screening and is never used for candidate selection."
            ),
            "canonical_input": (
                "Existing Phase 6.2B cross-period artifact"
            ),
            "strategy_selection": False,
            "parameter_optimization": False,
            "period_redefinition": False,
            "raw_data_reprocessing": False,
            "statistical_significance_test": False,
            "tradability_claim": False,
        },
        "results": analysis,
    }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(
    path: Path,
    analyses: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for item in analyses:
        row = {
            "timeframe": item["timeframe"],
            "horizon": item["horizon"],
            "bucket": item["bucket"],
            "statistic": item["statistic"],
            "training_sign": item["training_sign"],
            "training_sign_consistent": item[
                "training_sign_consistent"
            ],
            "training_mean": item["training_mean"],
            "training_median": item["training_median"],
            "training_min": item["training_min"],
            "training_max": item["training_max"],
            "training_range": item["training_range"],
            "training_mean_absolute": item[
                "training_mean_absolute"
            ],
            "training_between_period_stddev": item[
                "training_between_period_stddev"
            ],
            "positive_training_periods": item[
                "positive_training_periods"
            ],
            "negative_training_periods": item[
                "negative_training_periods"
            ],
            "zero_training_periods": item[
                "zero_training_periods"
            ],
            "held_out_estimate": item["held_out_estimate"],
            "held_out_lower_95": item["held_out_lower_95"],
            "held_out_upper_95": item["held_out_upper_95"],
            "held_out_observations": item[
                "held_out_observations"
            ],
            "held_out_sign": item["held_out_sign"],
            "held_out_agrees_with_training": item[
                "held_out_agrees_with_training"
            ],
            "temporal_stability_pass": item[
                "temporal_stability_pass"
            ],
        }

        for period_row in item["training_estimates"]:
            period = period_row["period"]
            prefix = period.replace("-", "_")

            row[f"{prefix}_estimate"] = period_row["estimate"]
            row[f"{prefix}_lower_95"] = period_row["lower_95"]
            row[f"{prefix}_upper_95"] = period_row["upper_95"]
            row[f"{prefix}_observations"] = period_row["observations"]

        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 6.2B training-period temporal "
            "stability screen"
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    args = parser.parse_args()

    payload = load_json(args.input)
    validate_input(payload, args.input)

    analysis = build_analysis(payload)

    json_path = (
        args.output_root
        / "xauusd_volatility_direction_temporal_stability_M15.json"
    )

    csv_path = (
        args.output_root
        / "xauusd_volatility_direction_temporal_stability_M15.csv"
    )

    write_json(json_path, analysis)
    write_csv(csv_path, analysis["analyses"])

    print("=" * 88)
    print(
        "PHASE 6.2B — TRAINING-PERIOD TEMPORAL STABILITY SCREEN"
    )
    print("=" * 88)
    print()
    print(
        "Training periods: "
        + ", ".join(TRAINING_PERIODS)
    )
    print(f"Held-out period:  {HELD_OUT_PERIOD}")
    print()
    print(f"Candidates:        {analysis['candidate_count']}")
    print(
        f"Training-stable:   {analysis['passing_candidates']}"
    )
    print()
    print(
        "IMPORTANT: 2024-2026 was not used for candidate selection."
    )
    print()
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
