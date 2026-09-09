#!/usr/bin/env python3
"""
Phase 6.3B — Robust Historical Stability Refinement

This module refines the initial historical stability screen.

It distinguishes:
    A+ = robust persistent relationship
    A  = persistent but weaker relationship
    B  = regime-dependent relationship
    C  = contradictory / unstable relationship
    D  = insufficient sample

The analysis uses the existing Phase 6.3B Step 3 output and does
not rerun historical market research.

Important:
This remains an in-sample historical stability screen. It is NOT
chronological walk-forward validation and does NOT establish
tradability.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "research_results"
    / "direction_stability"
    / "direction_stability_all.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "research_results"
    / "direction_stability_refined"
)

REGIME_ORDER = [
    "2010-2014",
    "2015-2019",
    "2020-2021",
    "2022-2023",
    "2024-2026",
]

MIN_SAMPLE_SIZE = 100

MEAN_EFFECT_THRESHOLD_PP = 2.0
MATERIAL_REVERSAL_THRESHOLD_PP = 1.0
ROBUST_PERIOD_COUNT = 4


CLASS_LABELS = {
    "A+": "robust_persistent_candidate",
    "A": "persistent_but_weaker_candidate",
    "B": "regime_dependent_candidate",
    "C": "unstable_or_contradictory",
    "D": "insufficient_sample",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sign(edge: float) -> int:
    if edge > 0:
        return 1
    if edge < 0:
        return -1
    return 0


def adequate_periods(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        period
        for period in row["periods"]
        if (
            period["available"]
            and period["directional_observations"]
            >= MIN_SAMPLE_SIZE
        )
    ]


def classify(
    *,
    periods: list[dict[str, Any]],
    mean_edge: float | None,
    median_edge: float | None,
    min_edge: float | None,
    max_edge: float | None,
) -> str:
    if len(periods) < 3:
        return "D"

    edges = [period["edge_pp"] for period in periods]

    positive = sum(edge > 0 for edge in edges)
    negative = sum(edge < 0 for edge in edges)

    if not edges:
        return "D"

    dominant_sign = (
        1 if positive > negative else
        -1 if negative > positive else
        0
    )

    dominant_count = max(positive, negative)

    if dominant_count < 3:
        return "C"

    if mean_edge is None or median_edge is None:
        return "D"

    # Robust persistent candidate:
    # At least 4/5 periods agree in sign AND the central tendency
    # has at least a 2pp effect AND the worst period does not
    # materially reverse the relationship.
    if (
        dominant_count >= ROBUST_PERIOD_COUNT
        and abs(mean_edge) >= MEAN_EFFECT_THRESHOLD_PP
        and abs(median_edge) >= MEAN_EFFECT_THRESHOLD_PP
    ):
        if dominant_sign > 0 and (min_edge or 0.0) > -MATERIAL_REVERSAL_THRESHOLD_PP:
            return "A+"

        if dominant_sign < 0 and (max_edge or 0.0) < MATERIAL_REVERSAL_THRESHOLD_PP:
            return "A+"

    # Persistent but weaker:
    # Sign remains dominant but magnitude does not satisfy A+.
    if dominant_count >= ROBUST_PERIOD_COUNT:
        return "A"

    # Regime-dependent:
    # A majority sign exists, but at least two periods disagree.
    if dominant_count >= 3:
        return "B"

    return "C"


def refine_row(row: dict[str, Any]) -> dict[str, Any]:
    periods = adequate_periods(row)

    edges = [period["edge_pp"] for period in periods]

    if edges:
        mean_edge = mean(edges)
        median_edge = median(edges)
        min_edge = min(edges)
        max_edge = max(edges)
        edge_std = pstdev(edges) if len(edges) > 1 else 0.0
    else:
        mean_edge = None
        median_edge = None
        min_edge = None
        max_edge = None
        edge_std = None

    positive_2pp = sum(
        edge >= MEAN_EFFECT_THRESHOLD_PP
        for edge in edges
    )

    negative_2pp = sum(
        edge <= -MEAN_EFFECT_THRESHOLD_PP
        for edge in edges
    )

    positive = sum(edge > 0 for edge in edges)
    negative = sum(edge < 0 for edge in edges)
    neutral = sum(edge == 0 for edge in edges)

    latest = next(
        (
            period
            for period in row["periods"]
            if period["regime"] == "2024-2026"
            and period["available"]
        ),
        None,
    )

    majority_sign = (
        "positive"
        if positive > negative
        else "negative"
        if negative > positive
        else "mixed"
    )

    latest_agrees = None

    if latest is not None and majority_sign != "mixed":
        latest_sign = sign(latest["edge_pp"])

        latest_agrees = (
            (majority_sign == "positive" and latest_sign > 0)
            or
            (majority_sign == "negative" and latest_sign < 0)
        )

    stability_class = classify(
        periods=periods,
        mean_edge=mean_edge,
        median_edge=median_edge,
        min_edge=min_edge,
        max_edge=max_edge,
    )

    return {
        "hypothesis": row["hypothesis"],
        "timeframe": row["timeframe"],
        "bucket": row["bucket"],
        "direction": row["direction"],
        "horizon_candles": row["horizon_candles"],
        "periods_available": len(periods),
        "positive_periods": positive,
        "negative_periods": negative,
        "neutral_periods": neutral,
        "positive_periods_ge_2pp": positive_2pp,
        "negative_periods_le_minus_2pp": negative_2pp,
        "sign_consistency": (
            max(positive, negative) / len(periods)
            if periods
            else 0.0
        ),
        "mean_edge_pp": mean_edge,
        "median_edge_pp": median_edge,
        "edge_std_pp": edge_std,
        "best_edge_pp": max_edge,
        "worst_edge_pp": min_edge,
        "edge_range_pp": (
            max_edge - min_edge
            if max_edge is not None and min_edge is not None
            else None
        ),
        "min_sample_size": min(
            (
                period["directional_observations"]
                for period in periods
            ),
            default=0,
        ),
        "majority_sign": majority_sign,
        "2024_2026_edge_pp": (
            latest["edge_pp"]
            if latest is not None
            else None
        ),
        "2024_2026_accuracy_pct": (
            latest["accuracy_pct"]
            if latest is not None
            else None
        ),
        "2024_2026_n": (
            latest["directional_observations"]
            if latest is not None
            else None
        ),
        "2024_2026_agrees_with_majority": latest_agrees,
        "stability_class": stability_class,
        "stability_label": CLASS_LABELS[stability_class],
        "periods": row["periods"],
    }


def analyze(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        refine_row(row)
        for row in data["results"]
    ]


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flattened = []

    for row in rows:
        output = {
            key: value
            for key, value in row.items()
            if key != "periods"
        }

        for period in row["periods"]:
            regime = period["regime"]

            if not period["available"]:
                output[f"{regime}_edge_pp"] = None
                output[f"{regime}_accuracy_pct"] = None
                output[f"{regime}_n"] = None
                continue

            output[f"{regime}_edge_pp"] = period["edge_pp"]
            output[f"{regime}_accuracy_pct"] = (
                period["accuracy_pct"]
            )
            output[f"{regime}_n"] = (
                period["directional_observations"]
            )

        flattened.append(output)

    if not flattened:
        return

    fieldnames = list(flattened[0].keys())

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
        writer.writerows(flattened)


def print_class_summary(rows: list[dict[str, Any]]) -> None:
    print()
    print("=" * 110)
    print("PHASE 6.3B — ROBUST STABILITY REFINEMENT")
    print("=" * 110)
    print()

    print(
        f"Minimum sample per period: {MIN_SAMPLE_SIZE}"
    )
    print(
        f"Meaningful effect threshold: "
        f"{MEAN_EFFECT_THRESHOLD_PP:.1f}pp"
    )
    print(
        f"Material reversal threshold: "
        f"{MATERIAL_REVERSAL_THRESHOLD_PP:.1f}pp"
    )
    print()

    for timeframe in sorted(
        {row["timeframe"] for row in rows}
    ):
        subset = [
            row
            for row in rows
            if row["timeframe"] == timeframe
        ]

        print("-" * 110)
        print(f"TIMEFRAME: {timeframe}")
        print("-" * 110)

        for code in ("A+", "A", "B", "C", "D"):
            count = sum(
                row["stability_class"] == code
                for row in subset
            )

            print(
                f"{code:<2} "
                f"{CLASS_LABELS[code]:<38} "
                f"{count}"
            )

        print()

        robust = [
            row
            for row in subset
            if row["stability_class"] == "A+"
        ]

        robust.sort(
            key=lambda row: abs(
                row["mean_edge_pp"] or 0.0
            ),
            reverse=True,
        )

        if robust:
            print("A+ robust candidates:")

            for row in robust:
                print(
                    f"  {row['hypothesis']:<42} "
                    f"mean={row['mean_edge_pp']:+.3f}pp "
                    f"median={row['median_edge_pp']:+.3f}pp "
                    f"worst={row['worst_edge_pp']:+.3f}pp "
                    f"best={row['best_edge_pp']:+.3f}pp "
                    f"sign={row['positive_periods']}+/"
                    f"{row['negative_periods']}- "
                    f"2024-26="
                    f"{row['2024_2026_edge_pp']:+.3f}pp"
                )

        print()

    print("=" * 110)
    print("H4 / H6 FOCUS — 2024-2026")
    print("=" * 110)

    focus = [
        row
        for row in rows
        if row["timeframe"] in {"H4", "H6"}
        and row["2024_2026_edge_pp"] is not None
    ]

    focus.sort(
        key=lambda row: abs(
            row["2024_2026_edge_pp"]
        ),
        reverse=True,
    )

    for row in focus[:20]:
        latest_edge = row["2024_2026_edge_pp"]
        historical_mean = row["mean_edge_pp"]

        latest_text = (
            f"{latest_edge:+.3f}pp"
            if latest_edge is not None
            else "N/A"
        )

        historical_text = (
            f"{historical_mean:+.3f}pp"
            if historical_mean is not None
            else "N/A"
        )

        print(
            f"  {row['hypothesis']:<42} "
            f"class={row['stability_class']:<2} "
            f"2024-26={latest_text:<10} "
            f"historical_mean={historical_text:<10} "
            f"n={row['2024_2026_n']}"
        )

    print()
    print("=" * 110)
    print("INTERPRETATION WARNING")
    print("=" * 110)
    print(
        "A+ means historically persistent under the defined "
        "screening thresholds."
    )
    print(
        "It does NOT mean the relationship is tradable, causal, "
        "or validated out-of-sample."
    )
    print(
        "Serial dependence, overlapping horizons, execution costs, "
        "and chronological OOS testing remain unresolved."
    )
    print()


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_json(INPUT_FILE)
    rows = analyze(data)

    output = {
        "phase": "6.3B",
        "step": "3B",
        "title": "Robust Historical Stability Refinement",
        "source": INPUT_FILE.name,
        "historical_periods": REGIME_ORDER,
        "minimum_sample_size_per_period": MIN_SAMPLE_SIZE,
        "meaningful_effect_threshold_pp": (
            MEAN_EFFECT_THRESHOLD_PP
        ),
        "material_reversal_threshold_pp": (
            MATERIAL_REVERSAL_THRESHOLD_PP
        ),
        "classification": CLASS_LABELS,
        "hypotheses_analyzed": len(rows),
        "results": rows,
    }

    write_json(
        OUTPUT_DIR / "direction_stability_refined_all.json",
        output,
    )

    write_csv(
        OUTPUT_DIR / "direction_stability_refined_all.csv",
        rows,
    )

    print_class_summary(rows)

    print(f"Output directory: {OUTPUT_DIR}")
    print("Generated:")
    print("  direction_stability_refined_all.json")
    print("  direction_stability_refined_all.csv")


if __name__ == "__main__":
    main()
