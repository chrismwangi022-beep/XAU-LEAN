#!/usr/bin/env python3
"""
Phase 6.3B — Momentum Statistical Validation

Step 3:
Historical stability analysis of the Phase 6.3A directional hypotheses.

This module operates entirely on the existing Phase 6.3B statistical
results and does NOT rerun historical market research.

A hypothesis is evaluated across chronological historical regimes:

    2010-2014
    2015-2019
    2020-2021
    2022-2023
    2024-2026

The purpose is to distinguish:
    - stable relationships,
    - regime-dependent relationships,
    - contradictory relationships,
    - insufficient-sample relationships.

Statistical significance alone is never treated as proof of tradability.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "research_results"
    / "direction_statistics"
    / "direction_statistics_all.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "research_results"
    / "direction_stability"
)

REGIME_ORDER = [
    "2010-2014",
    "2015-2019",
    "2020-2021",
    "2022-2023",
    "2024-2026",
]

MIN_SAMPLE_SIZE = 100

STABILITY_CLASSES = {
    "A": "stable_candidate",
    "B": "regime_dependent_candidate",
    "C": "unstable_or_contradictory",
    "D": "insufficient_sample",
}


def load_json(path: Path) -> dict[str, Any]:
    """Load the Phase 6.3B statistical results."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def hypothesis_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    """
    Remove the historical regime from a cell and create the comparable
    hypothesis identifier.
    """
    return (
        row["timeframe"],
        row["bucket"],
        row["direction"],
        int(row["horizon_candles"]),
    )


def extract_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten all timeframe reports into one list."""
    rows: list[dict[str, Any]] = []

    for report in data["reports"]:
        timeframe = report["metadata"]["timeframe"]

        for row in report["results"]:
            copied = dict(row)
            copied["timeframe"] = timeframe
            rows.append(copied)

    return rows


def build_groups(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str, int], list[dict[str, Any]]]:
    """Group observations by comparable hypothesis."""
    groups: dict[
        tuple[str, str, str, int],
        list[dict[str, Any]],
    ] = {}

    for row in rows:
        key = hypothesis_key(row)
        groups.setdefault(key, []).append(row)

    for group_rows in groups.values():
        group_rows.sort(
            key=lambda row: REGIME_ORDER.index(row["regime"])
        )

    return groups


def sign_of_edge(edge_pp: float) -> int:
    """Return +1, 0, or -1 for an observed directional edge."""
    if edge_pp > 0:
        return 1

    if edge_pp < 0:
        return -1

    return 0


def classify_stability(
    *,
    available_periods: int,
    adequate_sample_periods: int,
    positive_periods: int,
    negative_periods: int,
    neutral_periods: int,
    significant_positive_periods: int,
    significant_negative_periods: int,
    min_sample_size: int,
) -> str:
    """
    Classify historical stability.

    Class A:
        Positive relationship is observed in at least 4 chronological
        periods, with adequate sample sizes in at least 4 periods.

    Class B:
        Relationship is materially positive in multiple periods but
        concentrated in a subset of regimes.

    Class C:
        Evidence is contradictory, including meaningful sign reversals.

    Class D:
        Too little adequate data to make a stability judgement.
    """
    if (
        available_periods < 4
        or adequate_sample_periods < 4
        or min_sample_size < MIN_SAMPLE_SIZE
    ):
        return "D"

    if positive_periods >= 4 and negative_periods <= 1:
        return "A"

    if negative_periods >= 4 and positive_periods <= 1:
        return "A"

    if positive_periods >= 2 and negative_periods >= 2:
        return "C"

    if (
        significant_positive_periods >= 2
        and significant_negative_periods == 0
    ):
        return "B"

    if (
        significant_negative_periods >= 2
        and significant_positive_periods == 0
    ):
        return "B"

    if positive_periods >= 2 or negative_periods >= 2:
        return "B"

    return "D"


def analyze_group(
    key: tuple[str, str, str, int],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Analyze one hypothesis across chronological regimes."""
    timeframe, bucket, direction, horizon = key

    by_regime = {
        row["regime"]: row
        for row in rows
    }

    period_rows = []

    for regime in REGIME_ORDER:
        row = by_regime.get(regime)

        if row is None:
            period_rows.append(
                {
                    "regime": regime,
                    "available": False,
                }
            )
            continue

        period_rows.append(
            {
                "regime": regime,
                "available": True,
                "directional_observations": row[
                    "directional_observations"
                ],
                "directional_accuracy": row[
                    "directional_accuracy"
                ],
                "accuracy_pct": row[
                    "directional_accuracy"
                ] * 100.0,
                "edge_pp": row["edge_pp"],
                "ci_95_lower_pct": row["ci_95_lower_pct"],
                "ci_95_upper_pct": row["ci_95_upper_pct"],
                "p_value": row["p_value"],
                "p_value_fdr": row.get("p_value_fdr"),
                "fdr_significant_05": row.get(
                    "fdr_significant_05",
                    False,
                ),
            }
        )

    available = [
        row
        for row in period_rows
        if row["available"]
    ]

    adequate = [
        row
        for row in available
        if row["directional_observations"] >= MIN_SAMPLE_SIZE
    ]

    edges = [
        row["edge_pp"]
        for row in adequate
    ]

    positive_periods = sum(
        sign_of_edge(row["edge_pp"]) > 0
        for row in adequate
    )

    negative_periods = sum(
        sign_of_edge(row["edge_pp"]) < 0
        for row in adequate
    )

    neutral_periods = sum(
        sign_of_edge(row["edge_pp"]) == 0
        for row in adequate
    )

    significant_positive_periods = sum(
        row["fdr_significant_05"]
        and row["edge_pp"] > 0
        for row in adequate
    )

    significant_negative_periods = sum(
        row["fdr_significant_05"]
        and row["edge_pp"] < 0
        for row in adequate
    )

    min_sample_size = min(
        (
            row["directional_observations"]
            for row in adequate
        ),
        default=0,
    )

    max_edge = max(edges, default=None)
    min_edge = min(edges, default=None)

    result = {
        "timeframe": timeframe,
        "bucket": bucket,
        "direction": direction,
        "horizon_candles": horizon,
        "hypothesis": (
            f"{timeframe}|{bucket}|{direction}|H{horizon}"
        ),
        "periods_expected": len(REGIME_ORDER),
        "periods_available": len(available),
        "periods_adequate_sample": len(adequate),
        "positive_periods": positive_periods,
        "negative_periods": negative_periods,
        "neutral_periods": neutral_periods,
        "significant_positive_periods": (
            significant_positive_periods
        ),
        "significant_negative_periods": (
            significant_negative_periods
        ),
        "sign_consistency": (
            max(positive_periods, negative_periods)
            / len(adequate)
            if adequate
            else 0.0
        ),
        "mean_edge_pp": mean(edges) if edges else None,
        "best_edge_pp": max_edge,
        "worst_edge_pp": min_edge,
        "edge_range_pp": (
            max_edge - min_edge
            if max_edge is not None and min_edge is not None
            else None
        ),
        "min_sample_size": min_sample_size,
        "total_sample_size": sum(
            row["directional_observations"]
            for row in available
        ),
        "stability_class": classify_stability(
            available_periods=len(available),
            adequate_sample_periods=len(adequate),
            positive_periods=positive_periods,
            negative_periods=negative_periods,
            neutral_periods=neutral_periods,
            significant_positive_periods=(
                significant_positive_periods
            ),
            significant_negative_periods=(
                significant_negative_periods
            ),
            min_sample_size=min_sample_size,
        ),
        "stability_label": None,
        "periods": period_rows,
    }

    result["stability_label"] = STABILITY_CLASSES[
        result["stability_class"]
    ]

    return result


def analyze(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Analyze all comparable hypotheses."""
    rows = extract_rows(data)
    groups = build_groups(rows)

    results = [
        analyze_group(key, group_rows)
        for key, group_rows in groups.items()
    ]

    results.sort(
        key=lambda row: (
            row["timeframe"],
            row["bucket"],
            row["direction"],
            row["horizon_candles"],
        )
    )

    return results


def write_json(path: Path, data: Any) -> None:
    """Write formatted JSON."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def flatten_for_csv(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create one flat CSV row per hypothesis."""
    rows = []

    for result in results:
        row = {
            key: value
            for key, value in result.items()
            if key != "periods"
        }

        for period in result["periods"]:
            regime = period["regime"]

            if not period["available"]:
                row[f"{regime}_accuracy_pct"] = None
                row[f"{regime}_edge_pp"] = None
                row[f"{regime}_n"] = None
                row[f"{regime}_p_value"] = None
                row[f"{regime}_fdr_p"] = None
                continue

            row[f"{regime}_accuracy_pct"] = (
                period["accuracy_pct"]
            )
            row[f"{regime}_edge_pp"] = period["edge_pp"]
            row[f"{regime}_n"] = (
                period["directional_observations"]
            )
            row[f"{regime}_p_value"] = period["p_value"]
            row[f"{regime}_fdr_p"] = period["p_value_fdr"]

        rows.append(row)

    return rows


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    """Write flattened stability CSV."""
    rows = flatten_for_csv(results)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

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


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print stability summary."""
    print()
    print("=" * 110)
    print("PHASE 6.3B — HISTORICAL STABILITY ANALYSIS")
    print("=" * 110)
    print()
    print(
        "Historical periods:",
        ", ".join(REGIME_ORDER),
    )
    print(
        f"Minimum adequate sample per period: {MIN_SAMPLE_SIZE}"
    )
    print()

    for timeframe in sorted(
        {row["timeframe"] for row in results}
    ):
        timeframe_rows = [
            row
            for row in results
            if row["timeframe"] == timeframe
        ]

        print("-" * 110)
        print(f"TIMEFRAME: {timeframe}")
        print("-" * 110)

        for class_code in ("A", "B", "C", "D"):
            count = sum(
                row["stability_class"] == class_code
                for row in timeframe_rows
            )

            print(
                f"Class {class_code} "
                f"({STABILITY_CLASSES[class_code]}): {count}"
            )

        print()

        candidates = [
            row
            for row in timeframe_rows
            if row["stability_class"] in {"A", "B"}
        ]

        candidates.sort(
            key=lambda row: (
                row["stability_class"],
                -abs(row["mean_edge_pp"] or 0.0),
            )
        )

        if candidates:
            print("Most interesting surviving hypotheses:")

            for row in candidates[:10]:
                print(
                    f"  {row['hypothesis']:<42} "
                    f"class={row['stability_class']} "
                    f"mean_edge={row['mean_edge_pp']:+.3f}pp "
                    f"best={row['best_edge_pp']:+.3f}pp "
                    f"worst={row['worst_edge_pp']:+.3f}pp "
                    f"sign={row['positive_periods']}+/"
                    f"{row['negative_periods']}- "
                    f"min_n={row['min_sample_size']}"
                )

        print()

    print("=" * 110)
    print("CROSS-TIMEFRAME STABILITY SUMMARY")
    print("=" * 110)

    for class_code in ("A", "B", "C", "D"):
        count = sum(
            row["stability_class"] == class_code
            for row in results
        )

        print(
            f"Class {class_code} "
            f"({STABILITY_CLASSES[class_code]}): {count}"
        )

    print()
    print("IMPORTANT")
    print("=" * 110)
    print(
        "This is a historical stability screen, not an "
        "out-of-sample validation."
    )
    print(
        "A stable historical relationship can still fail "
        "under chronological walk-forward testing."
    )
    print(
        "Economic significance, execution costs, serial dependence, "
        "and OOS validation remain unresolved."
    )
    print()


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Required input file not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_json(INPUT_FILE)
    results = analyze(data)

    output = {
        "phase": "6.3B",
        "step": "3",
        "title": "Historical Stability Analysis",
        "source": INPUT_FILE.name,
        "historical_periods": REGIME_ORDER,
        "minimum_sample_size_per_period": MIN_SAMPLE_SIZE,
        "classification": STABILITY_CLASSES,
        "hypotheses_analyzed": len(results),
        "results": results,
    }

    write_json(
        OUTPUT_DIR / "direction_stability_all.json",
        output,
    )

    write_csv(
        OUTPUT_DIR / "direction_stability_all.csv",
        results,
    )

    for timeframe in sorted(
        {row["timeframe"] for row in results}
    ):
        subset = [
            row
            for row in results
            if row["timeframe"] == timeframe
        ]

        write_json(
            OUTPUT_DIR
            / f"direction_stability_{timeframe.lower()}.json",
            {
                "phase": "6.3B",
                "step": "3",
                "timeframe": timeframe,
                "historical_periods": REGIME_ORDER,
                "minimum_sample_size_per_period": MIN_SAMPLE_SIZE,
                "results": subset,
            },
        )

    print_summary(results)

    print(f"Output directory: {OUTPUT_DIR}")
    print("Generated:")
    print("  direction_stability_all.json")
    print("  direction_stability_all.csv")
    print("  direction_stability_<timeframe>.json")


if __name__ == "__main__":
    main()
