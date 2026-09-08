#!/usr/bin/env python3
"""
Phase 6.3B — Momentum Statistical Validation

Step 1 + Step 2:
- Directional accuracy confidence intervals
- Exact two-sided binomial test against 50%
- Directional edge in percentage points
- Benjamini-Hochberg false-discovery-rate correction
- JSON and CSV output

This module intentionally uses only the Python standard library.
It operates on already-generated Phase 6.3A JSON reports and does
not rerun historical research.

Multiple-testing correction is applied globally across all cells from
the four Phase 6.3A timeframe reports.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "research_results" / "direction"
OUTPUT_DIR = PROJECT_ROOT / "research_results" / "direction_statistics"

DEFAULT_FILES = [
    INPUT_DIR
    / "xauusd_momentum_direction_m15_20100101T000000Z_20260821T000000Z.json",
    INPUT_DIR
    / "xauusd_momentum_direction_h1_20100101T000000Z_20260821T000000Z.json",
    INPUT_DIR
    / "xauusd_momentum_direction_h4_20100101T000000Z_20260821T000000Z.json",
    INPUT_DIR
    / "xauusd_momentum_direction_h6_20100101T000000Z_20260821T000000Z.json",
]

NULL_PROBABILITY = 0.5
CONFIDENCE_Z = 1.959963984540054

FDR_Q_05 = 0.05
FDR_Q_10 = 0.10


def load_json(path: Path) -> dict[str, Any]:
    """Load one Phase 6.3A JSON report."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def wilson_interval(
    successes: int,
    trials: int,
    z: float = CONFIDENCE_Z,
) -> tuple[float, float]:
    """
    Calculate a two-sided Wilson score confidence interval.

    Returns:
        (lower_bound, upper_bound)
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between 0 and trials")

    p = successes / trials
    z2 = z * z

    denominator = 1.0 + z2 / trials
    centre = (p + z2 / (2.0 * trials)) / denominator

    margin = (
        z
        * math.sqrt(
            (p * (1.0 - p) / trials)
            + (z2 / (4.0 * trials * trials))
        )
        / denominator
    )

    return max(0.0, centre - margin), min(1.0, centre + margin)


def binomial_probability(
    successes: int,
    trials: int,
    probability: float = NULL_PROBABILITY,
) -> float:
    """Return P(X = successes) for a binomial distribution."""
    if trials < 0:
        raise ValueError("trials must be non-negative")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between 0 and trials")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")

    if probability == 0.0:
        return 1.0 if successes == 0 else 0.0

    if probability == 1.0:
        return 1.0 if successes == trials else 0.0

    log_probability = (
        math.lgamma(trials + 1)
        - math.lgamma(successes + 1)
        - math.lgamma(trials - successes + 1)
        + successes * math.log(probability)
        + (trials - successes) * math.log(1.0 - probability)
    )

    return math.exp(log_probability)


def exact_binomial_two_sided_pvalue(
    successes: int,
    trials: int,
    probability: float = NULL_PROBABILITY,
) -> float:
    """
    Calculate the exact two-sided binomial-test p-value.

    The two-sided p-value is the sum of probabilities of all outcomes
    whose probability under the null is less than or equal to the
    probability of the observed outcome.

    This is suitable for testing:

        H0: p = 0.50
        H1: p != 0.50
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between 0 and trials")

    observed_probability = binomial_probability(
        successes,
        trials,
        probability,
    )

    total = 0.0

    for k in range(trials + 1):
        probability_k = binomial_probability(k, trials, probability)

        if probability_k <= observed_probability + 1e-15:
            total += probability_k

    return min(1.0, total)


def benjamini_hochberg(
    p_values: list[float],
) -> list[float]:
    """
    Calculate Benjamini-Hochberg FDR-adjusted p-values.

    The returned list is in the same order as the input list.

    For m hypotheses sorted by ascending raw p-value:

        q_i = min_{j >= i}(m / j) * p_j

    with ranks j starting at 1.

    This implementation is deterministic and uses only the standard
    library.
    """
    m = len(p_values)

    if m == 0:
        return []

    for p_value in p_values:
        if not 0.0 <= p_value <= 1.0:
            raise ValueError("all p-values must be between 0 and 1")

    indexed = sorted(
        enumerate(p_values),
        key=lambda item: (item[1], item[0]),
    )

    adjusted_sorted = [0.0] * m
    running_min = 1.0

    for rank in range(m, 0, -1):
        original_index, p_value = indexed[rank - 1]

        adjusted = min(
            1.0,
            p_value * m / rank,
        )

        running_min = min(running_min, adjusted)
        adjusted_sorted[rank - 1] = running_min

    adjusted_original_order = [0.0] * m

    for sorted_position, (original_index, _) in enumerate(indexed):
        adjusted_original_order[original_index] = (
            adjusted_sorted[sorted_position]
        )

    return adjusted_original_order


def classify_statistical_result(
    accuracy: float,
    ci_lower: float,
    ci_upper: float,
    p_value: float,
) -> str:
    """
    Give a deliberately limited uncorrected statistical classification.

    This does NOT imply economic significance, tradability, or
    out-of-sample robustness.
    """
    if ci_lower > NULL_PROBABILITY and p_value < 0.05:
        return "statistically_above_50"

    if ci_upper < NULL_PROBABILITY and p_value < 0.05:
        return "statistically_below_50"

    return "not_statistically_distinguishable_from_50"


def classify_fdr_result(
    edge_pp: float,
    fdr_p_value: float,
    q: float,
) -> str:
    """Classify a result after FDR correction."""
    if fdr_p_value < q:
        if edge_pp > 0:
            return "fdr_significant_above_50"
        if edge_pp < 0:
            return "fdr_significant_below_50"

    return "fdr_not_significant"


def analyze_cell(
    cell_key: str,
    cell: dict[str, Any],
) -> dict[str, Any]:
    """Calculate statistical validation metrics for one research cell."""
    positive = int(cell["positive"])
    negative = int(cell["negative"])
    flat = int(cell.get("flat", 0))

    directional_observations = positive + negative
    total_observations = int(
        cell.get("observations", directional_observations)
    )

    if directional_observations <= 0:
        raise ValueError(
            f"Cell {cell_key} has no directional observations"
        )

    accuracy = positive / directional_observations
    edge_pp = (accuracy - NULL_PROBABILITY) * 100.0

    ci_lower, ci_upper = wilson_interval(
        positive,
        directional_observations,
    )

    p_value = exact_binomial_two_sided_pvalue(
        positive,
        directional_observations,
    )

    return {
        "cell": cell_key,
        "regime": cell["regime"],
        "bucket": cell["bucket"],
        "direction": cell["direction"],
        "horizon_candles": int(cell["horizon_candles"]),
        "positive": positive,
        "negative": negative,
        "flat": flat,
        "directional_observations": directional_observations,
        "total_observations": total_observations,
        "directional_accuracy": accuracy,
        "edge_pp": edge_pp,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "ci_95_lower_pct": ci_lower * 100.0,
        "ci_95_upper_pct": ci_upper * 100.0,
        "p_value": p_value,
        "p_value_fdr": None,
        "fdr_significant_05": False,
        "fdr_significant_10": False,
        "fdr_result_05": "pending",
        "fdr_result_10": "pending",
        "mean_signed_return_pct": float(
            cell["mean_signed_return_pct"]
        ),
        "mean_abs_return_pct": float(
            cell["mean_abs_return_pct"]
        ),
        "mean_directional_efficiency": float(
            cell["mean_directional_efficiency"]
        ),
        "statistical_result": classify_statistical_result(
            accuracy,
            ci_lower,
            ci_upper,
            p_value,
        ),
    }


def analyze_report(path: Path) -> dict[str, Any]:
    """Analyze every cell in one Phase 6.3A report."""
    data = load_json(path)

    metadata = data["metadata"]
    report = data["report"]

    results = []

    for cell_key, cell in report["cells"].items():
        results.append(analyze_cell(cell_key, cell))

    results.sort(
        key=lambda row: (
            row["regime"],
            row["bucket"],
            row["direction"],
            row["horizon_candles"],
        )
    )

    return {
        "source_file": path.name,
        "metadata": {
            "timeframe": metadata["timeframe"],
            "instrument": metadata["instrument"],
            "source": metadata["source"],
            "requested_window": metadata["requested_window"],
            "phase": metadata["phase"],
        },
        "null_hypothesis": "directional_accuracy = 50%",
        "alternative_hypothesis": "directional_accuracy != 50%",
        "confidence_level": 0.95,
        "test": "exact_two_sided_binomial",
        "confidence_interval": "wilson_score",
        "cells_analyzed": len(results),
        "results": results,
    }


def apply_global_fdr(
    analyzed_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Apply BH correction globally across every analyzed cell.

    Returns an audit summary describing the correction universe.
    """
    all_rows: list[dict[str, Any]] = []

    for report in analyzed_reports:
        for row in report["results"]:
            all_rows.append(row)

    p_values = [row["p_value"] for row in all_rows]
    adjusted_p_values = benjamini_hochberg(p_values)

    for row, adjusted_p_value in zip(
        all_rows,
        adjusted_p_values,
    ):
        row["p_value_fdr"] = adjusted_p_value

        row["fdr_significant_05"] = (
            adjusted_p_value < FDR_Q_05
        )

        row["fdr_significant_10"] = (
            adjusted_p_value < FDR_Q_10
        )

        row["fdr_result_05"] = classify_fdr_result(
            row["edge_pp"],
            adjusted_p_value,
            FDR_Q_05,
        )

        row["fdr_result_10"] = classify_fdr_result(
            row["edge_pp"],
            adjusted_p_value,
            FDR_Q_10,
        )

    significant_05 = [
        row
        for row in all_rows
        if row["fdr_significant_05"]
    ]

    significant_10 = [
        row
        for row in all_rows
        if row["fdr_significant_10"]
    ]

    above_05 = [
        row
        for row in significant_05
        if row["edge_pp"] > 0
    ]

    below_05 = [
        row
        for row in significant_05
        if row["edge_pp"] < 0
    ]

    above_10 = [
        row
        for row in significant_10
        if row["edge_pp"] > 0
    ]

    below_10 = [
        row
        for row in significant_10
        if row["edge_pp"] < 0
    ]

    return {
        "method": "benjamini_hochberg",
        "correction_scope": "all_480_phase_6_3a_cells",
        "hypotheses_tested": len(all_rows),
        "q_threshold_05": FDR_Q_05,
        "q_threshold_10": FDR_Q_10,
        "fdr_significant_at_05": len(significant_05),
        "fdr_significant_above_50_at_05": len(above_05),
        "fdr_significant_below_50_at_05": len(below_05),
        "fdr_significant_at_10": len(significant_10),
        "fdr_significant_above_50_at_10": len(above_10),
        "fdr_significant_below_50_at_10": len(below_10),
    }


def write_json(path: Path, result: dict[str, Any]) -> None:
    """Write formatted JSON output."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


def write_csv(path: Path, result: dict[str, Any]) -> None:
    """Write flat CSV output."""
    rows = result["results"]

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    all_results: list[dict[str, Any]],
    fdr_summary: dict[str, Any],
) -> None:
    """Print a concise human-readable summary."""
    print()
    print("=" * 110)
    print("PHASE 6.3B — MOMENTUM STATISTICAL VALIDATION")
    print("=" * 110)
    print()
    print("Null hypothesis: directional accuracy = 50%")
    print("Test: exact two-sided binomial test")
    print("Confidence interval: 95% Wilson score interval")
    print("Multiple testing: Benjamini-Hochberg FDR")
    print()

    total_cells = sum(
        report["cells_analyzed"]
        for report in all_results
    )

    print(f"Total hypothesis cells: {total_cells}")
    print(
        "FDR correction scope: all Phase 6.3A cells globally"
    )
    print()

    for report in all_results:
        print("-" * 110)
        print(f"TIMEFRAME: {report['metadata']['timeframe']}")
        print("-" * 110)

        statistically_above = [
            row
            for row in report["results"]
            if row["statistical_result"] == "statistically_above_50"
        ]

        statistically_below = [
            row
            for row in report["results"]
            if row["statistical_result"] == "statistically_below_50"
        ]

        fdr_above = [
            row
            for row in report["results"]
            if row["fdr_result_05"]
            == "fdr_significant_above_50"
        ]

        fdr_below = [
            row
            for row in report["results"]
            if row["fdr_result_05"]
            == "fdr_significant_below_50"
        ]

        print(f"Cells analyzed: {report['cells_analyzed']}")
        print(
            f"Uncorrected above 50%: "
            f"{len(statistically_above)}"
        )
        print(
            f"Uncorrected below 50%: "
            f"{len(statistically_below)}"
        )
        print(
            f"FDR-significant above 50% at q<0.05: "
            f"{len(fdr_above)}"
        )
        print(
            f"FDR-significant below 50% at q<0.05: "
            f"{len(fdr_below)}"
        )

        print()
        print("Strongest observed positive edges:")

        top_positive = sorted(
            report["results"],
            key=lambda row: row["edge_pp"],
            reverse=True,
        )[:5]

        for row in top_positive:
            print(
                f"  {row['cell']}: "
                f"accuracy={row['directional_accuracy'] * 100:.3f}% | "
                f"edge={row['edge_pp']:+.3f}pp | "
                f"n={row['directional_observations']} | "
                f"raw p={row['p_value']:.6g} | "
                f"FDR p={row['p_value_fdr']:.6g} | "
                f"FDR05="
                f"{'YES' if row['fdr_significant_05'] else 'NO'}"
            )

        print()
        print("Strongest observed negative edges:")

        top_negative = sorted(
            report["results"],
            key=lambda row: row["edge_pp"],
        )[:5]

        for row in top_negative:
            print(
                f"  {row['cell']}: "
                f"accuracy={row['directional_accuracy'] * 100:.3f}% | "
                f"edge={row['edge_pp']:+.3f}pp | "
                f"n={row['directional_observations']} | "
                f"raw p={row['p_value']:.6g} | "
                f"FDR p={row['p_value_fdr']:.6g} | "
                f"FDR05="
                f"{'YES' if row['fdr_significant_05'] else 'NO'}"
            )

    print()
    print("=" * 110)
    print("GLOBAL FDR SUMMARY")
    print("=" * 110)
    print(
        f"Hypotheses tested: "
        f"{fdr_summary['hypotheses_tested']}"
    )
    print(
        f"FDR-significant at q<0.05: "
        f"{fdr_summary['fdr_significant_at_05']}"
    )
    print(
        f"  Above 50%: "
        f"{fdr_summary['fdr_significant_above_50_at_05']}"
    )
    print(
        f"  Below 50%: "
        f"{fdr_summary['fdr_significant_below_50_at_05']}"
    )
    print(
        f"FDR-significant at q<0.10: "
        f"{fdr_summary['fdr_significant_at_10']}"
    )
    print(
        f"  Above 50%: "
        f"{fdr_summary['fdr_significant_above_50_at_10']}"
    )
    print(
        f"  Below 50%: "
        f"{fdr_summary['fdr_significant_below_50_at_10']}"
    )

    print()
    print("=" * 110)
    print("IMPORTANT")
    print("=" * 110)
    print(
        "FDR correction controls the expected proportion of false "
        "discoveries among rejected hypotheses."
    )
    print(
        "FDR significance does NOT establish economic significance, "
        "tradability, causality, or out-of-sample robustness."
    )
    print(
        "The 480-cell analysis remains exploratory and subject to "
        "data-snooping concerns. Chronological walk-forward validation "
        "is required before accepting any directional hypothesis."
    )
    print(
        "Trading costs and bid/ask execution effects remain separate "
        "economic-validation steps."
    )
    print()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    analyzed_reports = []

    for input_path in DEFAULT_FILES:
        if not input_path.exists():
            raise FileNotFoundError(
                f"Required Phase 6.3A report not found: {input_path}"
            )

        result = analyze_report(input_path)
        analyzed_reports.append(result)

    fdr_summary = apply_global_fdr(
        analyzed_reports
    )

    for result in analyzed_reports:
        timeframe = result["metadata"]["timeframe"].lower()

        json_path = (
            OUTPUT_DIR
            / f"direction_statistics_{timeframe}.json"
        )

        csv_path = (
            OUTPUT_DIR
            / f"direction_statistics_{timeframe}.csv"
        )

        write_json(json_path, result)
        write_csv(csv_path, result)

    combined = {
        "phase": "6.3B",
        "step": "1+2",
        "title": "Momentum Statistical Validation",
        "null_hypothesis": "directional_accuracy = 50%",
        "alternative_hypothesis": "directional_accuracy != 50%",
        "confidence_level": 0.95,
        "test": "exact_two_sided_binomial",
        "confidence_interval": "wilson_score",
        "multiple_testing": fdr_summary,
        "reports": analyzed_reports,
    }

    write_json(
        OUTPUT_DIR / "direction_statistics_all.json",
        combined,
    )

    print_summary(
        analyzed_reports,
        fdr_summary,
    )

    print(f"Output directory: {OUTPUT_DIR}")
    print("Generated:")
    print("  direction_statistics_all.json")
    print("  direction_statistics_<timeframe>.json")
    print("  direction_statistics_<timeframe>.csv")


if __name__ == "__main__":
    main()
