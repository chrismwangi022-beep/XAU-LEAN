#!/usr/bin/env python3
"""
Phase 6.2B — Cross-Period Volatility × Direction Robustness

Aggregates the five pre-defined chronological period artifacts for M15.

This is a descriptive robustness analysis.

It does NOT:
- pool raw observations across periods,
- perform strategy selection,
- optimize thresholds or horizons,
- redefine research periods,
- manufacture null-hypothesis p-values,
- modify the canonical research engine.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


PERIOD_FILES = {
    "2010-2014": (
        "xauusd_volatility_direction_formal_"
        "20100101T000000Z_20150101T000000Z.json"
    ),
    "2015-2019": (
        "xauusd_volatility_direction_formal_"
        "20150101T000000Z_20200101T000000Z.json"
    ),
    "2020-2021": (
        "xauusd_volatility_direction_formal_"
        "20200101T000000Z_20220101T000000Z.json"
    ),
    "2022-2023": (
        "xauusd_volatility_direction_formal_"
        "20220101T000000Z_20240101T000000Z.json"
    ),
    "2024-2026": (
        "xauusd_volatility_direction_formal_"
        "20240101T000000Z_20260821T000000Z.json"
    ),
}

PERIODS = list(PERIOD_FILES.keys())
HORIZONS = ("H1", "H3", "H5")
BUCKETS = ("elevated", "high", "extreme")
STATISTICS = (
    "accuracy_interaction",
    "signed_return_interaction",
)


def validate_period_file(
    payload: dict[str, Any],
    period: str,
    path: Path,
) -> None:
    if "results" not in payload:
        raise ValueError(f"{path}: missing 'results'")

    results = payload["results"]

    if "M15" not in results:
        raise ValueError(f"{path}: missing M15 results")

    if period not in results["M15"]:
        raise ValueError(
            f"{path}: missing expected period {period!r}"
        )


def load_period_results(
    data_root: Path,
) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}

    for period, filename in PERIOD_FILES.items():
        path = data_root / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing required period artifact: {path}"
            )

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        validate_period_file(payload, period, path)

        # IMPORTANT:
        # The formal script writes the full period-key structure even
        # when only one period was actually populated by a run.
        #
        # Therefore select the period corresponding to this exact
        # chronological artifact rather than trusting every key.
        loaded[period] = payload["results"]["M15"][period]

    return loaded


def classify_robustness(
    values: list[float],
    ci_excludes_zero: list[bool],
) -> str:
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    ci_positive = sum(
        value > 0 and excludes
        for value, excludes in zip(values, ci_excludes_zero)
    )
    ci_negative = sum(
        value < 0 and excludes
        for value, excludes in zip(values, ci_excludes_zero)
    )

    total = len(values)

    if positive == total:
        sign_description = "all_positive"
    elif negative == total:
        sign_description = "all_negative"
    elif positive >= 4:
        sign_description = "mostly_positive"
    elif negative >= 4:
        sign_description = "mostly_negative"
    else:
        sign_description = "mixed_sign"

    if ci_positive + ci_negative == total:
        ci_description = "all_periods_CI_excludes_zero"
    elif ci_positive + ci_negative >= 4:
        ci_description = "most_periods_CI_excludes_zero"
    elif ci_positive + ci_negative >= 1:
        ci_description = "some_periods_CI_excludes_zero"
    else:
        ci_description = "no_period_CI_excludes_zero"

    return f"{sign_description}; {ci_description}"


def aggregate_statistic(
    period_results: dict[str, dict[str, Any]],
    horizon: str,
    bucket: str,
    statistic: str,
) -> dict[str, Any]:
    period_rows: list[dict[str, Any]] = []

    values: list[float] = []
    ci_flags: list[bool] = []

    for period in PERIODS:
        period_data = period_results[period]

        if horizon not in period_data:
            raise ValueError(
                f"{period}: missing horizon {horizon}"
            )

        horizon_data = period_data[horizon]

        if bucket not in horizon_data:
            raise ValueError(
                f"{period}/{horizon}: missing bucket {bucket}"
            )

        bucket_data = horizon_data[bucket]

        if statistic not in bucket_data:
            raise ValueError(
                f"{period}/{horizon}/{bucket}: "
                f"missing statistic {statistic}"
            )

        stat_data = bucket_data[statistic]

        value = float(stat_data["statistic"])
        lower = float(stat_data["lower_95"])
        upper = float(stat_data["upper_95"])

        ci_excludes_zero = (
            lower > 0.0 or upper < 0.0
        )

        observations = int(
            bucket_data.get("observations", 0)
        )

        values.append(value)
        ci_flags.append(ci_excludes_zero)

        period_rows.append(
            {
                "period": period,
                "statistic": statistic,
                "estimate": value,
                "lower_95": lower,
                "upper_95": upper,
                "ci_excludes_zero": ci_excludes_zero,
                "observations": observations,
            }
        )

    positive_periods = sum(value > 0 for value in values)
    negative_periods = sum(value < 0 for value in values)
    zero_periods = sum(
        math.isclose(value, 0.0, abs_tol=1e-15)
        for value in values
    )

    ci_excludes_zero_periods = sum(ci_flags)

    if len(values) > 1:
        between_period_stddev = stdev(values)
    else:
        between_period_stddev = 0.0

    return {
        "horizon": horizon,
        "bucket": bucket,
        "statistic": statistic,
        "period_estimates": period_rows,
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "mean_absolute": mean(abs(value) for value in values),
        "between_period_stddev": between_period_stddev,
        "positive_periods": positive_periods,
        "negative_periods": negative_periods,
        "zero_periods": zero_periods,
        "ci_excludes_zero_periods": ci_excludes_zero_periods,
        "sign_consistent": (
            positive_periods == len(values)
            or negative_periods == len(values)
        ),
        "robustness_description": classify_robustness(
            values,
            ci_flags,
        ),
    }


def build_analysis(
    period_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    analyses: list[dict[str, Any]] = []

    for horizon in HORIZONS:
        for bucket in BUCKETS:
            for statistic in STATISTICS:
                analyses.append(
                    aggregate_statistic(
                        period_results=period_results,
                        horizon=horizon,
                        bucket=bucket,
                        statistic=statistic,
                    )
                )

    return {
        "timeframe": "M15",
        "periods": PERIODS,
        "horizons": list(HORIZONS),
        "buckets": list(BUCKETS),
        "statistics": list(STATISTICS),
        "analyses": analyses,
    }


def write_json(
    output_path: Path,
    analysis: dict[str, Any],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "metadata": {
            "phase": "6.2B",
            "analysis": (
                "Cross-period volatility × direction "
                "robustness"
            ),
            "timeframe": "M15",
            "periods": PERIODS,
            "method": (
                "Descriptive chronological comparison "
                "of five pre-defined period artifacts"
            ),
            "aggregation": (
                "Period-level estimates are summarized "
                "without pooling raw observations"
            ),
            "statistics": list(STATISTICS),
            "null_p_values": "not calculated",
            "strategy_selection": False,
            "parameter_optimization": False,
            "period_redefinition": False,
            "raw_data_reprocessing": False,
        },
        "results": analysis,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
        )


def write_csv(
    output_path: Path,
    analysis: dict[str, Any],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict[str, Any]] = []

    for item in analysis["analyses"]:
        for period_row in item["period_estimates"]:
            rows.append(
                {
                    "horizon": item["horizon"],
                    "bucket": item["bucket"],
                    "statistic": item["statistic"],
                    "period": period_row["period"],
                    "estimate": period_row["estimate"],
                    "lower_95": period_row["lower_95"],
                    "upper_95": period_row["upper_95"],
                    "ci_excludes_zero": (
                        period_row["ci_excludes_zero"]
                    ),
                    "observations": period_row["observations"],
                    "cross_period_mean": item["mean"],
                    "cross_period_median": item["median"],
                    "cross_period_min": item["min"],
                    "cross_period_max": item["max"],
                    "cross_period_range": item["range"],
                    "cross_period_mean_absolute": (
                        item["mean_absolute"]
                    ),
                    "between_period_stddev": (
                        item["between_period_stddev"]
                    ),
                    "positive_periods": (
                        item["positive_periods"]
                    ),
                    "negative_periods": (
                        item["negative_periods"]
                    ),
                    "zero_periods": item["zero_periods"],
                    "ci_excludes_zero_periods": (
                        item["ci_excludes_zero_periods"]
                    ),
                    "sign_consistent": (
                        item["sign_consistent"]
                    ),
                    "robustness_description": (
                        item["robustness_description"]
                    ),
                }
            )

    fieldnames = list(rows[0].keys()) if rows else []

    with output_path.open(
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


def print_summary(
    analysis: dict[str, Any],
) -> None:
    print()
    print("=" * 90)
    print("PHASE 6.2B — CROSS-PERIOD VOLATILITY × DIRECTION")
    print("=" * 90)
    print("Timeframe: M15")
    print("Periods:", ", ".join(PERIODS))
    print()

    for item in analysis["analyses"]:
        print(
            f"{item['horizon']} | "
            f"{item['bucket']:8s} | "
            f"{item['statistic']:25s} | "
            f"mean={item['mean']:+.6f} | "
            f"median={item['median']:+.6f} | "
            f"range={item['range']:.6f} | "
            f"signs=+{item['positive_periods']}/"
            f"-{item['negative_periods']} | "
            f"CI_excl_zero="
            f"{item['ci_excludes_zero_periods']}/5"
        )

    print()
    print("=" * 90)
    print("END OF DESCRIPTIVE CROSS-PERIOD ANALYSIS")
    print("=" * 90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-period robustness analysis for "
            "Phase 6.2B volatility × direction."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("research_results/volatility_direction_formal"),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "research_results/volatility_direction_cross_period"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    period_results = load_period_results(
        args.data_root
    )

    analysis = build_analysis(
        period_results
    )

    json_path = (
        args.output_root
        / "xauusd_volatility_direction_cross_period_M15.json"
    )

    csv_path = (
        args.output_root
        / "xauusd_volatility_direction_cross_period_M15.csv"
    )

    write_json(
        json_path,
        analysis,
    )

    write_csv(
        csv_path,
        analysis,
    )

    print_summary(
        analysis
    )

    print()
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()