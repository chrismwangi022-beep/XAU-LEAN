"""
Phase 6.3B — Dependence-Aware Directional Validation.

Purpose:
    Reconstruct the exact Phase 6.3A chronological observations for
    the robust A+ candidates and test whether their apparent edge
    survives temporal dependence.

Methodology:
    - Same Dukascopy XAUUSD BID/ASK M1 source as Phase 6.3A.
    - Same UTC calendar-aligned timeframe aggregation.
    - Same ATR(14) / previous-20-ATR baseline.
    - Same frozen volatility buckets.
    - Same signal-after-close / next-candle-open entry.
    - Same H1/H3/H5 horizons.
    - Observations remain chronological.
    - Serial dependence is measured using autocorrelation.
    - Effective sample size is estimated from positive autocorrelation.
    - Moving-block bootstrap is used for dependence-aware confidence
      intervals and sign probabilities.

This is validation research, not strategy optimization.
No thresholds or candidates are optimized against bootstrap results.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from xau_lean.data.dukascopy import DukascopyAdapter
from xau_lean.research.regimes import REGIMES, get_regime, validate_regimes
from xau_lean.research.timeframe import (
    Timeframe,
    aggregate_timeframe,
    aggregate_timeframes,
)


# ---------------------------------------------------------------------------
# Frozen Phase 6.3A parameters
# ---------------------------------------------------------------------------

ATR_PERIOD = 14
BASELINE_PERIOD = 20
HORIZONS = (1, 3, 5)

BUCKETS = (
    ("normal", 0.0, 1.25),
    ("elevated", 1.25, 1.50),
    ("high", 1.50, 2.00),
    ("extreme", 2.00, float("inf")),
)

SOURCE_NAME = "Dukascopy XAUUSD BID/ASK M1"
TIMEZONE_NAME = "UTC"

DEFAULT_DATA_ROOT = Path(
    "data/external/dukascopy/Market-Data-Lab-main/xauusd"
)

DEFAULT_STABILITY_ROOT = Path(
    "research_results/direction_stability_refined"
)

DEFAULT_OUTPUT_ROOT = Path(
    "research_results/direction_dependence"
)

BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 63042

MIN_BLOCK_LENGTH = 5
MAX_BLOCK_LENGTH = 60

TIMEFRAMES = (
    Timeframe.M15,
    Timeframe.H1,
    Timeframe.H4,
    Timeframe.H6,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_datetime(value: str) -> datetime:
    text = value.strip()

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    parsed = datetime.fromisoformat(text)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def format_timestamp(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .strftime("%Y%m%dT%H%M%SZ")
    )


def classify_bucket(ratio: float) -> str:
    for name, lower, upper in BUCKETS:
        if lower <= ratio < upper:
            return name

    raise ValueError(f"Unable to classify expansion ratio: {ratio}")


def candle_direction(candle: Any) -> str:
    if candle.bid_close > candle.bid_open:
        return "bullish"

    if candle.bid_close < candle.bid_open:
        return "bearish"

    return "flat"


def true_range(
    high: float,
    low: float,
    previous_close: float | None,
) -> float:
    if previous_close is None:
        return high - low

    return max(
        high - low,
        abs(high - previous_close),
        abs(low - previous_close),
    )


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot calculate percentile of empty data.")

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


# ---------------------------------------------------------------------------
# Candidate definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    timeframe: str
    bucket: str
    direction: str
    horizon: int

    @property
    def key(self) -> str:
        return (
            f"{self.timeframe}|{self.bucket}|"
            f"{self.direction}|H{self.horizon}"
        )


# ---------------------------------------------------------------------------
# Chronological observation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    timestamp: datetime
    signed_return: float

    @property
    def outcome(self) -> int:
        if self.signed_return > 0.0:
            return 1

        if self.signed_return < 0.0:
            return 0

        return -1


@dataclass
class PendingSignal:
    regime: str
    bucket: str
    direction: str
    signal_timestamp: datetime
    signal_close: float

    entry_price: float | None = None
    entry_timestamp: datetime | None = None
    candles_since_entry: int = 0


# ---------------------------------------------------------------------------
# Exact Phase 6.3A reconstruction
# ---------------------------------------------------------------------------

class ObservationCollector:
    """
    Reconstruct Phase 6.3A observations while retaining timestamps.

    The calculation intentionally mirrors run_direction_research.py.
    """

    def __init__(
        self,
        timeframe: Timeframe,
        candidates: set[str],
    ) -> None:
        self.timeframe = timeframe
        self.candidates = candidates

        self.previous_close: float | None = None

        self.tr_window: deque[float] = deque(maxlen=ATR_PERIOD)
        self.atr_history: deque[float] = deque(maxlen=BASELINE_PERIOD)

        self.pending: dict[int, list[PendingSignal]] = {
            horizon: []
            for horizon in HORIZONS
        }

        self.observations: dict[str, list[Observation]] = {}

        self.candles = 0
        self.complete_candles = 0
        self.signals_created = 0
        self.signals_resolved = 0

    def process(self, candle: Any) -> None:
        self.candles += 1

        if not getattr(candle, "complete", True):
            return

        self.complete_candles += 1

        self._process_pending(candle)

        bid_open = float(candle.bid_open)
        bid_high = float(candle.bid_high)
        bid_low = float(candle.bid_low)
        bid_close = float(candle.bid_close)

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

        regime = get_regime(candle.timestamp)

        if regime is None:
            self.atr_history.append(atr)
            return

        bucket = classify_bucket(ratio)
        direction = candle_direction(candle)

        if direction != "flat":
            for horizon in HORIZONS:
                candidate_key = (
                    f"{self.timeframe.name}|{bucket}|"
                    f"{direction}|H{horizon}"
                )

                if candidate_key not in self.candidates:
                    continue

                self.pending[horizon].append(
                    PendingSignal(
                        regime=regime.name,
                        bucket=bucket,
                        direction=direction,
                        signal_timestamp=candle.timestamp,
                        signal_close=bid_close,
                    )
                )

                self.signals_created += 1

        self.atr_history.append(atr)

    def _process_pending(self, candle: Any) -> None:
        current_open = float(candle.bid_open)
        current_close = float(candle.bid_close)

        for horizon in HORIZONS:
            queue = self.pending[horizon]

            if not queue:
                continue

            remaining: list[PendingSignal] = []

            for signal in queue:
                if signal.entry_price is None:
                    signal.entry_price = current_open
                    signal.entry_timestamp = candle.timestamp
                    signal.candles_since_entry = 1
                else:
                    signal.candles_since_entry += 1

                if signal.candles_since_entry < horizon:
                    remaining.append(signal)
                    continue

                if signal.entry_price <= 0.0:
                    continue

                raw_return = (
                    current_close - signal.entry_price
                ) / signal.entry_price

                if signal.direction == "bullish":
                    signed_return = raw_return
                else:
                    signed_return = -raw_return

                candidate_key = (
                    f"{self.timeframe.name}|{signal.bucket}|"
                    f"{signal.direction}|H{horizon}"
                )

                # Only retain shortlisted hypotheses.
                if candidate_key in self.candidates:
                    self.observations.setdefault(
                        candidate_key,
                        []
                    ).append(
                        Observation(
                            timestamp=candle.timestamp,
                            signed_return=signed_return,
                        )
                    )

                self.signals_resolved += 1

            self.pending[horizon] = remaining


# ---------------------------------------------------------------------------
# Candidate loading
# ---------------------------------------------------------------------------

def load_a_plus_candidates(
    stability_root: Path,
) -> list[Candidate]:
    path = stability_root / "direction_stability_refined_all.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Refined stability file not found: {path}"
        )

    payload = json.loads(path.read_text(encoding="utf-8"))

    candidates: list[Candidate] = []

    rows = payload.get("hypotheses")

    if rows is None:
        rows = payload.get("results")

    if rows is None:
        raise ValueError(
            "Could not find 'hypotheses' or 'results' in "
            f"{path}"
        )

    if isinstance(rows, dict):
        iterable = rows.values()
    else:
        iterable = rows

    for row in iterable:
        if not isinstance(row, dict):
            continue

        if row.get("stability_class") != "A+":
            continue

        hypothesis = row.get("hypothesis")

        if not hypothesis:
            continue

        parts = str(hypothesis).split("|")

        if len(parts) != 4:
            continue

        timeframe, bucket, direction, horizon_text = parts

        if not horizon_text.startswith("H"):
            continue

        candidates.append(
            Candidate(
                timeframe=timeframe,
                bucket=bucket,
                direction=direction,
                horizon=int(horizon_text[1:]),
            )
        )

    if not candidates:
        raise ValueError(
            "No A+ candidates found in refined stability output."
        )

    return sorted(
        candidates,
        key=lambda item: (
            item.timeframe,
            item.bucket,
            item.direction,
            item.horizon,
        ),
    )


# ---------------------------------------------------------------------------
# Dependence calculations
# ---------------------------------------------------------------------------

def autocorrelation(
    values: list[float],
    lag: int,
) -> float | None:
    if lag <= 0:
        raise ValueError("lag must be positive")

    if len(values) <= lag:
        return None

    x = values[:-lag]
    y = values[lag:]

    x_mean = mean(x)
    y_mean = mean(y)

    numerator = sum(
        (a - x_mean) * (b - y_mean)
        for a, b in zip(x, y)
    )

    denominator_x = sum(
        (a - x_mean) ** 2
        for a in x
    )

    denominator_y = sum(
        (b - y_mean) ** 2
        for b in y
    )

    denominator = math.sqrt(
        denominator_x * denominator_y
    )

    if denominator == 0.0:
        return 0.0

    return numerator / denominator


def effective_sample_size(
    values: list[float],
    max_lag: int = 100,
) -> float:
    """
    Estimate effective sample size using the initial positive
    autocorrelation sequence.

    This is a diagnostic, not the primary confidence interval method.
    """

    n = len(values)

    if n < 3:
        return float(n)

    total = 1.0

    limit = min(max_lag, n - 1)

    for lag in range(1, limit + 1):
        rho = autocorrelation(values, lag)

        if rho is None or rho <= 0.0:
            break

        total += 2.0 * rho

    if total <= 0.0:
        return float(n)

    return max(
        1.0,
        min(float(n), n / total),
    )


def choose_block_length(
    n: int,
    horizon: int,
) -> int:
    """
    Fixed transparent rule.

    sqrt(n) grows with sample size while remaining independent of
    observed performance. The horizon is a lower bound because H3/H5
    observations overlap by construction.
    """

    block = max(
        horizon,
        int(round(math.sqrt(n))),
        MIN_BLOCK_LENGTH,
    )

    return min(
        block,
        MAX_BLOCK_LENGTH,
        n,
    )


def moving_block_bootstrap(
    values: list[float],
    block_length: int,
    replicates: int,
    seed: int,
) -> list[float]:
    """
    Moving-block bootstrap of the sample mean.

    Blocks are contiguous in the original chronological sequence.
    The final bootstrap sample contains exactly n observations.
    """

    n = len(values)

    if n == 0:
        return []

    if block_length <= 0:
        raise ValueError("block_length must be positive")

    if block_length > n:
        block_length = n

    rng = random.Random(seed)

    starts = list(range(0, n - block_length + 1))

    results: list[float] = []

    for _ in range(replicates):
        sample: list[float] = []

        while len(sample) < n:
            start = rng.choice(starts)

            sample.extend(
                values[start:start + block_length]
            )

        sample = sample[:n]

        results.append(mean(sample))

    return results


def bootstrap_summary(
    values: list[float],
    block_length: int,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    bootstrap_means = moving_block_bootstrap(
        values=values,
        block_length=block_length,
        replicates=replicates,
        seed=seed,
    )

    lower = percentile(bootstrap_means, 0.025)
    upper = percentile(bootstrap_means, 0.975)

    probability_positive = sum(
        value > 0.0
        for value in bootstrap_means
    ) / len(bootstrap_means)

    probability_negative = sum(
        value < 0.0
        for value in bootstrap_means
    ) / len(bootstrap_means)

    return {
        "bootstrap_replicates": replicates,
        "bootstrap_ci_lower_pct": lower * 100.0,
        "bootstrap_ci_upper_pct": upper * 100.0,
        "bootstrap_probability_mean_positive": probability_positive,
        "bootstrap_probability_mean_negative": probability_negative,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def analyze_candidate(
    candidate: Candidate,
    observations: list[Observation],
) -> dict[str, Any]:
    if not observations:
        raise ValueError(
            f"No observations for {candidate.key}"
        )

    observations = sorted(
        observations,
        key=lambda item: item.timestamp,
    )

    returns = [
        item.signed_return
        for item in observations
    ]

    binary = [
        item.outcome
        for item in observations
        if item.outcome != -1
    ]

    directional_accuracy = (
        sum(binary) / len(binary)
        if binary
        else None
    )

    lag1 = autocorrelation(returns, 1)
    lag5 = autocorrelation(returns, 5)

    ess = effective_sample_size(returns)

    block_length = choose_block_length(
        n=len(returns),
        horizon=candidate.horizon,
    )

    mean_return = mean(returns)

    bootstrap = bootstrap_summary(
        values=returns,
        block_length=block_length,
        replicates=BOOTSTRAP_REPLICATES,
        seed=(
            BOOTSTRAP_SEED
            + sum(ord(char) for char in candidate.key)
        ),
    )

    ci_lower = float(
        bootstrap["bootstrap_ci_lower_pct"]
    )
    ci_upper = float(
        bootstrap["bootstrap_ci_upper_pct"]
    )

    if ci_lower > 0.0:
        survival = "SURVIVES"
    elif ci_upper < 0.0:
        survival = "SURVIVES_INVERSE"
    else:
        survival = "FAILS_DEPENDENCE_TEST"

    return {
        "hypothesis": candidate.key,
        "timeframe": candidate.timeframe,
        "bucket": candidate.bucket,
        "direction": candidate.direction,
        "horizon_candles": candidate.horizon,
        "observations": len(observations),
        "directional_observations": len(binary),
        "directional_accuracy": directional_accuracy,
        "mean_signed_return_pct": mean_return * 100.0,
        "median_signed_return_pct": median(returns) * 100.0,
        "lag1_autocorrelation": lag1,
        "lag5_autocorrelation": lag5,
        "effective_sample_size": ess,
        "effective_sample_fraction": ess / len(returns),
        "block_length": block_length,
        **bootstrap,
        "bootstrap_ci_width_pp": ci_upper - ci_lower,
        "dependence_classification": survival,
    }


# ---------------------------------------------------------------------------
# Research runner
# ---------------------------------------------------------------------------

def collect_observations(
    start: datetime,
    end: datetime,
    data_root: Path,
    candidates: list[Candidate],
) -> tuple[dict[str, list[Observation]], dict[str, Any]]:
    validate_regimes(REGIMES)

    adapter = DukascopyAdapter(data_root)

    candidate_keys = {
        candidate.key
        for candidate in candidates
    }

    by_timeframe: dict[Timeframe, set[str]] = {}

    for candidate in candidates:
        by_timeframe.setdefault(
            Timeframe[candidate.timeframe],
            set(),
        ).add(candidate.key)

    all_observations: dict[str, list[Observation]] = {}

    diagnostics: dict[str, Any] = {}

    for timeframe in TIMEFRAMES:
        keys = by_timeframe.get(timeframe)

        if not keys:
            continue

        collector = ObservationCollector(
            timeframe=timeframe,
            candidates=keys,
        )

        bars = adapter.iter_bars(
            start,
            end,
            require_ask=True,
        )

        if len(by_timeframe) == 1:
            candles = aggregate_timeframe(
                bars,
                timeframe,
            )

            for candle in candles:
                collector.process(candle)

        else:
            # This branch is retained for completeness; the caller
            # currently processes one timeframe at a time so that each
            # pass consumes the same canonical data stream.
            candles = aggregate_timeframe(
                bars,
                timeframe,
            )

            for candle in candles:
                collector.process(candle)

        all_observations.update(collector.observations)

        diagnostics[timeframe.name] = {
            "candles": collector.candles,
            "complete_candles": collector.complete_candles,
            "signals_created": collector.signals_created,
            "signals_resolved": collector.signals_resolved,
        }

    # Ensure every requested candidate exists in the result map.
    for key in candidate_keys:
        all_observations.setdefault(key, [])

    return all_observations, diagnostics


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 6.3B dependence-aware validation of "
            "A+ directional candidates."
        )
    )

    parser.add_argument(
        "--start",
        required=True,
    )

    parser.add_argument(
        "--end",
        required=True,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "data/external/dukascopy/"
            "Market-Data-Lab-main/xauusd"
        ),
    )

    parser.add_argument(
        "--stability-root",
        type=Path,
        default=DEFAULT_STABILITY_ROOT,
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

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.bootstrap_replicates < 100:
        parser.error(
            "--bootstrap-replicates must be >= 100"
        )

    global BOOTSTRAP_REPLICATES
    BOOTSTRAP_REPLICATES = args.bootstrap_replicates

    start = parse_datetime(args.start)
    end = parse_datetime(args.end)

    if end <= start:
        parser.error("--end must be after --start")

    candidates = load_a_plus_candidates(
        args.stability_root
    )

    print()
    print("=" * 110)
    print("PHASE 6.3B — DEPENDENCE-AWARE DIRECTIONAL VALIDATION")
    print("=" * 110)
    print(f"Source:              {SOURCE_NAME}")
    print(f"Window:              {start.isoformat()} -> {end.isoformat()}")
    print(f"A+ candidates:       {len(candidates)}")
    print(f"Bootstrap replicates: {BOOTSTRAP_REPLICATES}")
    print("Bootstrap:           moving block bootstrap")
    print("Block rule:          max(horizon, round(sqrt(n)), 5), capped at 60")
    print()

    print("A+ CANDIDATES")
    print("-" * 110)

    for candidate in candidates:
        print(f"  {candidate.key}")

    observations, diagnostics = collect_observations(
        start=start,
        end=end,
        data_root=args.data_root,
        candidates=candidates,
    )

    results: list[dict[str, Any]] = []

    print()
    print("DEPENDENCE RESULTS")
    print("-" * 110)

    for candidate in candidates:
        result = analyze_candidate(
            candidate=candidate,
            observations=observations[candidate.key],
        )

        results.append(result)

        print(
            f"{candidate.key:<42} "
            f"n={result['observations']:>5} "
            f"ESS={result['effective_sample_size']:>7.1f} "
            f"rho1={result['lag1_autocorrelation']:+.3f} "
            f"CI=["
            f"{result['bootstrap_ci_lower_pct']:+.3f},"
            f"{result['bootstrap_ci_upper_pct']:+.3f}]pp "
            f"{result['dependence_classification']}"
        )

    output_root = args.output_root
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_at = datetime.now(timezone.utc)

    payload = {
        "metadata": {
            "phase": "6.3B",
            "research": "direction_dependence",
            "instrument": "XAUUSD",
            "source": SOURCE_NAME,
            "timezone": TIMEZONE_NAME,
            "requested_window": {
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
            "generated_at": generated_at.isoformat(),
            "candidate_source": (
                "Phase 6.3B refined A+ stability candidates"
            ),
            "methodology": {
                "signal_definition": (
                    "exact Phase 6.3A reconstruction"
                ),
                "entry": "next candle open",
                "forward_measurement": (
                    "direction-adjusted return from entry "
                    "to horizon close"
                ),
                "dependence_method": (
                    "moving block bootstrap"
                ),
                "effective_sample_size": (
                    "initial positive autocorrelation sequence"
                ),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "block_length_rule": (
                    "max(horizon, round(sqrt(n)), 5), "
                    "capped at 60"
                ),
                "optimization": False,
            },
        },
        "diagnostics": diagnostics,
        "results": results,
    }

    output_path = (
        output_root
        / (
            "direction_dependence_"
            f"{format_timestamp(start)}_"
            f"{format_timestamp(end)}.json"
        )
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")

    print()
    print("=" * 110)
    print("SUMMARY")
    print("=" * 110)

    survives = sum(
        result["dependence_classification"] == "SURVIVES"
        for result in results
    )

    inverse = sum(
        result["dependence_classification"]
        == "SURVIVES_INVERSE"
        for result in results
    )

    fails = sum(
        result["dependence_classification"]
        == "FAILS_DEPENDENCE_TEST"
        for result in results
    )

    print(f"Total candidates:       {len(results)}")
    print(f"Positive-edge survives: {survives}")
    print(f"Inverse-edge survives:  {inverse}")
    print(f"Fails dependence test:  {fails}")
    print()
    print(f"Output: {output_path}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
