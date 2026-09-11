#!/usr/bin/env python3
"""
Phase 6.3B — Step 6
Chronological Out-of-Sample Direction Validation

Purpose
-------
Test whether direction/volatility relationships discovered in historical
training data persist in genuinely unseen chronological data.

This module deliberately does NOT use the full-sample Phase 6.3B A/A+
classification for candidate selection because that would leak future
information into earlier walk-forward windows.

Walk-forward windows
--------------------
1. Train: 2010-01-01 -> 2018-12-31
   OOS:   2019-01-01 -> 2021-12-31

2. Train: 2010-01-01 -> 2021-12-31
   OOS:   2022-01-01 -> 2023-12-31

3. Train: 2010-01-01 -> 2023-12-31
   OOS:   2024-01-01 -> 2026-08-21

Selection is performed using TRAINING data only.

Training candidate selection rule
---------------------------------
A candidate must:

    - have >= MIN_PERIOD_SAMPLE observations in at least
      two adequate historical training subperiods;

    - have positive directional edge in a strict majority
      of adequate training subperiods;

    - have mean training edge >= MIN_TRAIN_EDGE_PP;

    - have no adequate training period with a material
      reversal worse than -MAX_TRAIN_REVERSAL_PP.

This is intentionally less restrictive than requiring every historical
period to have the same sign and >=2pp edge. The purpose is to allow
genuine regime-dependent effects to reach the chronological OOS test
without allowing severe historical reversals.

OOS is then completely frozen.

Economic OOS:
    LONG  = ASK entry -> BID exit
    SHORT = BID entry -> ASK exit

Slippage:
    0.0, 0.5, 1.0 and 2.0 bps per side

No optimization is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from xau_lean.data.dukascopy import DukascopyAdapter
from xau_lean.research.regimes import get_regime, validate_regimes, REGIMES
from xau_lean.research.timeframe import (
    Timeframe,
    aggregate_timeframes,
)

from scripts.run_direction_research import (
    ATR_PERIOD,
    BASELINE_PERIOD,
    HORIZONS,
    BUCKETS,
    classify_bucket,
    candle_direction,
    true_range,
)


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
    / "direction_walkforward"
)

MIN_PERIOD_SAMPLE = 100

# Minimum mean historical directional edge required for selection.
MIN_TRAIN_EDGE_PP = 2.0

# A single historical period may be negative, but not materially so.
MAX_TRAIN_REVERSAL_PP = 2.0

BOOTSTRAP_REPLICATES = 3000
BOOTSTRAP_SEED = 6306

SLIPPAGE_BPS = (0.0, 0.5, 1.0, 2.0)

PERIODS = (
    ("2010-2014", "2010-01-01T00:00:00Z", "2015-01-01T00:00:00Z"),
    ("2015-2019", "2015-01-01T00:00:00Z", "2020-01-01T00:00:00Z"),
    ("2020-2021", "2020-01-01T00:00:00Z", "2022-01-01T00:00:00Z"),
    ("2022-2023", "2022-01-01T00:00:00Z", "2024-01-01T00:00:00Z"),
    ("2024-2026", "2024-01-01T00:00:00Z", "2026-08-22T00:00:00Z"),
)

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
        "oos_end": "2026-08-21T00:00:00Z",
    },
)


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------

def parse_datetime(value: str) -> datetime:
    text = value.strip()

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    parsed = datetime.fromisoformat(text)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    timeframe: str
    regime: str
    bucket: str
    direction: str
    horizon_candles: int

    signal_timestamp: datetime
    entry_timestamp: datetime

    bid_entry: float
    ask_entry: float
    bid_exit: float
    ask_exit: float

    signed_return: float

    @property
    def raw_return(self) -> float:
        return (
            self.bid_exit - self.bid_entry
        ) / self.bid_entry

    @property
    def abs_return(self) -> float:
        return abs(self.raw_return)


# ---------------------------------------------------------------------------
# Pending signal
# ---------------------------------------------------------------------------

@dataclass
class PendingSignal:
    timeframe: str
    regime: str
    bucket: str
    direction: str
    horizon_candles: int

    signal_timestamp: datetime

    entry_timestamp: datetime | None = None

    bid_entry: float | None = None
    ask_entry: float | None = None

    candles_since_entry: int = 0


# ---------------------------------------------------------------------------
# Exact Phase 6.3A mechanics, with executable prices retained
# ---------------------------------------------------------------------------

class ObservationEngine:
    """
    Reconstruct Phase 6.3A observations while retaining BID/ASK prices.

    The signal definition is deliberately identical to run_direction_research:
        ATR(14)
        previous 20 ATR values
        frozen volatility buckets
        candle direction
        entry at next candle OPEN
        horizons 1/3/5
    """

    def __init__(self, timeframe: Timeframe) -> None:
        self.timeframe = timeframe

        self.previous_close: float | None = None

        self.tr_window: deque[float] = deque(maxlen=ATR_PERIOD)
        self.atr_history: deque[float] = deque(maxlen=BASELINE_PERIOD)

        self.pending: dict[int, list[PendingSignal]] = {
            horizon: []
            for horizon in HORIZONS
        }

        self.observations: list[Observation] = []

        self.candles = 0
        self.complete_candles = 0
        self.atr_ready = 0
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

        ask_open = (
            float(candle.ask_open)
            if candle.ask_open is not None
            else None
        )

        ask_close = (
            float(candle.ask_close)
            if candle.ask_close is not None
            else None
        )

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

        self.atr_ready += 1

        regime = get_regime(candle.timestamp)

        if regime is None:
            self.atr_history.append(atr)
            return

        bucket = classify_bucket(ratio)
        direction = candle_direction(candle)

        if direction != "flat":
            for horizon in HORIZONS:
                self.pending[horizon].append(
                    PendingSignal(
                        timeframe=self.timeframe.name,
                        regime=regime.name,
                        bucket=bucket,
                        direction=direction,
                        horizon_candles=horizon,
                        signal_timestamp=candle.timestamp,
                    )
                )

                self.signals_created += 1

        self.atr_history.append(atr)

    def _process_pending(self, candle: Any) -> None:
        bid_open = float(candle.bid_open)
        bid_close = float(candle.bid_close)

        ask_open = (
            float(candle.ask_open)
            if candle.ask_open is not None
            else None
        )

        ask_close = (
            float(candle.ask_close)
            if candle.ask_close is not None
            else None
        )

        for horizon in HORIZONS:
            queue = self.pending[horizon]

            if not queue:
                continue

            remaining: list[PendingSignal] = []

            for signal in queue:
                if signal.entry_timestamp is None:
                    if ask_open is None:
                        continue

                    signal.entry_timestamp = candle.timestamp
                    signal.bid_entry = bid_open
                    signal.ask_entry = ask_open
                    signal.candles_since_entry = 1
                else:
                    signal.candles_since_entry += 1

                if signal.candles_since_entry < horizon:
                    remaining.append(signal)
                    continue

                if (
                    signal.bid_entry is None
                    or signal.ask_entry is None
                    or ask_close is None
                    or bid_close <= 0.0
                ):
                    continue

                if signal.direction == "bullish":
                    reference_return = (
                        bid_close - signal.bid_entry
                    ) / signal.bid_entry
                else:
                    reference_return = (
                        signal.bid_entry - bid_close
                    ) / signal.bid_entry

                self.observations.append(
                    Observation(
                        timeframe=self.timeframe.name,
                        regime=signal.regime,
                        bucket=signal.bucket,
                        direction=signal.direction,
                        horizon_candles=signal.horizon_candles,
                        signal_timestamp=signal.signal_timestamp,
                        entry_timestamp=signal.entry_timestamp,
                        bid_entry=signal.bid_entry,
                        ask_entry=signal.ask_entry,
                        bid_exit=bid_close,
                        ask_exit=ask_close,
                        signed_return=reference_return,
                    )
                )

                self.signals_resolved += 1

            self.pending[horizon] = remaining


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson_interval(
    positive: int,
    negative: int,
    confidence_z: float = 1.959963984540054,
) -> tuple[float | None, float | None]:
    n = positive + negative

    if n == 0:
        return None, None

    p = positive / n
    z = confidence_z

    denominator = 1.0 + (z * z / n)

    centre = (
        p + (z * z / (2.0 * n))
    ) / denominator

    margin = (
        z
        * math.sqrt(
            (
                p * (1.0 - p) / n
                + z * z / (4.0 * n * n)
            )
        )
        / denominator
    )

    return (
        max(0.0, centre - margin),
        min(1.0, centre + margin),
    )


def directional_stats(
    observations: list[Observation],
) -> dict[str, Any]:
    positive = sum(
        observation.signed_return > 0
        for observation in observations
    )

    negative = sum(
        observation.signed_return < 0
        for observation in observations
    )

    flat = sum(
        observation.signed_return == 0
        for observation in observations
    )

    directional_n = positive + negative

    ci_low, ci_high = wilson_interval(
        positive,
        negative,
    )

    if not observations:
        return {
            "observations": 0,
            "positive": 0,
            "negative": 0,
            "flat": 0,
            "directional_observations": 0,
            "accuracy_pct": None,
            "edge_pp": None,
            "mean_signed_return_pct": None,
            "median_signed_return_pct": None,
            "mean_abs_return_pct": None,
            "ci95_low_pct": None,
            "ci95_high_pct": None,
        }

    signed = [
        observation.signed_return
        for observation in observations
    ]

    absolute = [
        observation.abs_return
        for observation in observations
    ]

    return {
        "observations": len(observations),
        "positive": positive,
        "negative": negative,
        "flat": flat,
        "directional_observations": directional_n,
        "accuracy_pct": (
            positive / directional_n * 100.0
            if directional_n
            else None
        ),
        "edge_pp": (
            positive / directional_n * 100.0 - 50.0
            if directional_n
            else None
        ),
        "mean_signed_return_pct": (
            sum(signed) / len(signed) * 100.0
        ),
        "median_signed_return_pct": (
            median(signed) * 100.0
        ),
        "mean_abs_return_pct": (
            sum(absolute) / len(absolute) * 100.0
        ),
        "ci95_low_pct": (
            ci_low * 100.0
            if ci_low is not None
            else None
        ),
        "ci95_high_pct": (
            ci_high * 100.0
            if ci_high is not None
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

def candidate_key(observation: Observation) -> tuple:
    """
    Return the structural identity of a directional candidate.

    Chronological research periods are evaluation partitions, not
    candidate-defining features. Candidate identity must therefore
    remain constant across historical training periods.

    Candidate identity:
        timeframe | volatility bucket | direction | horizon
    """

    return (
        observation.timeframe,
        observation.bucket,
        observation.direction,
        observation.horizon_candles,
    )


def key_string(key: tuple) -> str:
    timeframe, bucket, direction, horizon = key

    return (
        f"{timeframe}|{bucket}|"
        f"{direction}|H{horizon}"
    )


def training_periods(
    train_start: datetime,
    train_end: datetime,
) -> list[tuple[str, datetime, datetime]]:
    result = []

    for name, raw_start, raw_end in PERIODS:
        period_start = parse_datetime(raw_start)
        period_end = parse_datetime(raw_end)

        start = max(period_start, train_start)
        end = min(period_end, train_end)

        if start < end:
            result.append((name, start, end))

    return result


def select_training_candidates(
    observations: list[Observation],
    train_start: datetime,
    train_end: datetime,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Select candidates using training data only.

    Selection rule:

    1. At least two adequate historical training periods.
    2. Positive training edge in a strict majority of adequate periods.
    3. Mean training edge >= MIN_TRAIN_EDGE_PP.
    4. No adequate training period may have a material reversal
       worse than -MAX_TRAIN_REVERSAL_PP.

    Negative-edge candidates are NOT inverted. They are rejected because
    the research question is whether the original directional relationship
    survives chronologically.
    """

    train_observations = [
        observation
        for observation in observations
        if (
            train_start
            <= observation.signal_timestamp
            < train_end
        )
    ]

    groups: dict[tuple, list[Observation]] = {}

    for observation in train_observations:
        groups.setdefault(
            candidate_key(observation),
            [],
        ).append(observation)

    selected = []
    rejected = []

    periods = training_periods(
        train_start,
        train_end,
    )

    for key, group in sorted(groups.items()):
        period_results = []

        for name, period_start, period_end in periods:
            subset = [
                observation
                for observation in group
                if (
                    period_start
                    <= observation.signal_timestamp
                    < period_end
                )
            ]

            stats = directional_stats(subset)

            period_results.append(
                {
                    "period": name,
                    "start": iso(period_start),
                    "end": iso(period_end),
                    **stats,
                }
            )

        adequate = [
            period
            for period in period_results
            if period["directional_observations"]
            >= MIN_PERIOD_SAMPLE
        ]

        reasons = []

        if len(adequate) < 2:
            reasons.append(
                "insufficient_training_periods"
            )

        edges = [
            period["edge_pp"]
            for period in adequate
            if period["edge_pp"] is not None
        ]

        positive_periods = sum(
            1
            for edge in edges
            if edge > 0.0
        )

        negative_periods = sum(
            1
            for edge in edges
            if edge < 0.0
        )

        mean_training_edge = (
            sum(edges) / len(edges)
            if edges
            else None
        )

        worst_training_edge = (
            min(edges)
            if edges
            else None
        )

        if not edges:
            reasons.append(
                "no_training_edge"
            )
        else:
            if positive_periods <= len(edges) / 2:
                reasons.append(
                    "positive_edge_not_in_majority"
                )

            if (
                mean_training_edge is not None
                and mean_training_edge < MIN_TRAIN_EDGE_PP
            ):
                reasons.append(
                    "mean_training_effect_below_threshold"
                )

            if (
                worst_training_edge is not None
                and worst_training_edge
                < -MAX_TRAIN_REVERSAL_PP
            ):
                reasons.append(
                    "material_training_reversal"
                )

        train_stats = directional_stats(group)

        result = {
            "candidate": key_string(key),
            "timeframe": key[0],
            "bucket": key[1],
            "direction": key[2],
            "horizon_candles": key[3],
            "training_periods": period_results,
            "adequate_periods": len(adequate),
            "positive_training_periods": positive_periods,
            "negative_training_periods": negative_periods,
            "mean_training_edge_pp": mean_training_edge,
            "worst_training_edge_pp": worst_training_edge,
            "training": train_stats,
            "selection_rule": {
                "minimum_period_sample": MIN_PERIOD_SAMPLE,
                "minimum_adequate_periods": 2,
                "minimum_mean_training_edge_pp": (
                    MIN_TRAIN_EDGE_PP
                ),
                "positive_edge_majority": True,
                "maximum_material_reversal_pp": (
                    MAX_TRAIN_REVERSAL_PP
                ),
                "training_only": True,
            },
        }

        if not reasons:
            result["selection_status"] = "SELECTED"
            selected.append(result)
        else:
            result["selection_status"] = "REJECTED"
            result["rejection_reasons"] = reasons
            rejected.append(result)

    return selected, rejected


# ---------------------------------------------------------------------------
# Dependence
# ---------------------------------------------------------------------------

def lag1_autocorrelation(values: list[float]) -> float | None:
    if len(values) < 3:
        return None

    mean_value = sum(values) / len(values)

    numerator = sum(
        (values[i] - mean_value)
        * (values[i - 1] - mean_value)
        for i in range(1, len(values))
    )

    denominator = sum(
        (value - mean_value) ** 2
        for value in values
    )

    if denominator <= 0:
        return 0.0

    return numerator / denominator


def effective_sample_size(
    n: int,
    rho1: float | None,
) -> float | None:
    if n <= 0:
        return None

    if rho1 is None:
        return float(n)

    denominator = 1.0 + 2.0 * rho1

    if denominator <= 0:
        return float(n)

    return n / denominator


# ---------------------------------------------------------------------------
# Moving block bootstrap
# ---------------------------------------------------------------------------

def moving_block_bootstrap_mean(
    values: list[float],
    *,
    horizon: int,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    n = len(values)

    if n < 10:
        return {
            "replicates": 0,
            "block_length": None,
            "ci95_low_pct": None,
            "ci95_high_pct": None,
            "probability_mean_gt_zero": None,
        }

    block_length = min(
        60,
        max(
            horizon,
            round(math.sqrt(n)),
            5,
        ),
    )

    blocks = [
        values[i:i + block_length]
        for i in range(
            0,
            n - block_length + 1,
        )
    ]

    if not blocks:
        return {
            "replicates": 0,
            "block_length": block_length,
            "ci95_low_pct": None,
            "ci95_high_pct": None,
            "probability_mean_gt_zero": None,
        }

    rng = random.Random(seed)

    bootstrap_means = []

    target = n

    for _ in range(replicates):
        sample: list[float] = []

        while len(sample) < target:
            block = rng.choice(blocks)
            sample.extend(block)

        sample = sample[:target]

        bootstrap_means.append(
            sum(sample) / target
        )

    bootstrap_means.sort()

    low_index = int(
        0.025 * (replicates - 1)
    )

    high_index = int(
        0.975 * (replicates - 1)
    )

    probability_positive = (
        sum(
            value > 0.0
            for value in bootstrap_means
        )
        / replicates
    )

    return {
        "replicates": replicates,
        "block_length": block_length,
        "ci95_low_pct": (
            bootstrap_means[low_index] * 100.0
        ),
        "ci95_high_pct": (
            bootstrap_means[high_index] * 100.0
        ),
        "probability_mean_gt_zero": probability_positive,
    }


# ---------------------------------------------------------------------------
# Economic evaluation
# ---------------------------------------------------------------------------

def executable_return(
    observation: Observation,
    slippage_bps: float,
) -> float:
    """
    Realistic executable return.

    LONG:
        entry = ASK
        exit  = BID

    SHORT:
        entry = BID
        exit  = ASK
    """

    slip = slippage_bps / 10000.0

    if observation.direction == "bullish":
        entry = observation.ask_entry * (1.0 + slip)
        exit_price = observation.bid_exit * (1.0 - slip)

        return (
            exit_price - entry
        ) / entry

    entry = observation.bid_entry * (1.0 - slip)
    exit_price = observation.ask_exit * (1.0 + slip)

    return (
        entry - exit_price
    ) / entry


def economic_evaluation(
    observations: list[Observation],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    scenarios = []

    for slippage in SLIPPAGE_BPS:
        returns = [
            executable_return(
                observation,
                slippage,
            )
            for observation in observations
        ]

        wins = sum(
            value > 0
            for value in returns
        )

        mean_return = (
            sum(returns) / len(returns)
            if returns
            else None
        )

        median_return = (
            median(returns)
            if returns
            else None
        )

        rho1 = lag1_autocorrelation(returns)

        bootstrap = (
            moving_block_bootstrap_mean(
                returns,
                horizon=observations[0].horizon_candles,
                replicates=bootstrap_replicates,
                seed=seed + int(slippage * 100),
            )
            if returns
            else {}
        )

        scenarios.append(
            {
                "slippage_bps_per_side": slippage,
                "observations": len(returns),
                "mean_return_pct": (
                    mean_return * 100.0
                    if mean_return is not None
                    else None
                ),
                "median_return_pct": (
                    median_return * 100.0
                    if median_return is not None
                    else None
                ),
                "win_rate_pct": (
                    wins / len(returns) * 100.0
                    if returns
                    else None
                ),
                "lag1_autocorrelation": rho1,
                "effective_sample_size": effective_sample_size(
                    len(returns),
                    rho1,
                ),
                "bootstrap": bootstrap,
            }
        )

    return {
        "scenarios": scenarios,
    }


# ---------------------------------------------------------------------------
# OOS classification
# ---------------------------------------------------------------------------

def classify_oos(
    *,
    directional: dict[str, Any],
    economics: dict[str, Any],
) -> str:
    n = directional["directional_observations"]

    if n < MIN_PERIOD_SAMPLE:
        return "OOS-E"

    edge = directional["edge_pp"]

    if edge is None:
        return "OOS-E"

    if edge < -MIN_TRAIN_EDGE_PP:
        return "OOS-D"

    one_bps = next(
        (
            scenario
            for scenario in economics["scenarios"]
            if scenario["slippage_bps_per_side"] == 1.0
        ),
        None,
    )

    if one_bps is None:
        return "OOS-C"

    mean_1bps = one_bps["mean_return_pct"]
    ci_low = one_bps["bootstrap"]["ci95_low_pct"]

    if (
        edge >= MIN_TRAIN_EDGE_PP
        and mean_1bps is not None
        and mean_1bps > 0.0
        and ci_low is not None
        and ci_low > 0.0
    ):
        return "OOS-A"

    if edge > 0.0:
        return "OOS-B"

    return "OOS-C"


# ---------------------------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------------------------

def evaluate_window(
    observations_by_timeframe: dict[str, list[Observation]],
    window: dict[str, str],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    train_start = parse_datetime(window["train_start"])
    train_end = parse_datetime(window["train_end"])
    oos_start = parse_datetime(window["oos_start"])
    oos_end = parse_datetime(window["oos_end"])

    selected_all = []
    rejected_all = []
    evaluated = []

    for timeframe, observations in sorted(
        observations_by_timeframe.items()
    ):
        selected, rejected = select_training_candidates(
            observations,
            train_start,
            train_end,
        )

        selected_all.extend(selected)
        rejected_all.extend(rejected)

        observation_map: dict[tuple, list[Observation]] = {}

        for observation in observations:
            observation_map.setdefault(
                candidate_key(observation),
                [],
            ).append(observation)

        for selection in selected:
            key = (
    selection["timeframe"],
    selection["bucket"],
    selection["direction"],
    selection["horizon_candles"],
)

            oos_observations = [
                observation
                for observation in observation_map.get(
                    key,
                    [],
                )
                if (
                    oos_start
                    <= observation.signal_timestamp
                    < oos_end
                )
            ]

            directional = directional_stats(
                oos_observations
            )

            signed_values = [
                observation.signed_return
                for observation in oos_observations
            ]

            rho1 = lag1_autocorrelation(
                signed_values
            )

            bootstrap = (
                moving_block_bootstrap_mean(
                    signed_values,
                    horizon=key[3],
                    replicates=bootstrap_replicates,
                    seed=seed,
                )
                if signed_values
                else {}
            )

            economics = economic_evaluation(
                oos_observations,
                bootstrap_replicates=bootstrap_replicates,
                seed=seed,
            )

            classification = classify_oos(
                directional=directional,
                economics=economics,
            )

            evaluated.append(
                {
                    "candidate": selection["candidate"],
                    "timeframe": timeframe,
                    "bucket": key[1],
                    "direction": key[2],
                    "horizon_candles": key[3],
                    "training": selection["training"],
                    "training_periods": selection["training_periods"],
                    "oos": {
                        "start": iso(oos_start),
                        "end": iso(oos_end),
                        **directional,
                        "lag1_autocorrelation": rho1,
                        "effective_sample_size": effective_sample_size(
                            directional[
                                "directional_observations"
                            ],
                            rho1,
                        ),
                        "moving_block_bootstrap": bootstrap,
                    },
                    "economic_oos": economics,
                    "oos_classification": classification,
                }
            )

    return {
        "window": {
            **window,
            "train_start": iso(train_start),
            "train_end": iso(train_end),
            "oos_start": iso(oos_start),
            "oos_end": iso(oos_end),
        },
        "candidate_selection": {
            "selected": len(selected_all),
            "rejected": len(rejected_all),
            "selected_candidates": selected_all,
            "rejected_candidates": rejected_all,
            "selection_rule": {
                "minimum_period_sample": MIN_PERIOD_SAMPLE,
                "minimum_adequate_periods": 2,
                "minimum_mean_training_edge_pp": (
                    MIN_TRAIN_EDGE_PP
                ),
                "positive_edge_majority": True,
                "maximum_material_reversal_pp": (
                    MAX_TRAIN_REVERSAL_PP
                ),
                "training_only": True,
            },
        },
        "selected_candidates": selected_all,
        "evaluations": evaluated,
    }


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def write_csv(
    path: Path,
    evaluations: list[dict[str, Any]],
) -> None:
    rows = []

    for evaluation in evaluations:
        row = {
            "walkforward_id": evaluation["window"]["id"],
            "candidate": evaluation["candidate"],
            "timeframe": evaluation["timeframe"],
            "bucket": evaluation["bucket"],
            "direction": evaluation["direction"],
            "horizon_candles": evaluation["horizon_candles"],
            "train_mean_edge_pp": evaluation[
                "training"
            ]["edge_pp"],
            "train_accuracy_pct": evaluation[
                "training"
            ]["accuracy_pct"],
            "train_n": evaluation[
                "training"
            ]["directional_observations"],
            "oos_edge_pp": evaluation[
                "oos"
            ]["edge_pp"],
            "oos_accuracy_pct": evaluation[
                "oos"
            ]["accuracy_pct"],
            "oos_mean_signed_return_pct": evaluation[
                "oos"
            ]["mean_signed_return_pct"],
            "oos_median_signed_return_pct": evaluation[
                "oos"
            ]["median_signed_return_pct"],
            "oos_n": evaluation[
                "oos"
            ]["directional_observations"],
            "oos_rho1": evaluation[
                "oos"
            ]["lag1_autocorrelation"],
            "oos_ess": evaluation[
                "oos"
            ]["effective_sample_size"],
            "oos_bootstrap_ci_low_pct": evaluation[
                "oos"
            ]["moving_block_bootstrap"
            ].get("ci95_low_pct"),
            "oos_bootstrap_ci_high_pct": evaluation[
                "oos"
            ]["moving_block_bootstrap"
            ].get("ci95_high_pct"),
            "oos_bootstrap_p_gt_zero": evaluation[
                "oos"
            ]["moving_block_bootstrap"
            ].get("probability_mean_gt_zero"),
            "oos_classification": evaluation[
                "oos_classification"
            ],
        }

        for scenario in evaluation[
            "economic_oos"
        ]["scenarios"]:
            suffix = str(
                scenario["slippage_bps_per_side"]
            ).replace(".", "_")

            row[
                f"slip_{suffix}_bps_mean_return_pct"
            ] = scenario["mean_return_pct"]

            row[
                f"slip_{suffix}_bps_win_rate_pct"
            ] = scenario["win_rate_pct"]

            row[
                f"slip_{suffix}_bps_ci_low_pct"
            ] = scenario[
                "bootstrap"
            ].get("ci95_low_pct")

            row[
                f"slip_{suffix}_bps_ci_high_pct"
            ] = scenario[
                "bootstrap"
            ].get("ci95_high_pct")

            row[
                f"slip_{suffix}_bps_p_gt_zero"
            ] = scenario[
                "bootstrap"
            ].get("probability_mean_gt_zero")

        rows.append(row)

    if not rows:
        return

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(
    windows: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluations = [
        evaluation
        for window in windows
        for evaluation in window["evaluations"]
    ]

    classifications = {
        label: sum(
            evaluation["oos_classification"] == label
            for evaluation in evaluations
        )
        for label in (
            "OOS-A",
            "OOS-B",
            "OOS-C",
            "OOS-D",
            "OOS-E",
        )
    }

    return {
        "walkforward_windows": len(windows),
        "total_oos_evaluations": len(evaluations),
        "classification_counts": classifications,
        "economically_positive_at_1bps": sum(
            any(
                scenario[
                    "slippage_bps_per_side"
                ] == 1.0
                and scenario[
                    "mean_return_pct"
                ] is not None
                and scenario[
                    "mean_return_pct"
                ] > 0.0
                for scenario in evaluation[
                    "economic_oos"
                ]["scenarios"]
            )
            for evaluation in evaluations
        ),
        "economically_positive_ci_lower_at_1bps": sum(
            any(
                scenario[
                    "slippage_bps_per_side"
                ] == 1.0
                and scenario[
                    "bootstrap"
                ].get("ci95_low_pct") is not None
                and scenario[
                    "bootstrap"
                ]["ci95_low_pct"] > 0.0
                for scenario in evaluation[
                    "economic_oos"
                ]["scenarios"]
            )
            for evaluation in evaluations
        ),
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_results(
    result: dict[str, Any],
) -> None:
    print()
    print("=" * 110)
    print(
        "PHASE 6.3B STEP 6 — CHRONOLOGICAL "
        "OUT-OF-SAMPLE VALIDATION"
    )
    print("=" * 110)
    print()

    for window in result["windows"]:
        info = window["window"]
        selection = window["candidate_selection"]

        print("-" * 110)
        print(
            f"{info['id']} | "
            f"TRAIN {info['train_start']} -> "
            f"{info['train_end']} | "
            f"OOS {info['oos_start']} -> "
            f"{info['oos_end']}"
        )
        print("-" * 110)

        print(
            f"Training candidates selected: "
            f"{selection['selected']}"
        )
        print(
            f"Training candidates rejected: "
            f"{selection['rejected']}"
        )
        print()

        for evaluation in window["evaluations"]:
            oos = evaluation["oos"]

            one_bps = next(
                (
                    scenario
                    for scenario in evaluation[
                        "economic_oos"
                    ]["scenarios"]
                    if scenario[
                        "slippage_bps_per_side"
                    ] == 1.0
                ),
                None,
            )

            one_bps_text = "N/A"

            if one_bps is not None:
                mean_value = one_bps[
                    "mean_return_pct"
                ]

                one_bps_text = (
                    f"{mean_value:+.5f}%"
                    if mean_value is not None
                    else "N/A"
                )

            print(
                f"{evaluation['candidate']:<48} "
                f"n={oos['directional_observations']:<6} "
                f"edge={oos['edge_pp']:+.3f}pp "
                f"mean={oos['mean_signed_return_pct']:+.5f}% "
                f"1bps={one_bps_text:<12} "
                f"{evaluation['oos_classification']}"
            )

        print()

    summary = result["summary"]

    print("=" * 110)
    print("SUMMARY")
    print("=" * 110)
    print(
        f"Total OOS evaluations: "
        f"{summary['total_oos_evaluations']}"
    )

    for label, count in summary[
        "classification_counts"
    ].items():
        print(f"{label}: {count}")

    print(
        "Economically positive at 1.0 bps/side: "
        f"{summary['economically_positive_at_1bps']}"
    )

    print(
        "1.0 bps/side bootstrap CI lower > 0: "
        f"{summary['economically_positive_ci_lower_at_1bps']}"
    )

    print()
    print("IMPORTANT:")
    print(
        "Candidate selection used training data only."
    )
    print(
        "Full-sample A/A+ stability labels were NOT used "
        "for OOS selection."
    )
    print(
        "Training selection allows a non-material historical "
        "reversal but rejects material reversals."
    )
    print(
        "Negative-edge candidates were NOT inverted."
    )
    print(
        "Positive OOS evidence is not automatically a "
        "tradable strategy."
    )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 6.3B Step 6 chronological "
            "direction walk-forward validation."
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
        "--timeframe",
        choices=[
            "ALL",
            *[timeframe.name for timeframe in Timeframe],
        ],
        default="ALL",
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
        "--seed",
        type=int,
        default=BOOTSTRAP_SEED,
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    if args.timeframe == "ALL":
        timeframes = tuple(Timeframe)
    else:
        timeframes = (Timeframe[args.timeframe],)

    print()
    print("=" * 110)
    print(
        "XAUUSD — PHASE 6.3B STEP 6 "
        "CHRONOLOGICAL WALK-FORWARD"
    )
    print("=" * 110)
    print(f"Source:       Dukascopy XAUUSD BID/ASK M1")
    print(
        f"Window:       {start.isoformat()} -> "
        f"{end.isoformat()}"
    )
    print(
        "Timeframes:   "
        + ", ".join(
            timeframe.name
            for timeframe in timeframes
        )
    )
    print(
        f"ATR:          {ATR_PERIOD}"
    )
    print(
        f"ATR baseline: {BASELINE_PERIOD}"
    )
    print(
        f"Horizons:     {HORIZONS}"
    )
    print(
        f"Min period n: {MIN_PERIOD_SAMPLE}"
    )
    print(
        f"Min adequate periods: 2"
    )
    print(
        f"Min mean train edge: "
        f"{MIN_TRAIN_EDGE_PP:.1f}pp"
    )
    print(
        f"Max material reversal: "
        f"-{MAX_TRAIN_REVERSAL_PP:.1f}pp"
    )
    print(
        f"Bootstrap:    {args.bootstrap_replicates}"
    )
    print()

    adapter = DukascopyAdapter(
        args.data_root
    )

    engines = {
        timeframe: ObservationEngine(timeframe)
        for timeframe in timeframes
    }

    bars = adapter.iter_bars(
        start,
        end,
        require_ask=True,
    )

    for timeframe, candle in aggregate_timeframes(
        bars,
        timeframes,
    ):
        engines[timeframe].process(candle)

    observations_by_timeframe = {
        timeframe.name: engine.observations
        for timeframe, engine in engines.items()
    }

    print("RECONSTRUCTION")
    print("-" * 110)

    for timeframe, engine in engines.items():
        print(
            f"{timeframe.name}: "
            f"candles={engine.candles:,} | "
            f"complete={engine.complete_candles:,} | "
            f"ATR-ready={engine.atr_ready:,} | "
            f"signals={engine.signals_created:,} | "
            f"resolved={engine.signals_resolved:,} | "
            f"observations={len(engine.observations):,}"
        )

    print()

    window_results = []

    for window in WALKFORWARD_WINDOWS:
        result = evaluate_window(
            observations_by_timeframe,
            window,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )

        window_results.append(result)

    summary = build_summary(
        window_results
    )

    payload = {
        "phase": "6.3B",
        "step": "6",
        "title": (
            "Chronological Out-of-Sample "
            "Direction Validation"
        ),
        "source": (
            "Dukascopy XAUUSD BID/ASK M1"
        ),
        "methodology": {
            "signal_definition": (
                "Phase 6.3A mechanics reproduced exactly"
            ),
            "candidate_selection": (
                "training data only"
            ),
            "random_split": False,
            "future_information_used_for_selection": False,
            "economic_entry_long": "ASK open",
            "economic_exit_long": "BID close",
            "economic_entry_short": "BID open",
            "economic_exit_short": "ASK close",
            "slippage_bps_per_side": list(
                SLIPPAGE_BPS
            ),
            "bootstrap_method": (
                "moving block bootstrap"
            ),
            "bootstrap_replicates": (
                args.bootstrap_replicates
            ),
            "bootstrap_seed": args.seed,
        },
        "selection_thresholds": {
            "minimum_period_sample": (
                MIN_PERIOD_SAMPLE
            ),
            "minimum_adequate_periods": 2,
            "minimum_mean_training_edge_pp": (
                MIN_TRAIN_EDGE_PP
            ),
            "positive_edge_majority": True,
            "maximum_material_reversal_pp": (
                MAX_TRAIN_REVERSAL_PP
            ),
            "training_only": True,
        },
        "walk_forward_windows": WALKFORWARD_WINDOWS,
        "windows": window_results,
        "summary": summary,
    }

    json_path = (
        args.output_root
        / "direction_walkforward_all.json"
    )

    csv_path = (
        args.output_root
        / "direction_walkforward_all.csv"
    )

    summary_path = (
        args.output_root
        / "direction_walkforward_summary.json"
    )

    with json_path.open(
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

    evaluations = [
        evaluation
        for window in window_results
        for evaluation in window["evaluations"]
    ]

    write_csv(
        csv_path,
        evaluations,
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")

    payload["windows"] = window_results

    print_results(payload)

    print(
        f"Output directory: {args.output_root}"
    )
    print(
        f"  {json_path.name}"
    )
    print(
        f"  {csv_path.name}"
    )
    print(
        f"  {summary_path.name}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())