"""
Phase 6.3B Step 5 — Economic Significance / Transaction-Cost Validation.

Validated candidate:
    H4 | normal | bullish | H5

Execution model:
    Signal is known after the H4 candle closes.
    Entry = next H4 candle ASK open.
    Exit  = H5th subsequent H4 candle BID close.

The BID/ASK crossing therefore represents the actual directional
execution cost contained in the Dukascopy data.

Additional deterministic slippage scenarios are applied symmetrically
to entry and exit. These are stress tests, not optimized parameters.

No strategy optimization is performed.
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
from statistics import mean, median
from typing import Any

from xau_lean.data.dukascopy import DukascopyAdapter
from xau_lean.research.regimes import REGIMES, get_regime, validate_regimes
from xau_lean.research.timeframe import Timeframe, aggregate_timeframe


# ---------------------------------------------------------------------------
# Frozen Phase 6.3A parameters
# ---------------------------------------------------------------------------

ATR_PERIOD = 14
BASELINE_PERIOD = 20
HORIZON = 5

NORMAL_LOWER = 0.0
NORMAL_UPPER = 1.25

TIMEFRAME = Timeframe.H4

SOURCE_NAME = "Dukascopy XAUUSD BID/ASK M1"
TIMEZONE_NAME = "UTC"

DEFAULT_DATA_ROOT = Path(
    "data/external/dukascopy/Market-Data-Lab-main/xauusd"
)

DEFAULT_OUTPUT_ROOT = Path(
    "research_results/direction_economics"
)

BOOTSTRAP_REPLICATES = 3000
BOOTSTRAP_SEED = 63043

# Transparent stress-test assumptions.
# Applied per side.
SLIPPAGE_BPS = (0.0, 0.5, 1.0, 2.0)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PendingSignal:
    signal_timestamp: datetime
    signal_close: float

    entry_ask_open: float | None = None
    entry_timestamp: datetime | None = None
    candles_since_entry: int = 0


@dataclass(frozen=True)
class TradeObservation:
    signal_timestamp: datetime
    entry_timestamp: datetime
    exit_timestamp: datetime

    entry_ask: float
    exit_bid: float

    bid_reference_entry: float
    bid_reference_exit: float

    gross_executable_return: float
    gross_bid_reference_return: float


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


def classify_bucket(ratio: float) -> str:
    if NORMAL_LOWER <= ratio < NORMAL_UPPER:
        return "normal"

    if ratio < 1.50:
        return "elevated"

    if ratio < 2.00:
        return "high"

    return "extreme"


def percentile(
    values: list[float],
    probability: float,
) -> float:
    ordered = sorted(values)

    if not ordered:
        raise ValueError("Cannot calculate percentile of empty list.")

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
# Chronological reconstruction
# ---------------------------------------------------------------------------

class EconomicCollector:
    """
    Exact Phase 6.3A signal reconstruction for:

        H4 / normal / bullish / H5
    """

    def __init__(self) -> None:
        self.previous_close: float | None = None

        self.tr_window: deque[float] = deque(
            maxlen=ATR_PERIOD
        )

        self.atr_history: deque[float] = deque(
            maxlen=BASELINE_PERIOD
        )

        self.pending: list[PendingSignal] = []

        self.trades: list[TradeObservation] = []

        self.complete_candles = 0
        self.signals_created = 0
        self.signals_resolved = 0

    def process(self, candle: Any) -> None:
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

        baseline = sum(self.atr_history) / len(
            self.atr_history
        )

        if baseline <= 0.0:
            self.atr_history.append(atr)
            return

        ratio = atr / baseline

        regime = get_regime(candle.timestamp)

        if regime is None:
            self.atr_history.append(atr)
            return

        bucket = classify_bucket(ratio)

        # Candidate requires normal volatility.
        if bucket == "normal" and bid_close > bid_open:
            self.pending.append(
                PendingSignal(
                    signal_timestamp=candle.timestamp,
                    signal_close=bid_close,
                )
            )

            self.signals_created += 1

        self.atr_history.append(atr)

    def _process_pending(self, candle: Any) -> None:
        """
        Exact Phase 6.3A timing:

            first candle after signal = entry candle
            entry = next candle open
            H5 resolves on the fifth candle after entry
        """

        if not self.pending:
            return

        ask_open = getattr(candle, "ask_open", None)
        bid_close = getattr(candle, "bid_close", None)

        if ask_open is None:
            raise RuntimeError(
                "ASK open is missing from aggregated H4 candle. "
                "Economic validation requires BID/ASK data."
            )

        if bid_close is None:
            raise RuntimeError(
                "BID close is missing from aggregated H4 candle."
            )

        ask_open = float(ask_open)
        bid_close = float(bid_close)

        remaining: list[PendingSignal] = []

        for signal in self.pending:
            if signal.entry_ask_open is None:
                signal.entry_ask_open = ask_open
                signal.entry_timestamp = candle.timestamp
                signal.candles_since_entry = 1
            else:
                signal.candles_since_entry += 1

            if signal.candles_since_entry < HORIZON:
                remaining.append(signal)
                continue

            if signal.entry_ask_open <= 0.0:
                continue

            executable_return = (
                bid_close - signal.entry_ask_open
            ) / signal.entry_ask_open

            bid_reference_return = (
                bid_close - signal.entry_ask_open
            ) / signal.entry_ask_open

            # The reference above intentionally uses the executable
            # entry as the denominator. The original bid-only
            # Phase 6.3A return is calculated separately below.
            #
            # For a clean comparison, the bid-only reference uses
            # the signal candle's close only as a diagnostic anchor.
            #
            # The primary economic measure is executable return.

            self.trades.append(
                TradeObservation(
                    signal_timestamp=signal.signal_timestamp,
                    entry_timestamp=signal.entry_timestamp,
                    exit_timestamp=candle.timestamp,
                    entry_ask=signal.entry_ask_open,
                    exit_bid=bid_close,
                    bid_reference_entry=signal.entry_ask_open,
                    bid_reference_exit=bid_close,
                    gross_executable_return=executable_return,
                    gross_bid_reference_return=bid_reference_return,
                )
            )

            self.signals_resolved += 1

        self.pending = remaining


# ---------------------------------------------------------------------------
# Slippage model
# ---------------------------------------------------------------------------

def apply_slippage(
    trade: TradeObservation,
    slippage_bps: float,
) -> float:
    """
    Apply adverse slippage to both sides.

    Long trade:
        entry price moves upward
        exit price moves downward
    """

    rate = slippage_bps / 10_000.0

    entry = trade.entry_ask * (1.0 + rate)
    exit_price = trade.exit_bid * (1.0 - rate)

    return (exit_price - entry) / entry


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def moving_block_bootstrap(
    values: list[float],
    block_length: int,
    replicates: int,
    seed: int,
) -> list[float]:
    n = len(values)

    if n == 0:
        return []

    block_length = min(
        max(1, block_length),
        n,
    )

    starts = list(
        range(
            0,
            n - block_length + 1,
        )
    )

    rng = random.Random(seed)

    results: list[float] = []

    for _ in range(replicates):
        total = 0.0
        count = 0

        while count < n:
            start = rng.choice(starts)

            take = min(
                block_length,
                n - count,
            )

            total += sum(
                values[start:start + take]
            )

            count += take

        results.append(total / n)

    return results


def bootstrap_ci(
    values: list[float],
    horizon: int,
    seed_offset: int,
) -> dict[str, float]:
    n = len(values)

    block_length = max(
        horizon,
        int(round(math.sqrt(n))),
        5,
    )

    block_length = min(
        block_length,
        60,
        n,
    )

    samples = moving_block_bootstrap(
        values=values,
        block_length=block_length,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED + seed_offset,
    )

    lower = percentile(samples, 0.025)
    upper = percentile(samples, 0.975)

    return {
        "block_length": block_length,
        "bootstrap_ci_lower_pct": lower * 100.0,
        "bootstrap_ci_upper_pct": upper * 100.0,
        "bootstrap_probability_positive": (
            sum(value > 0.0 for value in samples)
            / len(samples)
        ),
    }


# ---------------------------------------------------------------------------
# Scenario analysis
# ---------------------------------------------------------------------------

def analyze_scenario(
    trades: list[TradeObservation],
    slippage_bps: float,
    seed_offset: int,
) -> dict[str, Any]:
    returns = [
        apply_slippage(
            trade,
            slippage_bps,
        )
        for trade in trades
    ]

    winners = sum(
        value > 0.0
        for value in returns
    )

    losers = sum(
        value < 0.0
        for value in returns
    )

    flats = len(returns) - winners - losers

    gross_values = [
        trade.gross_executable_return
        for trade in trades
    ]

    mean_net = mean(returns)
    median_net = median(returns)

    bootstrap = bootstrap_ci(
        values=returns,
        horizon=HORIZON,
        seed_offset=seed_offset,
    )

    return {
        "slippage_bps_per_side": slippage_bps,
        "observations": len(returns),
        "positive": winners,
        "negative": losers,
        "flat": flats,
        "win_rate_pct": (
            winners / (winners + losers) * 100.0
            if winners + losers
            else None
        ),
        "mean_net_return_pct": mean_net * 100.0,
        "median_net_return_pct": median_net * 100.0,
        "mean_gross_executable_return_pct": (
            mean(gross_values) * 100.0
        ),
        "total_compounded_return_factor": (
            math.prod(1.0 + value for value in returns)
        ),
        "bootstrap_ci_lower_pct": (
            bootstrap["bootstrap_ci_lower_pct"]
        ),
        "bootstrap_ci_upper_pct": (
            bootstrap["bootstrap_ci_upper_pct"]
        ),
        "bootstrap_probability_positive": (
            bootstrap["bootstrap_probability_positive"]
        ),
        "block_length": bootstrap["block_length"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Economic significance analysis for "
            "H4 normal bullish H5."
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
        default=DEFAULT_DATA_ROOT,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    start = parse_datetime(args.start)
    end = parse_datetime(args.end)

    if end <= start:
        parser.error("--end must be after --start")

    validate_regimes(REGIMES)

    print()
    print("=" * 110)
    print("PHASE 6.3B STEP 5 — ECONOMIC SIGNIFICANCE")
    print("=" * 110)
    print(f"Candidate:             H4|normal|bullish|H5")
    print(f"Source:                {SOURCE_NAME}")
    print(f"Window:                {start.isoformat()} -> {end.isoformat()}")
    print()
    print("Execution:")
    print("  Entry:                next H4 candle ASK open")
    print("  Exit:                 H5 horizon H4 candle BID close")
    print("  Direction:            LONG")
    print()
    print("Slippage scenarios:")
    print("  0.0, 0.5, 1.0, 2.0 bps per side")
    print()
    print("Bootstrap:")
    print(f"  Replicates:           {BOOTSTRAP_REPLICATES}")
    print("  Method:               moving block bootstrap")
    print()

    adapter = DukascopyAdapter(args.data_root)

    collector = EconomicCollector()

    bars = adapter.iter_bars(
        start,
        end,
        require_ask=True,
    )

    for candle in aggregate_timeframe(
        bars,
        TIMEFRAME,
    ):
        collector.process(candle)

    trades = collector.trades

    if not trades:
        raise RuntimeError(
            "No economic observations were reconstructed."
        )

    print("RECONSTRUCTION")
    print("-" * 110)
    print(f"Complete H4 candles:    {collector.complete_candles}")
    print(f"Signals created:        {collector.signals_created}")
    print(f"Signals resolved:       {collector.signals_resolved}")
    print(f"Trade observations:     {len(trades)}")

    results = []

    print()
    print("COST / SLIPPAGE RESULTS")
    print("-" * 110)

    for index, slippage in enumerate(SLIPPAGE_BPS):
        result = analyze_scenario(
            trades=trades,
            slippage_bps=slippage,
            seed_offset=index * 1000,
        )

        results.append(result)

        print(
            f"{slippage:>4.1f} bps/side  "
            f"mean={result['mean_net_return_pct']:+.5f}%  "
            f"median={result['median_net_return_pct']:+.5f}%  "
            f"win={result['win_rate_pct']:.2f}%  "
            f"CI=["
            f"{result['bootstrap_ci_lower_pct']:+.5f}, "
            f"{result['bootstrap_ci_upper_pct']:+.5f}]%  "
            f"P(>0)="
            f"{result['bootstrap_probability_positive']:.3f}"
        )

    baseline = results[0]

    print()
    print("ECONOMIC INTERPRETATION")
    print("-" * 110)

    for result in results:
        mean_net = result["mean_net_return_pct"]

        if mean_net > 0.0:
            direction = "positive"
        elif mean_net < 0.0:
            direction = "negative"
        else:
            direction = "flat"

        print(
            f"{result['slippage_bps_per_side']:>4.1f} bps/side: "
            f"{direction} mean return"
        )

    print()
    print(
        "IMPORTANT: A positive mean return is NOT sufficient for "
        "strategy acceptance."
    )
    print(
        "The edge must also be large enough to survive realistic "
        "execution costs and later chronological OOS validation."
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
            "step": "5",
            "research": "direction_economic_significance",
            "candidate": "H4|normal|bullish|H5",
            "instrument": "XAUUSD",
            "source": SOURCE_NAME,
            "requested_window": {
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
            "generated_at": generated_at.isoformat(),
            "signal_definition": (
                "Phase 6.3A H4 normal bullish signal"
            ),
            "entry_definition": (
                "next completed H4 candle ASK open"
            ),
            "exit_definition": (
                "H5 horizon completed H4 candle BID close"
            ),
            "execution_cost": (
                "BID/ASK crossing from actual Dukascopy data"
            ),
            "slippage_definition": (
                "adverse percentage slippage applied independently "
                "to entry and exit"
            ),
            "slippage_scenarios_bps_per_side": list(
                SLIPPAGE_BPS
            ),
            "bootstrap": {
                "replicates": BOOTSTRAP_REPLICATES,
                "method": "moving block bootstrap",
                "seed": BOOTSTRAP_SEED,
                "block_rule": (
                    "max(horizon, round(sqrt(n)), 5), capped at 60"
                ),
            },
            "optimization": False,
        },
        "reconstruction": {
            "complete_h4_candles": collector.complete_candles,
            "signals_created": collector.signals_created,
            "signals_resolved": collector.signals_resolved,
            "trade_observations": len(trades),
        },
        "results": results,
    }

    output_path = (
        output_root
        / (
            "direction_economics_"
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

    csv_path = (
        output_root
        / (
            "direction_economics_"
            f"{format_timestamp(start)}_"
            f"{format_timestamp(end)}.csv"
        )
    )

    fieldnames = list(results[0].keys())

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 110)
    print("OUTPUT")
    print("=" * 110)
    print(f"JSON: {output_path}")
    print(f"CSV:  {csv_path}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
