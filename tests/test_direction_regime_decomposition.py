from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from scripts.analyze_direction_regime_decomposition import (
    RECONCILIATION_TOLERANCE,
    build_persistence_map,
    interaction_from_observations,
    reconcile,
)
from scripts.analyze_direction_walkforward import Observation

UTC = timezone.utc


def candle(ts, open_price, close_price):
    return SimpleNamespace(
        timestamp=ts,
        complete=True,
        bid_open=open_price,
        bid_high=max(open_price, close_price),
        bid_low=min(open_price, close_price),
        bid_close=close_price,
    )


def observation(ts, bucket, direction, signed_return):
    return Observation(
        timeframe="M15",
        regime="2010-2014",
        bucket=bucket,
        direction=direction,
        horizon_candles=1,
        signal_timestamp=ts,
        entry_timestamp=ts + timedelta(minutes=15),
        bid_entry=100.0,
        ask_entry=100.1,
        bid_exit=100.0 * (1.0 + signed_return),
        ask_exit=100.1,
        signed_return=signed_return,
    )


def test_persistence_skips_flats_and_classifies_nonflat_sequence():
    start = datetime(2020, 1, 1, tzinfo=UTC)
    candles = [
        candle(start, 100, 101),
        candle(start + timedelta(minutes=15), 101, 101),  # flat: ignored
        candle(start + timedelta(minutes=30), 101, 102),
        candle(start + timedelta(minutes=45), 102, 101),
        candle(start + timedelta(minutes=60), 101, 100),
    ]

    result = build_persistence_map(candles)

    assert result[candles[0].timestamp] == "unclassified"
    assert result[candles[2].timestamp] == "persistent"
    assert result[candles[3].timestamp] == "reversal"
    assert result[candles[4].timestamp] == "persistent"


def test_reconciliation_reproduces_exact_phase_6_3b_interaction():
    start = datetime(2020, 1, 1, tzinfo=UTC)
    observations = [
        observation(start + timedelta(minutes=0), "extreme", "bullish", 0.02),
        observation(start + timedelta(minutes=15), "extreme", "bearish", -0.01),
        observation(start + timedelta(minutes=30), "normal", "bullish", 0.005),
        observation(start + timedelta(minutes=45), "normal", "bearish", -0.004),
    ]

    persistent = [observations[0], observations[2]]
    reversal = [observations[1], observations[3]]

    original = interaction_from_observations(observations)
    pooled = interaction_from_observations([*persistent, *reversal])

    assert pooled[0] == pytest.approx(original[0], abs=RECONCILIATION_TOLERANCE)
    assert pooled[1] == pytest.approx(original[1], abs=RECONCILIATION_TOLERANCE)

    result = reconcile(observations, persistent, reversal)
    assert result["passed"] is True
    assert result["original_n"] == 4
    assert result["pooled_n"] == 4
    assert result["timestamps_match"] is True


def test_reconciliation_detects_missing_subgroup_observation():
    start = datetime(2020, 1, 1, tzinfo=UTC)
    observations = [
        observation(start + timedelta(minutes=0), "extreme", "bullish", 0.02),
        observation(start + timedelta(minutes=15), "extreme", "bearish", -0.01),
        observation(start + timedelta(minutes=30), "normal", "bullish", 0.005),
        observation(start + timedelta(minutes=45), "normal", "bearish", -0.004),
    ]

    result = reconcile(
        observations,
        observations[:2],
        observations[2:3],
    )

    assert result["passed"] is False
    assert result["original_n"] == 4
    assert result["pooled_n"] == 3
