from __future__ import annotations

from datetime import datetime, timezone

from scripts.analyze_direction_walkforward import (
    Observation,
    classify_oos,
    directional_stats,
    effective_sample_size,
    lag1_autocorrelation,
    moving_block_bootstrap_mean,
    select_training_candidates,
    training_periods,
    executable_return,
)


def make_observation(
    *,
    timestamp: str,
    signed_return: float,
    direction: str = "bullish",
    timeframe: str = "H4",
    bucket: str = "normal",
    regime: str = "REGIME",
    horizon: int = 5,
) -> Observation:
    ts = datetime.fromisoformat(
        timestamp.replace("Z", "+00:00")
    )

    bid_entry = 100.0
    ask_entry = 100.1

    if direction == "bullish":
        bid_exit = bid_entry * (1.0 + signed_return)
        ask_exit = bid_exit + 0.1
    else:
        bid_exit = bid_entry * (1.0 - signed_return)
        ask_exit = bid_exit + 0.1

    return Observation(
        timeframe=timeframe,
        regime=regime,
        bucket=bucket,
        direction=direction,
        horizon_candles=horizon,
        signal_timestamp=ts,
        entry_timestamp=ts,
        bid_entry=bid_entry,
        ask_entry=ask_entry,
        bid_exit=bid_exit,
        ask_exit=ask_exit,
        signed_return=signed_return,
    )


def test_directional_stats_accuracy_and_edge():
    observations = [
        make_observation(
            timestamp="2010-01-01T00:00:00Z",
            signed_return=0.01,
        ),
        make_observation(
            timestamp="2010-01-01T01:00:00Z",
            signed_return=0.02,
        ),
        make_observation(
            timestamp="2010-01-01T02:00:00Z",
            signed_return=-0.01,
        ),
        make_observation(
            timestamp="2010-01-01T03:00:00Z",
            signed_return=0.0,
        ),
    ]

    result = directional_stats(observations)

    assert result["observations"] == 4
    assert result["positive"] == 2
    assert result["negative"] == 1
    assert result["flat"] == 1
    assert result["directional_observations"] == 3
    assert result["accuracy_pct"] == 2 / 3 * 100
    assert result["edge_pp"] == 2 / 3 * 100 - 50


def test_training_periods_are_clipped_to_training_window():
    start = datetime(
        2010,
        1,
        1,
        tzinfo=timezone.utc,
    )

    end = datetime(
        2019,
        1,
        1,
        tzinfo=timezone.utc,
    )

    periods = training_periods(start, end)

    assert len(periods) == 2
    assert periods[0][0] == "2010-2014"
    assert periods[1][0] == "2015-2019"
    assert periods[1][2] == end


def test_effective_sample_size_reduces_with_positive_autocorrelation():
    assert effective_sample_size(100, 0.0) == 100.0
    assert effective_sample_size(100, 0.5) < 100.0


def test_lag1_autocorrelation_constant_series_is_zero():
    values = [1.0] * 10
    assert lag1_autocorrelation(values) == 0.0


def test_bootstrap_returns_valid_interval():
    values = [
        0.01,
        0.02,
        0.015,
        0.012,
        0.018,
        0.011,
        0.017,
        0.014,
        0.013,
        0.016,
    ]

    result = moving_block_bootstrap_mean(
        values,
        horizon=5,
        replicates=200,
        seed=123,
    )

    assert result["replicates"] == 200
    assert result["block_length"] >= 5
    assert (
        result["ci95_low_pct"]
        <= result["ci95_high_pct"]
    )
    assert 0.0 <= result["probability_mean_gt_zero"] <= 1.0


def test_executable_long_return_uses_ask_entry_and_bid_exit():
    observation = make_observation(
        timestamp="2024-01-01T00:00:00Z",
        signed_return=0.01,
        direction="bullish",
    )

    result = executable_return(
        observation,
        slippage_bps=0.0,
    )

    expected = (
        observation.bid_exit
        - observation.ask_entry
    ) / observation.ask_entry

    assert result == expected


def test_executable_short_return_uses_bid_entry_and_ask_exit():
    observation = make_observation(
        timestamp="2024-01-01T00:00:00Z",
        signed_return=0.01,
        direction="bearish",
    )

    result = executable_return(
        observation,
        slippage_bps=0.0,
    )

    expected = (
        observation.bid_entry
        - observation.ask_exit
    ) / observation.bid_entry

    assert result == expected


def test_training_selection_requires_same_sign():
    observations = []

    for index in range(120):
        observations.append(
            make_observation(
                timestamp=(
                    f"2010-01-01T{index % 24:02d}:00:00Z"
                ),
                signed_return=0.03,
            )
        )

    selected, rejected = select_training_candidates(
        observations,
        datetime(
            2010,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        datetime(
            2015,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )

    assert selected == []
    assert rejected


def test_oos_negative_edge_is_contradictory():
    directional = {
        "directional_observations": 200,
        "edge_pp": -5.0,
    }

    economics = {
        "scenarios": [
            {
                "slippage_bps_per_side": 1.0,
                "mean_return_pct": -0.1,
                "bootstrap": {
                    "ci95_low_pct": -0.2,
                },
            }
        ]
    }

    assert (
        classify_oos(
            directional=directional,
            economics=economics,
        )
        == "OOS-D"
    )


def test_oos_positive_economic_edge_can_be_oos_a():
    directional = {
        "directional_observations": 500,
        "edge_pp": 5.0,
    }

    economics = {
        "scenarios": [
            {
                "slippage_bps_per_side": 1.0,
                "mean_return_pct": 0.05,
                "bootstrap": {
                    "ci95_low_pct": 0.01,
                },
            }
        ]
    }

    assert (
        classify_oos(
            directional=directional,
            economics=economics,
        )
        == "OOS-A"
    )


def test_oos_positive_but_cost_sensitive_is_oos_b():
    directional = {
        "directional_observations": 500,
        "edge_pp": 3.0,
    }

    economics = {
        "scenarios": [
            {
                "slippage_bps_per_side": 1.0,
                "mean_return_pct": -0.01,
                "bootstrap": {
                    "ci95_low_pct": -0.05,
                },
            }
        ]
    }

    assert (
        classify_oos(
            directional=directional,
            economics=economics,
        )
        == "OOS-B"
    )
