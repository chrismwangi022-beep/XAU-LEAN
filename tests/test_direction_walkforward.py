from datetime import datetime, timedelta, timezone

from scripts.analyze_direction_walkforward import (
    MAX_TRAIN_REVERSAL_PP,
    MIN_PERIOD_SAMPLE,
    MIN_TRAIN_EDGE_PP,
    Observation,
    candidate_key,
    classify_oos,
    directional_stats,
    economic_evaluation,
    effective_sample_size,
    executable_return,
    key_string,
    lag1_autocorrelation,
    moving_block_bootstrap_mean,
    select_training_candidates,
    training_periods,
    wilson_interval,
)


UTC = timezone.utc


def make_observation(
    *,
    timestamp: datetime,
    signed_return: float,
    timeframe: str = "H4",
    bucket: str = "normal",
    regime: str = "REGIME",
    direction: str = "bullish",
    horizon: int = 5,
    bid_entry: float = 100.0,
    ask_entry: float = 100.1,
    bid_exit: float | None = None,
    ask_exit: float | None = None,
) -> Observation:
    """
    Construct an Observation using the actual production dataclass API.

    No production helper is assumed or invented.
    """
    if bid_exit is None:
        bid_exit = bid_entry * (1.0 + signed_return)

    if ask_exit is None:
        ask_exit = bid_exit + 0.1

    return Observation(
        timeframe=timeframe,
        regime=regime,
        bucket=bucket,
        direction=direction,
        horizon_candles=horizon,
        signal_timestamp=timestamp,
        entry_timestamp=timestamp + timedelta(minutes=1),
        bid_entry=bid_entry,
        ask_entry=ask_entry,
        bid_exit=bid_exit,
        ask_exit=ask_exit,
        signed_return=signed_return,
    )


def make_period_observations(
    *,
    start: datetime,
    end: datetime,
    positive: int,
    negative: int,
    timeframe: str = "H4",
    bucket: str = "normal",
    regime: str = "REGIME",
    direction: str = "bullish",
    horizon: int = 5,
) -> list[Observation]:
    """
    Generate deterministic synthetic observations inside one period.

    Each observation is one minute apart. Positive observations receive
    +1% signed return and negative observations receive -1% signed return.
    """
    observations: list[Observation] = []

    total = positive + negative

    for index in range(total):
        timestamp = start + timedelta(minutes=index)

        if timestamp >= end:
            raise AssertionError(
                "Synthetic test period does not contain enough timestamps."
            )

        signed_return = 0.01 if index < positive else -0.01

        observations.append(
            make_observation(
                timestamp=timestamp,
                signed_return=signed_return,
                timeframe=timeframe,
                bucket=bucket,
                regime=regime,
                direction=direction,
                horizon=horizon,
            )
        )

    return observations


def make_candidate_observations(
    *,
    periods: list[tuple[datetime, datetime, int, int]],
    timeframe: str = "H4",
    bucket: str = "normal",
    regime: str = "REGIME",
    direction: str = "bullish",
    horizon: int = 5,
) -> list[Observation]:
    observations: list[Observation] = []

    for start, end, positive, negative in periods:
        observations.extend(
            make_period_observations(
                start=start,
                end=end,
                positive=positive,
                negative=negative,
                timeframe=timeframe,
                bucket=bucket,
                regime=regime,
                direction=direction,
                horizon=horizon,
            )
        )

    return observations


def test_training_periods_are_chronological() -> None:
    periods = training_periods(
        datetime(2010, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 1, tzinfo=UTC),
    )

    assert periods

    for index, (_, start, end) in enumerate(periods):
        assert start < end

        if index > 0:
            _, previous_start, previous_end = periods[index - 1]

            assert start >= previous_start
            assert start >= previous_end


def test_training_periods_match_expected_structure() -> None:
    periods = training_periods(
        datetime(2010, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 1, tzinfo=UTC),
    )

    assert periods == [
        (
            "2010-2014",
            datetime(2010, 1, 1, tzinfo=UTC),
            datetime(2015, 1, 1, tzinfo=UTC),
        ),
        (
            "2015-2019",
            datetime(2015, 1, 1, tzinfo=UTC),
            datetime(2020, 1, 1, tzinfo=UTC),
        ),
        (
            "2020-2021",
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2022, 1, 1, tzinfo=UTC),
        ),
        (
            "2022-2023",
            datetime(2022, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),
        ),
        (
            "2024-2026",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 1, tzinfo=UTC),
        ),
    ]


def test_candidate_key_and_key_string_use_actual_observation_api() -> None:
    timestamp = datetime(2020, 1, 1, tzinfo=UTC)

    observation = make_observation(
        timestamp=timestamp,
        signed_return=0.01,
        timeframe="H4",
        bucket="normal",
        regime="TREND",
        direction="bullish",
        horizon=5,
    )

    key = candidate_key(observation)

    assert key == (
        "H4",
        "normal",
        "bullish",
        5,
    )

    assert key_string(key) == (
        "H4|normal|bullish|H5"
    )


def test_training_selection_requires_two_adequate_periods() -> None:
    observations = make_candidate_observations(
        periods=[
            (
                datetime(2010, 1, 1, tzinfo=UTC),
                datetime(2015, 1, 1, tzinfo=UTC),
                25,
                25,
            ),
            (
                datetime(2015, 1, 1, tzinfo=UTC),
                datetime(2020, 1, 1, tzinfo=UTC),
                25,
                25,
            ),
        ]
    )

    selected, rejected = select_training_candidates(
        observations,
        train_start=datetime(2010, 1, 1, tzinfo=UTC),
        train_end=datetime(2020, 1, 1, tzinfo=UTC),
    )

    assert selected == []
    assert len(rejected) == 1

    assert "insufficient_training_periods" in (
        rejected[0]["rejection_reasons"]
    )


def test_training_selection_respects_minimum_period_sample() -> None:
    observations = make_candidate_observations(
        periods=[
            (
                datetime(2010, 1, 1, tzinfo=UTC),
                datetime(2015, 1, 1, tzinfo=UTC),
                MIN_PERIOD_SAMPLE - 1,
                0,
            ),
            (
                datetime(2015, 1, 1, tzinfo=UTC),
                datetime(2020, 1, 1, tzinfo=UTC),
                MIN_PERIOD_SAMPLE - 1,
                0,
            ),
        ]
    )

    selected, rejected = select_training_candidates(
        observations,
        train_start=datetime(2010, 1, 1, tzinfo=UTC),
        train_end=datetime(2020, 1, 1, tzinfo=UTC),
    )

    assert selected == []
    assert len(rejected) == 1

    assert "insufficient_training_periods" in (
        rejected[0]["rejection_reasons"]
    )


def test_training_selection_requires_positive_edge_majority() -> None:
    observations = make_candidate_observations(
        periods=[
            (
                datetime(2010, 1, 1, tzinfo=UTC),
                datetime(2015, 1, 1, tzinfo=UTC),
                1000,
                0,
            ),
            (
                datetime(2015, 1, 1, tzinfo=UTC),
                datetime(2020, 1, 1, tzinfo=UTC),
                495,
                505,
            ),
            (
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2022, 1, 1, tzinfo=UTC),
                495,
                505,
            ),
        ]
    )

    selected, rejected = select_training_candidates(
        observations,
        train_start=datetime(2010, 1, 1, tzinfo=UTC),
        train_end=datetime(2022, 1, 1, tzinfo=UTC),
    )

    assert selected == []
    assert len(rejected) == 1

    assert "positive_edge_not_in_majority" in (
        rejected[0]["rejection_reasons"]
    )


def test_training_selection_requires_mean_edge_threshold() -> None:
    observations = make_candidate_observations(
        periods=[
            (
                datetime(2010, 1, 1, tzinfo=UTC),
                datetime(2015, 1, 1, tzinfo=UTC),
                509,
                491,
            ),
            (
                datetime(2015, 1, 1, tzinfo=UTC),
                datetime(2020, 1, 1, tzinfo=UTC),
                509,
                491,
            ),
        ]
    )

    selected, rejected = select_training_candidates(
        observations,
        train_start=datetime(2010, 1, 1, tzinfo=UTC),
        train_end=datetime(2020, 1, 1, tzinfo=UTC),
    )

    assert selected == []
    assert len(rejected) == 1

    assert "mean_training_effect_below_threshold" in (
        rejected[0]["rejection_reasons"]
    )


def test_training_selection_allows_non_material_reversal() -> None:
    observations = make_candidate_observations(
        periods=[
            (
                datetime(2010, 1, 1, tzinfo=UTC),
                datetime(2015, 1, 1, tzinfo=UTC),
                1000,
                0,
            ),
            (
                datetime(2015, 1, 1, tzinfo=UTC),
                datetime(2020, 1, 1, tzinfo=UTC),
                1000,
                0,
            ),
            (
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2022, 1, 1, tzinfo=UTC),
                495,
                505,
            ),
        ]
    )

    selected, rejected = select_training_candidates(
        observations,
        train_start=datetime(2010, 1, 1, tzinfo=UTC),
        train_end=datetime(2022, 1, 1, tzinfo=UTC),
    )

    assert len(selected) == 1
    assert rejected == []

    candidate = selected[0]

    assert candidate["positive_training_periods"] == 2
    assert candidate["adequate_periods"] == 3
    assert candidate["worst_training_edge_pp"] > (
        -MAX_TRAIN_REVERSAL_PP
    )
    assert candidate["mean_training_edge_pp"] >= (
        MIN_TRAIN_EDGE_PP
    )


def test_training_selection_rejects_material_reversal() -> None:
    observations = make_candidate_observations(
        periods=[
            (
                datetime(2010, 1, 1, tzinfo=UTC),
                datetime(2015, 1, 1, tzinfo=UTC),
                1000,
                0,
            ),
            (
                datetime(2015, 1, 1, tzinfo=UTC),
                datetime(2020, 1, 1, tzinfo=UTC),
                1000,
                0,
            ),
            (
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2022, 1, 1, tzinfo=UTC),
                470,
                530,
            ),
        ]
    )

    selected, rejected = select_training_candidates(
        observations,
        train_start=datetime(2010, 1, 1, tzinfo=UTC),
        train_end=datetime(2022, 1, 1, tzinfo=UTC),
    )

    assert selected == []
    assert len(rejected) == 1

    assert "material_training_reversal" in (
        rejected[0]["rejection_reasons"]
    )

    assert rejected[0]["worst_training_edge_pp"] <= (
        -MAX_TRAIN_REVERSAL_PP
    )


def test_training_selection_accepts_strict_positive_majority() -> None:
    observations = make_candidate_observations(
        periods=[
            (
                datetime(2010, 1, 1, tzinfo=UTC),
                datetime(2015, 1, 1, tzinfo=UTC),
                550,
                450,
            ),
            (
                datetime(2015, 1, 1, tzinfo=UTC),
                datetime(2020, 1, 1, tzinfo=UTC),
                550,
                450,
            ),
            (
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2022, 1, 1, tzinfo=UTC),
                495,
                505,
            ),
        ]
    )

    selected, rejected = select_training_candidates(
        observations,
        train_start=datetime(2010, 1, 1, tzinfo=UTC),
        train_end=datetime(2022, 1, 1, tzinfo=UTC),
    )

    assert len(selected) == 1
    assert rejected == []

    candidate = selected[0]

    assert candidate["positive_training_periods"] == 2
    assert candidate["negative_training_periods"] == 1
    assert candidate["adequate_periods"] == 3

    assert candidate["mean_training_edge_pp"] >= (
        MIN_TRAIN_EDGE_PP
    )

    assert candidate["worst_training_edge_pp"] > (
        -MAX_TRAIN_REVERSAL_PP
    )


def test_training_selection_rejects_negative_candidate_without_inversion() -> None:
    observations = make_candidate_observations(
        periods=[
            (
                datetime(2010, 1, 1, tzinfo=UTC),
                datetime(2015, 1, 1, tzinfo=UTC),
                400,
                600,
            ),
            (
                datetime(2015, 1, 1, tzinfo=UTC),
                datetime(2020, 1, 1, tzinfo=UTC),
                400,
                600,
            ),
        ]
    )

    selected, rejected = select_training_candidates(
        observations,
        train_start=datetime(2010, 1, 1, tzinfo=UTC),
        train_end=datetime(2020, 1, 1, tzinfo=UTC),
    )

    assert selected == []
    assert len(rejected) == 1

    candidate = rejected[0]

    assert candidate["direction"] == "bullish"
    assert candidate["mean_training_edge_pp"] < 0.0
    assert candidate["worst_training_edge_pp"] < 0.0


def test_training_selection_metadata_matches_revised_rule() -> None:
    observations = make_candidate_observations(
        periods=[
            (
                datetime(2010, 1, 1, tzinfo=UTC),
                datetime(2015, 1, 1, tzinfo=UTC),
                550,
                450,
            ),
            (
                datetime(2015, 1, 1, tzinfo=UTC),
                datetime(2020, 1, 1, tzinfo=UTC),
                550,
                450,
            ),
            (
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2022, 1, 1, tzinfo=UTC),
                495,
                505,
            ),
        ]
    )

    selected, rejected = select_training_candidates(
        observations,
        train_start=datetime(2010, 1, 1, tzinfo=UTC),
        train_end=datetime(2022, 1, 1, tzinfo=UTC),
    )

    assert len(selected) == 1
    assert rejected == []

    metadata = selected[0]["selection_rule"]

    assert metadata["minimum_period_sample"] == MIN_PERIOD_SAMPLE
    assert metadata["minimum_adequate_periods"] == 2
    assert metadata["minimum_mean_training_edge_pp"] == (
        MIN_TRAIN_EDGE_PP
    )
    assert metadata["positive_edge_majority"] is True
    assert metadata["maximum_material_reversal_pp"] == (
        MAX_TRAIN_REVERSAL_PP
    )
    assert metadata["training_only"] is True


def test_training_selection_uses_training_timestamps_only() -> None:
    observations = make_candidate_observations(
        periods=[
            (
                datetime(2010, 1, 1, tzinfo=UTC),
                datetime(2015, 1, 1, tzinfo=UTC),
                550,
                450,
            ),
            (
                datetime(2015, 1, 1, tzinfo=UTC),
                datetime(2020, 1, 1, tzinfo=UTC),
                550,
                450,
            ),
        ]
    )

    # Strong positive observations after train_end must not influence
    # candidate selection.
    observations.extend(
        make_period_observations(
            start=datetime(2020, 1, 1, tzinfo=UTC),
            end=datetime(2021, 1, 1, tzinfo=UTC),
            positive=1000,
            negative=0,
        )
    )

    selected, rejected = select_training_candidates(
        observations,
        train_start=datetime(2010, 1, 1, tzinfo=UTC),
        train_end=datetime(2020, 1, 1, tzinfo=UTC),
    )

    assert len(selected) == 1
    assert rejected == []

    candidate = selected[0]

    assert candidate["adequate_periods"] == 2
    assert candidate["training"]["directional_observations"] == 2000


def test_training_selection_handles_multiple_candidates() -> None:
    observations: list[Observation] = []

    observations.extend(
        make_candidate_observations(
            periods=[
                (
                    datetime(2010, 1, 1, tzinfo=UTC),
                    datetime(2015, 1, 1, tzinfo=UTC),
                    550,
                    450,
                ),
                (
                    datetime(2015, 1, 1, tzinfo=UTC),
                    datetime(2020, 1, 1, tzinfo=UTC),
                    550,
                    450,
                ),
                (
                    datetime(2020, 1, 1, tzinfo=UTC),
                    datetime(2022, 1, 1, tzinfo=UTC),
                    495,
                    505,
                ),
            ],
            timeframe="H4",
            bucket="normal",
            regime="TREND",
            direction="bullish",
            horizon=5,
        )
    )

    observations.extend(
        make_candidate_observations(
            periods=[
                (
                    datetime(2010, 1, 1, tzinfo=UTC),
                    datetime(2015, 1, 1, tzinfo=UTC),
                    400,
                    600,
                ),
                (
                    datetime(2015, 1, 1, tzinfo=UTC),
                    datetime(2020, 1, 1, tzinfo=UTC),
                    400,
                    600,
                ),
            ],
            timeframe="H4",
            bucket="high",
            regime="TREND",
            direction="bullish",
            horizon=5,
        )
    )

    selected, rejected = select_training_candidates(
        observations,
        train_start=datetime(2010, 1, 1, tzinfo=UTC),
        train_end=datetime(2020, 1, 1, tzinfo=UTC),
    )

    assert len(selected) == 1
    assert len(rejected) == 1

    assert selected[0]["bucket"] == "normal"
    assert rejected[0]["bucket"] == "high"


def test_directional_stats_calculates_accuracy_and_edge() -> None:
    start = datetime(2020, 1, 1, tzinfo=UTC)

    observations = [
        make_observation(
            timestamp=start + timedelta(minutes=index),
            signed_return=0.01 if index < 6 else -0.01,
        )
        for index in range(10)
    ]

    stats = directional_stats(observations)

    assert stats["observations"] == 10
    assert stats["positive"] == 6
    assert stats["negative"] == 4
    assert stats["directional_observations"] == 10
    assert stats["accuracy_pct"] == 60.0
    assert stats["edge_pp"] == 10.0


def test_wilson_interval_returns_valid_interval() -> None:
    low, high = wilson_interval(60, 40)

    assert low is not None
    assert high is not None
    assert 0.0 <= low < high <= 1.0


def test_lag1_autocorrelation_and_effective_sample_size() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    rho1 = lag1_autocorrelation(values)

    assert rho1 is not None
    assert rho1 > 0.0

    ess = effective_sample_size(len(values), rho1)

    assert ess is not None
    assert 0.0 < ess <= len(values)


def test_lag1_autocorrelation_returns_none_for_short_series() -> None:
    assert lag1_autocorrelation([]) is None
    assert lag1_autocorrelation([1.0]) is None
    assert lag1_autocorrelation([1.0, 2.0]) is None


def test_moving_block_bootstrap_is_deterministic() -> None:
    values = [
        0.001,
        0.002,
        -0.001,
        0.003,
        0.001,
        -0.002,
        0.002,
        0.004,
        -0.001,
        0.003,
        0.002,
        0.001,
    ]

    first = moving_block_bootstrap_mean(
        values,
        horizon=5,
        replicates=100,
        seed=6306,
    )

    second = moving_block_bootstrap_mean(
        values,
        horizon=5,
        replicates=100,
        seed=6306,
    )

    assert first == second
    assert first["replicates"] == 100
    assert first["block_length"] >= 5
    assert first["ci95_low_pct"] is not None
    assert first["ci95_high_pct"] is not None
    assert first["probability_mean_gt_zero"] is not None


def test_moving_block_bootstrap_handles_short_series() -> None:
    result = moving_block_bootstrap_mean(
        [0.001, 0.002, 0.003],
        horizon=5,
        replicates=100,
        seed=6306,
    )

    assert result["replicates"] == 0
    assert result["block_length"] is None
    assert result["ci95_low_pct"] is None
    assert result["ci95_high_pct"] is None
    assert result["probability_mean_gt_zero"] is None


def test_executable_return_long_uses_ask_entry_and_bid_exit() -> None:
    observation = make_observation(
        timestamp=datetime(2020, 1, 1, tzinfo=UTC),
        signed_return=0.0,
        direction="bullish",
        bid_entry=100.0,
        ask_entry=101.0,
        bid_exit=102.0,
        ask_exit=103.0,
    )

    result = executable_return(
        observation,
        slippage_bps=0.0,
    )

    assert result == (102.0 - 101.0) / 101.0


def test_executable_return_short_uses_bid_entry_and_ask_exit() -> None:
    observation = make_observation(
        timestamp=datetime(2020, 1, 1, tzinfo=UTC),
        signed_return=0.0,
        direction="bearish",
        bid_entry=101.0,
        ask_entry=102.0,
        bid_exit=99.0,
        ask_exit=100.0,
    )

    result = executable_return(
        observation,
        slippage_bps=0.0,
    )

    assert result == (101.0 - 100.0) / 101.0


def test_economic_evaluation_contains_all_slippage_scenarios() -> None:
    start = datetime(2020, 1, 1, tzinfo=UTC)

    observations = [
        make_observation(
            timestamp=start + timedelta(minutes=index),
            signed_return=0.01,
            direction="bullish",
            bid_entry=100.0,
            ask_entry=100.1,
            bid_exit=101.0,
            ask_exit=101.1,
        )
        for index in range(12)
    ]

    result = economic_evaluation(
        observations,
        bootstrap_replicates=100,
        seed=6306,
    )

    scenarios = result["scenarios"]

    assert len(scenarios) == 4

    assert [
        scenario["slippage_bps_per_side"]
        for scenario in scenarios
    ] == [0.0, 0.5, 1.0, 2.0]

    for scenario in scenarios:
        assert scenario["observations"] == 12
        assert scenario["mean_return_pct"] is not None
        assert scenario["median_return_pct"] is not None
        assert scenario["win_rate_pct"] is not None
        assert scenario["bootstrap"]["replicates"] == 100


def test_classify_oos_returns_e_for_insufficient_sample() -> None:
    directional = {
        "directional_observations": MIN_PERIOD_SAMPLE - 1,
        "edge_pp": 10.0,
    }

    economics = {
        "scenarios": [],
    }

    assert classify_oos(
        directional=directional,
        economics=economics,
    ) == "OOS-E"


def test_classify_oos_returns_d_for_material_negative_edge() -> None:
    directional = {
        "directional_observations": MIN_PERIOD_SAMPLE,
        "edge_pp": -MIN_TRAIN_EDGE_PP - 0.01,
    }

    economics = {
        "scenarios": [],
    }

    assert classify_oos(
        directional=directional,
        economics=economics,
    ) == "OOS-D"


def test_classify_oos_returns_a_only_with_strong_economic_confirmation() -> None:
    directional = {
        "directional_observations": MIN_PERIOD_SAMPLE,
        "edge_pp": MIN_TRAIN_EDGE_PP,
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
        ],
    }

    assert classify_oos(
        directional=directional,
        economics=economics,
    ) == "OOS-A"


def test_classify_oos_returns_b_for_positive_but_unconfirmed_edge() -> None:
    directional = {
        "directional_observations": MIN_PERIOD_SAMPLE,
        "edge_pp": 1.0,
    }

    economics = {
        "scenarios": [
            {
                "slippage_bps_per_side": 1.0,
                "mean_return_pct": 0.01,
                "bootstrap": {
                    "ci95_low_pct": -0.01,
                },
            }
        ],
    }

    assert classify_oos(
        directional=directional,
        economics=economics,
    ) == "OOS-B"


def test_classify_oos_returns_c_for_non_positive_edge() -> None:
    directional = {
        "directional_observations": MIN_PERIOD_SAMPLE,
        "edge_pp": 0.0,
    }

    economics = {
        "scenarios": [
            {
                "slippage_bps_per_side": 1.0,
                "mean_return_pct": -0.01,
                "bootstrap": {
                    "ci95_low_pct": -0.02,
                },
            }
        ],
    }

    assert classify_oos(
        directional=directional,
        economics=economics,
    ) == "OOS-C"