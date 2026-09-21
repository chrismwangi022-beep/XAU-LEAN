from __future__ import annotations

from datetime import datetime, timedelta, timezone

from scripts.analyze_volatility_direction import Observation

from scripts.analyze_volatility_direction_formal import (
    choose_block_length,
    interaction_from_observations,
    moving_block_sample,
    percentile,
)


def make_observation(
    *,
    bucket: str,
    direction: str,
    signed_return: float,
    minutes: int,
) -> Observation:
    timestamp = (
        datetime(
            2024,
            1,
            1,
            tzinfo=timezone.utc,
        )
        + timedelta(minutes=minutes)
    )

    return Observation(
        timeframe="M15",
        regime="2024-2026",
        bucket=bucket,
        direction=direction,
        horizon=1,
        signal_timestamp=timestamp,
        entry_timestamp=timestamp + timedelta(minutes=15),
        bid_entry=100.0,
        ask_entry=100.1,
        bid_exit=100.0 + signed_return,
        ask_exit=100.1 + signed_return,
        signed_return=signed_return,
    )


def test_choose_block_length_matches_canonical_rule():
    assert choose_block_length(25, 1) == 5
    assert choose_block_length(100, 3) == 10
    assert choose_block_length(10_000, 5) == 50


def test_percentile_linear_interpolation():
    values = [1.0, 2.0, 3.0, 4.0]

    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 1.0) == 4.0
    assert percentile(values, 0.5) == 2.5


def test_interaction_matches_difference_in_differences():
    observations = [
        # Bucket bullish: 3 / 4 correct
        make_observation(
            bucket="high",
            direction="bullish",
            signed_return=1.0,
            minutes=0,
        ),
        make_observation(
            bucket="high",
            direction="bullish",
            signed_return=1.0,
            minutes=1,
        ),
        make_observation(
            bucket="high",
            direction="bullish",
            signed_return=1.0,
            minutes=2,
        ),
        make_observation(
            bucket="high",
            direction="bullish",
            signed_return=-1.0,
            minutes=3,
        ),

        # Bucket bearish: 1 / 4 correct
        make_observation(
            bucket="high",
            direction="bearish",
            signed_return=-1.0,
            minutes=4,
        ),
        make_observation(
            bucket="high",
            direction="bearish",
            signed_return=1.0,
            minutes=5,
        ),
        make_observation(
            bucket="high",
            direction="bearish",
            signed_return=1.0,
            minutes=6,
        ),
        make_observation(
            bucket="high",
            direction="bearish",
            signed_return=1.0,
            minutes=7,
        ),

        # Normal bullish: 2 / 4 correct
        make_observation(
            bucket="normal",
            direction="bullish",
            signed_return=1.0,
            minutes=8,
        ),
        make_observation(
            bucket="normal",
            direction="bullish",
            signed_return=1.0,
            minutes=9,
        ),
        make_observation(
            bucket="normal",
            direction="bullish",
            signed_return=-1.0,
            minutes=10,
        ),
        make_observation(
            bucket="normal",
            direction="bullish",
            signed_return=-1.0,
            minutes=11,
        ),

        # Normal bearish: 2 / 4 correct
        make_observation(
            bucket="normal",
            direction="bearish",
            signed_return=-1.0,
            minutes=12,
        ),
        make_observation(
            bucket="normal",
            direction="bearish",
            signed_return=-1.0,
            minutes=13,
        ),
        make_observation(
            bucket="normal",
            direction="bearish",
            signed_return=1.0,
            minutes=14,
        ),
        make_observation(
            bucket="normal",
            direction="bearish",
            signed_return=1.0,
            minutes=15,
        ),
    ]

    accuracy, signed_return = interaction_from_observations(
        observations,
        "high",
    )

    # Bucket accuracy spread = .75 - .25 = .50
    # Normal accuracy spread = .50 - .50 = 0
    # Interaction = .50
    assert accuracy == 0.5

    # Bucket signed-return spread:
    # +0.5 - (+0.5) = 0
    #
    # Normal signed-return spread:
    # 0 - 0 = 0
    #
    # Interaction = 0
    assert signed_return == 0.0


def test_interaction_requires_all_four_groups():
    observations = [
        make_observation(
            bucket="high",
            direction="bullish",
            signed_return=1.0,
            minutes=0,
        ),
    ]

    accuracy, signed_return = interaction_from_observations(
        observations,
        "high",
    )

    assert accuracy != accuracy
    assert signed_return != signed_return


def test_moving_block_sample_is_deterministic():
    observations = [
        make_observation(
            bucket="normal",
            direction="bullish",
            signed_return=1.0,
            minutes=index,
        )
        for index in range(20)
    ]

    import random

    sample_a = moving_block_sample(
        observations,
        block_length=5,
        rng=random.Random(123),
    )

    sample_b = moving_block_sample(
        observations,
        block_length=5,
        rng=random.Random(123),
    )

    assert sample_a == sample_b
    assert len(sample_a) == len(observations)


def test_moving_block_sample_preserves_observation_objects():
    observations = [
        make_observation(
            bucket="normal",
            direction="bullish",
            signed_return=1.0,
            minutes=index,
        )
        for index in range(20)
    ]

    import random

    sample = moving_block_sample(
        observations,
        block_length=5,
        rng=random.Random(7),
    )

    original_ids = {
        id(item)
        for item in observations
    }

    assert len(sample) == len(observations)
    assert all(
        id(item) in original_ids
        for item in sample
    )
