from __future__ import annotations

from datetime import datetime, timezone

from scripts.analyze_direction_dependence import (
    Candidate,
    Observation,
    autocorrelation,
    choose_block_length,
    effective_sample_size,
    moving_block_bootstrap,
    percentile,
)


def make_observations(values):
    return [
        Observation(
            timestamp=datetime(
                2020,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            signed_return=value,
        )
        for value in values
    ]


def test_percentile_empty_rejected():
    try:
        percentile([], 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Empty percentile input should raise ValueError."
        )


def test_percentile_interpolates():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5


def test_autocorrelation_constant_series_is_zero():
    assert autocorrelation(
        [1.0, 1.0, 1.0, 1.0],
        1,
    ) == 0.0


def test_autocorrelation_requires_enough_data():
    assert autocorrelation([1.0], 1) is None


def test_effective_sample_size_never_exceeds_n():
    values = [float(index % 2) for index in range(100)]

    ess = effective_sample_size(values)

    assert 1.0 <= ess <= 100.0


def test_effective_sample_size_positive_autocorrelation_reduces_sample():
    values = []

    for _ in range(50):
        values.extend([1.0, 1.0, 0.0, 0.0])

    ess = effective_sample_size(values)

    assert ess < len(values)


def test_block_length_respects_horizon():
    assert choose_block_length(
        n=100,
        horizon=5,
    ) >= 5


def test_block_length_is_capped():
    assert choose_block_length(
        n=100000,
        horizon=5,
    ) <= 60


def test_block_length_never_exceeds_sample():
    assert choose_block_length(
        n=3,
        horizon=5,
    ) == 3


def test_candidate_key():
    candidate = Candidate(
        timeframe="H6",
        bucket="high",
        direction="bullish",
        horizon=3,
    )

    assert candidate.key == "H6|high|bullish|H3"


def test_observation_outcome_positive():
    observation = make_observations([0.01])[0]

    assert observation.outcome == 1


def test_observation_outcome_negative():
    observation = make_observations([-0.01])[0]

    assert observation.outcome == 0


def test_observation_outcome_flat():
    observation = make_observations([0.0])[0]

    assert observation.outcome == -1


def test_bootstrap_returns_requested_replicates():
    values = [0.01, -0.01, 0.02, -0.005] * 10

    result = moving_block_bootstrap(
        values=values,
        block_length=5,
        replicates=100,
        seed=123,
    )

    assert len(result) == 100


def test_bootstrap_is_reproducible():
    values = [0.01, -0.01, 0.02, -0.005] * 10

    first = moving_block_bootstrap(
        values=values,
        block_length=5,
        replicates=100,
        seed=123,
    )

    second = moving_block_bootstrap(
        values=values,
        block_length=5,
        replicates=100,
        seed=123,
    )

    assert first == second


def test_bootstrap_means_are_numeric():
    values = [0.01, 0.02, -0.01] * 10

    result = moving_block_bootstrap(
        values=values,
        block_length=4,
        replicates=25,
        seed=42,
    )

    assert all(isinstance(value, float) for value in result)
