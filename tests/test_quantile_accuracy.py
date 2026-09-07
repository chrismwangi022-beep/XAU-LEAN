from __future__ import annotations

import random

import pytest

from xau_lean.research.quantiles import (
    DEFAULT_QUANTILES,
    P2Quantile,
    StreamingQuantiles,
)


def exact_quantile(
    values: list[float],
    quantile: float,
) -> float:
    """Reference linear-interpolated sample quantile."""
    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = quantile * (len(ordered) - 1)

    lower = int(position)
    upper = lower + 1

    if upper >= len(ordered):
        return ordered[-1]

    fraction = position - lower

    return (
        ordered[lower]
        + fraction
        * (ordered[upper] - ordered[lower])
    )


@pytest.mark.parametrize("quantile", DEFAULT_QUANTILES)
def test_p2_accuracy_on_deterministic_uniform_stream(
    quantile: float,
) -> None:
    """P² should remain close to the exact quantile."""
    rng = random.Random(20260907)

    values = [
        rng.random()
        for _ in range(10_000)
    ]

    estimator = P2Quantile(quantile)

    for value in values:
        estimator.update(value)

    expected = exact_quantile(values, quantile)
    actual = estimator.value

    assert actual is not None

    # P² is an approximation, so test tolerance rather than equality.
    assert abs(actual - expected) < 0.01


@pytest.mark.parametrize("quantile", DEFAULT_QUANTILES)
def test_p2_accuracy_on_deterministic_normal_stream(
    quantile: float,
) -> None:
    """P² should remain accurate on a non-uniform distribution."""
    rng = random.Random(20260907)

    values = [
        rng.gauss(0.0, 1.0)
        for _ in range(10_000)
    ]

    estimator = P2Quantile(quantile)

    for value in values:
        estimator.update(value)

    expected = exact_quantile(values, quantile)
    actual = estimator.value

    assert actual is not None

    assert abs(actual - expected) < 0.05


def test_all_default_quantiles_are_monotonic() -> None:
    """Configured quantile estimates must remain ordered."""
    rng = random.Random(20260907)

    values = [
        rng.random()
        for _ in range(20_000)
    ]

    estimator = StreamingQuantiles()

    for value in values:
        estimator.update(value)

    result = estimator.to_dict()

    estimates = [
        result["p10"],
        result["p25"],
        result["p50"],
        result["p75"],
        result["p90"],
        result["p95"],
        result["p99"],
    ]

    assert all(
        left <= right
        for left, right in zip(
            estimates,
            estimates[1:],
        )
    )


def test_p99_remains_within_observed_data_range() -> None:
    """Tail estimates must never escape the observed range."""
    rng = random.Random(20260907)

    values = [
        rng.expovariate(1.0)
        for _ in range(20_000)
    ]

    estimator = P2Quantile(0.99)

    for value in values:
        estimator.update(value)

    assert estimator.value is not None
    assert min(values) <= estimator.value <= max(values)
