import math

import pytest

from xau_lean.research.quantiles import (
    DEFAULT_QUANTILES,
    P2Quantile,
    StreamingQuantiles,
)


def test_empty_estimator_returns_none():
    estimator = P2Quantile(0.50)

    assert estimator.value is None
    assert estimator.count == 0


def test_small_sample_uses_exact_quantile():
    estimator = P2Quantile(0.50)

    for value in [5, 1, 3, 2]:
        estimator.update(value)

    assert estimator.count == 4
    assert estimator.value == 2.5


def test_small_sample_percentile_is_exact():
    estimator = P2Quantile(0.90)

    values = [1, 2, 3, 4]

    for value in values:
        estimator.update(value)

    assert estimator.value == pytest.approx(3.7)


def test_constant_stream_is_exact():
    estimator = P2Quantile(0.95)

    for _ in range(1000):
        estimator.update(7.5)

    assert estimator.value == pytest.approx(7.5)


def test_monotonic_stream_stays_within_data_range():
    estimator = P2Quantile(0.50)

    for value in range(1, 1001):
        estimator.update(value)

    assert 1 <= estimator.value <= 1000
    assert estimator.value == pytest.approx(
        500.5,
        rel=0.01,
    )


def test_multiple_quantiles():
    quantiles = StreamingQuantiles()

    for value in range(1, 1001):
        quantiles.update(value)

    result = quantiles.to_dict()

    assert set(result) == {
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
    }

    assert result["p10"] < result["p25"]
    assert result["p25"] < result["p50"]
    assert result["p50"] < result["p75"]
    assert result["p75"] < result["p90"]
    assert result["p90"] < result["p95"]
    assert result["p95"] < result["p99"]


def test_invalid_quantile_is_rejected():
    with pytest.raises(ValueError):
        P2Quantile(-0.1)

    with pytest.raises(ValueError):
        P2Quantile(1.1)


def test_empty_quantile_collection_is_rejected():
    with pytest.raises(ValueError):
        StreamingQuantiles([])


def test_duplicate_quantiles_are_rejected():
    with pytest.raises(ValueError):
        StreamingQuantiles([0.5, 0.5])


def test_invalid_collection_quantile_is_rejected():
    with pytest.raises(ValueError):
        StreamingQuantiles([0.1, 1.1])


def test_default_quantiles_are_expected():
    assert DEFAULT_QUANTILES == (
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
    )
