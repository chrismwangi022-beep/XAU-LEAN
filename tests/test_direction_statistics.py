import math

import pytest

from scripts.analyze_direction_statistics import (
    analyze_cell,
    benjamini_hochberg,
    binomial_probability,
    classify_fdr_result,
    classify_statistical_result,
    exact_binomial_two_sided_pvalue,
    wilson_interval,
)


def test_binomial_probability_fair_coin():
    assert math.isclose(
        binomial_probability(5, 10, 0.5),
        252 / 1024,
        rel_tol=1e-12,
    )


def test_binomial_probability_extreme_outcome():
    assert math.isclose(
        binomial_probability(0, 10, 0.5),
        1 / 1024,
        rel_tol=1e-12,
    )


def test_exact_binomial_two_sided_pvalue_5_of_10():
    assert math.isclose(
        exact_binomial_two_sided_pvalue(5, 10),
        1.0,
        rel_tol=1e-12,
    )


def test_exact_binomial_two_sided_pvalue_10_of_10():
    assert math.isclose(
        exact_binomial_two_sided_pvalue(10, 10),
        2 / 1024,
        rel_tol=1e-12,
    )


def test_wilson_interval_is_valid():
    lower, upper = wilson_interval(50, 100)

    assert 0.0 <= lower < 0.5
    assert 0.5 < upper <= 1.0


def test_wilson_interval_bounds_extreme_success_rate():
    lower, upper = wilson_interval(100, 100)

    assert 0.0 <= lower < 1.0
    assert upper == pytest.approx(1.0)


def test_classification_above_50():
    result = classify_statistical_result(
        accuracy=0.55,
        ci_lower=0.53,
        ci_upper=0.57,
        p_value=0.001,
    )

    assert result == "statistically_above_50"


def test_classification_below_50():
    result = classify_statistical_result(
        accuracy=0.45,
        ci_lower=0.43,
        ci_upper=0.47,
        p_value=0.001,
    )

    assert result == "statistically_below_50"


def test_classification_not_significant():
    result = classify_statistical_result(
        accuracy=0.51,
        ci_lower=0.49,
        ci_upper=0.53,
        p_value=0.70,
    )

    assert result == "not_statistically_distinguishable_from_50"


def test_analyze_cell_excludes_flat_from_directional_n():
    cell = {
        "regime": "test",
        "bucket": "normal",
        "direction": "bullish",
        "directional_accuracy": 0.60,
        "flat": 10,
        "horizon_candles": 1,
        "mean_abs_return_pct": 0.10,
        "mean_directional_efficiency": 0.60,
        "mean_signed_return_pct": 0.02,
        "positive": 60,
        "negative": 40,
        "observations": 110,
    }

    result = analyze_cell("test|normal|bullish|H1", cell)

    assert result["directional_observations"] == 100
    assert result["total_observations"] == 110
    assert result["directional_accuracy"] == pytest.approx(0.60)
    assert result["edge_pp"] == pytest.approx(10.0)
    assert result["p_value_fdr"] is None
    assert result["fdr_significant_05"] is False


def test_analyze_cell_rejects_no_directional_observations():
    cell = {
        "regime": "test",
        "bucket": "normal",
        "direction": "bullish",
        "flat": 10,
        "horizon_candles": 1,
        "mean_abs_return_pct": 0.10,
        "mean_directional_efficiency": 0.0,
        "mean_signed_return_pct": 0.0,
        "positive": 0,
        "negative": 0,
        "observations": 10,
    }

    with pytest.raises(ValueError):
        analyze_cell("test|normal|bullish|H1", cell)


def test_benjamini_hochberg_empty():
    assert benjamini_hochberg([]) == []


def test_benjamini_hochberg_preserves_input_order():
    p_values = [0.04, 0.001, 0.20]

    adjusted = benjamini_hochberg(p_values)

    assert len(adjusted) == 3
    assert adjusted[0] == pytest.approx(0.06)
    assert adjusted[1] == pytest.approx(0.003)
    assert adjusted[2] == pytest.approx(0.20)


def test_benjamini_hochberg_monotonic_adjusted_values():
    p_values = [0.001, 0.01, 0.02, 0.5]

    adjusted = benjamini_hochberg(p_values)

    assert adjusted[0] <= adjusted[1]
    assert adjusted[1] <= adjusted[2]
    assert adjusted[2] <= adjusted[3]


def test_benjamini_hochberg_all_equal():
    p_values = [0.01, 0.01, 0.01, 0.01]

    adjusted = benjamini_hochberg(p_values)

    assert adjusted == pytest.approx(
        [0.01, 0.01, 0.01, 0.01]
    )


def test_benjamini_hochberg_rejects_invalid_p_values():
    with pytest.raises(ValueError):
        benjamini_hochberg([0.01, 1.1])

    with pytest.raises(ValueError):
        benjamini_hochberg([-0.01, 0.2])


def test_fdr_classification_positive():
    assert (
        classify_fdr_result(
            edge_pp=5.0,
            fdr_p_value=0.01,
            q=0.05,
        )
        == "fdr_significant_above_50"
    )


def test_fdr_classification_negative():
    assert (
        classify_fdr_result(
            edge_pp=-5.0,
            fdr_p_value=0.01,
            q=0.05,
        )
        == "fdr_significant_below_50"
    )


def test_fdr_classification_not_significant():
    assert (
        classify_fdr_result(
            edge_pp=5.0,
            fdr_p_value=0.08,
            q=0.05,
        )
        == "fdr_not_significant"
    )


def test_fdr_classification_zero_edge():
    assert (
        classify_fdr_result(
            edge_pp=0.0,
            fdr_p_value=0.01,
            q=0.05,
        )
        == "fdr_not_significant"
    )
