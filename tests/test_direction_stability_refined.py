from scripts.analyze_direction_stability_refined import (
    classify,
    refine_row,
)


def make_period(regime, edge, n=500):
    return {
        "regime": regime,
        "available": True,
        "directional_observations": n,
        "directional_accuracy": 0.50 + edge / 100,
        "accuracy_pct": 50.0 + edge,
        "edge_pp": edge,
        "ci_95_lower_pct": 45.0,
        "ci_95_upper_pct": 55.0,
        "p_value": 0.01,
        "p_value_fdr": 0.02,
        "fdr_significant_05": True,
    }


def make_row(edges):
    regimes = [
        "2010-2014",
        "2015-2019",
        "2020-2021",
        "2022-2023",
        "2024-2026",
    ]

    return {
        "hypothesis": "H6|high|bullish|H3",
        "timeframe": "H6",
        "bucket": "high",
        "direction": "bullish",
        "horizon_candles": 3,
        "periods": [
            make_period(regime, edge)
            for regime, edge in zip(regimes, edges)
        ],
    }


def test_robust_positive_classification():
    result = classify(
        periods=[
            make_period("2010-2014", 3.0),
            make_period("2015-2019", 4.0),
            make_period("2020-2021", 5.0),
            make_period("2022-2023", 4.0),
            make_period("2024-2026", 3.0),
        ],
        mean_edge=3.8,
        median_edge=4.0,
        min_edge=3.0,
        max_edge=5.0,
    )

    assert result == "A+"


def test_robust_negative_classification():
    result = classify(
        periods=[
            make_period("2010-2014", -3.0),
            make_period("2015-2019", -4.0),
            make_period("2020-2021", -5.0),
            make_period("2022-2023", -4.0),
            make_period("2024-2026", -3.0),
        ],
        mean_edge=-3.8,
        median_edge=-4.0,
        min_edge=-5.0,
        max_edge=-3.0,
    )

    assert result == "A+"


def test_sign_persistence_without_2pp_magnitude_is_A():
    result = classify(
        periods=[
            make_period("2010-2014", 0.5),
            make_period("2015-2019", 0.8),
            make_period("2020-2021", 1.0),
            make_period("2022-2023", 0.7),
            make_period("2024-2026", 0.9),
        ],
        mean_edge=0.78,
        median_edge=0.8,
        min_edge=0.5,
        max_edge=1.0,
    )

    assert result == "A"


def test_majority_with_two_reversals_is_B():
    result = classify(
        periods=[
            make_period("2010-2014", 5.0),
            make_period("2015-2019", 4.0),
            make_period("2020-2021", -3.0),
            make_period("2022-2023", -2.0),
            make_period("2024-2026", 3.0),
        ],
        mean_edge=1.4,
        median_edge=3.0,
        min_edge=-3.0,
        max_edge=5.0,
    )

    assert result == "B"


def test_mixed_sign_majority_is_B():
    result = classify(
        periods=[
            make_period("2010-2014", 5.0),
            make_period("2015-2019", -4.0),
            make_period("2020-2021", 3.0),
            make_period("2022-2023", -2.0),
            make_period("2024-2026", 1.0),
        ],
        mean_edge=0.6,
        median_edge=1.0,
        min_edge=-4.0,
        max_edge=5.0,
    )

    assert result == "B"


def test_insufficient_periods_is_D():
    result = classify(
        periods=[
            make_period("2024-2026", 5.0),
            make_period("2022-2023", 4.0),
        ],
        mean_edge=4.5,
        median_edge=4.5,
        min_edge=4.0,
        max_edge=5.0,
    )

    assert result == "D"


def test_refine_row_tracks_2024_2026():
    result = refine_row(
        make_row([3.0, 4.0, 5.0, 4.0, 3.0])
    )

    assert result["stability_class"] == "A+"
    assert result["2024_2026_edge_pp"] == 3.0
    assert result["2024_2026_accuracy_pct"] == 53.0
    assert result["2024_2026_n"] == 500
    assert result["2024_2026_agrees_with_majority"] is True


def test_refine_row_detects_negative_majority():
    result = refine_row(
        make_row([-3.0, -4.0, -5.0, -4.0, -3.0])
    )

    assert result["majority_sign"] == "negative"
    assert result["stability_class"] == "A+"
