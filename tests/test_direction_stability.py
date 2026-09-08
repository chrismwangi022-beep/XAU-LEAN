from scripts.analyze_direction_stability import (
    MIN_SAMPLE_SIZE,
    REGIME_ORDER,
    analyze_group,
    classify_stability,
    hypothesis_key,
    sign_of_edge,
)


def make_row(
    regime,
    timeframe="H4",
    bucket="elevated",
    direction="bullish",
    horizon=3,
    edge=5.0,
    n=500,
    fdr=False,
):
    return {
        "regime": regime,
        "timeframe": timeframe,
        "bucket": bucket,
        "direction": direction,
        "horizon_candles": horizon,
        "edge_pp": edge,
        "directional_accuracy": 0.50 + edge / 100,
        "ci_95_lower_pct": 50.0,
        "ci_95_upper_pct": 60.0,
        "p_value": 0.01,
        "p_value_fdr": 0.02,
        "fdr_significant_05": fdr,
        "directional_observations": n,
    }


def test_sign_of_edge():
    assert sign_of_edge(2.0) == 1
    assert sign_of_edge(-2.0) == -1
    assert sign_of_edge(0.0) == 0


def test_hypothesis_key_excludes_regime():
    row = make_row("2024-2026")

    assert hypothesis_key(row) == (
        "H4",
        "elevated",
        "bullish",
        3,
    )


def test_class_a_positive_stability():
    assert (
        classify_stability(
            available_periods=5,
            adequate_sample_periods=5,
            positive_periods=5,
            negative_periods=0,
            neutral_periods=0,
            significant_positive_periods=3,
            significant_negative_periods=0,
            min_sample_size=500,
        )
        == "A"
    )


def test_class_a_negative_stability():
    assert (
        classify_stability(
            available_periods=5,
            adequate_sample_periods=5,
            positive_periods=0,
            negative_periods=5,
            neutral_periods=0,
            significant_positive_periods=0,
            significant_negative_periods=3,
            min_sample_size=500,
        )
        == "A"
    )


def test_class_c_contradictory():
    assert (
        classify_stability(
            available_periods=5,
            adequate_sample_periods=5,
            positive_periods=3,
            negative_periods=2,
            neutral_periods=0,
            significant_positive_periods=1,
            significant_negative_periods=1,
            min_sample_size=500,
        )
        == "C"
    )


def test_class_d_insufficient_sample():
    assert (
        classify_stability(
            available_periods=5,
            adequate_sample_periods=2,
            positive_periods=2,
            negative_periods=0,
            neutral_periods=0,
            significant_positive_periods=1,
            significant_negative_periods=0,
            min_sample_size=20,
        )
        == "D"
    )


def test_analyze_group_preserves_chronological_periods():
    rows = [
        make_row("2024-2026", edge=8.0),
        make_row("2010-2014", edge=2.0),
        make_row("2020-2021", edge=5.0),
        make_row("2015-2019", edge=3.0),
        make_row("2022-2023", edge=4.0),
    ]

    result = analyze_group(
        ("H4", "elevated", "bullish", 3),
        rows,
    )

    assert [
        period["regime"]
        for period in result["periods"]
    ] == REGIME_ORDER

    assert result["periods_available"] == 5
    assert result["periods_adequate_sample"] == 5
    assert result["positive_periods"] == 5
    assert result["negative_periods"] == 0
    assert result["stability_class"] == "A"
    assert result["min_sample_size"] == 500
    assert result["best_edge_pp"] == 8.0
    assert result["worst_edge_pp"] == 2.0


def test_small_sample_period_is_not_adequate():
    rows = [
        make_row(
            "2010-2014",
            edge=20.0,
            n=20,
        ),
        make_row(
            "2015-2019",
            edge=5.0,
            n=500,
        ),
        make_row(
            "2020-2021",
            edge=5.0,
            n=500,
        ),
        make_row(
            "2022-2023",
            edge=5.0,
            n=500,
        ),
        make_row(
            "2024-2026",
            edge=5.0,
            n=500,
        ),
    ]

    result = analyze_group(
        ("H4", "elevated", "bullish", 3),
        rows,
    )

    assert result["periods_available"] == 5
    assert result["periods_adequate_sample"] == 4
    assert result["min_sample_size"] == 500
    assert result["stability_class"] == "A"


def test_missing_period_is_handled():
    rows = [
        make_row("2010-2014"),
        make_row("2015-2019"),
        make_row("2020-2021"),
        make_row("2024-2026"),
    ]

    result = analyze_group(
        ("H4", "elevated", "bullish", 3),
        rows,
    )

    assert result["periods_available"] == 4

    missing = [
        period
        for period in result["periods"]
        if period["regime"] == "2022-2023"
    ][0]

    assert missing["available"] is False
