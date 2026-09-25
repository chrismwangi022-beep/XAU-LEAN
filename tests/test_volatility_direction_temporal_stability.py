from __future__ import annotations

import pytest

from scripts.analyze_volatility_direction_temporal_stability import (
    HELD_OUT_PERIOD,
    TRAINING_PERIODS,
    screen_candidate,
)


def make_period_rows(
    training_estimates,
    held_out_estimate,
):
    rows = []

    for period, estimate in zip(
        TRAINING_PERIODS,
        training_estimates,
    ):
        rows.append(
            {
                "period": period,
                "estimate": estimate,
                "lower_95": estimate - 1.0,
                "upper_95": estimate + 1.0,
                "observations": 500,
            }
        )

    rows.append(
        {
            "period": HELD_OUT_PERIOD,
            "estimate": held_out_estimate,
            "lower_95": held_out_estimate - 1.0,
            "upper_95": held_out_estimate + 1.0,
            "observations": 500,
        }
    )

    return rows


def test_positive_candidate_passes_training_stability():
    result = screen_candidate(
        horizon="H1",
        bucket="high",
        statistic="accuracy_interaction",
        period_estimates=make_period_rows(
            [1.0, 2.0, 3.0, 2.0],
            1.5,
        ),
    )

    assert result["temporal_stability_pass"] is True
    assert result["training_sign"] == "positive"
    assert result["positive_training_periods"] == 4
    assert result["negative_training_periods"] == 0
    assert result["zero_training_periods"] == 0
    assert result["training_mean"] == 2.0
    assert result["held_out_agrees_with_training"] is True


def test_negative_candidate_passes_training_stability():
    result = screen_candidate(
        horizon="H3",
        bucket="extreme",
        statistic="signed_return_interaction",
        period_estimates=make_period_rows(
            [-1.0, -2.0, -3.0, -2.0],
            -0.5,
        ),
    )

    assert result["temporal_stability_pass"] is True
    assert result["training_sign"] == "negative"
    assert result["negative_training_periods"] == 4
    assert result["positive_training_periods"] == 0


def test_mixed_sign_candidate_fails():
    result = screen_candidate(
        horizon="H5",
        bucket="elevated",
        statistic="accuracy_interaction",
        period_estimates=make_period_rows(
            [1.0, -2.0, 3.0, 2.0],
            2.0,
        ),
    )

    assert result["temporal_stability_pass"] is False
    assert result["training_sign"] == "mixed_or_zero"
    assert result["positive_training_periods"] == 3
    assert result["negative_training_periods"] == 1


def test_zero_training_period_fails():
    result = screen_candidate(
        horizon="H1",
        bucket="high",
        statistic="accuracy_interaction",
        period_estimates=make_period_rows(
            [1.0, 0.0, 2.0, 3.0],
            2.0,
        ),
    )

    assert result["temporal_stability_pass"] is False
    assert result["zero_training_periods"] == 1


def test_held_out_period_does_not_change_training_pass():
    positive_held_out = screen_candidate(
        horizon="H1",
        bucket="high",
        statistic="accuracy_interaction",
        period_estimates=make_period_rows(
            [1.0, 2.0, 3.0, 4.0],
            5.0,
        ),
    )

    negative_held_out = screen_candidate(
        horizon="H1",
        bucket="high",
        statistic="accuracy_interaction",
        period_estimates=make_period_rows(
            [1.0, 2.0, 3.0, 4.0],
            -5.0,
        ),
    )

    assert positive_held_out["temporal_stability_pass"] is True
    assert negative_held_out["temporal_stability_pass"] is True

    assert positive_held_out["held_out_agrees_with_training"] is True
    assert negative_held_out["held_out_agrees_with_training"] is False
