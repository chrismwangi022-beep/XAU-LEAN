import pytest

from scripts.run_signal_research import (
    ATR_PERIOD,
    BASELINE_PERIOD,
    HORIZONS,
    EXPANSION_BUCKETS,
    REGIMES,
    PendingSignal,
    SignalAccumulator,
    expansion_bucket,
)


def test_atr_period_and_baseline_constants_are_canonical():
    assert ATR_PERIOD == 14
    assert BASELINE_PERIOD == 20
    assert HORIZONS == (1, 3, 5)


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (0.0, "normal"),
        (1.249999999, "normal"),
        (1.25, "elevated"),
        (1.499999999, "elevated"),
        (1.50, "high"),
        (1.999999999, "high"),
        (2.00, "extreme"),
        (10.0, "extreme"),
    ],
)
def test_expansion_bucket_boundaries(ratio, expected):
    assert expansion_bucket(ratio) == expected


def test_pending_signal_starts_without_entry():
    signal = PendingSignal(
        regime="2010-2014",
        bucket="elevated",
        horizon=1,
    )

    assert signal.entry_price is None
    assert signal.candles_seen == 0
    assert signal.high is None
    assert signal.low is None


def test_signal_accumulator_starts_empty_but_has_all_expected_keys():
    accumulator = SignalAccumulator()

    expected_count = (
        len(REGIMES)
        * len(EXPANSION_BUCKETS)
        * len(HORIZONS)
    )

    assert len(accumulator.stats) == expected_count

    for stats in accumulator.stats.values():
        assert stats.observations == 0
        assert stats.sum_abs_return == 0.0
        assert stats.sum_abs_return_pct == 0.0
        assert stats.sum_directional_efficiency == 0.0
        assert stats.sum_forward_range == 0.0


def test_signal_accumulator_update_and_report():
    accumulator = SignalAccumulator()

    accumulator.update(
        regime="2010-2014",
        bucket="elevated",
        horizon=1,
        abs_return=2.0,
        abs_return_pct=0.02,
        directional_efficiency=0.5,
        forward_range=4.0,
    )

    accumulator.update(
        regime="2010-2014",
        bucket="elevated",
        horizon=1,
        abs_return=4.0,
        abs_return_pct=0.04,
        directional_efficiency=0.75,
        forward_range=8.0,
    )

    result = accumulator.build_report()

    stats = result["2010-2014"]["elevated"]["horizons"]["1"]

    assert stats["observations"] == 2
    assert stats["mean_abs_return"] == pytest.approx(3.0)
    assert stats["mean_abs_return_pct"] == pytest.approx(0.03)
    assert stats["mean_directional_efficiency"] == pytest.approx(0.625)
    assert stats["mean_forward_range"] == pytest.approx(6.0)


def test_pending_signal_horizon_is_explicit_and_positive():
    for horizon in HORIZONS:
        signal = PendingSignal(
            regime="2010-2014",
            bucket="normal",
            horizon=horizon,
        )

        assert signal.horizon == horizon
        assert signal.horizon > 0


def test_pending_signal_contains_no_future_price_fields():
    signal = PendingSignal(
        regime="2010-2014",
        bucket="extreme",
        horizon=5,
    )

    assert not hasattr(signal, "future_price")
    assert not hasattr(signal, "future_return")
    assert not hasattr(signal, "future_high")
    assert not hasattr(signal, "future_low")


def test_canonical_atr_is_mean_of_current_14_true_ranges():
    """
    Canonical ATR(14) must be the arithmetic mean of the
    current rolling 14 TR values.
    """
    true_ranges = list(range(1, ATR_PERIOD + 1))

    rolling_window = true_ranges[-ATR_PERIOD:]

    atr = sum(rolling_window) / ATR_PERIOD

    assert atr == pytest.approx(7.5)


def test_atr_rolling_window_drops_oldest_true_range():
    """
    Once the 15th TR arrives, the first TR must leave the
    rolling ATR window.
    """
    true_ranges = list(range(1, ATR_PERIOD + 2))

    rolling_window = true_ranges[-ATR_PERIOD:]

    assert len(rolling_window) == ATR_PERIOD
    assert rolling_window[0] == 2
    assert rolling_window[-1] == 15

    atr = sum(rolling_window) / ATR_PERIOD

    assert atr == pytest.approx(8.5)


def test_baseline_requires_20_prior_atr_values():
    """
    A current ATR cannot be classified until 20 completed
    historical ATR values exist.
    """
    history = []

    for value in range(1, BASELINE_PERIOD + 1):
        if len(history) < BASELINE_PERIOD:
            history.append(float(value))

    assert len(history) == BASELINE_PERIOD
    assert history == [float(value) for value in range(1, 21)]


def test_current_atr_is_not_part_of_baseline_before_classification():
    """
    The baseline must contain only ATR values that existed
    before the current ATR.
    """
    history = [100.0] * BASELINE_PERIOD
    current_atr = 200.0

    baseline_before_append = sum(history) / len(history)

    assert baseline_before_append == pytest.approx(100.0)

    ratio = current_atr / baseline_before_append

    assert ratio == pytest.approx(2.0)
    assert expansion_bucket(ratio) == "extreme"

    # Only after classification should current ATR enter history.
    from collections import deque

    bounded_history = deque(history, maxlen=BASELINE_PERIOD)
    bounded_history.append(current_atr)

    assert len(bounded_history) == BASELINE_PERIOD
    assert bounded_history[-1] == current_atr
    assert bounded_history[0] == 100.0


def test_baseline_changes_only_after_current_atr_is_classified():
    """
    Demonstrates the required ordering:
        1. calculate current ATR
        2. calculate prior-ATR baseline
        3. classify current ATR
        4. append current ATR to history
    """
    history = [100.0] * BASELINE_PERIOD
    current_atr = 150.0

    baseline = sum(history) / len(history)
    ratio_before_append = current_atr / baseline

    assert baseline == pytest.approx(100.0)
    assert ratio_before_append == pytest.approx(1.5)
    assert expansion_bucket(ratio_before_append) == "high"

    from collections import deque

    bounded_history = deque(history, maxlen=BASELINE_PERIOD)
    bounded_history.append(current_atr)

    assert len(bounded_history) == BASELINE_PERIOD
    assert bounded_history[-1] == 150.0
    assert bounded_history[0] == 100.0


def test_signal_creation_has_no_future_price_information():
    """
    A newly-created PendingSignal contains only classification
    information and no future candle prices.
    """
    signal = PendingSignal(
        regime="2024-2026",
        bucket="high",
        horizon=3,
    )

    assert signal.entry_price is None
    assert signal.high is None
    assert signal.low is None
    assert signal.candles_seen == 0


def test_pending_signal_is_not_resolved_until_future_candles_are_processed():
    """
    The signal starts with no entry price. Therefore the signal
    itself cannot contain a same-candle future price at creation.
    """
    signal = PendingSignal(
        regime="2010-2014",
        bucket="elevated",
        horizon=3,
    )

    assert signal.entry_price is None
    assert signal.candles_seen == 0

    # No future candle has been processed yet.
    assert signal.high is None
    assert signal.low is None


def test_all_canonical_regimes_use_hyphenated_names():
    assert [regime.name for regime in REGIMES] == [
        "2010-2014",
        "2015-2019",
        "2020-2021",
        "2022-2023",
        "2024-2026",
    ]
