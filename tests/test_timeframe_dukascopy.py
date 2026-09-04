from datetime import datetime, timezone
from pathlib import Path

from xau_lean.data.dukascopy import DukascopyAdapter
from xau_lean.research.timeframe import (
    Timeframe,
    aggregate_bars,
    interval_start,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DUKASCOPY_ROOT = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "dukascopy"
    / "Market-Data-Lab-main"
    / "xauusd"
)

START = datetime(
    2014,
    5,
    1,
    0,
    0,
    tzinfo=timezone.utc,
)

END = datetime(
    2014,
    5,
    2,
    0,
    0,
    tzinfo=timezone.utc,
)


def test_may_2014_dukascopy_timeframe_aggregation():
    adapter = DukascopyAdapter(DUKASCOPY_ROOT)

    m1_bars = list(
        adapter.iter_bars(
            START,
            END,
            require_ask=True,
        )
    )

    # May 1, 2014 contains complete one-minute coverage
    # in the canonical Dukascopy BID/ASK dataset.
    assert len(m1_bars) == 1440

    candles_by_timeframe = {}

    for timeframe in Timeframe:
        candles = list(
            aggregate_bars(
                adapter.iter_bars(
                    START,
                    END,
                    require_ask=True,
                ),
                timeframe,
            )
        )

        candles_by_timeframe[timeframe] = candles

    for timeframe, candles in candles_by_timeframe.items():
        assert len(candles) > 0

        for candle in candles:
            assert candle.timestamp.tzinfo == timezone.utc

            assert (
                interval_start(candle.timestamp, timeframe)
                == candle.timestamp
            )

            assert candle.expected_m1 == timeframe.minutes
            assert 1 <= candle.observed_m1 <= timeframe.minutes
            assert 0.0 < candle.completeness_ratio <= 1.0

            # Complete M1 coverage means every emitted candle
            # should contain the full expected number of minutes.
            assert candle.complete

    # No timeframe may fabricate or duplicate M1 observations.
    for timeframe, candles in candles_by_timeframe.items():
        total_observed = sum(
            candle.observed_m1
            for candle in candles
        )

        assert total_observed == len(m1_bars)


def test_may_2014_dukascopy_bid_ask_are_aggregated_separately():
    adapter = DukascopyAdapter(DUKASCOPY_ROOT)

    candles = list(
        aggregate_bars(
            adapter.iter_bars(
                START,
                END,
                require_ask=True,
            ),
            Timeframe.H1,
        )
    )

    assert candles

    for candle in candles:
        assert candle.bid_open is not None
        assert candle.bid_high is not None
        assert candle.bid_low is not None
        assert candle.bid_close is not None

        assert candle.ask_open is not None
        assert candle.ask_high is not None
        assert candle.ask_low is not None
        assert candle.ask_close is not None

        # Normal quote relationship.
        assert candle.ask_close >= candle.bid_close


def test_may_2014_dukascopy_has_no_fabricated_missing_minutes():
    adapter = DukascopyAdapter(DUKASCOPY_ROOT)

    for timeframe in Timeframe:
        candles = list(
            aggregate_bars(
                adapter.iter_bars(
                    START,
                    END,
                    require_ask=True,
                ),
                timeframe,
            )
        )

        assert all(
            candle.observed_m1 == timeframe.minutes
            for candle in candles
        )
