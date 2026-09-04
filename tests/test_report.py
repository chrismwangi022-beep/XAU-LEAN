from datetime import datetime, timezone

from xau_lean.research.report import ResearchMetadata, ResearchReport
from xau_lean.research.timeframe import Timeframe


def test_research_metadata():
    metadata = ResearchMetadata(
        source="Dukascopy",
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2020, 2, 1, tzinfo=timezone.utc),
        timeframe=Timeframe.H1,
        timezone="UTC",
        aggregation="fixed UTC calendar intervals",
        weekend_policy="preserve",
        spread_warning=1.0,
        generated_at=datetime(2026, 9, 4, tzinfo=timezone.utc),
    )

    assert metadata.source == "Dukascopy"
    assert metadata.timeframe is Timeframe.H1
    assert metadata.timezone == "UTC"


def test_research_report_to_dict():
    metadata = ResearchMetadata(
        source="Dukascopy",
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2020, 2, 1, tzinfo=timezone.utc),
        timeframe=Timeframe.M15,
        timezone="UTC",
        aggregation="fixed UTC calendar intervals",
        weekend_policy="preserve",
        spread_warning=1.0,
        generated_at=datetime(2026, 9, 4, tzinfo=timezone.utc),
    )

    report = ResearchReport(
        metadata=metadata,
        candle_count=100,
        complete_candle_count=95,
        incomplete_candle_count=5,
        bullish_count=50,
        bearish_count=45,
        flat_count=5,
        persistence_rate=0.55,
        reversal_rate=0.45,
        mean_range=10.0,
        median_range=9.0,
        mean_body=6.0,
        median_body=5.5,
        mean_body_ratio=0.60,
        realized_volatility=0.01,
        mean_atr=12.0,
        mean_spread=0.50,
        median_spread=0.40,
        mean_spread_to_range=0.05,
        spread_ge_050_pct=0.40,
        spread_ge_100_pct=0.10,
        spread_ge_200_pct=0.02,
        spread_ge_500_pct=0.01,
        candles_per_day=10.0,
        candles_per_week=50.0,
        candles_per_month=200.0,
    )

    result = report.to_dict()

    assert result["metadata"]["timeframe"] == "M15"
    assert result["candle_count"] == 100
    assert result["complete_candle_count"] == 95
    assert result["incomplete_candle_count"] == 5
    assert result["mean_range"] == 10.0
    assert result["spread_ge_100_pct"] == 0.10
