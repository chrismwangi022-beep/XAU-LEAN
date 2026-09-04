from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .timeframe import Timeframe


@dataclass(frozen=True, slots=True)
class ResearchMetadata:
    """
    Reproducibility metadata for a timeframe research run.
    """

    source: str
    start: datetime
    end: datetime
    timeframe: Timeframe
    timezone: str
    aggregation: str
    weekend_policy: str
    spread_warning: float
    generated_at: datetime


@dataclass(frozen=True, slots=True)
class ResearchReport:
    """
    Standardized research output for one timeframe.

    The report intentionally stores descriptive statistics rather
    than a composite score. Timeframe selection must be evidence-based,
    not driven by arbitrary weighting.
    """

    metadata: ResearchMetadata

    candle_count: int
    complete_candle_count: int
    incomplete_candle_count: int

    bullish_count: int
    bearish_count: int
    flat_count: int

    persistence_rate: float | None
    reversal_rate: float | None

    mean_range: float | None
    median_range: float | None
    mean_body: float | None
    median_body: float | None
    mean_body_ratio: float | None

    realized_volatility: float | None
    mean_atr: float | None

    mean_spread: float | None
    median_spread: float | None
    mean_spread_to_range: float | None

    spread_ge_050_pct: float
    spread_ge_100_pct: float
    spread_ge_200_pct: float
    spread_ge_500_pct: float

    candles_per_day: float | None
    candles_per_week: float | None
    candles_per_month: float | None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the report into a deterministic serializable dictionary.
        """

        return {
            "metadata": {
                "source": self.metadata.source,
                "start": self.metadata.start.isoformat(),
                "end": self.metadata.end.isoformat(),
                "timeframe": self.metadata.timeframe.value,
                "timezone": self.metadata.timezone,
                "aggregation": self.metadata.aggregation,
                "weekend_policy": self.metadata.weekend_policy,
                "spread_warning": self.metadata.spread_warning,
                "generated_at": self.metadata.generated_at.isoformat(),
            },
            "candle_count": self.candle_count,
            "complete_candle_count": self.complete_candle_count,
            "incomplete_candle_count": self.incomplete_candle_count,
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "flat_count": self.flat_count,
            "persistence_rate": self.persistence_rate,
            "reversal_rate": self.reversal_rate,
            "mean_range": self.mean_range,
            "median_range": self.median_range,
            "mean_body": self.mean_body,
            "median_body": self.median_body,
            "mean_body_ratio": self.mean_body_ratio,
            "realized_volatility": self.realized_volatility,
            "mean_atr": self.mean_atr,
            "mean_spread": self.mean_spread,
            "median_spread": self.median_spread,
            "mean_spread_to_range": self.mean_spread_to_range,
            "spread_ge_050_pct": self.spread_ge_050_pct,
            "spread_ge_100_pct": self.spread_ge_100_pct,
            "spread_ge_200_pct": self.spread_ge_200_pct,
            "spread_ge_500_pct": self.spread_ge_500_pct,
            "candles_per_day": self.candles_per_day,
            "candles_per_week": self.candles_per_week,
            "candles_per_month": self.candles_per_month,
        }
