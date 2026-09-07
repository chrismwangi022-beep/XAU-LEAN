from **future** import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from xau_lean.research.regime_accumulator import (
RegimeDistributionAccumulator,
)
from xau_lean.research.regimes import REGIMES

def make_candle(
timestamp: datetime,
*,
open_price: float,
high: float,
low: float,
close: float,
spread: float = 0.50,
complete: bool = True,
include_ask: bool = True,
):
values = {
"timestamp": timestamp,
"bid_open": open_price,
"bid_high": high,
"bid_low": low,
"bid_close": close,
"complete": complete,
}

```
if include_ask:
    values.update(
        {
            "ask_open": open_price + spread,
            "ask_high": high + spread,
            "ask_low": low + spread,
            "ask_close": close + spread,
        }
    )

return SimpleNamespace(**values)
```

def make_timestamp(
*,
year: int = 2010,
month: int = 1,
day: int = 4,
hour: int = 10,
minute: int = 0,
) -> datetime:
return datetime(
year,
month,
day,
hour,
minute,
tzinfo=timezone.utc,
)

def test_empty_accumulator_builds_valid_report():
accumulator = RegimeDistributionAccumulator(
REGIMES[0],
"M5",
)

```
report = accumulator.build_report()

assert report["timeframe"] == "M5"
assert report["regime"]["name"] == "2010-2014"

assert report["counts"]["candles"] == 0
assert report["counts"]["complete"] == 0
assert report["counts"]["incomplete"] == 0
assert report["counts"]["completeness_ratio"] is None

assert report["range"]["mean"] is None
assert report["body"]["mean"] is None
assert report["body_ratio"]["mean"] is None

assert report["volatility"]["log_return_std"] is None
assert report["volatility"]["atr_mean"] is None

assert report["direction"]["bullish"] == 0
assert report["direction"]["bearish"] == 0
assert report["direction"]["flat"] == 0
```

def test_accumulator_tracks_basic_distribution_statistics():
accumulator = RegimeDistributionAccumulator(
REGIMES[0],
"M5",
)

```
accumulator.update(
    make_candle(
        make_timestamp(),
        open_price=100.0,
        high=102.0,
        low=99.0,
        close=101.0,
    )
)

report = accumulator.build_report()

assert report["counts"]["candles"] == 1
assert report["counts"]["complete"] == 1
assert report["counts"]["incomplete"] == 0
assert report["counts"]["completeness_ratio"] == pytest.approx(1.0)

assert report["range"]["mean"] == pytest.approx(3.0)
assert report["body"]["mean"] == pytest.approx(1.0)
assert report["body_ratio"]["mean"] == pytest.approx(1 / 3)

assert report["direction"]["bullish"] == 1
assert report["direction"]["bearish"] == 0
assert report["direction"]["flat"] == 0

assert report["spread"]["mean"] == pytest.approx(0.50)
```

def test_accumulator_tracks_directional_persistence_and_reversal():
accumulator = RegimeDistributionAccumulator(
REGIMES[0],
"M5",
)

```
base = make_timestamp()

candles = [
    (100.0, 102.0, 99.0, 101.0),
    (101.0, 103.0, 100.0, 102.0),
    (102.0, 103.0, 99.0, 100.0),
]

for index, values in enumerate(candles):
    accumulator.update(
        make_candle(
            base.replace(minute=index * 5),
            open_price=values[0],
            high=values[1],
            low=values[2],
            close=values[3],
        )
    )

report = accumulator.build_report()

assert report["direction"]["bullish"] == 2
assert report["direction"]["bearish"] == 1
assert report["direction"]["flat"] == 0

assert report["directional_behavior"]["persistence"] == 1
assert report["directional_behavior"]["reversals"] == 1
assert report["directional_behavior"]["average_run_length"] == pytest.approx(1.5)
assert report["directional_behavior"]["max_run_length"] == 2
```

def test_flat_candles_are_counted_but_do_not_create_directional_runs():
accumulator = RegimeDistributionAccumulator(
REGIMES[0],
"M5",
)

```
base = make_timestamp()

candles = [
    (100.0, 101.0, 99.0, 101.0),
    (101.0, 101.0, 101.0, 101.0),
    (101.0, 102.0, 100.0, 100.0),
]

for index, values in enumerate(candles):
    accumulator.update(
        make_candle(
            base.replace(minute=index * 5),
            open_price=values[0],
            high=values[1],
            low=values[2],
            close=values[3],
        )
    )

report = accumulator.build_report()

assert report["direction"]["bullish"] == 1
assert report["direction"]["bearish"] == 1
assert report["direction"]["flat"] == 1

assert report["directional_behavior"]["persistence"] == 0
assert report["directional_behavior"]["reversals"] == 1
assert report["directional_behavior"]["max_run_length"] == 1
```

def test_accumulator_tracks_hour_and_weekday():
accumulator = RegimeDistributionAccumulator(
REGIMES[0],
"H1",
)

```
accumulator.update(
    make_candle(
        make_timestamp(hour=13),
        open_price=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
    )
)

report = accumulator.build_report()

assert report["hour_of_day"]["13"] == 1

# 2010-01-04 was Monday.
assert report["weekday"]["0"] == 1
```

def test_accumulator_rejects_naive_timestamp():
accumulator = RegimeDistributionAccumulator(
REGIMES[0],
"M5",
)

```
with pytest.raises(ValueError, match="timezone-aware"):
    accumulator.update(
        make_candle(
            datetime(
                2010,
                1,
                4,
                10,
                0,
            ),
            open_price=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
        )
    )
```

def test_accumulator_rejects_outside_regime():
accumulator = RegimeDistributionAccumulator(
REGIMES[0],
"M5",
)

```
timestamp = datetime(
    2015,
    1,
    1,
    0,
    0,
    tzinfo=timezone.utc,
)

with pytest.raises(ValueError, match="outside regime"):
    accumulator.update(
        make_candle(
            timestamp,
            open_price=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
        )
    )
```

def test_accumulator_rejects_non_chronological_input():
accumulator = RegimeDistributionAccumulator(
REGIMES[0],
"M5",
)

```
first = make_timestamp(minute=0)
second = make_timestamp(minute=0)

accumulator.update(
    make_candle(
        first,
        open_price=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
    )
)

with pytest.raises(
    ValueError,
    match="strictly chronological",
):
    accumulator.update(
        make_candle(
            second,
            open_price=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
        )
    )
```

def test_quantile_output_contains_default_quantiles():
accumulator = RegimeDistributionAccumulator(
REGIMES[0],
"M5",
)

```
base = make_timestamp()

for index in range(10):
    price = 100.0 + index

    accumulator.update(
        make_candle(
            base.replace(minute=index),
            open_price=price,
            high=price + 2.0,
            low=price - 1.0,
            close=price + 1.0,
        )
    )

report = accumulator.build_report()

expected_keys = {
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
    "p99",
}

assert set(report["range"]["quantiles"]) == expected_keys
```

def test_zero_range_has_zero_body_ratio():
accumulator = RegimeDistributionAccumulator(
REGIMES[0],
"M5",
)

```
accumulator.update(
    make_candle(
        make_timestamp(),
        open_price=100.0,
        high=100.0,
        low=100.0,
        close=100.0,
    )
)

report = accumulator.build_report()

assert report["range"]["mean"] == 0.0
assert report["body"]["mean"] == 0.0
assert report["body_ratio"]["mean"] == 0.0
```

def test_incomplete_candles_are_tracked():
accumulator = RegimeDistributionAccumulator(
REGIMES[0],
"M5",
)

```
accumulator.update(
    make_candle(
        make_timestamp(),
        open_price=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        complete=False,
    )
)

report = accumulator.build_report()

assert report["counts"]["candles"] == 1
assert report["counts"]["complete"] == 0
assert report["counts"]["incomplete"] == 1
assert report["counts"]["completeness_ratio"] == pytest.approx(0.0)
```

def test_missing_ask_data_is_tracked_without_breaking_bid_statistics():
accumulator = RegimeDistributionAccumulator(
REGIMES[0],
"M5",
)

```
accumulator.update(
    make_candle(
        make_timestamp(),
        open_price=100.0,
        high=102.0,
        low=99.0,
        close=101.0,
        include_ask=False,
    )
)

report = accumulator.build_report()

assert report["counts"]["missing_ask"] == 1
assert report["spread"]["mean"] is None
assert report["range"]["mean"] == pytest.approx(3.0)
```

def test_abnormal_spread_is_counted():
accumulator = RegimeDistributionAccumulator(
REGIMES[0],
"M5",
)

```
accumulator.update(
    make_candle(
        make_timestamp(),
        open_price=100.0,
        high=102.0,
        low=99.0,
        close=101.0,
        spread=1.50,
    )
)

report = accumulator.build_report()

assert report["counts"]["abnormal_spread"] == 1
assert report["spread"]["mean"] == pytest.approx(1.50)
```

def test_atr_is_not_available_until_period_is_reached():
accumulator = RegimeDistributionAccumulator(
REGIMES[0],
"M5",
atr_period=3,
)

```
base = make_timestamp()

for index in range(2):
    price = 100.0 + index

    accumulator.update(
        make_candle(
            base.replace(minute=index * 5),
            open_price=price,
            high=price + 2.0,
            low=price - 1.0,
            close=price + 1.0,
        )
    )

report = accumulator.build_report()

assert report["volatility"]["atr_mean"] is None
```

def test_atr_is_calculated_after_period_is_reached():
accumulator = RegimeDistributionAccumulator(
REGIMES[0],
"M5",
atr_period=3,
)

```
base = make_timestamp()

for index in range(3):
    price = 100.0 + index

    accumulator.update(
        make_candle(
            base.replace(minute=index * 5),
            open_price=price,
            high=price + 2.0,
            low=price - 1.0,
            close=price + 1.0,
        )
    )

report = accumulator.build_report()

assert report["volatility"]["atr_mean"] == pytest.approx(3.0)
```

def test_density_counts_unique_days_weeks_and_months():
accumulator = RegimeDistributionAccumulator(
REGIMES[0],
"M5",
)

```
timestamps = [
    datetime(2010, 1, 4, 10, 0, tzinfo=timezone.utc),
    datetime(2010, 1, 4, 10, 5, tzinfo=timezone.utc),
    datetime(2010, 1, 5, 10, 0, tzinfo=timezone.utc),
    datetime(2010, 2, 1, 10, 0, tzinfo=timezone.utc),
]

for index, timestamp in enumerate(timestamps):
    accumulator.update(
        make_candle(
            timestamp,
            open_price=100.0 + index,
            high=102.0 + index,
            low=99.0 + index,
            close=101.0 + index,
        )
    )

report = accumulator.build_report()

assert report["density"]["candles_per_day"] == pytest.approx(4 / 3)
assert report["density"]["candles_per_month"] == pytest.approx(2.0)
```

def test_acf_fields_are_present():
accumulator = RegimeDistributionAccumulator(
REGIMES[0],
"M5",
)

```
base = make_timestamp()

price = 100.0

for index in range(20):
    direction = 1.0 if index % 2 == 0 else -1.0
    close = price + direction

    accumulator.update(
        make_candle(
            base.replace(minute=index),
            open_price=price,
            high=max(price, close) + 1.0,
            low=min(price, close) - 1.0,
            close=close,
        )
    )

    price = close

report = accumulator.build_report()

abs_acf = report["volatility_clustering"]["abs_log_return_acf"]
squared_acf = report["volatility_clustering"]["squared_return_acf"]

assert "lag_1" in abs_acf
assert "lag_5" in abs_acf

assert "lag_1" in squared_acf
assert "lag_5" in squared_acf

assert abs_acf["lag_1"] is not None
assert abs_acf["lag_5"] is not None

assert squared_acf["lag_1"] is not None
assert squared_acf["lag_5"] is not None
```

def test_acf_constant_series_returns_zero():
accumulator = RegimeDistributionAccumulator(
REGIMES[0],
"M5",
)

```
base = make_timestamp()

price = 100.0

for index in range(10):
    accumulator.update(
        make_candle(
            base.replace(minute=index),
            open_price=price,
            high=price + 1.0,
            low=price - 1.0,
            close=price,
        )
    )

report = accumulator.build_report()

assert (
    report["volatility_clustering"]["abs_log_return_acf"]["lag_1"]
    == 0.0
)

assert (
    report["volatility_clustering"]["squared_return_acf"]["lag_1"]
    == 0.0
)
```

def test_invalid_constructor_arguments_are_rejected():
with pytest.raises(ValueError, match="atr_period"):
RegimeDistributionAccumulator(
REGIMES[0],
"M5",
atr_period=0,
)

```
with pytest.raises(ValueError, match="timeframe"):
    RegimeDistributionAccumulator(
        REGIMES[0],
        "",
    )

with pytest.raises(ValueError, match="acf_lags must not be empty"):
    RegimeDistributionAccumulator(
        REGIMES[0],
        "M5",
        acf_lags=(),
    )

with pytest.raises(
    ValueError,
    match="positive integers",
):
    RegimeDistributionAccumulator(
        REGIMES[0],
        "M5",
        acf_lags=(0, 1),
    )

with pytest.raises(
    ValueError,
    match="unique",
):
    RegimeDistributionAccumulator(
        REGIMES[0],
        "M5",
        acf_lags=(1, 1),
    )
```

def test_build_report_does_not_mutate_current_run():
accumulator = RegimeDistributionAccumulator(
REGIMES[0],
"M5",
)

```
base = make_timestamp()

for index in range(3):
    accumulator.update(
        make_candle(
            base.replace(minute=index * 5),
            open_price=100.0 + index,
            high=102.0 + index,
            low=99.0 + index,
            close=101.0 + index,
        )
    )

first_report = accumulator.build_report()
second_report = accumulator.build_report()

assert (
    first_report["directional_behavior"]
    == second_report["directional_behavior"]
)

assert accumulator.candle_count == 3
assert accumulator._run_count == 0
assert accumulator._current_run == 3
```
