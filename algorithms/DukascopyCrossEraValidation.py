from AlgorithmImports import *
from datetime import datetime, timezone, timedelta
from pathlib import Path


BASE_PATH = Path(
    "/Lean/workspace/data/external/dukascopy/Market-Data-Lab-main/xauusd"
)


# Representative eras across the complete Dukascopy history.
TARGET_MONTHS = [
    (2010, 1),
    (2014, 5),
    (2018, 1),
    (2020, 3),
    (2022, 1),
    (2024, 1),
    (2026, 8),
]


class DukascopyBid(PythonData):

    def GetSource(self, config, date, isLive):

        path = (
            BASE_PATH
            / "bid"
            / "m1"
            / f"xauusd_bid_m1_{date.year:04d}_{date.month:02d}.csv"
        )

        return SubscriptionDataSource(
            str(path),
            SubscriptionTransportMedium.LocalFile,
            FileFormat.Csv
        )

    def Reader(self, config, line, date, isLive):

        if not line or line.startswith("timestamp"):
            return None

        parts = line.strip().split(",")

        if len(parts) != 5:
            return None

        try:
            timestamp = int(parts[0])

            dt = datetime.fromtimestamp(
                timestamp / 1000,
                tz=timezone.utc
            )

            data = DukascopyBid()

            data.Symbol = config.Symbol
            data.Time = dt
            data.EndTime = dt + timedelta(minutes=1)

            data.Value = float(parts[4])

            data["Open"] = float(parts[1])
            data["High"] = float(parts[2])
            data["Low"] = float(parts[3])
            data["Close"] = float(parts[4])

            return data

        except Exception:
            return None


class DukascopyAsk(PythonData):

    def GetSource(self, config, date, isLive):

        path = (
            BASE_PATH
            / "ask"
            / "m1"
            / f"xauusd_ask_m1_{date.year:04d}_{date.month:02d}.csv"
        )

        return SubscriptionDataSource(
            str(path),
            SubscriptionTransportMedium.LocalFile,
            FileFormat.Csv
        )

    def Reader(self, config, line, date, isLive):

        if not line or line.startswith("timestamp"):
            return None

        parts = line.strip().split(",")

        if len(parts) != 5:
            return None

        try:
            timestamp = int(parts[0])

            dt = datetime.fromtimestamp(
                timestamp / 1000,
                tz=timezone.utc
            )

            data = DukascopyAsk()

            data.Symbol = config.Symbol
            data.Time = dt
            data.EndTime = dt + timedelta(minutes=1)

            data.Value = float(parts[4])

            data["Open"] = float(parts[1])
            data["High"] = float(parts[2])
            data["Low"] = float(parts[3])
            data["Close"] = float(parts[4])

            return data

        except Exception:
            return None


class DukascopyCrossEraValidation(QCAlgorithm):

    def Initialize(self):

        # The test runs ONE representative month at a time.
        # This prevents LEAN from traversing the entire 2010-2026 history.
        year, month = TARGET_MONTHS[0]

        self.SetStartDate(year, month, 1)

        if month == 12:
            next_year = year + 1
            next_month = 1
        else:
            next_year = year
            next_month = month + 1

        self.SetEndDate(next_year, next_month, 1)

        self.SetCash(100000)

        self.bid_symbol = self.AddData(
            DukascopyBid,
            "XAUUSD_BID_CROSSERA",
            Resolution.Minute,
            None,
            False
        ).Symbol

        self.ask_symbol = self.AddData(
            DukascopyAsk,
            "XAUUSD_ASK_CROSSERA",
            Resolution.Minute,
            None,
            False
        ).Symbol

        self.target_month = f"{year:04d}-{month:02d}"

        self.bid_count = 0
        self.ask_count = 0
        self.aligned_count = 0
        self.misaligned_count = 0

        self.positive_spreads = 0
        self.zero_spreads = 0
        self.negative_spreads = 0

        self.first_bid = None
        self.last_bid = None
        self.first_ask = None
        self.last_ask = None

        self.Debug("========================================")
        self.Debug("DUKASCOPY CROSS-ERA VALIDATION")
        self.Debug("========================================")
        self.Debug(f"TARGET MONTH: {self.target_month}")
        self.Debug("========================================")

    def OnData(self, data):

        has_bid = data.ContainsKey(self.bid_symbol)
        has_ask = data.ContainsKey(self.ask_symbol)

        if has_bid:

            bid = data[self.bid_symbol]

            self.bid_count += 1

            if self.first_bid is None:
                self.first_bid = bid.EndTime

                self.Debug(
                    f"FIRST BID: {bid.EndTime} | "
                    f"PRICE={bid.Value}"
                )

            self.last_bid = bid.EndTime

        if has_ask:

            ask = data[self.ask_symbol]

            self.ask_count += 1

            if self.first_ask is None:
                self.first_ask = ask.EndTime

                self.Debug(
                    f"FIRST ASK: {ask.EndTime} | "
                    f"PRICE={ask.Value}"
                )

            self.last_ask = ask.EndTime

        if has_bid and has_ask:

            bid = data[self.bid_symbol]
            ask = data[self.ask_symbol]

            if bid.EndTime == ask.EndTime:

                self.aligned_count += 1

                spread = ask.Value - bid.Value

                if spread > 0:
                    self.positive_spreads += 1

                elif spread == 0:
                    self.zero_spreads += 1

                else:
                    self.negative_spreads += 1

            else:

                self.misaligned_count += 1

    def OnEndOfAlgorithm(self):

        self.Debug("========================================")
        self.Debug(f"RESULTS: {self.target_month}")
        self.Debug("========================================")

        self.Debug(
            f"BID BARS: {self.bid_count}"
        )

        self.Debug(
            f"ASK BARS: {self.ask_count}"
        )

        self.Debug(
            f"ALIGNED BARS: {self.aligned_count}"
        )

        self.Debug(
            f"MISALIGNED BARS: {self.misaligned_count}"
        )

        self.Debug(
            f"POSITIVE SPREADS: {self.positive_spreads}"
        )

        self.Debug(
            f"ZERO SPREADS: {self.zero_spreads}"
        )

        self.Debug(
            f"NEGATIVE SPREADS: {self.negative_spreads}"
        )

        self.Debug(
            f"FIRST BID: {self.first_bid}"
        )

        self.Debug(
            f"LAST BID: {self.last_bid}"
        )

        self.Debug(
            f"FIRST ASK: {self.first_ask}"
        )

        self.Debug(
            f"LAST ASK: {self.last_ask}"
        )

        # Basic validation for this representative month.
        passed = (
            self.bid_count > 0
            and self.ask_count > 0
            and self.aligned_count > 0
            and self.misaligned_count == 0
            and self.positive_spreads > 0
            and self.negative_spreads == 0
        )

        if passed:
            self.Debug(
                f"MONTH RESULT: PASS — {self.target_month}"
            )
        else:
            self.Debug(
                f"MONTH RESULT: FAIL — {self.target_month}"
            )

        self.Debug("========================================")
        self.Debug("NOTE:")
        self.Debug(
            "This executable test validates one representative "
            "month per run. The seven target months are listed "
            "in TARGET_MONTHS for the cross-era campaign."
        )
        self.Debug("========================================")


# ============================================================
# CROSS-ERA CAMPAIGN PLAN
# ============================================================
#
# Run this same algorithm with each target month below:
#
# 2010-01  -> oldest available era
# 2014-05  -> previously validated era
# 2018-01  -> mid-history
# 2020-03  -> extreme-volatility era
# 2022-01  -> recent historical era
# 2024-01  -> modern era
# 2026-08  -> latest available era
#
# Keeping each month isolated makes failures much easier to
# diagnose and prevents a 16-year continuous backtest.
#
