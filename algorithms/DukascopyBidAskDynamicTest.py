from AlgorithmImports import *
from datetime import datetime, timezone, timedelta
from pathlib import Path


BASE_PATH = Path(
    "/Lean/workspace/data/external/dukascopy/Market-Data-Lab-main/xauusd"
)


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


class DukascopyBidAskDynamicTest(QCAlgorithm):

    def Initialize(self):

        self.SetStartDate(2014, 5, 1)
        self.SetEndDate(2014, 8, 1)

        self.SetCash(100000)

        self.bid_symbol = self.AddData(
            DukascopyBid,
            "XAUUSD_BID_DYNAMIC",
            Resolution.Minute,
            None,
            False
        ).Symbol

        self.ask_symbol = self.AddData(
            DukascopyAsk,
            "XAUUSD_ASK_DYNAMIC",
            Resolution.Minute,
            None,
            False
        ).Symbol

        self.bid_count = 0
        self.ask_count = 0
        self.aligned_count = 0
        self.misaligned_count = 0

        self.months_seen = set()
        self.first_bar_by_month = {}

        self.Debug("========================================")
        self.Debug("DUKASCOPY DYNAMIC BID + ASK TEST")
        self.Debug("========================================")
        self.Debug("TEST PERIOD: 2014-05 through 2014-08-01")
        self.Debug("========================================")

    def OnData(self, data):

        has_bid = data.ContainsKey(self.bid_symbol)
        has_ask = data.ContainsKey(self.ask_symbol)

        if has_bid:

            self.bid_count += 1

            bid = data[self.bid_symbol]

            month_key = bid.EndTime.strftime("%Y-%m")
            self.months_seen.add(month_key)

            if month_key not in self.first_bar_by_month:

                self.first_bar_by_month[month_key] = bid.EndTime

                self.Debug(
                    f"NEW BID MONTH: {month_key} "
                    f"FIRST BAR={bid.EndTime} "
                    f"PRICE={bid.Value}"
                )

        if has_ask:
            self.ask_count += 1

        if has_bid and has_ask:

            bid = data[self.bid_symbol]
            ask = data[self.ask_symbol]

            if bid.EndTime == ask.EndTime:

                self.aligned_count += 1

                if self.aligned_count == 1:

                    spread = ask.Value - bid.Value

                    self.Debug(
                        f"FIRST ALIGNED BAR: {bid.EndTime} "
                        f"BID={bid.Value} "
                        f"ASK={ask.Value} "
                        f"SPREAD={spread}"
                    )

            else:

                self.misaligned_count += 1

    def OnEndOfAlgorithm(self):

        self.Debug("========================================")
        self.Debug("TEST COMPLETE")
        self.Debug(f"BID BARS: {self.bid_count}")
        self.Debug(f"ASK BARS: {self.ask_count}")
        self.Debug(f"ALIGNED BARS: {self.aligned_count}")
        self.Debug(f"MISALIGNED BARS: {self.misaligned_count}")
        self.Debug(f"MONTHS SEEN: {sorted(self.months_seen)}")

        for month in sorted(self.first_bar_by_month):

            self.Debug(
                f"MONTH CHECK: {month} "
                f"FIRST BAR={self.first_bar_by_month[month]}"
            )

        if (
            self.bid_count > 0
            and self.ask_count > 0
            and self.aligned_count > 0
            and self.misaligned_count == 0
            and len(self.months_seen) == 4
        ):

            self.Debug("RESULT: PASS")

        else:

            self.Debug("RESULT: FAIL")

        self.Debug("========================================")
