from AlgorithmImports import *
from datetime import datetime, timezone, timedelta
from pathlib import Path


class DukascopyBid(PythonData):

    def GetSource(self, config, date, isLive):

        path = (
            Path("/Lean/workspace")
            / "data/external/dukascopy/Market-Data-Lab-main/xauusd/bid/m1/xauusd_bid_m1_2014_05.csv"
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
            Path("/Lean/workspace")
            / "data/external/dukascopy/Market-Data-Lab-main/xauusd/ask/m1/xauusd_ask_m1_2014_05.csv"
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


class DukascopyBidAskFullMonthTest(QCAlgorithm):

    def Initialize(self):

        self.SetStartDate(2014, 5, 1)
        self.SetEndDate(2014, 5, 11)

        self.SetCash(100000)

        self.bid_symbol = self.AddData(
            DukascopyBid,
            "XAUUSD_BID_TEST",
            Resolution.Minute,
            None,
            False
        ).Symbol

        self.ask_symbol = self.AddData(
            DukascopyAsk,
            "XAUUSD_ASK_TEST",
            Resolution.Minute,
            None,
            False
        ).Symbol

        self.bid_count = 0
        self.ask_count = 0
        self.aligned_count = 0
        self.misaligned_count = 0
        self.first_logged = False

        self.Debug("========================================")
        self.Debug("DUKASCOPY BID + ASK FULL MONTH TEST")
        self.Debug("SUBSCRIPTIONS CREATED")
        self.Debug("========================================")

    def OnData(self, data):

        has_bid = data.ContainsKey(self.bid_symbol)
        has_ask = data.ContainsKey(self.ask_symbol)

        if has_bid:
            self.bid_count += 1

        if has_ask:
            self.ask_count += 1

        if has_bid and has_ask:

            bid = data[self.bid_symbol]
            ask = data[self.ask_symbol]

            if bid.EndTime == ask.EndTime:

                self.aligned_count += 1

                spread = ask.Value - bid.Value

                if not self.first_logged:

                    self.Debug(
                        f"FIRST ALIGNED BAR: {bid.EndTime} "
                        f"BID={bid.Value} "
                        f"ASK={ask.Value} "
                        f"SPREAD={spread}"
                    )

                    self.first_logged = True

            else:

                self.misaligned_count += 1

    def OnEndOfAlgorithm(self):

        self.Debug("========================================")
        self.Debug("TEST COMPLETE")
        self.Debug(f"BID BARS: {self.bid_count}")
        self.Debug(f"ASK BARS: {self.ask_count}")
        self.Debug(f"ALIGNED BARS: {self.aligned_count}")
        self.Debug(f"MISALIGNED BARS: {self.misaligned_count}")

        if (
            self.bid_count > 0
            and self.ask_count > 0
            and self.aligned_count > 0
            and self.misaligned_count == 0
        ):
            self.Debug("RESULT: PASS")
        else:
            self.Debug("RESULT: FAIL")

        self.Debug("========================================")
