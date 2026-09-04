from AlgorithmImports import *
from datetime import datetime, timezone, timedelta
from pathlib import Path


class DukascopyBid10(PythonData):

    def GetSource(self, config, date, isLive):
        path = (
            Path("/Lean/workspace")
            / "data/external/dukascopy/test/bid_10.csv"
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

        data = DukascopyBid10()
        data.Symbol = config.Symbol
        data.Time = dt
        data.EndTime = dt + timedelta(minutes=1)

        data.Value = float(parts[4])

        data["Open"] = float(parts[1])
        data["High"] = float(parts[2])
        data["Low"] = float(parts[3])
        data["Close"] = float(parts[4])

        return data


class DukascopyAsk10(PythonData):

    def GetSource(self, config, date, isLive):
        path = (
            Path("/Lean/workspace")
            / "data/external/dukascopy/test/ask_10.csv"
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

        data = DukascopyAsk10()
        data.Symbol = config.Symbol
        data.Time = dt
        data.EndTime = dt + timedelta(minutes=1)

        data.Value = float(parts[4])

        data["Open"] = float(parts[1])
        data["High"] = float(parts[2])
        data["Low"] = float(parts[3])
        data["Close"] = float(parts[4])

        return data


class DukascopyPythonData10Test(QCAlgorithm):

    def Initialize(self):

        self.SetStartDate(2014, 5, 1)
        self.SetEndDate(2014, 5, 1)

        self.SetCash(100000)

        self.bid = self.AddData(
            DukascopyBid10,
            "XAUUSD_BID",
            Resolution.Minute,
            None,
            False
        )

        self.ask = self.AddData(
            DukascopyAsk10,
            "XAUUSD_ASK",
            Resolution.Minute,
            None,
            False
        )

        self.bid_count = 0
        self.ask_count = 0

        self.Debug("=== DUKASCOPY 10-ROW PYTHONDATA TEST ===")
        self.Debug("SUBSCRIPTIONS CREATED")

    def OnData(self, data):

        if data.ContainsKey(self.bid.Symbol):
            self.bid_count += 1

            bar = data[self.bid.Symbol]

            if self.bid_count <= 3:
                self.Debug(
                    f"BID #{self.bid_count}: "
                    f"{bar.EndTime} "
                    f"O={bar.Open} "
                    f"H={bar.High} "
                    f"L={bar.Low} "
                    f"C={bar.Close}"
                )

        if data.ContainsKey(self.ask.Symbol):
            self.ask_count += 1

            bar = data[self.ask.Symbol]

            if self.ask_count <= 3:
                self.Debug(
                    f"ASK #{self.ask_count}: "
                    f"{bar.EndTime} "
                    f"O={bar.Open} "
                    f"H={bar.High} "
                    f"L={bar.Low} "
                    f"C={bar.Close}"
                )

    def OnEndOfAlgorithm(self):

        self.Debug("=== TEST COMPLETE ===")
        self.Debug(f"BID COUNT: {self.bid_count}")
        self.Debug(f"ASK COUNT: {self.ask_count}")

        if self.bid_count > 0 and self.ask_count > 0:
            self.Debug("RESULT: PASS — PythonData delivered BID and ASK")
        else:
            self.Debug("RESULT: FAIL — PythonData delivered no usable bars")
