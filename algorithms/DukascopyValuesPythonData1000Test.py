from AlgorithmImports import *
from datetime import datetime, timezone, timedelta
from pathlib import Path


class DukascopyValues(PythonData):

    def GetSource(self, config, date, isLive):

        path = (
            Path("/Lean/workspace")
            / "data/external/dukascopy/test/bid_10000.csv"
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

        data = DukascopyValues()

        data.Symbol = config.Symbol
        data.Time = dt
        data.EndTime = dt + timedelta(minutes=1)

        data.Value = float(parts[4])

        data["Open"] = float(parts[1])
        data["High"] = float(parts[2])
        data["Low"] = float(parts[3])
        data["Close"] = float(parts[4])

        return data


class DukascopyValuesPythonDataTest(QCAlgorithm):

    def Initialize(self):

        self.SetStartDate(2014, 5, 1)
        self.SetEndDate(2014, 5, 1)

        self.SetCash(100000)

        self.symbol = self.AddData(
            DukascopyValues,
            "XAUUSD_BID_TEST",
            Resolution.Minute,
            None,
            False
        ).Symbol

        self.count = 0

        self.Debug("========================================")
        self.Debug("DUKASCOPY VALUES PYTHONDATA TEST")
        self.Debug("SUBSCRIPTION CREATED")
        self.Debug("========================================")

    def OnData(self, data):

        if data.ContainsKey(self.symbol):

            self.count += 1

            bar = data[self.symbol]

            self.Debug(
                f"BAR #{self.count}: "
                f"{bar.EndTime} "
                f"O={bar.Open} "
                f"H={bar.High} "
                f"L={bar.Low} "
                f"C={bar.Close}"
            )

    def OnEndOfAlgorithm(self):

        self.Debug("========================================")
        self.Debug("TEST COMPLETE")
        self.Debug(f"BARS RECEIVED: {self.count}")

        if self.count > 0:
            self.Debug("RESULT: PASS")
        else:
            self.Debug("RESULT: FAIL")

        self.Debug("========================================")
