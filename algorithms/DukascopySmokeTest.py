from AlgorithmImports import *
from pathlib import Path
from datetime import datetime, timezone, timedelta


class DukascopyBid(PythonData):

    def GetSource(self, config, date, isLive):
        path = (
            Path("/Lean/workspace")
            / "data/external/dukascopy/Market-Data-Lab-main/xauusd"
            / "bid/m1"
            / "xauusd_bid_m1_2014_05.csv"
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
        dt = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)

        # Only emit data belonging to the requested date.
        if dt.date() != date.date():
            return None

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
            / "data/external/dukascopy/Market-Data-Lab-main/xauusd"
            / "ask/m1"
            / "xauusd_ask_m1_2014_05.csv"
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
        dt = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)

        # Only emit data belonging to the requested date.
        if dt.date() != date.date():
            return None

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


class DukascopySmokeTest(QCAlgorithm):

    def Initialize(self):

        self.SetStartDate(2014, 5, 1)
        self.SetEndDate(2014, 5, 2)
        self.SetCash(100000)
        self.SetTimeZone("UTC")

        self.bid = self.AddData(
            DukascopyBid,
            "XAUUSD_BID",
            Resolution.Minute,
            None,
            False
        )

        self.ask = self.AddData(
            DukascopyAsk,
            "XAUUSD_ASK",
            Resolution.Minute,
            None,
            False
        )

        self.bid_symbol = self.bid.Symbol
        self.ask_symbol = self.ask.Symbol

        self.count_bid = 0
        self.count_ask = 0

        self.first_bid = None
        self.first_ask = None
        self.last_bid = None
        self.last_ask = None

        self.Log("=" * 80)
        self.Log("XAU-LEAN — DUKASCOPY LEAN SMOKE TEST")
        self.Log("=" * 80)

    def OnData(self, data):

        if data.ContainsKey(self.bid_symbol):

            bar = data[self.bid_symbol]
            self.count_bid += 1

            if self.first_bid is None:

                self.first_bid = bar

                self.Log(
                    f"FIRST BID: {bar.EndTime} "
                    f"O={bar['Open']} "
                    f"H={bar['High']} "
                    f"L={bar['Low']} "
                    f"C={bar['Close']}"
                )

            self.last_bid = bar

        if data.ContainsKey(self.ask_symbol):

            bar = data[self.ask_symbol]
            self.count_ask += 1

            if self.first_ask is None:

                self.first_ask = bar

                self.Log(
                    f"FIRST ASK: {bar.EndTime} "
                    f"O={bar['Open']} "
                    f"H={bar['High']} "
                    f"L={bar['Low']} "
                    f"C={bar['Close']}"
                )

            self.last_ask = bar

    def OnEndOfAlgorithm(self):

        self.Log("")
        self.Log("=" * 80)
        self.Log("FINAL RESULTS")
        self.Log("=" * 80)

        self.Log(f"BID bars received: {self.count_bid}")
        self.Log(f"ASK bars received: {self.count_ask}")

        if self.last_bid:

            self.Log(
                f"LAST BID: {self.last_bid.EndTime} "
                f"C={self.last_bid['Close']}"
            )

        if self.last_ask:

            self.Log(
                f"LAST ASK: {self.last_ask.EndTime} "
                f"C={self.last_ask['Close']}"
            )

        if self.count_bid == self.count_ask:
            self.Log("BID/ASK COUNT: PASS")
        else:
            self.Log("BID/ASK COUNT: REVIEW")

        self.Log("=" * 80)
