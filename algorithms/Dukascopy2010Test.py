from AlgorithmImports import *
from datetime import datetime, timezone, timedelta
from pathlib import Path


BASE_PATH = Path(
    "/Lean/workspace/data/external/dukascopy/test"
)


class DukascopyBid(PythonData):

    def GetSource(self, config, date, isLive):

        path = BASE_PATH / "bid_2010_1000.csv"

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

        path = BASE_PATH / "ask_2010_1000.csv"

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


class Dukascopy2010Test(QCAlgorithm):

    def Initialize(self):

        # Keep the algorithm date range tightly around
        # the actual timestamps in the test files.
        self.SetStartDate(2010, 1, 1)
        self.SetEndDate(2010, 1, 2)

        self.SetCash(100000)

        self.bid_symbol = self.AddData(
            DukascopyBid,
            "XAUUSD_BID_2010",
            Resolution.Minute,
            None,
            False
        ).Symbol

        self.ask_symbol = self.AddData(
            DukascopyAsk,
            "XAUUSD_ASK_2010",
            Resolution.Minute,
            None,
            False
        ).Symbol

        self.bid_count = 0
        self.ask_count = 0
        self.aligned_count = 0
        self.misaligned_count = 0

        self.positive_spreads = 0
        self.zero_spreads = 0
        self.negative_spreads = 0

        self.first_bid = None
        self.first_ask = None
        self.last_bid = None
        self.last_ask = None

        self.Debug("========================================")
        self.Debug("DUKASCOPY 2010 ISOLATION TEST")
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
        self.Debug("2010 ISOLATION TEST RESULTS")
        self.Debug("========================================")

        self.Debug(f"BID BARS: {self.bid_count}")
        self.Debug(f"ASK BARS: {self.ask_count}")
        self.Debug(f"ALIGNED BARS: {self.aligned_count}")
        self.Debug(f"MISALIGNED BARS: {self.misaligned_count}")

        self.Debug(
            f"POSITIVE SPREADS: {self.positive_spreads}"
        )

        self.Debug(
            f"ZERO SPREADS: {self.zero_spreads}"
        )

        self.Debug(
            f"NEGATIVE SPREADS: {self.negative_spreads}"
        )

        self.Debug(f"FIRST BID: {self.first_bid}")
        self.Debug(f"LAST BID: {self.last_bid}")
        self.Debug(f"FIRST ASK: {self.first_ask}")
        self.Debug(f"LAST ASK: {self.last_ask}")

        passed = (
            self.bid_count == 1000
            and self.ask_count == 1000
            and self.aligned_count == 1000
            and self.misaligned_count == 0
            and self.positive_spreads > 0
            and self.negative_spreads == 0
        )

        if passed:
            self.Debug("RESULT: PASS")
        else:
            self.Debug("RESULT: FAIL")

        self.Debug("========================================")
