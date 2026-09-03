from AlgorithmImports import *


class DataCoverageDiagnostic(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2014, 5, 1)
        self.SetEndDate(2014, 5, 31)
        self.SetCash(100000)

        self.symbol = self.AddCfd(
            "XAUUSD",
            Resolution.Minute,
            Market.Oanda,
            fillForward=False,
            leverage=1
        ).Symbol

        self.bar_count = 0
        self.first_time = None
        self.last_time = None
        self.daily_counts = {}

        self.Debug("=" * 60)
        self.Debug("XAUUSD QUOTE DATA COVERAGE DIAGNOSTIC")
        self.Debug("=" * 60)
        self.Debug(f"Symbol: {self.symbol}")
        self.Debug(
            f"Exchange Time Zone: "
            f"{self.Securities[self.symbol].Exchange.TimeZone}"
        )
        self.Debug(f"Algorithm Time Zone: {self.TimeZone}")
        self.Debug("=" * 60)

    def OnData(self, data: Slice):

        if not data.QuoteBars.ContainsKey(self.symbol):
            return

        quote = data.QuoteBars[self.symbol]

        self.bar_count += 1

        if self.first_time is None:
            self.first_time = quote.EndTime

            self.Debug(
                f"FIRST QUOTE BAR | "
                f"Time: {quote.EndTime} | "
                f"Bid: {quote.Bid.Close} | "
                f"Ask: {quote.Ask.Close}"
            )

        self.last_time = quote.EndTime

        date = quote.EndTime.date()

        if date not in self.daily_counts:
            self.daily_counts[date] = 0

        self.daily_counts[date] += 1

        if self.bar_count <= 5:
            self.Debug(
                f"QUOTE {self.bar_count} | "
                f"Time: {quote.EndTime} | "
                f"Bid: {quote.Bid.Close} | "
                f"Ask: {quote.Ask.Close}"
            )

    def OnEndOfAlgorithm(self):

        self.Debug("=" * 60)
        self.Debug("FINAL XAUUSD QUOTE COVERAGE RESULTS")
        self.Debug("=" * 60)

        self.Debug(f"Total XAUUSD minute quote bars: {self.bar_count}")
        self.Debug(f"First quote: {self.first_time}")
        self.Debug(f"Last quote: {self.last_time}")

        self.Debug("=" * 60)
        self.Debug("QUOTES PER DATE")
        self.Debug("=" * 60)

        for date in sorted(self.daily_counts):
            self.Debug(
                f"{date} -> {self.daily_counts[date]} quote bars"
            )

        self.Debug("=" * 60)
        self.Debug("END OF DIAGNOSTIC")
        self.Debug("=" * 60)