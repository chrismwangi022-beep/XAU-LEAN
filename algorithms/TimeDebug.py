from AlgorithmImports import *


class TimeDebug(QCAlgorithm):

    def Initialize(self):

        self.SetStartDate(2014, 5, 15)
        self.SetEndDate(2014, 5, 16)
        self.SetCash(100000)

        self.xau = self.AddCfd(
            "XAUUSD",
            Resolution.Minute,
            Market.Oanda,
            fillForward=False
        )

        exchange = self.xau.Exchange
        config = self.xau.SubscriptionDataConfig

        self.Debug("========================================")
        self.Debug("XAUUSD TIMEZONE DEBUG")
        self.Debug("========================================")

        self.Debug(f"Symbol: {self.xau.Symbol}")
        self.Debug(f"Exchange Time Zone: {exchange.TimeZone.Id}")
        self.Debug(f"Data Time Zone: {config.DataTimeZone.Id}")
        self.Debug(f"Resolution: {config.Resolution}")
        self.Debug(f"Fill Forward: {config.FillDataForward}")
        self.Debug(
            f"Extended Market Hours: "
            f"{config.ExtendedMarketHours}"
        )

        self.Debug("========================================")

    def OnData(self, data):
        pass