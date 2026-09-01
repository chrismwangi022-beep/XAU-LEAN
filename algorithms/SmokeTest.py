from AlgorithmImports import *


class SmokeTest(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2014, 5, 1)
        self.SetEndDate(2014, 5, 2)
        self.SetCash(100000)

        self.xau = self.AddCfd(
            "XAUUSD",
            Resolution.Minute,
            Market.Oanda
        )

        self.received = 0

        self.Debug("XAU-LEAN DATA TEST: ENGINE STARTED")
        self.Debug("XAU-LEAN DATA TEST: XAUUSD SUBSCRIPTION CREATED")

    def OnData(self, data):
        if data.ContainsKey(self.xau.Symbol):
            self.received += 1

            if self.received <= 5:
                bar = data[self.xau.Symbol]

                self.Debug(
                    f"XAUUSD DATA RECEIVED: "
                    f"{bar.EndTime} "
                    f"O={bar.Open} "
                    f"H={bar.High} "
                    f"L={bar.Low} "
                    f"C={bar.Close}"
                )

    def OnEndOfAlgorithm(self):
        self.Debug(
            f"XAU-LEAN DATA TEST COMPLETE: "
            f"XAUUSD DATA POINTS RECEIVED = {self.received}"
        )