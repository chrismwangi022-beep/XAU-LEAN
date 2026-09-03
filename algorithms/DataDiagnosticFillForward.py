from AlgorithmImports import *


class DataDiagnostic(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2014, 5, 1)
        self.SetEndDate(2014, 5, 2)
        self.SetCash(100000)

        # IMPORTANT:
        # Fill-forward is deliberately DISABLED.
        # We want to inspect what is actually present
        # in the historical data.
        self.xau = self.AddCfd(
            "XAUUSD",
            Resolution.Minute,
            Market.Oanda,
            fillForward=True
        )

        self.received = 0
        self.gaps = []
        self.previous_end_time = None

        self.first_time = None
        self.last_time = None

        self.Debug("========================================")
        self.Debug("XAU-LEAN PHASE 1.3 DATA DIAGNOSTIC")
        self.Debug("========================================")
        self.Debug("Symbol: XAUUSD")
        self.Debug("Resolution: Minute")
        self.Debug("Fill-Forward: ENABLED")
        self.Debug("Period: 2014-05-01 to 2014-05-02")
        self.Debug("========================================")

    def OnData(self, data):

        if not data.ContainsKey(self.xau.Symbol):
            return

        bar = data[self.xau.Symbol]

        self.received += 1

        current_time = bar.EndTime

        if self.first_time is None:
            self.first_time = current_time

            self.Debug(
                f"FIRST BAR: {current_time} "
                f"O={bar.Open} "
                f"H={bar.High} "
                f"L={bar.Low} "
                f"C={bar.Close}"
            )

        self.last_time = current_time

        # Check continuity between consecutive bars.
        if self.previous_end_time is not None:

            gap = current_time - self.previous_end_time

            # Normal consecutive minute bars should be approximately
            # one minute apart. Anything larger is reported.
            if gap > timedelta(minutes=1):

                self.gaps.append(
                    (
                        self.previous_end_time,
                        current_time,
                        gap
                    )
                )

                self.Debug(
                    f"GAP DETECTED: "
                    f"{self.previous_end_time} -> "
                    f"{current_time} "
                    f"Duration={gap}"
                )

        self.previous_end_time = current_time

    def OnEndOfAlgorithm(self):

        self.Debug("========================================")
        self.Debug("DATA DIAGNOSTIC COMPLETE")
        self.Debug("========================================")

        self.Debug(
            f"TOTAL BARS RECEIVED = {self.received}"
        )

        self.Debug(
            f"FIRST BAR TIME = {self.first_time}"
        )

        self.Debug(
            f"LAST BAR TIME = {self.last_time}"
        )

        self.Debug(
            f"GAPS GREATER THAN 1 MINUTE = {len(self.gaps)}"
        )

        if self.gaps:

            self.Debug("---------- GAP SUMMARY ----------")

            for previous_time, current_time, gap in self.gaps:
                self.Debug(
                    f"{previous_time} -> "
                    f"{current_time} | "
                    f"Gap = {gap}"
                )

        else:

            self.Debug(
                "NO GAPS GREATER THAN 1 MINUTE DETECTED"
            )

        self.Debug("========================================")
