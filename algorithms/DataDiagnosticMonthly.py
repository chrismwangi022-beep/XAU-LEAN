from AlgorithmImports import *


class DataDiagnosticMonthly(QCAlgorithm):

    def Initialize(self):
        # Requested audit period
        self.requested_start = datetime(2014, 5, 1)
        self.requested_end = datetime(2014, 6, 1)

        self.SetStartDate(2014, 5, 1)
        self.SetEndDate(2014, 6, 1)
        self.SetCash(100000)

        # Raw-data audit:
        # Fill-forward deliberately disabled so we measure
        # what is actually present in the historical dataset.
        self.xau = self.AddCfd(
            "XAUUSD",
            Resolution.Minute,
            Market.Oanda,
            fillForward=False
        )

        # Core statistics
        self.total_bars = 0
        self.first_time = None
        self.last_time = None

        # Daily statistics
        self.daily_bar_counts = {}

        # Gap tracking
        self.gaps = []

        self.gaps_2min = 0
        self.gaps_3min = 0
        self.gaps_4min = 0
        self.gaps_5_to_59min = 0
        self.gaps_60min_plus = 0

        # Known recurring session/data boundary
        self.recurring_66min_breaks = 0

        # Weekend discontinuities
        self.weekend_gaps = 0

        # Largest overall gap
        self.largest_gap = timedelta(0)
        self.largest_gap_start = None
        self.largest_gap_end = None

        # Largest non-weekend gap
        self.largest_non_weekend_gap = timedelta(0)
        self.largest_non_weekend_gap_start = None
        self.largest_non_weekend_gap_end = None

        self.previous_end_time = None

        self.trading_dates = set()

        self.Debug("========================================")
        self.Debug("XAU-LEAN MONTHLY DATA QUALITY AUDIT")
        self.Debug("========================================")
        self.Debug("Symbol: XAUUSD")
        self.Debug("Resolution: Minute")
        self.Debug("Fill-Forward: DISABLED")
        self.Debug(
            f"Requested period: "
            f"{self.requested_start} -> {self.requested_end}"
        )
        self.Debug("========================================")

    def OnData(self, data):

        if not data.ContainsKey(self.xau.Symbol):
            return

        bar = data[self.xau.Symbol]
        current_time = bar.EndTime

        self.total_bars += 1

        # Track first and last ACTUAL observed bars
        if self.first_time is None:
            self.first_time = current_time

        self.last_time = current_time

        # Track daily bar counts
        current_date = current_time.date()

        self.trading_dates.add(current_date)

        if current_date not in self.daily_bar_counts:
            self.daily_bar_counts[current_date] = 0

        self.daily_bar_counts[current_date] += 1

        # ----------------------------------------
        # GAP ANALYSIS
        # ----------------------------------------

        if self.previous_end_time is not None:

            gap = current_time - self.previous_end_time

            if gap > timedelta(minutes=1):

                self.gaps.append(
                    (
                        self.previous_end_time,
                        current_time,
                        gap
                    )
                )

                total_minutes = gap.total_seconds() / 60

                # Categorize gap
                if total_minutes == 2:
                    self.gaps_2min += 1

                elif total_minutes == 3:
                    self.gaps_3min += 1

                elif total_minutes == 4:
                    self.gaps_4min += 1

                elif total_minutes < 60:
                    self.gaps_5_to_59min += 1

                else:
                    self.gaps_60min_plus += 1

                # Known recurring 16:58 -> 18:04 break
                if (
                    self.previous_end_time.hour == 16
                    and self.previous_end_time.minute == 58
                    and current_time.hour == 18
                    and current_time.minute == 4
                    and gap == timedelta(hours=1, minutes=6)
                ):
                    self.recurring_66min_breaks += 1

                # Friday -> Sunday gap
                is_weekend_gap = (
                    self.previous_end_time.weekday() == 4
                    and current_time.weekday() == 6
                )

                if is_weekend_gap:
                    self.weekend_gaps += 1

                # Track largest overall gap
                if gap > self.largest_gap:
                    self.largest_gap = gap
                    self.largest_gap_start = self.previous_end_time
                    self.largest_gap_end = current_time

                # Track largest NON-WEEKEND gap
                if (
                    not is_weekend_gap
                    and gap > self.largest_non_weekend_gap
                ):
                    self.largest_non_weekend_gap = gap
                    self.largest_non_weekend_gap_start = (
                        self.previous_end_time
                    )
                    self.largest_non_weekend_gap_end = current_time

        self.previous_end_time = current_time

    def OnEndOfAlgorithm(self):

        # ----------------------------------------
        # DAILY STATISTICS
        # ----------------------------------------

        if self.daily_bar_counts:

            daily_counts = list(
                self.daily_bar_counts.values()
            )

            average_bars_per_day = (
                self.total_bars / len(daily_counts)
            )

            minimum_bars_per_day = min(daily_counts)
            maximum_bars_per_day = max(daily_counts)

        else:

            average_bars_per_day = 0
            minimum_bars_per_day = 0
            maximum_bars_per_day = 0

        # ----------------------------------------
        # DATA COVERAGE
        # ----------------------------------------

        if self.first_time is not None and self.last_time is not None:

            observed_duration = (
                self.last_time - self.first_time
            )

            requested_duration = (
                self.requested_end - self.requested_start
            )

            if requested_duration.total_seconds() > 0:

                coverage_ratio = (
                    observed_duration.total_seconds()
                    / requested_duration.total_seconds()
                )

                coverage_percent = coverage_ratio * 100

            else:
                coverage_percent = 0

        else:

            observed_duration = timedelta(0)
            coverage_percent = 0

        # ----------------------------------------
        # FINAL REPORT
        # ----------------------------------------

        self.Debug("")
        self.Debug("========================================")
        self.Debug("XAU-LEAN DATA QUALITY REPORT")
        self.Debug("========================================")

        self.Debug(
            f"Requested start: {self.requested_start}"
        )

        self.Debug(
            f"Requested end:   {self.requested_end}"
        )

        self.Debug("")

        self.Debug(
            f"Actual first XAUUSD bar: {self.first_time}"
        )

        self.Debug(
            f"Actual last XAUUSD bar:  {self.last_time}"
        )

        self.Debug("")

        self.Debug(
            f"Total XAUUSD minute bars: {self.total_bars}"
        )

        self.Debug(
            f"Trading dates observed: {len(self.trading_dates)}"
        )

        self.Debug(
            f"Average bars per trading day: "
            f"{average_bars_per_day:.2f}"
        )

        self.Debug(
            f"Minimum bars in a trading day: "
            f"{minimum_bars_per_day}"
        )

        self.Debug(
            f"Maximum bars in a trading day: "
            f"{maximum_bars_per_day}"
        )

        self.Debug("")

        self.Debug(
            f"Observed data span: {observed_duration}"
        )

        self.Debug(
            f"Approximate temporal coverage: "
            f"{coverage_percent:.2f}%"
        )

        self.Debug("")
        self.Debug("---------- GAP SUMMARY ----------")

        self.Debug(
            f"Total gaps > 1 minute: {len(self.gaps)}"
        )

        self.Debug(
            f"2-minute gaps: {self.gaps_2min}"
        )

        self.Debug(
            f"3-minute gaps: {self.gaps_3min}"
        )

        self.Debug(
            f"4-minute gaps: {self.gaps_4min}"
        )

        self.Debug(
            f"5-59 minute gaps: {self.gaps_5_to_59min}"
        )

        self.Debug(
            f"60+ minute gaps: {self.gaps_60min_plus}"
        )

        self.Debug("")
        self.Debug("---------- RECURRING SESSION BREAK ----------")

        self.Debug(
            f"16:58 -> 18:04 recurring breaks: "
            f"{self.recurring_66min_breaks}"
        )

        self.Debug("")
        self.Debug("---------- WEEKEND ----------")

        self.Debug(
            f"Friday -> Sunday large gaps: "
            f"{self.weekend_gaps}"
        )

        self.Debug("")
        self.Debug("---------- LARGEST GAPS ----------")

        self.Debug(
            f"Largest overall gap: {self.largest_gap}"
        )

        self.Debug(
            f"Largest overall gap start: "
            f"{self.largest_gap_start}"
        )

        self.Debug(
            f"Largest overall gap end: "
            f"{self.largest_gap_end}"
        )

        self.Debug("")

        self.Debug(
            f"Largest non-weekend gap: "
            f"{self.largest_non_weekend_gap}"
        )

        self.Debug(
            f"Largest non-weekend gap start: "
            f"{self.largest_non_weekend_gap_start}"
        )

        self.Debug(
            f"Largest non-weekend gap end: "
            f"{self.largest_non_weekend_gap_end}"
        )

        # ----------------------------------------
        # INTERPRETATION
        # ----------------------------------------

        self.Debug("")
        self.Debug("========================================")
        self.Debug("DATA QUALITY INTERPRETATION")
        self.Debug("========================================")

        if self.first_time is None:

            self.Debug(
                "FAIL: No XAUUSD data was received."
            )

        else:

            self.Debug(
                "PASS: XAUUSD data successfully received."
            )

        if self.recurring_66min_breaks > 0:

            self.Debug(
                "PASS: Recurring 16:58 -> 18:04 break confirmed."
            )

            self.Debug(
                "Interpretation: recurring session/data boundary, "
                "not automatically corruption."
            )

        else:

            self.Debug(
                "INFO: Recurring 16:58 -> 18:04 break not detected."
            )

        if self.weekend_gaps > 0:

            self.Debug(
                "PASS: Weekend discontinuity confirmed."
            )

            self.Debug(
                "Weekend gaps are expected and should not "
                "be treated as missing intraday bars."
            )

        if (
            self.gaps_2min
            + self.gaps_3min
            + self.gaps_4min
            > 0
        ):

            self.Debug(
                "REVIEW: Small intra-session gaps are present."
            )

        if self.gaps_5_to_59min > 0:

            self.Debug(
                "REVIEW: 5-59 minute gaps are present."
            )

        if self.gaps_60min_plus > 0:

            self.Debug(
                "REVIEW: Large gaps are present and require "
                "session-aware interpretation."
            )

        # Final status
        if self.total_bars == 0:

            final_status = "FAIL"

        elif self.largest_non_weekend_gap > timedelta(minutes=4):

            final_status = "REVIEW"

        else:

            final_status = "PASS"

        self.Debug("")
        self.Debug(
            f"FINAL DATA QUALITY STATUS: {final_status}"
        )

        self.Debug("")
        self.Debug(
            "Raw-data audit complete. "
            "No fill-forward was used."
        )

        self.Debug("========================================")
