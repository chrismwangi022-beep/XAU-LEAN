from AlgorithmImports import *
from pathlib import Path
from datetime import datetime, timezone


class DukascopyFileAccessTest(QCAlgorithm):

    def Initialize(self):

        self.SetStartDate(2014, 5, 1)
        self.SetEndDate(2014, 5, 1)
        self.SetCash(100000)
        self.SetTimeZone("UTC")

        path = Path(
            "/Lean/workspace/data/external/dukascopy/"
            "Market-Data-Lab-main/xauusd/bid/m1/"
            "xauusd_bid_m1_2014_05.csv"
        )

        self.Log("=" * 80)
        self.Log("XAU-LEAN — DUKASCOPY DIRECT FILE ACCESS TEST")
        self.Log("=" * 80)
        self.Log(f"FILE: {path}")
        self.Log(f"EXISTS: {path.exists()}")

        if not path.exists():
            self.Error("Dukascopy file does not exist")
            return

        self.Log(f"SIZE: {path.stat().st_size} bytes")

        count = 0

        with path.open("r") as f:

            header = f.readline().strip()
            self.Log(f"HEADER: {header}")

            for line in f:

                if count >= 5:
                    break

                parts = line.strip().split(",")

                if len(parts) != 5:
                    self.Error(f"BAD ROW: {line.strip()}")
                    continue

                timestamp = int(parts[0])
                dt = datetime.fromtimestamp(
                    timestamp / 1000,
                    tz=timezone.utc
                )

                self.Log(
                    f"ROW {count + 1}: "
                    f"{dt} "
                    f"O={parts[1]} "
                    f"H={parts[2]} "
                    f"L={parts[3]} "
                    f"C={parts[4]}"
                )

                count += 1

        self.Log(f"ROWS READ: {count}")
        self.Log("=" * 80)
        self.Log("DIRECT FILE ACCESS TEST COMPLETE")
        self.Log("=" * 80)
