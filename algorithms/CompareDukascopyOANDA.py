from pathlib import Path
from datetime import datetime, timezone, timedelta
import zipfile
import csv
import statistics


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path.home() / "XAU-LEAN"

DUKASCOPY_BASE = (
    PROJECT_ROOT
    / "data/external/dukascopy/Market-Data-Lab-main/xauusd"
)

OANDA_BASE = (
    PROJECT_ROOT
    / "data/cfd/oanda/minute/xauusd"
)


# ============================================================
# HELPERS
# ============================================================

def parse_unix_ms(value):
    """Convert Unix milliseconds to timezone-aware UTC datetime."""
    return datetime.fromtimestamp(
        int(value) / 1000,
        tz=timezone.utc
    )


def parse_oanda_timestamp(value, base_date=None):
    """
    Parse OANDA/LEAN timestamps.

    OANDA minute files may store timestamps as:
      - Unix milliseconds
      - Unix seconds
      - milliseconds since midnight of the file date
      - ISO timestamps

    Returns timezone-aware UTC datetime.
    """

    value = value.strip()

    # Numeric timestamp
    if value.isdigit():

        number = int(value)

        # Unix milliseconds
        if number > 10**12:
            return datetime.fromtimestamp(
                number / 1000,
                tz=timezone.utc
            )

        # Unix seconds
        if number > 10**9:
            return datetime.fromtimestamp(
                number,
                tz=timezone.utc
            )

        # Milliseconds since midnight
        if base_date is not None:

            midnight = datetime.combine(
                base_date,
                datetime.min.time(),
                tzinfo=timezone.utc
            )

            return midnight + timedelta(
                milliseconds=number
            )

    # ISO timestamp
    value = value.replace("Z", "+00:00")

    dt = datetime.fromisoformat(value)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


# ============================================================
# LOAD DUKASCOPY
# ============================================================

def load_dukascopy():

    bid_file = (
        DUKASCOPY_BASE
        / "bid/m1/xauusd_bid_m1_2014_05.csv"
    )

    ask_file = (
        DUKASCOPY_BASE
        / "ask/m1/xauusd_ask_m1_2014_05.csv"
    )

    bid = {}
    ask = {}

    print("Loading Dukascopy BID...")

    with bid_file.open("r", newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            dt = parse_unix_ms(row["timestamp"])

            bid[dt] = {
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"])
            }

    print("Loading Dukascopy ASK...")

    with ask_file.open("r", newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            dt = parse_unix_ms(row["timestamp"])

            ask[dt] = {
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"])
            }

    return bid, ask


# ============================================================
# LOAD OANDA
# ============================================================

def load_oanda():

    oanda = {}

    files = sorted(
        OANDA_BASE.glob("201405*_quote.zip")
    )

    print()
    print(f"OANDA ZIP files found: {len(files)}")

    for zip_path in files:

        print(f"  Reading {zip_path.name}")

        # Extract YYYYMMDD from filename
        file_date = datetime.strptime(
            zip_path.stem[:8],
            "%Y%m%d"
        ).date()

        with zipfile.ZipFile(zip_path, "r") as z:

            names = z.namelist()

            if not names:
                continue

            with z.open(names[0]) as raw:

                text = (
                    line.decode("utf-8")
                    for line in raw
                )

                reader = csv.reader(text)

                for row in reader:

                    if not row:
                        continue

                    # Skip possible header
                    if row[0].lower() in (
                        "timestamp",
                        "time",
                        "date"
                    ):
                        continue

                    if len(row) < 9:
                        continue

                    try:

                        dt = parse_oanda_timestamp(
                            row[0],
                            base_date=file_date
                        )

                    except Exception:
                        continue

                    oanda[dt] = {

                        "bid_open": float(row[1]),
                        "bid_high": float(row[2]),
                        "bid_low": float(row[3]),
                        "bid_close": float(row[4]),

                        "ask_open": float(row[6]),
"ask_high": float(row[7]),
"ask_low": float(row[8]),
"ask_close": float(row[9])
                    }

    return oanda


# ============================================================
# STATISTICS
# ============================================================

def describe(values):

    if not values:
        return None

    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values)
    }


# ============================================================
# MAIN COMPARISON
# ============================================================

def main():

    print()
    print("=" * 78)
    print("XAUUSD — DUKASCOPY vs OANDA")
    print("May 2014 Provider Comparison")
    print("=" * 78)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    duka_bid, duka_ask = load_dukascopy()
    oanda = load_oanda()

    print()
    print("RAW DATA COUNTS")
    print("-" * 78)

    print(f"Dukascopy BID : {len(duka_bid):,}")
    print(f"Dukascopy ASK : {len(duka_ask):,}")
    print(f"OANDA         : {len(oanda):,}")

    # --------------------------------------------------------
    # Timestamp sets
    # --------------------------------------------------------

    bid_times = set(duka_bid)
    ask_times = set(duka_ask)
    oanda_times = set(oanda)

    duka_times = bid_times & ask_times
    common = duka_times & oanda_times

    missing_oanda = duka_times - oanda_times
    missing_duka = oanda_times - duka_times

    print()
    print("TIMESTAMP COVERAGE")
    print("-" * 78)

    print(f"Dukascopy BID+ASK : {len(duka_times):,}")
    print(f"OANDA             : {len(oanda_times):,}")
    print(f"COMMON            : {len(common):,}")

    print()
    print(
        "Dukascopy minutes missing from OANDA:",
        f"{len(missing_oanda):,}"
    )

    print(
        "OANDA minutes missing from Dukascopy:",
        f"{len(missing_duka):,}"
    )

    # --------------------------------------------------------
    # Date ranges
    # --------------------------------------------------------

    print()
    print("DATE RANGES")
    print("-" * 78)

    if duka_times:

        print(
            "Dukascopy:",
            min(duka_times),
            "->",
            max(duka_times)
        )

    if oanda_times:

        print(
            "OANDA    :",
            min(oanda_times),
            "->",
            max(oanda_times)

        )

    if not oanda_times:

        print()
        print("ERROR: No valid OANDA timestamps were loaded.")
        return

    if not common:

        print()
        print("ERROR: No common timestamps between providers.")
        return

    # --------------------------------------------------------
    # Compare bid prices
    # --------------------------------------------------------

    bid_open_diff = []
    bid_high_diff = []
    bid_low_diff = []
    bid_close_diff = []

    ask_open_diff = []
    ask_high_diff = []
    ask_low_diff = []
    ask_close_diff = []

    spread_values = []

    for dt in sorted(common):

        d_bid = duka_bid[dt]
        d_ask = duka_ask[dt]

        o = oanda[dt]

        bid_open_diff.append(
            o["bid_open"] - d_bid["open"]
        )

        bid_high_diff.append(
            o["bid_high"] - d_bid["high"]
        )

        bid_low_diff.append(
            o["bid_low"] - d_bid["low"]
        )

        bid_close_diff.append(
            o["bid_close"] - d_bid["close"]
        )

        ask_open_diff.append(
            o["ask_open"] - d_ask["open"]
        )

        ask_high_diff.append(
            o["ask_high"] - d_ask["high"]
        )

        ask_low_diff.append(
            o["ask_low"] - d_ask["low"]
        )

        ask_close_diff.append(
            o["ask_close"] - d_ask["close"]
        )

        spread_values.append(
            o["ask_close"] - o["bid_close"]
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()
    print("PRICE DIFFERENCE — OANDA minus DUKASCOPY")
    print("-" * 78)

    print()
    print("BID OPEN")
    print(describe(bid_open_diff))

    print()
    print("BID HIGH")
    print(describe(bid_high_diff))

    print()
    print("BID LOW")
    print(describe(bid_low_diff))

    print()
    print("BID CLOSE")
    print(describe(bid_close_diff))

    print()
    print("ASK OPEN")
    print(describe(ask_open_diff))

    print()
    print("ASK HIGH")
    print(describe(ask_high_diff))

    print()
    print("ASK LOW")
    print(describe(ask_low_diff))

    print()
    print("ASK CLOSE")
    print(describe(ask_close_diff))

    # --------------------------------------------------------
    # Spread
    # --------------------------------------------------------

    print()
    print("OANDA BID/ASK SPREAD — CLOSE")
    print("-" * 78)

    print(describe(spread_values))

    # --------------------------------------------------------
    # Exact / tolerance checks
    # --------------------------------------------------------

    def count_within(values, tolerance):
        return sum(
            abs(x) <= tolerance
            for x in values
        )

    print()
    print("BID CLOSE DIFFERENCE TOLERANCE")
    print("-" * 78)

    for tolerance in (
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        1.00
    ):

        count = count_within(
            bid_close_diff,
            tolerance
        )

        pct = (
            count / len(bid_close_diff) * 100
            if bid_close_diff
            else 0
        )

        print(
            f"<= ${tolerance:.2f}: "
            f"{count:,} / {len(bid_close_diff):,} "
            f"({pct:.2f}%)"
        )

    # --------------------------------------------------------
    # Sample comparisons
    # --------------------------------------------------------

    print()
    print("SAMPLE COMMON TIMESTAMPS")
    print("-" * 78)

    for dt in sorted(common)[:10]:

        d_bid = duka_bid[dt]
        d_ask = duka_ask[dt]
        o = oanda[dt]

        print()
        print(dt)

        print(
            f"  Dukascopy BID: "
            f"O={d_bid['open']:.3f} "
            f"H={d_bid['high']:.3f} "
            f"L={d_bid['low']:.3f} "
            f"C={d_bid['close']:.3f}"
        )

        print(
            f"  OANDA BID:     "
            f"O={o['bid_open']:.3f} "
            f"H={o['bid_high']:.3f} "
            f"L={o['bid_low']:.3f} "
            f"C={o['bid_close']:.3f}"
        )

        print(
            f"  Dukascopy ASK: "
            f"O={d_ask['open']:.3f} "
            f"H={d_ask['high']:.3f} "
            f"L={d_ask['low']:.3f} "
            f"C={d_ask['close']:.3f}"
        )

        print(
            f"  OANDA ASK:     "
            f"O={o['ask_open']:.3f} "
            f"H={o['ask_high']:.3f} "
            f"L={o['ask_low']:.3f} "
            f"C={o['ask_close']:.3f}"
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("COMPARISON COMPLETE")
    print("=" * 78)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()