from pathlib import Path
from datetime import datetime, timedelta, timezone
import zipfile
import csv


OANDA = Path("data/cfd/oanda/minute/xauusd/20140501_quote.zip")

DUK_BID = Path(
    "data/external/dukascopy/Market-Data-Lab-main/"
    "xauusd/bid/m1/xauusd_bid_m1_2014_05.csv"
)

DUK_ASK = Path(
    "data/external/dukascopy/Market-Data-Lab-main/"
    "xauusd/ask/m1/xauusd_ask_m1_2014_05.csv"
)

START = datetime(2014, 5, 1, tzinfo=timezone.utc)


# ============================================================================
# LOAD OANDA TIMESTAMPS
# ============================================================================

oanda_times = set()

with zipfile.ZipFile(OANDA) as z:
    with z.open(z.namelist()[0]) as f:
        reader = csv.reader(line.decode("utf-8") for line in f)

        for row in reader:
            if len(row) < 10:
                continue

            try:
                ms = int(float(row[0]))

                dt = START + timedelta(milliseconds=ms)
                dt = dt.replace(second=0, microsecond=0)

                oanda_times.add(dt)

            except (ValueError, TypeError):
                pass


# ============================================================================
# EXPECTED MINUTES
# ============================================================================

expected = {
    START + timedelta(minutes=i)
    for i in range(1440)
}


missing = sorted(expected - oanda_times)


# ============================================================================
# LOAD DUKASCOPY BID
# ============================================================================

duk_bid = {}

with DUK_BID.open("r", errors="ignore") as f:
    reader = csv.reader(f)

    for row in reader:

        if not row:
            continue

        if row[0] == "timestamp":
            continue

        try:
            ts = int(row[0])

            dt = datetime.fromtimestamp(
                ts / 1000,
                tz=timezone.utc
            )

            dt = dt.replace(second=0, microsecond=0)

            if dt.date() != START.date():
                continue

            duk_bid[dt] = {
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
            }

        except (ValueError, TypeError, OverflowError):
            pass


# ============================================================================
# LOAD DUKASCOPY ASK
# ============================================================================

duk_ask = {}

with DUK_ASK.open("r", errors="ignore") as f:
    reader = csv.reader(f)

    for row in reader:

        if not row:
            continue

        if row[0] == "timestamp":
            continue

        try:
            ts = int(row[0])

            dt = datetime.fromtimestamp(
                ts / 1000,
                tz=timezone.utc
            )

            dt = dt.replace(second=0, microsecond=0)

            if dt.date() != START.date():
                continue

            duk_ask[dt] = {
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
            }

        except (ValueError, TypeError, OverflowError):
            pass


# ============================================================================
# REPORT
# ============================================================================

print()
print("=" * 78)
print("OANDA-MISSING MINUTES — DUKASCOPY PRICE VALIDATION")
print("=" * 78)

print()
print("Missing OANDA minutes :", len(missing))
print("Dukascopy BID matches :", sum(t in duk_bid for t in missing))
print("Dukascopy ASK matches :", sum(t in duk_ask for t in missing))

print()
print("PRICE DATA DURING OANDA GAPS")
print("-" * 78)

print(
    f"{'TIME':<8}"
    f"{'BID O':>10}"
    f"{'BID H':>10}"
    f"{'BID L':>10}"
    f"{'BID C':>10}"
    f"{'ASK O':>10}"
    f"{'ASK H':>10}"
    f"{'ASK L':>10}"
    f"{'ASK C':>10}"
)

print("-" * 78)


for dt in missing:

    bid = duk_bid.get(dt)
    ask = duk_ask.get(dt)

    if bid is None or ask is None:
        print(
            f"{dt:%H:%M}   "
            f"NO COMPLETE DUKASCOPY DATA"
        )
        continue

    print(
        f"{dt:%H:%M}"
        f"{bid['open']:>10.3f}"
        f"{bid['high']:>10.3f}"
        f"{bid['low']:>10.3f}"
        f"{bid['close']:>10.3f}"
        f"{ask['open']:>10.3f}"
        f"{ask['high']:>10.3f}"
        f"{ask['low']:>10.3f}"
        f"{ask['close']:>10.3f}"
    )


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

complete = [
    dt for dt in missing
    if dt in duk_bid and dt in duk_ask
]

spreads = [
    duk_ask[dt]["close"] - duk_bid[dt]["close"]
    for dt in complete
]

if spreads:

    print()
    print("=" * 78)
    print("DUKASCOPY SPREAD DURING OANDA GAPS")
    print("=" * 78)

    print(f"Minutes analysed : {len(spreads)}")
    print(f"Min spread       : ${min(spreads):.3f}")
    print(f"Max spread       : ${max(spreads):.3f}")
    print(f"Mean spread      : ${sum(spreads) / len(spreads):.3f}")


print()
print("=" * 78)
print("VALIDATION VERDICT")
print("=" * 78)

if len(complete) == len(missing):

    print("PASS — Dukascopy contains complete BID/ASK price data")
    print("for every minute missing from OANDA.")

else:

    print("WARNING — Some OANDA gaps also lack complete")
    print("Dukascopy BID/ASK data.")

print("=" * 78)
