#!/usr/bin/env python3

import csv
import re
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta


ROOT = Path("data/external/dukascopy/Market-Data-Lab-main/xauusd")
BID_DIR = ROOT / "bid" / "m1"
ASK_DIR = ROOT / "ask" / "m1"

EXPECTED_START = "2010-01"
EXPECTED_END = "2026-08"

FILE_RE_BID = re.compile(r"^xauusd_bid_m1_(\d{4})_(\d{2})\.csv$")
FILE_RE_ASK = re.compile(r"^xauusd_ask_m1_(\d{4})_(\d{2})\.csv$")

KNOWN_ASK_ONLY = 1786924740000  # 2026-08-16 23:59:00 UTC


def month_range(start_year, start_month, end_year, end_month):
    months = []
    y, m = start_year, start_month

    while (y, m) <= (end_year, end_month):
        months.append(f"{y:04d}-{m:02d}")

        m += 1
        if m == 13:
            m = 1
            y += 1

    return months


EXPECTED_MONTHS = month_range(2010, 1, 2026, 8)


def timestamp_to_dt(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def validate_ohlc(o, h, l, c):
    return (
        o > 0
        and h > 0
        and l > 0
        and c > 0
        and h >= o
        and h >= c
        and h >= l
        and l <= o
        and l <= c
        and l <= h
    )


def scan_file(path):
    rows = 0
    invalid_rows = 0
    duplicate_timestamps = 0
    out_of_order = 0
    invalid_ohlc = 0
    non_positive = 0
    gaps_gt_1m = 0

    first_ts = None
    last_ts = None

    with path.open("r", newline="") as f:
        reader = csv.reader(f)

        header = next(reader, None)

        if header != ["timestamp", "open", "high", "low", "close"]:
            raise RuntimeError(
                f"BAD HEADER: {path} -> {header}"
            )

        for line_no, row in enumerate(reader, start=2):
            rows += 1

            if len(row) != 5:
                invalid_rows += 1
                continue

            try:
                ts = int(row[0])
                o = float(row[1])
                h = float(row[2])
                l = float(row[3])
                c = float(row[4])
            except Exception:
                invalid_rows += 1
                continue

            if first_ts is None:
                first_ts = ts

            if last_ts is not None:
                delta = ts - last_ts

                if delta == 0:
                    duplicate_timestamps += 1

                if delta < 0:
                    out_of_order += 1

                if delta > 60_000:
                    gaps_gt_1m += 1

            if not validate_ohlc(o, h, l, c):
                invalid_ohlc += 1

            if o <= 0 or h <= 0 or l <= 0 or c <= 0:
                non_positive += 1

            last_ts = ts

    return {
        "rows": rows,
        "invalid_rows": invalid_rows,
        "duplicate_timestamps": duplicate_timestamps,
        "out_of_order": out_of_order,
        "invalid_ohlc": invalid_ohlc,
        "non_positive": non_positive,
        "gaps_gt_1m": gaps_gt_1m,
        "first_ts": first_ts,
        "last_ts": last_ts,
    }


def compare_bid_ask(bid_path, ask_path):
    aligned = 0
    bid_only = 0
    ask_only = 0

    positive_spread = 0
    zero_spread = 0
    negative_spread = 0

    spread_ge_1 = 0
    spread_ge_5 = 0

    first_bid_only = []
    first_ask_only = []

    with bid_path.open("r", newline="") as bf, ask_path.open("r", newline="") as af:

        br = csv.reader(bf)
        ar = csv.reader(af)

        next(br, None)
        next(ar, None)

        b = next(br, None)
        a = next(ar, None)

        while b is not None or a is not None:

            if b is None:
                ask_only += 1

                if len(first_ask_only) < 5:
                    first_ask_only.append(int(a[0]))

                a = next(ar, None)
                continue

            if a is None:
                bid_only += 1

                if len(first_bid_only) < 5:
                    first_bid_only.append(int(b[0]))

                b = next(br, None)
                continue

            bts = int(b[0])
            ats = int(a[0])

            if bts == ats:
                aligned += 1

                bid_close = float(b[4])
                ask_close = float(a[4])

                spread = ask_close - bid_close

                if spread > 0:
                    positive_spread += 1
                elif spread == 0:
                    zero_spread += 1
                else:
                    negative_spread += 1

                if spread >= 1:
                    spread_ge_1 += 1

                if spread >= 5:
                    spread_ge_5 += 1

                b = next(br, None)
                a = next(ar, None)

            elif bts < ats:
                bid_only += 1

                if len(first_bid_only) < 5:
                    first_bid_only.append(bts)

                b = next(br, None)

            else:
                ask_only += 1

                if len(first_ask_only) < 5:
                    first_ask_only.append(ats)

                a = next(ar, None)

    return {
        "aligned": aligned,
        "bid_only": bid_only,
        "ask_only": ask_only,
        "positive_spread": positive_spread,
        "zero_spread": zero_spread,
        "negative_spread": negative_spread,
        "spread_ge_1": spread_ge_1,
        "spread_ge_5": spread_ge_5,
        "first_bid_only": first_bid_only,
        "first_ask_only": first_ask_only,
    }


def fmt_ts(ts):
    if ts is None:
        return "N/A"
    return timestamp_to_dt(ts).strftime("%Y-%m-%d %H:%M:%S UTC")


def discover_month_files(directory, regex):
    result = {}

    for path in directory.glob("*.csv"):
        match = regex.match(path.name)

        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            key = f"{year:04d}-{month:02d}"

            result[key] = path

    return result


def main():

    start = time.perf_counter()

    print()
    print("=" * 80)
    print("XAUUSD — DUKASCOPY FULL-HISTORY VALIDATION")
    print("=" * 80)
    print()
    print(f"Dataset root : {ROOT}")
    print(f"Expected     : {EXPECTED_START} → {EXPECTED_END}")
    print(f"Expected mo. : {len(EXPECTED_MONTHS)}")
    print()

    if not BID_DIR.exists():
        raise SystemExit(f"FAIL: BID directory does not exist: {BID_DIR}")

    if not ASK_DIR.exists():
        raise SystemExit(f"FAIL: ASK directory does not exist: {ASK_DIR}")

    bid_files = discover_month_files(BID_DIR, FILE_RE_BID)
    ask_files = discover_month_files(ASK_DIR, FILE_RE_ASK)

    print(f"Monthly BID files discovered: {len(bid_files)}")
    print(f"Monthly ASK files discovered: {len(ask_files)}")
    print()

    expected_set = set(EXPECTED_MONTHS)
    bid_set = set(bid_files)
    ask_set = set(ask_files)

    missing_bid = sorted(expected_set - bid_set)
    missing_ask = sorted(expected_set - ask_set)

    unexpected_bid = sorted(bid_set - expected_set)
    unexpected_ask = sorted(ask_set - expected_set)

    print("MONTH COVERAGE")
    print("-" * 80)

    print(f"Missing BID months    : {missing_bid or 'NONE'}")
    print(f"Missing ASK months    : {missing_ask or 'NONE'}")
    print(f"Unexpected BID months : {unexpected_bid or 'NONE'}")
    print(f"Unexpected ASK months : {unexpected_ask or 'NONE'}")
    print()

    common_months = [
        m for m in EXPECTED_MONTHS
        if m in bid_files and m in ask_files
    ]

    total_bid_rows = 0
    total_ask_rows = 0

    total_bid_invalid = 0
    total_ask_invalid = 0

    total_bid_duplicates = 0
    total_ask_duplicates = 0

    total_bid_out_of_order = 0
    total_ask_out_of_order = 0

    total_bid_invalid_ohlc = 0
    total_ask_invalid_ohlc = 0

    total_bid_non_positive = 0
    total_ask_non_positive = 0

    total_bid_gaps = 0
    total_ask_gaps = 0

    total_aligned = 0
    total_bid_only = 0
    total_ask_only = 0

    total_positive_spread = 0
    total_zero_spread = 0
    total_negative_spread = 0

    total_spread_ge_1 = 0
    total_spread_ge_5 = 0

    first_global_bid = None
    last_global_bid = None
    first_global_ask = None
    last_global_ask = None

    unexpected_alignment = []

    print("PROCESSING")
    print("-" * 80)

    for index, month in enumerate(common_months, start=1):

        bid_path = bid_files[month]
        ask_path = ask_files[month]

        print(f"[{index:03d}/{len(common_months):03d}] {month}", end=" ... ", flush=True)

        bid = scan_file(bid_path)
        ask = scan_file(ask_path)

        comparison = compare_bid_ask(bid_path, ask_path)

        total_bid_rows += bid["rows"]
        total_ask_rows += ask["rows"]

        total_bid_invalid += bid["invalid_rows"]
        total_ask_invalid += ask["invalid_rows"]

        total_bid_duplicates += bid["duplicate_timestamps"]
        total_ask_duplicates += ask["duplicate_timestamps"]

        total_bid_out_of_order += bid["out_of_order"]
        total_ask_out_of_order += ask["out_of_order"]

        total_bid_invalid_ohlc += bid["invalid_ohlc"]
        total_ask_invalid_ohlc += ask["invalid_ohlc"]

        total_bid_non_positive += bid["non_positive"]
        total_ask_non_positive += ask["non_positive"]

        total_bid_gaps += bid["gaps_gt_1m"]
        total_ask_gaps += ask["gaps_gt_1m"]

        total_aligned += comparison["aligned"]
        total_bid_only += comparison["bid_only"]
        total_ask_only += comparison["ask_only"]

        total_positive_spread += comparison["positive_spread"]
        total_zero_spread += comparison["zero_spread"]
        total_negative_spread += comparison["negative_spread"]

        total_spread_ge_1 += comparison["spread_ge_1"]
        total_spread_ge_5 += comparison["spread_ge_5"]

        if bid["first_ts"] is not None:
            if first_global_bid is None or bid["first_ts"] < first_global_bid:
                first_global_bid = bid["first_ts"]

        if bid["last_ts"] is not None:
            if last_global_bid is None or bid["last_ts"] > last_global_bid:
                last_global_bid = bid["last_ts"]

        if ask["first_ts"] is not None:
            if first_global_ask is None or ask["first_ts"] < first_global_ask:
                first_global_ask = ask["first_ts"]

        if ask["last_ts"] is not None:
            if last_global_ask is None or ask["last_ts"] > last_global_ask:
                last_global_ask = ask["last_ts"]

        if comparison["bid_only"] or comparison["ask_only"]:
            unexpected_alignment.append(
                (
                    month,
                    comparison["bid_only"],
                    comparison["ask_only"],
                    comparison["first_bid_only"],
                    comparison["first_ask_only"],
                )
            )

        print(
            f"BID={bid['rows']:,} "
            f"ASK={ask['rows']:,} "
            f"ALIGNED={comparison['aligned']:,} "
            f"BID_ONLY={comparison['bid_only']:,} "
            f"ASK_ONLY={comparison['ask_only']:,}"
        )

    elapsed = time.perf_counter() - start

    print()
    print("=" * 80)
    print("FULL-HISTORY TOTALS")
    print("=" * 80)

    print(f"BID rows                 : {total_bid_rows:,}")
    print(f"ASK rows                 : {total_ask_rows:,}")
    print(f"Combined rows            : {total_bid_rows + total_ask_rows:,}")
    print()

    print(f"BID invalid rows         : {total_bid_invalid:,}")
    print(f"ASK invalid rows         : {total_ask_invalid:,}")

    print(f"BID duplicate timestamps : {total_bid_duplicates:,}")
    print(f"ASK duplicate timestamps : {total_ask_duplicates:,}")

    print(f"BID out-of-order         : {total_bid_out_of_order:,}")
    print(f"ASK out-of-order         : {total_ask_out_of_order:,}")

    print(f"BID invalid OHLC         : {total_bid_invalid_ohlc:,}")
    print(f"ASK invalid OHLC         : {total_ask_invalid_ohlc:,}")

    print(f"BID non-positive         : {total_bid_non_positive:,}")
    print(f"ASK non-positive         : {total_ask_non_positive:,}")

    print()
    print(f"BID gaps > 1 minute      : {total_bid_gaps:,}")
    print(f"ASK gaps > 1 minute      : {total_ask_gaps:,}")

    print()
    print(f"ALIGNED timestamps       : {total_aligned:,}")
    print(f"BID-only timestamps      : {total_bid_only:,}")
    print(f"ASK-only timestamps      : {total_ask_only:,}")

    print()
    print(f"Positive spreads         : {total_positive_spread:,}")
    print(f"Zero spreads             : {total_zero_spread:,}")
    print(f"Negative spreads         : {total_negative_spread:,}")
    print(f"Spread >= $1             : {total_spread_ge_1:,}")
    print(f"Spread >= $5             : {total_spread_ge_5:,}")

    print()
    print(f"FIRST BID timestamp      : {fmt_ts(first_global_bid)}")
    print(f"LAST BID timestamp       : {fmt_ts(last_global_bid)}")
    print(f"FIRST ASK timestamp      : {fmt_ts(first_global_ask)}")
    print(f"LAST ASK timestamp       : {fmt_ts(last_global_ask)}")

    print()
    print("=" * 80)
    print("ALIGNMENT EXCEPTIONS")
    print("=" * 80)

    if not unexpected_alignment:
        print("NONE")
    else:
        for month, bid_only, ask_only, bid_examples, ask_examples in unexpected_alignment:
            print()
            print(f"{month}:")
            print(f"  BID-only : {bid_only}")
            print(f"  ASK-only : {ask_only}")

            if bid_examples:
                print("  BID-only examples:")
                for ts in bid_examples:
                    print(f"    {fmt_ts(ts)}")

            if ask_examples:
                print("  ASK-only examples:")
                for ts in ask_examples:
                    print(f"  ASK-only examples:")
                    for ts in ask_examples:
                        print(f"    {fmt_ts(ts)}")

    print()
    print("=" * 80)
    print("KNOWN SOURCE ANOMALY CHECK")
    print("=" * 80)

    known_seen = False

    for month, bid_only, ask_only, bid_examples, ask_examples in unexpected_alignment:
        if KNOWN_ASK_ONLY in ask_examples:
            known_seen = True

    print(
        "Known Aug-2026 ASK-only timestamp "
        f"({fmt_ts(KNOWN_ASK_ONLY)}): "
        + ("FOUND" if known_seen else "NOT FOUND")
    )

    print()
    print("=" * 80)
    print("FINAL GATE")
    print("=" * 80)

    failures = []

    if len(bid_files) != 200:
        failures.append(f"Expected 200 monthly BID files, found {len(bid_files)}")

    if len(ask_files) != 200:
        failures.append(f"Expected 200 monthly ASK files, found {len(ask_files)}")

    if missing_bid:
        failures.append(f"Missing BID months: {missing_bid}")

    if missing_ask:
        failures.append(f"Missing ASK months: {missing_ask}")

    if unexpected_bid:
        failures.append(f"Unexpected BID months: {unexpected_bid}")

    if unexpected_ask:
        failures.append(f"Unexpected ASK months: {unexpected_ask}")

    if total_bid_invalid != 0:
        failures.append(f"BID invalid rows: {total_bid_invalid}")

    if total_ask_invalid != 0:
        failures.append(f"ASK invalid rows: {total_ask_invalid}")

    if total_bid_duplicates != 0:
        failures.append(f"BID duplicate timestamps: {total_bid_duplicates}")

    if total_ask_duplicates != 0:
        failures.append(f"ASK duplicate timestamps: {total_ask_duplicates}")

    if total_bid_out_of_order != 0:
        failures.append(f"BID out-of-order rows: {total_bid_out_of_order}")

    if total_ask_out_of_order != 0:
        failures.append(f"ASK out-of-order rows: {total_ask_out_of_order}")

    if total_bid_invalid_ohlc != 0:
        failures.append(f"BID invalid OHLC: {total_bid_invalid_ohlc}")

    if total_ask_invalid_ohlc != 0:
        failures.append(f"ASK invalid OHLC: {total_ask_invalid_ohlc}")

    if total_bid_non_positive != 0:
        failures.append(f"BID non-positive prices: {total_bid_non_positive}")

    if total_ask_non_positive != 0:
        failures.append(f"ASK non-positive prices: {total_ask_non_positive}")

    if total_negative_spread != 0:
        failures.append(f"Negative spreads: {total_negative_spread}")

    # The one known source anomaly is allowed.
    unexpected_ask_only = total_ask_only
    unexpected_bid_only = total_bid_only

    if unexpected_bid_only != 0:
        failures.append(
            f"Unexpected BID-only timestamps: {unexpected_bid_only}"
        )

    if unexpected_ask_only != 1:
        failures.append(
            f"Expected exactly 1 ASK-only source anomaly, found {unexpected_ask_only}"
        )
    elif not known_seen:
        failures.append(
            "ASK-only timestamp exists but is not the known Aug-2026 anomaly"
        )

    print()

    if failures:
        print("RESULT: FAIL")
        print()
        for failure in failures:
            print(f" - {failure}")
    else:
        print("RESULT: PASS")
        print()
        print("Dukascopy full-history data foundation is VALIDATED.")
        print("2010-01 → 2026-08")
        print("200 BID months + 200 ASK months")
        print("No structural data failures detected.")
        print("One known isolated Aug-2026 ASK-only source anomaly accepted.")

    print()
    print("=" * 80)
    print(f"Runtime: {elapsed:.2f} seconds")
    print(
        f"Throughput: "
        f"{(total_bid_rows + total_ask_rows) / elapsed:,.0f} rows/sec"
    )
    print("=" * 80)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
