"""
XAU-LEAN — Dukascopy XAUUSD M1 Data Validator

Purpose:
    Read-only validation of the canonical Dukascopy XAUUSD
    1-minute BID/ASK historical dataset.

Checks:
    1. File coverage
    2. BID/ASK file pairing
    3. Timestamp validity
    4. Timestamp ordering
    5. Duplicate timestamps
    6. Expected 1-minute spacing
    7. OHLC validity
    8. BID/ASK timestamp alignment
    9. Spread statistics and anomalies
    10. Monthly/yearly coverage
    11. Weekend/session observations

IMPORTANT:
    This script does NOT modify the source data.
"""

from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import csv
import math
import statistics
import re


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "dukascopy"
    / "Market-Data-Lab-main"
    / "xauusd"
)

BID_DIR = DATA_ROOT / "bid" / "m1"
ASK_DIR = DATA_ROOT / "ask" / "m1"

BID_PATTERN = re.compile(r"xauusd_bid_m1_(\d{4})_(\d{2})\.csv$")
ASK_PATTERN = re.compile(r"xauusd_ask_m1_(\d{4})_(\d{2})\.csv$")

EXPECTED_COLUMNS = ["timestamp", "open", "high", "low", "close"]

# Spread thresholds are diagnostic only.
# They do NOT automatically classify the data as invalid.
SPREAD_WARNING = 1.00
SPREAD_CRITICAL = 5.00

# Gap threshold:
# Anything greater than 1 minute is reported.
EXPECTED_INTERVAL_MS = 60_000


# ============================================================
# HELPERS
# ============================================================

def fmt_ts(ms):
    """Convert Unix milliseconds to UTC datetime string."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def parse_price(value):
    try:
        x = float(value)
        if not math.isfinite(x):
            return None
        return x
    except (TypeError, ValueError):
        return None


def discover_files(directory, pattern):
    result = {}

    for path in directory.glob("*.csv"):
        match = pattern.match(path.name)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            result[(year, month)] = path

    return result


def month_label(year, month):
    return f"{year:04d}-{month:02d}"


# ============================================================
# FILE COVERAGE
# ============================================================

def coverage_audit():

    print("\n" + "=" * 80)
    print("1. FILE COVERAGE")
    print("=" * 80)

    if not BID_DIR.exists():
        raise FileNotFoundError(f"BID directory not found: {BID_DIR}")

    if not ASK_DIR.exists():
        raise FileNotFoundError(f"ASK directory not found: {ASK_DIR}")

    bid_files = discover_files(BID_DIR, BID_PATTERN)
    ask_files = discover_files(ASK_DIR, ASK_PATTERN)

    bid_months = set(bid_files)
    ask_months = set(ask_files)

    both = bid_months & ask_months
    bid_only = bid_months - ask_months
    ask_only = ask_months - bid_months

    print(f"BID files discovered : {len(bid_files)}")
    print(f"ASK files discovered : {len(ask_files)}")
    print(f"Paired months        : {len(both)}")

    if bid_only:
        print("\nBID-only months:")
        print(", ".join(month_label(*x) for x in sorted(bid_only)))

    if ask_only:
        print("\nASK-only months:")
        print(", ".join(month_label(*x) for x in sorted(ask_only)))

    if not bid_only and not ask_only:
        print("BID/ASK monthly coverage: PASS")

    if both:
        first = min(both)
        last = max(both)

        print(f"First paired month: {month_label(*first)}")
        print(f"Last paired month : {month_label(*last)}")

    return bid_files, ask_files, both


# ============================================================
# SINGLE FILE AUDIT
# ============================================================

def audit_file(path):

    result = {
        "rows": 0,
        "invalid_rows": 0,
        "duplicate_timestamps": 0,
        "out_of_order": 0,
        "gaps": 0,
        "largest_gap_ms": 0,
        "first_ts": None,
        "last_ts": None,
        "bad_ohlc": 0,
        "negative_prices": 0,
        "timestamps": [],
    }

    previous_ts = None

    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames != EXPECTED_COLUMNS:
                result["header_error"] = (
                    f"Expected {EXPECTED_COLUMNS}, got {reader.fieldnames}"
                )
                return result

            for row in reader:

                result["rows"] += 1

                try:
                    ts = int(row["timestamp"])

                    o = parse_price(row["open"])
                    h = parse_price(row["high"])
                    l = parse_price(row["low"])
                    c = parse_price(row["close"])

                    if None in (o, h, l, c):
                        result["invalid_rows"] += 1
                        continue

                except Exception:
                    result["invalid_rows"] += 1
                    continue

                if result["first_ts"] is None:
                    result["first_ts"] = ts

                result["last_ts"] = ts

                result["timestamps"].append(ts)

                if previous_ts is not None:

                    delta = ts - previous_ts

                    if delta == 0:
                        result["duplicate_timestamps"] += 1

                    elif delta < 0:
                        result["out_of_order"] += 1

                    elif delta > EXPECTED_INTERVAL_MS:
                        result["gaps"] += 1

                        if delta > result["largest_gap_ms"]:
                            result["largest_gap_ms"] = delta

                previous_ts = ts

                # OHLC validity
                if h < max(o, c) or l > min(o, c) or h < l:
                    result["bad_ohlc"] += 1

                if min(o, h, l, c) <= 0:
                    result["negative_prices"] += 1

    except Exception as e:
        result["file_error"] = str(e)

    return result


# ============================================================
# BID / ASK ALIGNMENT
# ============================================================

def alignment_audit(bid_path, ask_path):

    bid_ts = set()
    ask_ts = set()

    try:
        with bid_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    bid_ts.add(int(row["timestamp"]))
                except Exception:
                    pass

        with ask_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    ask_ts.add(int(row["timestamp"]))
                except Exception:
                    pass

    except Exception as e:
        return {
            "error": str(e),
            "bid_rows": 0,
            "ask_rows": 0,
            "missing_bid": 0,
            "missing_ask": 0,
        }

    return {
        "bid_rows": len(bid_ts),
        "ask_rows": len(ask_ts),
        "missing_bid": len(ask_ts - bid_ts),
        "missing_ask": len(bid_ts - ask_ts),
        "intersection": len(bid_ts & ask_ts),
    }


# ============================================================
# SPREAD AUDIT
# ============================================================

def spread_audit(bid_path, ask_path):

    bid = {}
    ask = {}

    try:

        with bid_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    bid[int(row["timestamp"])] = parse_price(row["close"])
                except Exception:
                    pass

        with ask_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    ask[int(row["timestamp"])] = parse_price(row["close"])
                except Exception:
                    pass

    except Exception as e:
        return {"error": str(e)}

    spreads = []

    for ts in bid.keys() & ask.keys():

        b = bid[ts]
        a = ask[ts]

        if b is None or a is None:
            continue

        spread = a - b

        if math.isfinite(spread):
            spreads.append(spread)

    if not spreads:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "warning": 0,
            "critical": 0,
            "negative": 0,
        }

    warning = sum(x >= SPREAD_WARNING for x in spreads)
    critical = sum(x >= SPREAD_CRITICAL for x in spreads)
    negative = sum(x < 0 for x in spreads)

    return {
        "count": len(spreads),
        "min": min(spreads),
        "max": max(spreads),
        "mean": statistics.mean(spreads),
        "median": statistics.median(spreads),
        "warning": warning,
        "critical": critical,
        "negative": negative,
    }


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():

    print("\n" + "=" * 80)
    print("XAU-LEAN — DUKASCOPY XAUUSD M1 DATA VALIDATOR")
    print("=" * 80)

    print(f"Data root:")
    print(f"  {DATA_ROOT}")

    bid_files, ask_files, paired_months = coverage_audit()

    if not paired_months:
        print("\nERROR: No paired BID/ASK months found.")
        return

    print("\n" + "=" * 80)
    print("2. MONTHLY DATA QUALITY AUDIT")
    print("=" * 80)

    totals = defaultdict(int)
    monthly_results = []

    for i, ym in enumerate(sorted(paired_months), start=1):

        year, month = ym

        bid_path = bid_files[ym]
        ask_path = ask_files[ym]

        print(
            f"[{i:03d}/{len(paired_months):03d}] "
            f"{month_label(year, month)}",
            end=" ",
            flush=True,
        )

        bid = audit_file(bid_path)
        ask = audit_file(ask_path)
        alignment = alignment_audit(bid_path, ask_path)
        spread = spread_audit(bid_path, ask_path)

        monthly = {
            "year": year,
            "month": month,
            "bid": bid,
            "ask": ask,
            "alignment": alignment,
            "spread": spread,
        }

        monthly_results.append(monthly)

        # Aggregate
        for side in ("bid", "ask"):
            for key in (
                "rows",
                "invalid_rows",
                "duplicate_timestamps",
                "out_of_order",
                "gaps",
                "bad_ohlc",
                "negative_prices",
            ):
                totals[f"{side}_{key}"] += bid[key] if side == "bid" else ask[key]

        totals["alignment_missing_bid"] += alignment.get("missing_bid", 0)
        totals["alignment_missing_ask"] += alignment.get("missing_ask", 0)

        if spread.get("count"):
            totals["spread_count"] += spread["count"]
            totals["spread_warning"] += spread["warning"]
            totals["spread_critical"] += spread["critical"]
            totals["spread_negative"] += spread["negative"]

        problems = []

        if bid["invalid_rows"] or ask["invalid_rows"]:
            problems.append("INVALID")

        if bid["duplicate_timestamps"] or ask["duplicate_timestamps"]:
            problems.append("DUP")

        if bid["out_of_order"] or ask["out_of_order"]:
            problems.append("ORDER")

        if bid["gaps"] or ask["gaps"]:
            problems.append(f"GAPS={bid['gaps'] + ask['gaps']}")

        if bid["bad_ohlc"] or ask["bad_ohlc"]:
            problems.append("OHLC")

        if (
            alignment.get("missing_bid", 0)
            or alignment.get("missing_ask", 0)
        ):
            problems.append("ALIGN")

        if spread.get("critical", 0):
            problems.append(f"SPREAD>={SPREAD_CRITICAL}")

        if problems:
            print(" | ".join(problems))
        else:
            print("PASS")

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print("3. FULL DATASET SUMMARY")
    print("=" * 80)

    print(f"BID rows                 : {totals['bid_rows']:,}")
    print(f"ASK rows                 : {totals['ask_rows']:,}")

    print(f"BID invalid rows         : {totals['bid_invalid_rows']:,}")
    print(f"ASK invalid rows         : {totals['ask_invalid_rows']:,}")

    print(f"BID duplicate timestamps : {totals['bid_duplicate_timestamps']:,}")
    print(f"ASK duplicate timestamps : {totals['ask_duplicate_timestamps']:,}")

    print(f"BID out-of-order rows    : {totals['bid_out_of_order']:,}")
    print(f"ASK out-of-order rows    : {totals['ask_out_of_order']:,}")

    print(f"BID gaps > 1 minute      : {totals['bid_gaps']:,}")
    print(f"ASK gaps > 1 minute      : {totals['ask_gaps']:,}")

    print(f"BID invalid OHLC         : {totals['bid_bad_ohlc']:,}")
    print(f"ASK invalid OHLC         : {totals['ask_bad_ohlc']:,}")

    print(f"BID non-positive prices  : {totals['bid_negative_prices']:,}")
    print(f"ASK non-positive prices  : {totals['ask_negative_prices']:,}")

    print("\nBID/ASK timestamp alignment:")
    print(f"  Missing BID timestamps : {totals['alignment_missing_bid']:,}")
    print(f"  Missing ASK timestamps : {totals['alignment_missing_ask']:,}")

    print("\nSpread:")
    print(f"  Observations           : {totals['spread_count']:,}")
    print(f"  >= ${SPREAD_WARNING:.2f}            : {totals['spread_warning']:,}")
    print(f"  >= ${SPREAD_CRITICAL:.2f}            : {totals['spread_critical']:,}")
    print(f"  Negative spreads       : {totals['spread_negative']:,}")

    # ========================================================
    # YEARLY SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print("4. YEARLY COVERAGE")
    print("=" * 80)

    yearly = defaultdict(
        lambda: {
            "months": 0,
            "bid_rows": 0,
            "ask_rows": 0,
            "gaps": 0,
            "duplicates": 0,
            "alignment": 0,
        }
    )

    for item in monthly_results:

        year = item["year"]

        yearly[year]["months"] += 1
        yearly[year]["bid_rows"] += item["bid"]["rows"]
        yearly[year]["ask_rows"] += item["ask"]["rows"]

        yearly[year]["gaps"] += (
            item["bid"]["gaps"] + item["ask"]["gaps"]
        )

        yearly[year]["duplicates"] += (
            item["bid"]["duplicate_timestamps"]
            + item["ask"]["duplicate_timestamps"]
        )

        yearly[year]["alignment"] += (
            item["alignment"].get("missing_bid", 0)
            + item["alignment"].get("missing_ask", 0)
        )

    print(
        f"{'YEAR':<8}"
        f"{'MONTHS':>10}"
        f"{'BID ROWS':>15}"
        f"{'ASK ROWS':>15}"
        f"{'GAPS':>12}"
        f"{'DUP':>10}"
        f"{'ALIGN':>12}"
    )

    print("-" * 82)

    for year in sorted(yearly):

        x = yearly[year]

        print(
            f"{year:<8}"
            f"{x['months']:>10}"
            f"{x['bid_rows']:>15,}"
            f"{x['ask_rows']:>15,}"
            f"{x['gaps']:>12,}"
            f"{x['duplicates']:>10,}"
            f"{x['alignment']:>12,}"
        )

    # ========================================================
    # FINAL VERDICT
    # ========================================================

    print("\n" + "=" * 80)
    print("5. VALIDATION VERDICT")
    print("=" * 80)

    hard_failures = (
        totals["bid_invalid_rows"]
        + totals["ask_invalid_rows"]
        + totals["bid_duplicate_timestamps"]
        + totals["ask_duplicate_timestamps"]
        + totals["bid_out_of_order"]
        + totals["ask_out_of_order"]
        + totals["bid_bad_ohlc"]
        + totals["ask_bad_ohlc"]
        + totals["bid_negative_prices"]
        + totals["ask_negative_prices"]
        + totals["alignment_missing_bid"]
        + totals["alignment_missing_ask"]
        + totals["spread_negative"]
    )

    if hard_failures == 0:
        print("CORE DATA INTEGRITY: PASS")
    else:
        print(f"CORE DATA INTEGRITY: REVIEW ({hard_failures:,} findings)")

    print()
    print("NOTE:")
    print("Gaps > 1 minute are NOT automatically failures.")
    print("Forex/CFD markets naturally contain market-closure periods")
    print("and other periods without a bar.")
    print("They must be interpreted by trading-session context.")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
