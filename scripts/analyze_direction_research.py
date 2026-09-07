import json
from pathlib import Path
from collections import defaultdict

INPUT_ROOT = Path("research_results/direction")

FILES = {
    "M15": INPUT_ROOT / "xauusd_momentum_direction_m15_20100101T000000Z_20260821T000000Z.json",
    "H1": INPUT_ROOT / "xauusd_momentum_direction_h1_20100101T000000Z_20260821T000000Z.json",
    "H4": INPUT_ROOT / "xauusd_momentum_direction_h4_20100101T000000Z_20260821T000000Z.json",
    "H6": INPUT_ROOT / "xauusd_momentum_direction_h6_20100101T000000Z_20260821T000000Z.json",
}


def load(path):
    with path.open() as f:
        return json.load(f)


def directional_stats(rows):
    observations = sum(r["observations"] for r in rows)
    positive = sum(r["positive"] for r in rows)
    negative = sum(r["negative"] for r in rows)
    flat = sum(r["flat"] for r in rows)

    if observations == 0:
        return {
            "observations": 0,
            "positive": 0,
            "negative": 0,
            "flat": 0,
            "accuracy": 0.0,
            "mean_signed_return_pct": 0.0,
            "directional_edge_pct": 0.0,
        }

    # Bullish rows: positive return is correct.
    # Bearish rows: negative return is correct.
    correct = 0

    weighted_signed = 0.0

    for r in rows:
        if r["direction"] == "bullish":
            correct += r["positive"]
            weighted_signed += (
                r["mean_signed_return_pct"] * r["observations"]
            )
        elif r["direction"] == "bearish":
            correct += r["negative"]
            weighted_signed += (
                -r["mean_signed_return_pct"] * r["observations"]
            )

    accuracy = correct / observations

    directional_edge = accuracy - 0.5

    mean_signed_return = (
        weighted_signed / observations
    )

    return {
        "observations": observations,
        "positive": positive,
        "negative": negative,
        "flat": flat,
        "accuracy": accuracy,
        "mean_signed_return_pct": mean_signed_return,
        "directional_edge_pct": directional_edge,
    }


def print_header(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def main():

    datasets = {}

    for timeframe, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(path)

        data = load(path)
        datasets[timeframe] = data["report"]

    # ---------------------------------------------------------------
    # 1. Correct directional accuracy
    # ---------------------------------------------------------------
    print_header("1. CORRECTED DIRECTIONAL ACCURACY")

    for timeframe, report in datasets.items():
        print(f"\n{timeframe}")

        for horizon in [1, 3, 5]:

            rows = [
                cell
                for cell in report["cells"].values()
                if cell["horizon_candles"] == horizon
            ]

            s = directional_stats(rows)

            print(
                f"  H{horizon}: "
                f"accuracy={s['accuracy'] * 100:.2f}% | "
                f"edge={s['directional_edge_pct'] * 100:.2f}pp | "
                f"mean_favorable_return={s['mean_signed_return_pct']:.5f}% | "
                f"obs={s['observations']:,}"
            )

    # ---------------------------------------------------------------
    # 2. Bullish vs bearish
    # ---------------------------------------------------------------
    print_header("2. BULLISH VS BEARISH — CORRECTED")

    for timeframe, report in datasets.items():
        print(f"\n{timeframe}")

        for horizon in [1, 3, 5]:

            for direction in ["bullish", "bearish"]:

                rows = [
                    cell
                    for cell in report["cells"].values()
                    if cell["horizon_candles"] == horizon
                    and cell["direction"] == direction
                ]

                s = directional_stats(rows)

                print(
                    f"  H{horizon} {direction:8s}: "
                    f"accuracy={s['accuracy'] * 100:.2f}% | "
                    f"edge={s['directional_edge_pct'] * 100:.2f}pp | "
                    f"favorable_return={s['mean_signed_return_pct']:.5f}% | "
                    f"obs={s['observations']:,}"
                )

    # ---------------------------------------------------------------
    # 3. Historical stability
    # ---------------------------------------------------------------
    print_header("3. HISTORICAL REGIME STABILITY")

    for timeframe, report in datasets.items():

        print(f"\n{timeframe}")

        regimes = sorted({
            cell["regime"]
            for cell in report["cells"].values()
        })

        for regime in regimes:

            rows = [
                cell
                for cell in report["cells"].values()
                if cell["regime"] == regime
            ]

            s = directional_stats(rows)

            print(
                f"  {regime:12s}: "
                f"accuracy={s['accuracy'] * 100:.2f}% | "
                f"edge={s['directional_edge_pct'] * 100:.2f}pp | "
                f"favorable_return={s['mean_signed_return_pct']:.5f}% | "
                f"obs={s['observations']:,}"
            )

    # ---------------------------------------------------------------
    # 4. Volatility buckets
    # ---------------------------------------------------------------
    print_header("4. VOLATILITY BUCKETS")

    buckets = ["normal", "elevated", "high", "extreme"]

    for timeframe, report in datasets.items():

        print(f"\n{timeframe}")

        for bucket in buckets:

            rows = [
                cell
                for cell in report["cells"].values()
                if cell["bucket"] == bucket
            ]

            s = directional_stats(rows)

            print(
                f"  {bucket:10s}: "
                f"accuracy={s['accuracy'] * 100:.2f}% | "
                f"edge={s['directional_edge_pct'] * 100:.2f}pp | "
                f"favorable_return={s['mean_signed_return_pct']:.5f}% | "
                f"obs={s['observations']:,}"
            )

    # ---------------------------------------------------------------
    # 5. Best robust cells
    # ---------------------------------------------------------------
    print_header("5. BEST CELLS — MINIMUM 500 OBSERVATIONS")

    for timeframe, report in datasets.items():

        rows = []

        for cell in report["cells"].values():

            if cell["observations"] < 500:
                continue

            s = directional_stats([cell])

            rows.append({
                "cell": cell,
                "stats": s,
            })

        rows.sort(
            key=lambda x: (
                x["stats"]["accuracy"],
                x["stats"]["mean_signed_return_pct"],
            ),
            reverse=True,
        )

        print(f"\n{timeframe}")

        for item in rows[:15]:

            c = item["cell"]
            s = item["stats"]

            print(
                f"  {c['regime']:12s} | "
                f"{c['bucket']:9s} | "
                f"{c['direction']:8s} | "
                f"H{c['horizon_candles']} | "
                f"accuracy={s['accuracy'] * 100:.2f}% | "
                f"edge={s['directional_edge_pct'] * 100:.2f}pp | "
                f"return={s['mean_signed_return_pct']:.5f}% | "
                f"obs={s['observations']:,}"
            )

    # ---------------------------------------------------------------
    # 6. Worst cells
    # ---------------------------------------------------------------
    print_header("6. WORST CELLS — MINIMUM 500 OBSERVATIONS")

    for timeframe, report in datasets.items():

        rows = []

        for cell in report["cells"].values():

            if cell["observations"] < 500:
                continue

            s = directional_stats([cell])

            rows.append({
                "cell": cell,
                "stats": s,
            })

        rows.sort(
            key=lambda x: (
                x["stats"]["accuracy"],
                x["stats"]["mean_signed_return_pct"],
            )
        )

        print(f"\n{timeframe}")

        for item in rows[:15]:

            c = item["cell"]
            s = item["stats"]

            print(
                f"  {c['regime']:12s} | "
                f"{c['bucket']:9s} | "
                f"{c['direction']:8s} | "
                f"H{c['horizon_candles']} | "
                f"accuracy={s['accuracy'] * 100:.2f}% | "
                f"edge={s['directional_edge_pct'] * 100:.2f}pp | "
                f"return={s['mean_signed_return_pct']:.5f}% | "
                f"obs={s['observations']:,}"
            )


if __name__ == "__main__":
    main()
