"""Live data - is the REST API the same data as the frozen archive files?

Run:  python -m src.live.check_source        (writes results/live_source_check.json)

WHY. Live candles come from the REST API, the frozen ones from Binance Vision
archives. Joining the two is only safe if both give the same candles. This
fetches the last CHECK_CANDLES candles of every frozen series again through the
REST API and compares every column, value by value. It changes no data.
"""
import json

import numpy as np
import pandas as pd

from src import config
from src.live import binance_rest, store

# 300 candles reach back at most 300 days (Daily), so every compared candle is
# from 2025 or later, when the archives switched to microsecond timestamps; the
# API is asked for microseconds too, so even the close times must match exactly.
CHECK_CANDLES = 300
# The archives were read with pandas' CSV parser, which does not always round a
# long decimal string to the nearest float; the API text is converted exactly.
# On large totals (quote_volume ~ 1e9 with 8 decimals) this leaves differences
# of 1-2 units in the last binary digit (relative ~2e-16). Such a column counts
# as equal "within float rounding"; every other difference fails the check.
FLOAT_ROUNDING = 1e-15


def main() -> None:
    client = binance_rest.BinanceClient()
    now = binance_rest.server_time(client)
    results = []
    for symbol in config.SYMBOLS:
        for interval in config.INTERVALS:
            frozen = pd.read_parquet(store.frozen_path(symbol, interval)).tail(CHECK_CANDLES)
            frozen = frozen.reset_index(drop=True)
            fetched = binance_rest.fetch_closed(client, symbol, interval,
                                                frozen["open_time"].iloc[0], now)
            fetched = fetched[fetched["open_time"] <= frozen["open_time"].iloc[-1]]
            joined = frozen.merge(fetched, on="open_time", how="outer",
                                  suffixes=("_archive", "_rest"), indicator=True)
            only_one = int((joined["_merge"] != "both").sum())
            both = joined[joined["_merge"] == "both"]
            exact_diffs, rounding = {}, {}
            for c in config.RAW_COLUMNS[1:]:
                a, b = both[f"{c}_archive"], both[f"{c}_rest"]
                n_diff = int((a != b).sum())
                if not n_diff:
                    continue
                if c in ("close_time", "num_trades"):
                    exact_diffs[c] = n_diff
                    continue
                relative = float((np.abs(a - b) / np.abs(b).clip(lower=1e-300)).max())
                if relative <= FLOAT_ROUNDING:
                    rounding[c] = {"candles": n_diff, "max_relative_difference": relative}
                else:
                    exact_diffs[c] = n_diff
            equal = only_one == 0 and not exact_diffs
            status = ("IDENTICAL" if equal and not rounding else
                      "EQUAL (within float rounding)" if equal else "DIFFERENT")
            results.append({"symbol": symbol, "interval": interval, "candles": len(frozen),
                            "from": str(frozen["open_time"].iloc[0]),
                            "to": str(frozen["open_time"].iloc[-1]),
                            "missing_in_one_source": only_one,
                            "real_differences": exact_diffs,
                            "float_rounding_only": rounding, "status": status, "equal": equal})
            print(f"{symbol} {interval:>3}: {len(frozen)} candles "
                  f"({frozen['open_time'].iloc[0]:%Y-%m-%d} -> {frozen['open_time'].iloc[-1]:%Y-%m-%d}): "
                  f"{status}" + ("" if equal else f"  missing {only_one}, differing {exact_diffs}"))
    all_equal = all(r["equal"] for r in results)
    with open(config.RESULTS_DIR / "live_source_check.json", "w") as file:
        json.dump({"checked_at_binance_time": str(now), "candles_per_series": CHECK_CANDLES,
                   "float_rounding_tolerance": FLOAT_ROUNDING, "series": results,
                   "all_equal": all_equal}, file, indent=2)
    print(f"\nREST API and archive agree on every candle: {all_equal}. "
          "Saved: results/live_source_check.json")


if __name__ == "__main__":
    main()
