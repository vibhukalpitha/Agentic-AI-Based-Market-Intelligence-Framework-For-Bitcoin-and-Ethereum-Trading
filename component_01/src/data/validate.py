"""M1 - Integrity checks on the raw Parquet files.

Run:  python -m src.data.validate

WHY: every later result depends on these files. This script is the evidence
that the data is complete and well-formed, and it must be re-run whenever the
data is re-downloaded. It only reads; it never changes a file.
"""
import sys

import pandas as pd

from src import config


def check_file(symbol: str, interval: str) -> bool:
    """Print one summary line and return True if every hard check passes.

    Gaps and zero-volume candles are reported but are NOT failures: they are
    genuine exchange outages in the source data, and are stated as a limitation.
    """
    df = pd.read_parquet(config.PARQUET_DIR / f"{symbol}_{interval}.parquet")
    t = df["open_time"]

    gaps = int((t.diff().dropna() != config.INTERVAL_STEP[interval]).sum())
    bad_ohlc = int(((df["high"] < df[["open", "close"]].max(axis=1))
                    | (df["low"] > df[["open", "close"]].min(axis=1))).sum())
    problems = []
    if list(df.columns) != config.RAW_COLUMNS:
        problems.append("columns differ from schema")
    if len(df) != config.EXPECTED_ROWS[interval]:
        problems.append(f"expected {config.EXPECTED_ROWS[interval]} rows")
    if t.duplicated().any():
        problems.append("duplicate timestamps")
    if not t.is_monotonic_increasing:
        problems.append("not sorted by time")
    if df.isna().any().any():
        problems.append("null values")
    if bad_ohlc:
        problems.append(f"{bad_ohlc} candles with high/low inconsistent")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        problems.append("non-positive price")

    print(f"{symbol}_{interval:<3} rows={len(df):>7,} "
          f"{t.min():%Y-%m-%d} -> {t.max():%Y-%m-%d} "
          f"gaps={gaps:<3} zero_volume={int((df['volume'] == 0).sum()):<3} "
          f"close {df['close'].min():>9,.2f} .. {df['close'].max():>10,.2f}  "
          f"{'OK' if not problems else 'FAIL: ' + '; '.join(problems)}")
    return not problems


def main() -> None:
    results = [check_file(s, i) for s in config.SYMBOLS for i in config.INTERVALS]
    print(f"\n{sum(results)}/{len(results)} files passed")
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
