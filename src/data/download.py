"""M1 - Download BTCUSDT / ETHUSDT candles from Binance Vision.

Run:  python -m src.data.download            (skips files that already exist)
      python -m src.data.download --force    (re-downloads everything)

WHY this script is in Git and the data is not: the Parquet files are about
145 MB, but anyone can rebuild them exactly with this script. That is what
makes the study reproducible.

Source: https://data.binance.vision  (public, free, no API key)
"""
import argparse
import io
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from src import config


def parse_timestamps(raw: pd.Series) -> pd.Series:
    """Convert Binance epoch integers to datetimes.

    Binance SPOT files use milliseconds before 2025-01-01 and microseconds from
    2025-01-01. The unit is detected per row by magnitude, so a file that
    contains both is still parsed correctly.
    """
    raw = raw.astype("int64")
    # Everything is brought to microseconds so no precision is thrown away.
    micros = np.where(raw > config.MICROSECOND_THRESHOLD, raw, raw * 1000)
    return pd.Series(pd.to_datetime(micros, unit="us"), index=raw.index).astype("datetime64[ns]")


def download_month(symbol: str, interval: str, year: int, month: int) -> pd.DataFrame | None:
    """Download one monthly archive. Returns None if Binance has no file for it."""
    url = config.BINANCE_VISION_URL.format(
        symbol=symbol, interval=interval, year=year, month=month)
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        with archive.open(archive.namelist()[0]) as csv_file:
            df = pd.read_csv(csv_file, header=None, names=config.BINANCE_CSV_COLUMNS)

    # Some archives carry a header line; it shows up as a non-numeric first row.
    df = df[pd.to_numeric(df["open_time"], errors="coerce").notna()]
    df = df.drop(columns="ignore")
    df["open_time"] = parse_timestamps(df["open_time"])
    df["close_time"] = parse_timestamps(df["close_time"])
    df["num_trades"] = df["num_trades"].astype("int64")
    float_columns = [c for c in config.RAW_COLUMNS
                     if c not in ("open_time", "close_time", "num_trades")]
    df[float_columns] = df[float_columns].astype("float64")
    return df.reset_index(drop=True)


def download_series(symbol: str, interval: str) -> pd.DataFrame:
    """Download every month for one symbol and timeframe and join them."""
    months = pd.period_range(config.DOWNLOAD_START[interval], config.DOWNLOAD_END, freq="M")
    parts = []
    for period in months:
        part = download_month(symbol, interval, period.year, period.month)
        if part is None:
            print(f"  {symbol} {interval} {period}: no file on Binance Vision")
            continue
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="re-download files that already exist")
    args = parser.parse_args()

    config.PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in config.SYMBOLS:
        for interval in config.INTERVALS:
            path = config.PARQUET_DIR / f"{symbol}_{interval}.parquet"
            if path.exists() and not args.force:
                print(f"{path.name}: already exists, skipped")
                continue
            print(f"{path.name}: downloading ...")
            df = download_series(symbol, interval)
            df.to_parquet(path, index=False)
            print(f"{path.name}: {len(df):,} rows saved")
    print("\nNow run:  python -m src.data.validate")


if __name__ == "__main__":
    main()
