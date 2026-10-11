
from pathlib import Path
import pandas as pd


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
INPUT_DIR = PROCESSED_DIR / "1m"

CHUNK_SIZE = 200_000

DATASETS = {
    "BTCUSDT": INPUT_DIR / "BTCUSDT_1m_preprocessed.csv",
    "ETHUSDT": INPUT_DIR / "ETHUSDT_1m_preprocessed.csv",
}

TIMEFRAMES = {
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}


# ==================================================
# 2. COLUMN AGGREGATION RULES
# ==================================================

AGGREGATION = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",

    "volume": "sum",
    "quote_volume": "sum",
    "num_trades": "sum",

    "taker_buy_base_vol": "sum",
    "taker_buy_quote_vol": "sum",
    "taker_sell_base_vol": "sum",
    "taker_sell_quote_vol": "sum",

    "mark_open": "first",
    "mark_high": "max",
    "mark_low": "min",
    "mark_close": "last",

    "index_open": "first",
    "index_high": "max",
    "index_low": "min",
    "index_close": "last",
}


# ==================================================
# 3. AGGREGATE CANDLES BY TIME BUCKET
# ==================================================

def aggregate_buckets(frame):
    """Aggregate rows that belong to the same timeframe."""

    return (
        frame.groupby("_bucket", sort=True)[list(AGGREGATION)]
        .agg(AGGREGATION)
    )


# ==================================================
# 4. RESAMPLE ONE DATASET
# ==================================================

def resample_dataset(symbol, input_path, timeframe, frequency):

    output_dir = PROCESSED_DIR / timeframe
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{symbol}.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    # Protect existing outputs from accidental overwriting.
    if output_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            "Move/delete it intentionally before rerunning."
        )

    print(f"\n{symbol} | {timeframe}")
    print(f"Input: {input_path}")

    total_input_rows = 0
    total_output_rows = 0

    # Keep the last bucket between chunks because a candle
    # may span a chunk boundary.
    carry = None
    first_output = True

    try:
        for chunk in pd.read_csv(
            input_path,
            chunksize=CHUNK_SIZE,
        ):

            chunk["timestamp_utc"] = pd.to_datetime(
                chunk["timestamp_utc"],
                utc=True,
                errors="raise",
            )

            total_input_rows += len(chunk)

            # Assign every 1-minute candle to its UTC bucket.
            chunk["_bucket"] = chunk["timestamp_utc"].dt.floor(
                frequency
            )

            grouped = aggregate_buckets(chunk)

            # Combine the previous partial bucket with this chunk.
            if carry is not None:
                grouped = pd.concat([carry, grouped])
                grouped = (
                    grouped.groupby(level=0, sort=True)
                    .agg(AGGREGATION)
                )

            # Keep the final bucket for the next chunk.
            # All earlier buckets are complete relative to the
            # current input chunk.
            carry = grouped.iloc[[-1]].copy()
            ready = grouped.iloc[:-1].copy()

            if not ready.empty:
                ready.index.name = "timestamp_utc"
                ready = ready.reset_index()

                ready["timestamp_utc"] = (
                    ready["timestamp_utc"].map(
                        lambda ts: ts.isoformat()
                    )
                )

                ready.to_csv(
                    output_path,
                    mode="w" if first_output else "a",
                    header=first_output,
                    index=False,
                )

                total_output_rows += len(ready)
                first_output = False

        # Write the final remaining bucket.
        if carry is not None and not carry.empty:
            final_chunk = carry.copy()
            final_chunk.index.name = "timestamp_utc"
            final_chunk = final_chunk.reset_index()

            final_chunk["timestamp_utc"] = (
                final_chunk["timestamp_utc"].map(
                    lambda ts: ts.isoformat()
                )
            )

            final_chunk.to_csv(
                output_path,
                mode="w" if first_output else "a",
                header=first_output,
                index=False,
            )

            total_output_rows += len(final_chunk)

    except Exception:
        # Remove incomplete output if processing fails.
        if output_path.exists():
            output_path.unlink()
        raise

    print(f"Input rows: {total_input_rows:,}")
    print(f"Output candles: {total_output_rows:,}")
    print(f"Saved: {output_path}")

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "input_rows": total_input_rows,
        "output_rows": total_output_rows,
        "output_path": str(output_path),
    }


# ==================================================
# 5. RUN ALL TIMEFRAMES FOR BTC AND ETH
# ==================================================

def main():

    results = []

    for symbol, input_path in DATASETS.items():

        for timeframe, frequency in TIMEFRAMES.items():

            result = resample_dataset(
                symbol=symbol,
                input_path=input_path,
                timeframe=timeframe,
                frequency=frequency,
            )

            results.append(result)

    print("\n" + "=" * 55)
    print("RESAMPLING SUMMARY")
    print("=" * 55)

    for result in results:
        print(
            f"{result['symbol']} | "
            f"{result['timeframe']} | "
            f"{result['output_rows']:,} candles"
        )

    print("\nAll resampling jobs completed.")


if __name__ == "__main__":
    main()
