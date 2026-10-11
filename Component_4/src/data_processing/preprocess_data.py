
from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. DATASET CONFIGURATION
# --------------------------------------------------

DATASETS = {
    "BTCUSDT": RAW_DATA_DIR / "BTCUSDT.csv",
    "ETHUSDT": RAW_DATA_DIR / "ETHUSDT.csv",
}

CHUNK_SIZE = 200_000

TIMESTAMP_COLUMN = "timestamp_utc"

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "num_trades",
    "taker_buy_base_vol",
    "taker_buy_quote_vol",
    "taker_sell_base_vol",
    "taker_sell_quote_vol",
    "mark_open",
    "mark_high",
    "mark_low",
    "mark_close",
    "index_open",
    "index_high",
    "index_low",
    "index_close",
]


# --------------------------------------------------
# 3. PREPROCESS ONE DATASET
# --------------------------------------------------

def preprocess_dataset(symbol, input_path):

    print(f"\nProcessing {symbol}...")
    print(f"Input: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {input_path}"
        )

    output_path = (
        PROCESSED_DATA_DIR / f"{symbol}_1m_preprocessed.csv"
    )

    if output_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            "Rename or move the existing output before rerunning."
        )

    total_rows = 0
    first_chunk = True

    try:
        for chunk in pd.read_csv(
            input_path,
            chunksize=CHUNK_SIZE,
        ):

            # Convert timestamps to UTC datetime.
            timestamps = pd.to_datetime(
                chunk[TIMESTAMP_COLUMN],
                utc=True,
                errors="raise",
            )

            # Convert numeric columns without filling missing values.
            for column in NUMERIC_COLUMNS:
                chunk[column] = pd.to_numeric(
                    chunk[column],
                    errors="raise",
                )

            # Store timestamps in a consistent ISO-8601 UTC format.
            chunk[TIMESTAMP_COLUMN] = timestamps.map(
                lambda value: value.isoformat()
            )

            # Save each chunk without loading the entire dataset.
            chunk.to_csv(
                output_path,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
            )

            total_rows += len(chunk)
            first_chunk = False

            print(f"Rows processed: {total_rows:,}")

    except Exception:
        # Avoid leaving a partial output that looks complete.
        if output_path.exists():
            output_path.unlink()
        raise

    print(f"Completed: {symbol}")
    print(f"Total rows: {total_rows:,}")
    print(f"Output: {output_path}")

    return {
        "symbol": symbol,
        "total_rows": total_rows,
        "output_path": str(output_path),
    }


# --------------------------------------------------
# 4. RUN PREPROCESSING
# --------------------------------------------------

def main():

    results = []

    for symbol, input_path in DATASETS.items():
        result = preprocess_dataset(symbol, input_path)
        results.append(result)

    print("\n" + "=" * 50)
    print("PREPROCESSING SUMMARY")
    print("=" * 50)

    for result in results:
        print(
            f"{result['symbol']}: "
            f"{result['total_rows']:,} rows"
        )
        print(f"Output: {result['output_path']}")


if __name__ == "__main__":
    main()
