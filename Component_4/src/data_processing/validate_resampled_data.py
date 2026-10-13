from __future__ import annotations

import csv
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ==================================================
# 1. PROJECT PATHS AND DATASET CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

RAW_START = datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)
RAW_END = datetime(2025, 8, 31, 23, 59, tzinfo=timezone.utc)

SYMBOLS = ("BTCUSDT", "ETHUSDT")

SOURCE_1M_FILES = {
    symbol: PROCESSED_DIR / "1m" / f"{symbol}_1m.csv"
    for symbol in SYMBOLS
}

TIMEFRAMES = {
    "15m": {
        "interval": timedelta(minutes=15),
        "expected_rows": 198_720,
    },
    "1h": {
        "interval": timedelta(hours=1),
        "expected_rows": 49_680,
    },
    "4h": {
        "interval": timedelta(hours=4),
        "expected_rows": 12_420,
    },
    "1d": {
        "interval": timedelta(days=1),
        "expected_rows": 2_070,
    },
}

EXPECTED_COLUMNS = [
    "timestamp_utc",
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

MAIN_NUMERIC_COLUMNS = [
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
]

ACTIVITY_COLUMNS = [
    "volume",
    "quote_volume",
    "num_trades",
    "taker_buy_base_vol",
    "taker_buy_quote_vol",
    "taker_sell_base_vol",
    "taker_sell_quote_vol",
]

OPTIONAL_COLUMNS = [
    "mark_open",
    "mark_high",
    "mark_low",
    "mark_close",
    "index_open",
    "index_high",
    "index_low",
    "index_close",
]


# ==================================================
# 2. SMALL REPORTING HELPERS
# ==================================================

def print_check(status: str, message: str) -> None:
    print(f"  [{status}] {message}")


def parse_utc_timestamp(value: str) -> datetime:
    normalized = value.strip()

    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    timestamp = datetime.fromisoformat(normalized)

    if timestamp.tzinfo is None:
        raise ValueError("timestamp is timezone-naive")

    return timestamp.astimezone(timezone.utc)


def parse_required_number(value: str) -> float:
    if value is None or value.strip() == "":
        raise ValueError("missing value")

    number = float(value)

    if not math.isfinite(number):
        raise ValueError("non-finite value")

    return number


def expected_final_start(interval: timedelta) -> datetime:
    total_minutes = int((RAW_END - RAW_START).total_seconds() // 60)
    interval_minutes = int(interval.total_seconds() // 60)
    completed_intervals = total_minutes // interval_minutes

    return RAW_START + completed_intervals * interval


def format_examples(examples: list[str]) -> str:
    if not examples:
        return ""

    return " Examples: " + "; ".join(examples[:3])


# ==================================================
# 3. VALIDATE ONE RESAMPLED CSV FILE
# ==================================================

def validate_resampled_file(symbol: str, timeframe: str, csv_path: Path) -> tuple[int, int]:
    interval = TIMEFRAMES[timeframe]["interval"]
    expected_rows = TIMEFRAMES[timeframe]["expected_rows"]
    expected_last_timestamp = expected_final_start(interval)

    print("\n" + "=" * 72)
    print(f"{symbol} | {timeframe}")
    print(f"File: {csv_path}")

    failures = 0
    warnings = 0

    if not csv_path.exists():
        print_check("FAIL", "File does not exist.")
        return 1, 0

    print_check("PASS", "File exists.")

    row_count = 0
    previous_timestamp = None
    first_timestamp = None
    last_timestamp = None
    seen_timestamps = set()

    timestamp_parse_errors = 0
    duplicate_timestamps = 0
    chronological_errors = 0
    spacing_errors = 0

    missing_main_values = {column: 0 for column in MAIN_NUMERIC_COLUMNS}
    non_finite_main_values = {column: 0 for column in MAIN_NUMERIC_COLUMNS}
    negative_activity_values = {column: 0 for column in ACTIVITY_COLUMNS}
    optional_missing_values = {column: 0 for column in OPTIONAL_COLUMNS}

    ohlc_errors = 0

    timestamp_examples = []
    duplicate_examples = []
    spacing_examples = []
    ohlc_examples = []
    main_value_examples = []
    negative_examples = []

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames != EXPECTED_COLUMNS:
            print_check("FAIL", f"Columns do not match expected schema: {reader.fieldnames}")
            failures += 1
            return failures, warnings

        print_check("PASS", "Columns match expected schema.")

        for row_number, row in enumerate(reader, start=2):
            row_count += 1

            try:
                timestamp = parse_utc_timestamp(row["timestamp_utc"])
            except Exception as exc:
                timestamp_parse_errors += 1
                if len(timestamp_examples) < 3:
                    timestamp_examples.append(f"row {row_number}: {row['timestamp_utc']} ({exc})")
                continue

            if first_timestamp is None:
                first_timestamp = timestamp

            last_timestamp = timestamp

            if timestamp in seen_timestamps:
                duplicate_timestamps += 1
                if len(duplicate_examples) < 3:
                    duplicate_examples.append(f"row {row_number}: {timestamp.isoformat()}")
            else:
                seen_timestamps.add(timestamp)

            if previous_timestamp is not None:
                actual_gap = timestamp - previous_timestamp

                if timestamp <= previous_timestamp:
                    chronological_errors += 1
                    if len(spacing_examples) < 3:
                        spacing_examples.append(
                            f"row {row_number}: {previous_timestamp.isoformat()} -> "
                            f"{timestamp.isoformat()}"
                        )

                if actual_gap != interval:
                    spacing_errors += 1
                    if len(spacing_examples) < 3:
                        spacing_examples.append(
                            f"row {row_number}: gap {actual_gap}, expected {interval}"
                        )

            previous_timestamp = timestamp

            parsed_values = {}

            for column in MAIN_NUMERIC_COLUMNS:
                raw_value = row[column]

                if raw_value is None or raw_value.strip() == "":
                    missing_main_values[column] += 1
                    if len(main_value_examples) < 3:
                        main_value_examples.append(f"row {row_number}: missing {column}")
                    continue

                try:
                    parsed_values[column] = parse_required_number(raw_value)
                except ValueError as exc:
                    non_finite_main_values[column] += 1
                    if len(main_value_examples) < 3:
                        main_value_examples.append(f"row {row_number}: {column}={raw_value} ({exc})")
                    continue

                if column in ACTIVITY_COLUMNS and parsed_values[column] < 0:
                    negative_activity_values[column] += 1
                    if len(negative_examples) < 3:
                        negative_examples.append(
                            f"row {row_number}: {column}={parsed_values[column]}"
                        )

            for column in OPTIONAL_COLUMNS:
                raw_value = row[column]
                if raw_value is None or raw_value.strip() == "":
                    optional_missing_values[column] += 1

            if all(column in parsed_values for column in ("open", "high", "low", "close")):
                open_price = parsed_values["open"]
                high_price = parsed_values["high"]
                low_price = parsed_values["low"]
                close_price = parsed_values["close"]

                if (
                    high_price < open_price
                    or high_price < close_price
                    or high_price < low_price
                    or low_price > open_price
                    or low_price > close_price
                ):
                    ohlc_errors += 1
                    if len(ohlc_examples) < 3:
                        ohlc_examples.append(
                            f"row {row_number}: open={open_price}, high={high_price}, "
                            f"low={low_price}, close={close_price}"
                        )

    if row_count == expected_rows:
        print_check("PASS", f"Row count is {row_count:,}.")
    else:
        print_check("FAIL", f"Row count is {row_count:,}; expected {expected_rows:,}.")
        failures += 1

    if timestamp_parse_errors == 0:
        print_check("PASS", "All timestamps parse as timezone-aware UTC values.")
    else:
        print_check(
            "FAIL",
            f"{timestamp_parse_errors:,} timestamp parse errors."
            f"{format_examples(timestamp_examples)}",
        )
        failures += 1

    if first_timestamp == RAW_START:
        print_check("PASS", f"First timestamp is {RAW_START.isoformat()}.")
    else:
        print_check(
            "FAIL",
            f"First timestamp is {first_timestamp}; expected {RAW_START.isoformat()}.",
        )
        failures += 1

    if last_timestamp == expected_last_timestamp:
        print_check("PASS", f"Final candle starts at {expected_last_timestamp.isoformat()}.")
    else:
        print_check(
            "FAIL",
            f"Final candle starts at {last_timestamp}; expected "
            f"{expected_last_timestamp.isoformat()}.",
        )
        failures += 1

    final_candle_end = None
    if last_timestamp is not None:
        final_candle_end = last_timestamp + interval - timedelta(minutes=1)

    if final_candle_end == RAW_END:
        print_check(
            "PASS",
            f"Final candle is complete through {RAW_END.isoformat()}.",
        )
    else:
        print_check(
            "FAIL",
            f"Final candle is not complete against raw end {RAW_END.isoformat()}; "
            f"computed final covered minute is {final_candle_end}.",
        )
        failures += 1

    if duplicate_timestamps == 0:
        print_check("PASS", "No duplicate timestamps.")
    else:
        print_check(
            "FAIL",
            f"{duplicate_timestamps:,} duplicate timestamps."
            f"{format_examples(duplicate_examples)}",
        )
        failures += 1

    if chronological_errors == 0:
        print_check("PASS", "Timestamps are strictly chronological.")
    else:
        print_check(
            "FAIL",
            f"{chronological_errors:,} chronological ordering errors."
            f"{format_examples(spacing_examples)}",
        )
        failures += 1

    if spacing_errors == 0:
        print_check("PASS", f"All timestamps are spaced by {interval}.")
    else:
        print_check(
            "FAIL",
            f"{spacing_errors:,} timestamp spacing errors."
            f"{format_examples(spacing_examples)}",
        )
        failures += 1

    total_missing_main = sum(missing_main_values.values())
    total_non_finite_main = sum(non_finite_main_values.values())

    if total_missing_main == 0 and total_non_finite_main == 0:
        print_check("PASS", "No missing or non-finite values in main OHLCV/activity columns.")
    else:
        print_check(
            "FAIL",
            f"Main columns contain {total_missing_main:,} missing and "
            f"{total_non_finite_main:,} non-finite values."
            f"{format_examples(main_value_examples)}",
        )
        failures += 1

    total_negative_activity = sum(negative_activity_values.values())

    if total_negative_activity == 0:
        print_check("PASS", "No negative volume or trade-activity values.")
    else:
        print_check(
            "FAIL",
            f"{total_negative_activity:,} negative volume/trade-activity values."
            f"{format_examples(negative_examples)}",
        )
        failures += 1

    if ohlc_errors == 0:
        print_check("PASS", "OHLC relationships are consistent.")
    else:
        print_check(
            "FAIL",
            f"{ohlc_errors:,} OHLC consistency errors."
            f"{format_examples(ohlc_examples)}",
        )
        failures += 1

    total_optional_missing = sum(optional_missing_values.values())

    if total_optional_missing == 0:
        print_check("PASS", "No missing values in optional mark/index columns.")
    else:
        warnings += 1
        missing_summary = ", ".join(
            f"{column}={count:,}"
            for column, count in optional_missing_values.items()
            if count
        )
        print_check(
            "WARNING",
            "Optional mark/index columns contain missing values: "
            f"{missing_summary}.",
        )

    return failures, warnings


# ==================================================
# 4. RUN VALIDATION FOR ALL RESAMPLED DATASETS
# ==================================================

def main() -> int:
    total_failures = 0
    total_warnings = 0

    print("=" * 72)
    print("RESAMPLED DATA VALIDATION")
    print("=" * 72)
    print(f"Project directory: {PROJECT_DIR}")
    print(f"Known raw coverage: {RAW_START.isoformat()} through {RAW_END.isoformat()}")
    print("\nChecking actual 1-minute source files used by this project:")

    for symbol, source_path in SOURCE_1M_FILES.items():
        if source_path.exists():
            print_check("PASS", f"{symbol} source exists: {source_path}")
        else:
            print_check("FAIL", f"{symbol} source is missing: {source_path}")
            total_failures += 1

    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            csv_path = PROCESSED_DIR / timeframe / f"{symbol}_{timeframe}.csv"
            failures, warnings = validate_resampled_file(symbol, timeframe, csv_path)
            total_failures += failures
            total_warnings += warnings

    print("\n" + "=" * 72)
    print("VALIDATION SUMMARY")
    print("=" * 72)
    print(f"FAIL conditions: {total_failures}")
    print(f"WARNING conditions: {total_warnings}")

    if total_failures:
        print("Overall result: FAIL")
        return 1

    print("Overall result: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
