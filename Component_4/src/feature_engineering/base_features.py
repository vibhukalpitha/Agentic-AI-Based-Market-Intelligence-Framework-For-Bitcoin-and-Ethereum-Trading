from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import unittest
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ==================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FEATURES_DIR = PROJECT_DIR / "data" / "features"
REPORT_DIR = PROJECT_DIR / "reports" / "feature_engineering"
REPORT_PATH = REPORT_DIR / "phase1_summary.json"

SYMBOLS = ("BTCUSDT", "ETHUSDT")

TIMEFRAMES = {
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
}

REQUIRED_COLUMNS = [
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
]

OPTIONAL_SOURCE_COLUMNS = [
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

NUMERIC_SOURCE_COLUMNS = [
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

OPTIONAL_MARK_INDEX_COLUMNS = [
    "mark_open",
    "mark_high",
    "mark_low",
    "mark_close",
    "index_open",
    "index_high",
    "index_low",
    "index_close",
]

FEATURE_COLUMNS = [
    "available_at_utc",
    "candle_range",
    "signed_body",
    "body_size",
    "upper_wick",
    "lower_wick",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "close_location_value",
    "true_range",
    "atr_14_sma",
    "normalized_atr",
    "atr_normalized_body",
    "relative_volume_20",
    "relative_trades_20",
    "taker_buy_ratio",
    "taker_sell_ratio",
    "taker_volume_imbalance",
    "close_to_close_return",
    "rolling_mean_return_20",
    "previous_20_high",
    "previous_20_low",
    "range_normalized_by_atr",
]

FORMULAS = {
    "available_at_utc": "timestamp_utc + timeframe interval.",
    "candle_range": "high - low.",
    "signed_body": "close - open.",
    "body_size": "abs(close - open).",
    "upper_wick": "high - max(open, close).",
    "lower_wick": "min(open, close) - low.",
    "body_ratio": "body_size / candle_range; missing when candle_range is zero.",
    "upper_wick_ratio": "upper_wick / candle_range; missing when candle_range is zero.",
    "lower_wick_ratio": "lower_wick / candle_range; missing when candle_range is zero.",
    "close_location_value": "(2 * close - high - low) / candle_range; missing when candle_range is zero.",
    "true_range": "max(high-low, abs(high-previous_close), abs(low-previous_close)); first row uses high-low.",
    "atr_14_sma": "Simple moving average of the latest 14 true_range values, including the current candle.",
    "normalized_atr": "atr_14_sma / close.",
    "atr_normalized_body": "signed_body / atr_14_sma.",
    "relative_volume_20": "current volume / average volume of the previous 20 candles.",
    "relative_trades_20": "current num_trades / average num_trades of the previous 20 candles.",
    "taker_buy_ratio": "taker_buy_base_vol / volume.",
    "taker_sell_ratio": "taker_sell_base_vol / volume.",
    "taker_volume_imbalance": "(taker_buy_base_vol - taker_sell_base_vol) / volume.",
    "close_to_close_return": "(close - previous_close) / previous_close.",
    "rolling_mean_return_20": "Average close_to_close_return from the previous 20 candles, excluding current candle.",
    "previous_20_high": "Maximum high from the previous 20 candles, excluding current candle.",
    "previous_20_low": "Minimum low from the previous 20 candles, excluding current candle.",
    "range_normalized_by_atr": "candle_range / atr_14_sma.",
}


# ==================================================
# 2. BASIC PARSING AND NUMERIC HELPERS
# ==================================================

def parse_utc_timestamp(value: str) -> datetime:
    normalized = value.strip()

    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    timestamp = datetime.fromisoformat(normalized)

    if timestamp.tzinfo is None:
        raise ValueError("timestamp is timezone-naive")

    return timestamp.astimezone(timezone.utc)


def format_utc_timestamp(timestamp: datetime) -> str:
    return timestamp.astimezone(timezone.utc).isoformat()


def parse_optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None

    number = float(value)

    if not math.isfinite(number):
        return None

    return number


def safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None

    if denominator == 0:
        return None

    result = numerator / denominator

    if not math.isfinite(result):
        return None

    return result


def mean_if_complete(values: deque[float | None], expected_length: int) -> float | None:
    if len(values) != expected_length:
        return None

    if any(value is None for value in values):
        return None

    return sum(value for value in values if value is not None) / expected_length


def max_if_complete(values: deque[float | None], expected_length: int) -> float | None:
    if len(values) != expected_length or any(value is None for value in values):
        return None

    return max(value for value in values if value is not None)


def min_if_complete(values: deque[float | None], expected_length: int) -> float | None:
    if len(values) != expected_length or any(value is None for value in values):
        return None

    return min(value for value in values if value is not None)


def format_feature_value(value: float | None) -> str:
    if value is None:
        return ""

    return f"{value:.12g}"


# ==================================================
# 3. ROW-LEVEL FEATURE ENGINEERING
# ==================================================

def compute_feature_rows(
    rows: list[dict[str, str]],
    timeframe: str,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    interval = TIMEFRAMES[timeframe]

    previous_close = None
    true_range_window: deque[float | None] = deque(maxlen=14)
    volume_window: deque[float | None] = deque(maxlen=20)
    trades_window: deque[float | None] = deque(maxlen=20)
    return_window: deque[float | None] = deque(maxlen=20)
    high_window: deque[float | None] = deque(maxlen=20)
    low_window: deque[float | None] = deque(maxlen=20)

    feature_rows = []
    missing_counts = {column: 0 for column in FEATURE_COLUMNS}
    leading_missing_counts = {column: 0 for column in FEATURE_COLUMNS}
    seen_non_missing = {column: False for column in FEATURE_COLUMNS}
    optional_missing_counts = {column: 0 for column in OPTIONAL_MARK_INDEX_COLUMNS}
    warnings = []
    anomaly_counts = {
        "negative_volume_or_activity": 0,
        "taker_buy_exceeds_volume": 0,
        "taker_sell_exceeds_volume": 0,
        "taker_buy_sell_sum_mismatch": 0,
        "timestamp_order_errors": 0,
    }
    previous_timestamp = None

    for row_number, row in enumerate(rows, start=2):
        output_row = dict(row)
        timestamp = parse_utc_timestamp(row["timestamp_utc"])
        output_row["timestamp_utc"] = format_utc_timestamp(timestamp)
        output_row["available_at_utc"] = format_utc_timestamp(timestamp + interval)

        if previous_timestamp is not None and timestamp <= previous_timestamp:
            anomaly_counts["timestamp_order_errors"] += 1

        previous_timestamp = timestamp

        values = {
            column: parse_optional_float(row.get(column))
            for column in NUMERIC_SOURCE_COLUMNS
        }

        for column in OPTIONAL_MARK_INDEX_COLUMNS:
            if parse_optional_float(row.get(column)) is None:
                optional_missing_counts[column] += 1

        open_price = values["open"]
        high_price = values["high"]
        low_price = values["low"]
        close_price = values["close"]
        volume = values["volume"]
        num_trades = values["num_trades"]
        taker_buy_base = values["taker_buy_base_vol"]
        taker_sell_base = values["taker_sell_base_vol"]

        if any(
            value is not None and value < 0
            for value in (
                volume,
                values["quote_volume"],
                num_trades,
                taker_buy_base,
                values["taker_buy_quote_vol"],
                taker_sell_base,
                values["taker_sell_quote_vol"],
            )
        ):
            anomaly_counts["negative_volume_or_activity"] += 1

        if volume is not None and taker_buy_base is not None and taker_buy_base > volume + 1e-9:
            anomaly_counts["taker_buy_exceeds_volume"] += 1

        if volume is not None and taker_sell_base is not None and taker_sell_base > volume + 1e-9:
            anomaly_counts["taker_sell_exceeds_volume"] += 1

        if (
            volume is not None
            and taker_buy_base is not None
            and taker_sell_base is not None
            and abs((taker_buy_base + taker_sell_base) - volume) > max(1e-6, abs(volume) * 1e-9)
        ):
            anomaly_counts["taker_buy_sell_sum_mismatch"] += 1

        candle_range = None
        signed_body = None
        body_size = None
        upper_wick = None
        lower_wick = None

        if None not in (open_price, high_price, low_price, close_price):
            candle_range = high_price - low_price
            signed_body = close_price - open_price
            body_size = abs(signed_body)
            upper_wick = high_price - max(open_price, close_price)
            lower_wick = min(open_price, close_price) - low_price

        true_range = None

        if None not in (high_price, low_price):
            high_low_range = high_price - low_price

            if previous_close is None:
                true_range = high_low_range
            else:
                true_range = max(
                    high_low_range,
                    abs(high_price - previous_close),
                    abs(low_price - previous_close),
                )

        true_range_window.append(true_range)
        atr_14_sma = mean_if_complete(true_range_window, 14)

        previous_volume_average = mean_if_complete(volume_window, 20)
        previous_trades_average = mean_if_complete(trades_window, 20)
        previous_return_average = mean_if_complete(return_window, 20)
        previous_20_high = max_if_complete(high_window, 20)
        previous_20_low = min_if_complete(low_window, 20)

        close_to_close_return = None

        if close_price is not None and previous_close is not None:
            close_to_close_return = safe_divide(close_price - previous_close, previous_close)

        feature_values = {
            "candle_range": candle_range,
            "signed_body": signed_body,
            "body_size": body_size,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "body_ratio": safe_divide(body_size, candle_range),
            "upper_wick_ratio": safe_divide(upper_wick, candle_range),
            "lower_wick_ratio": safe_divide(lower_wick, candle_range),
            "close_location_value": safe_divide(
                None if None in (close_price, high_price, low_price) else (2 * close_price - high_price - low_price),
                candle_range,
            ),
            "true_range": true_range,
            "atr_14_sma": atr_14_sma,
            "normalized_atr": safe_divide(atr_14_sma, close_price),
            "atr_normalized_body": safe_divide(signed_body, atr_14_sma),
            "relative_volume_20": safe_divide(volume, previous_volume_average),
            "relative_trades_20": safe_divide(num_trades, previous_trades_average),
            "taker_buy_ratio": safe_divide(taker_buy_base, volume),
            "taker_sell_ratio": safe_divide(taker_sell_base, volume),
            "taker_volume_imbalance": safe_divide(
                None if None in (taker_buy_base, taker_sell_base) else taker_buy_base - taker_sell_base,
                volume,
            ),
            "close_to_close_return": close_to_close_return,
            "rolling_mean_return_20": previous_return_average,
            "previous_20_high": previous_20_high,
            "previous_20_low": previous_20_low,
            "range_normalized_by_atr": safe_divide(candle_range, atr_14_sma),
        }

        for column, value in feature_values.items():
            output_row[column] = format_feature_value(value)

        for column in FEATURE_COLUMNS:
            if column == "available_at_utc":
                value_is_missing = output_row[column] == ""
            else:
                value_is_missing = feature_values[column] is None

            if value_is_missing:
                missing_counts[column] += 1
                if not seen_non_missing[column]:
                    leading_missing_counts[column] += 1
            else:
                seen_non_missing[column] = True

        feature_rows.append(output_row)

        previous_close = close_price
        volume_window.append(volume)
        trades_window.append(num_trades)
        return_window.append(close_to_close_return)
        high_window.append(high_price)
        low_window.append(low_price)

    for name, count in anomaly_counts.items():
        if count:
            warnings.append(f"{name}: {count}")

    stats = {
        "row_count": len(rows),
        "feature_columns": FEATURE_COLUMNS,
        "formulas": FORMULAS,
        "missing_feature_counts": missing_counts,
        "leading_missing_feature_counts": leading_missing_counts,
        "optional_source_missing_counts": optional_missing_counts,
        "warnings": warnings,
        "anomaly_counts": anomaly_counts,
        "timestamp_checks": {
            "chronological_order_errors": anomaly_counts["timestamp_order_errors"],
        },
        "atr_method": "Simple moving average over 14 true_range values, including the current completed candle.",
    }

    return feature_rows, stats


# ==================================================
# 4. FILE PROCESSING
# ==================================================

def validate_input_columns(path: Path, fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise ValueError(f"{path} has no CSV header.")

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in fieldnames
    ]

    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")


def process_file(symbol: str, timeframe: str, overwrite: bool = False) -> dict[str, object]:
    input_path = PROCESSED_DIR / timeframe / f"{symbol}_{timeframe}.csv"
    output_dir = FEATURES_DIR / timeframe
    output_path = output_dir / f"{symbol}_{timeframe}_features.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Expected input file does not exist: {input_path}")

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Feature output already exists: {output_path}. "
            "Use --overwrite to replace feature outputs intentionally."
        )

    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        validate_input_columns(input_path, reader.fieldnames)
        source_columns = list(reader.fieldnames or [])
        rows = list(reader)

    feature_rows, stats = compute_feature_rows(rows, timeframe)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_columns = source_columns + [
        column for column in FEATURE_COLUMNS
        if column not in source_columns
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(feature_rows)

    output_row_count = len(feature_rows)

    if output_row_count != stats["row_count"]:
        raise RuntimeError(
            f"Output row count mismatch for {symbol} {timeframe}: "
            f"{output_row_count} != {stats['row_count']}"
        )

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "input_rows": stats["row_count"],
        "output_rows": output_row_count,
        "source_columns": source_columns,
        "feature_columns_created": FEATURE_COLUMNS,
        "missing_feature_counts": stats["missing_feature_counts"],
        "leading_missing_feature_counts": stats["leading_missing_feature_counts"],
        "optional_source_missing_counts": stats["optional_source_missing_counts"],
        "timestamp_checks": stats["timestamp_checks"],
        "warnings": stats["warnings"],
        "anomaly_counts": stats["anomaly_counts"],
        "atr_method": stats["atr_method"],
        "formulas": FORMULAS,
    }


def run_tests() -> dict[str, object]:
    tests_dir = PROJECT_DIR / "tests"
    suite = unittest.defaultTestLoader.discover(str(tests_dir), pattern="test_base_features.py")
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)

    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "successful": result.wasSuccessful(),
    }


def write_summary(report: dict[str, object]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def run_feature_engineering(overwrite: bool = False, include_tests: bool = True) -> dict[str, object]:
    results = []

    for timeframe in TIMEFRAMES:
        for symbol in SYMBOLS:
            print(f"Processing {symbol} {timeframe}...")
            results.append(process_file(symbol, timeframe, overwrite=overwrite))

    report: dict[str, object] = {
        "phase": "Component 4 - Phase 1 Base Feature Engineering",
        "generated_at_utc": format_utc_timestamp(datetime.now(timezone.utc)),
        "project_dir": str(PROJECT_DIR),
        "datasets": results,
        "tests": None,
    }

    if include_tests:
        print("\nRunning Phase 1 tests...")
        report["tests"] = run_tests()

    write_summary(report)
    print(f"\nSaved summary report: {REPORT_PATH}")

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Phase 1 base market features for Component 4."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing feature outputs. Source processed data is never modified.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Generate features without running the Phase 1 unit tests.",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    report = run_feature_engineering(
        overwrite=args.overwrite,
        include_tests=not args.skip_tests,
    )

    test_result = report.get("tests")

    if isinstance(test_result, dict) and not test_result.get("successful", False):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
