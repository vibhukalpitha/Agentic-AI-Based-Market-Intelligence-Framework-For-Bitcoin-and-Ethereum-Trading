from __future__ import annotations

import copy
import math
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_engineering.base_features import compute_feature_rows, parse_utc_timestamp


def make_row(
    minute: int,
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
    volume: float = 100.0,
    trades: float = 10.0,
    taker_buy: float = 40.0,
    mark_open: str = "100",
    index_open: str = "100",
) -> dict[str, str]:
    timestamp = datetime(2020, 1, 1, 0, minute, tzinfo=timezone.utc)
    taker_sell = volume - taker_buy

    return {
        "timestamp_utc": timestamp.isoformat(),
        "open": str(open_price),
        "high": str(high_price),
        "low": str(low_price),
        "close": str(close_price),
        "volume": str(volume),
        "quote_volume": str(volume * close_price),
        "num_trades": str(trades),
        "taker_buy_base_vol": str(taker_buy),
        "taker_buy_quote_vol": str(taker_buy * close_price),
        "taker_sell_base_vol": str(taker_sell),
        "taker_sell_quote_vol": str(taker_sell * close_price),
        "mark_open": mark_open,
        "mark_high": mark_open,
        "mark_low": mark_open,
        "mark_close": mark_open,
        "index_open": index_open,
        "index_high": index_open,
        "index_low": index_open,
        "index_close": index_open,
    }


def feature_float(row: dict[str, str], column: str) -> float | None:
    value = row[column]

    if value == "":
        return None

    return float(value)


class BaseFeatureTests(unittest.TestCase):
    def test_candle_body_and_wicks(self) -> None:
        rows = [make_row(0, 10, 15, 8, 12)]
        features, _ = compute_feature_rows(rows, "15m")
        row = features[0]

        self.assertEqual(feature_float(row, "candle_range"), 7)
        self.assertEqual(feature_float(row, "signed_body"), 2)
        self.assertEqual(feature_float(row, "body_size"), 2)
        self.assertEqual(feature_float(row, "upper_wick"), 3)
        self.assertEqual(feature_float(row, "lower_wick"), 2)
        self.assertTrue(math.isclose(feature_float(row, "body_ratio"), 2 / 7))
        self.assertTrue(math.isclose(feature_float(row, "upper_wick_ratio"), 3 / 7))
        self.assertTrue(math.isclose(feature_float(row, "lower_wick_ratio"), 2 / 7))
        self.assertTrue(math.isclose(feature_float(row, "close_location_value"), 1 / 7))

    def test_true_range_and_atr_sma(self) -> None:
        rows = [
            make_row(i, 100 + i, 110 + i, 95 + i, 100 + i)
            for i in range(14)
        ]
        features, _ = compute_feature_rows(rows, "15m")

        self.assertEqual(feature_float(features[0], "true_range"), 15)
        self.assertIsNone(feature_float(features[12], "atr_14_sma"))
        self.assertEqual(feature_float(features[13], "atr_14_sma"), 15)

    def test_previous_volume_and_trade_windows_exclude_current(self) -> None:
        rows = [
            make_row(i, 100, 101, 99, 100, volume=100 + i, trades=10 + i)
            for i in range(21)
        ]
        features, _ = compute_feature_rows(rows, "15m")

        expected_volume_average = sum(100 + i for i in range(20)) / 20
        expected_trade_average = sum(10 + i for i in range(20)) / 20

        self.assertIsNone(feature_float(features[19], "relative_volume_20"))
        self.assertTrue(
            math.isclose(
                feature_float(features[20], "relative_volume_20"),
                120 / expected_volume_average,
            )
        )
        self.assertTrue(
            math.isclose(
                feature_float(features[20], "relative_trades_20"),
                30 / expected_trade_average,
            )
        )

    def test_zero_range_and_zero_denominators(self) -> None:
        rows = [
            make_row(0, 100, 100, 100, 100, volume=0, trades=0, taker_buy=0),
        ]
        features, _ = compute_feature_rows(rows, "15m")
        row = features[0]

        self.assertEqual(feature_float(row, "candle_range"), 0)
        self.assertIsNone(feature_float(row, "body_ratio"))
        self.assertIsNone(feature_float(row, "close_location_value"))
        self.assertIsNone(feature_float(row, "taker_buy_ratio"))
        self.assertIsNone(feature_float(row, "taker_sell_ratio"))
        self.assertIsNone(feature_float(row, "taker_volume_imbalance"))

    def test_missing_optional_mark_index_values_are_preserved_in_stats(self) -> None:
        rows = [
            make_row(0, 100, 101, 99, 100, mark_open="", index_open=""),
            make_row(1, 100, 101, 99, 100),
        ]
        _, stats = compute_feature_rows(rows, "15m")

        self.assertEqual(stats["optional_source_missing_counts"]["mark_open"], 1)
        self.assertEqual(stats["optional_source_missing_counts"]["index_open"], 1)

    def test_available_at_utc_by_timeframe(self) -> None:
        row = make_row(0, 100, 101, 99, 100)

        rows_15m, _ = compute_feature_rows([copy.deepcopy(row)], "15m")
        rows_1h, _ = compute_feature_rows([copy.deepcopy(row)], "1h")
        rows_4h, _ = compute_feature_rows([copy.deepcopy(row)], "4h")

        self.assertEqual(
            parse_utc_timestamp(rows_15m[0]["available_at_utc"]),
            datetime(2020, 1, 1, 0, 15, tzinfo=timezone.utc),
        )
        self.assertEqual(
            parse_utc_timestamp(rows_1h[0]["available_at_utc"]),
            datetime(2020, 1, 1, 1, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(
            parse_utc_timestamp(rows_4h[0]["available_at_utc"]),
            datetime(2020, 1, 1, 4, 0, tzinfo=timezone.utc),
        )

    def test_warmup_period_handling(self) -> None:
        rows = [
            make_row(i, 100, 101, 99, 100 + i, volume=100, trades=10)
            for i in range(22)
        ]
        features, stats = compute_feature_rows(rows, "15m")

        self.assertIsNone(feature_float(features[12], "atr_14_sma"))
        self.assertIsNotNone(feature_float(features[13], "atr_14_sma"))
        self.assertIsNone(feature_float(features[20], "rolling_mean_return_20"))
        self.assertIsNotNone(feature_float(features[21], "rolling_mean_return_20"))
        self.assertEqual(stats["leading_missing_feature_counts"]["atr_14_sma"], 13)

    def test_no_future_data_leakage(self) -> None:
        rows = [
            make_row(i, 100, 101, 99, 100 + i, volume=100 + i, trades=10 + i)
            for i in range(25)
        ]
        altered = copy.deepcopy(rows)
        altered[-1]["high"] = "9999"
        altered[-1]["volume"] = "9999"
        altered[-1]["close"] = "9999"

        original_features, _ = compute_feature_rows(rows, "15m")
        altered_features, _ = compute_feature_rows(altered, "15m")

        self.assertEqual(original_features[20], altered_features[20])

    def test_row_counts_and_timestamp_ordering(self) -> None:
        rows = [
            make_row(i, 100, 101, 99, 100 + i)
            for i in range(5)
        ]
        features, stats = compute_feature_rows(rows, "15m")

        self.assertEqual(len(features), len(rows))
        self.assertEqual(stats["row_count"], len(rows))
        self.assertEqual(stats["timestamp_checks"]["chronological_order_errors"], 0)


if __name__ == "__main__":
    unittest.main()
