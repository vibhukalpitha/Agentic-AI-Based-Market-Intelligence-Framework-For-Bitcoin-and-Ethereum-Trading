from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from itertools import islice

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from zone_detection.imbalance import (
    Config, FeatureCandle, REQUIRED_COLUMNS, detect_candidates, displacement_candidate,
    fvg_candidate, paths_for, preflight, process_dataset, read_candles,
)
from zone_detection.swings import INTERVALS, utc

START = datetime(2020, 1, 1, tzinfo=timezone.utc)


def candle(index, opening=100, high=102, low=99, close=101, atr=2):
    return FeatureCandle(START + timedelta(minutes=15 * index),
                         START + timedelta(minutes=15 * (index + 1)),
                         opening, high, low, close, atr,
                         abs(close - opening) / (high - low) if high > low else None)


def bullish():
    return [candle(0, 99, 101, 98, 100), candle(1, 100, 106, 100, 105),
            candle(2, 105, 107, 104, 106)]


def bearish():
    return [candle(0, 105, 106, 104, 105), candle(1, 105, 105, 99, 100),
            candle(2, 100, 101, 98, 99)]


def fvg(bars):
    return fvg_candidate(*bars, "BTCUSDT", "15m", Config())


def source_rows(bars):
    return [{"timestamp_utc": b.timestamp.isoformat(), "available_at_utc": b.available_at.isoformat(),
             "open": b.open, "high": b.high, "low": b.low, "close": b.close,
             "atr_14_sma": b.atr, "body_ratio": b.body_ratio,
             "atr_normalized_body": (b.close - b.open) / b.atr if b.atr else None}
            for b in bars]


def write_input(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def events(bars, include_body=True):
    return list(detect_candidates(bars, "BTCUSDT", "15m", Config(), include_body))


class GeometryTests(unittest.TestCase):
    def test_bullish_gap(self):
        event = fvg(bullish())
        self.assertEqual(event["imbalance_type"], "bullish_fvg")
        self.assertEqual((event["lower_price"], event["upper_price"]), (101, 104))
        self.assertEqual(event["gap_width"], 3)
        self.assertEqual(event["gap_width_atr"], 1.5)

    def test_bearish_gap(self):
        event = fvg(bearish())
        self.assertEqual(event["imbalance_type"], "bearish_fvg")
        self.assertEqual((event["lower_price"], event["upper_price"]), (101, 104))
        self.assertTrue(event["aligned_displacement"])

    def test_overlapping_and_equal_boundaries_have_no_fvg(self):
        for bars in ([candle(i) for i in range(3)],
                     [candle(0), candle(1), candle(2, 103, 104, 102, 103)],
                     [candle(0), candle(1), candle(2, 98, 99, 97, 98)]):
            with self.subTest(bars=bars):
                self.assertIsNone(fvg(bars))

    def test_c3_close_availability_and_c2_atr(self):
        bars = bullish()
        bars[0] = replace(bars[0], atr=100)
        bars[2] = replace(bars[2], atr=1000)
        event = fvg(bars)
        self.assertEqual(event["available_at_utc"], (START + timedelta(minutes=45)).isoformat())
        self.assertEqual(event["reference_candle_time_utc"], bars[1].timestamp.isoformat())
        self.assertEqual(event["reference_atr_available_at_utc"], bars[1].available_at.isoformat())
        self.assertEqual(event["displacement_body_atr"], 2.5)
        self.assertEqual(event["gap_width_atr"], 1.5)
        self.assertTrue(event["aligned_displacement"])

    def test_weak_or_opposite_displacement_keeps_fvg_and_metrics(self):
        bars = bullish()
        bars[1] = candle(1, 103, 106, 100, 102)
        event = fvg(bars)
        self.assertFalse(event["aligned_displacement"])
        self.assertEqual(event["displacement_body_atr"], 0.5)
        self.assertAlmostEqual(event["displacement_body_ratio"], 1 / 6)
        bars[1] = candle(1, 105, 106, 100, 100)
        self.assertFalse(fvg(bars)["aligned_displacement"])

    def test_missing_and_invalid_atr_does_not_suppress_fvg(self):
        for atr in (None, 0, -1, math.nan, math.inf):
            with self.subTest(atr=atr):
                bars = bullish()
                bars[1] = replace(bars[1], atr=atr)
                event = fvg(bars)
                self.assertFalse(event["aligned_displacement"])
                self.assertIsNone(event["gap_width_atr"])
                self.assertIsNone(event["displacement_body_atr"])
                self.assertIsNone(displacement_candidate(bars[1], "BTCUSDT", "15m", Config()))

    def test_zero_range_does_not_displace(self):
        bars = bullish()
        bars[1] = candle(1, 103, 103, 103, 103)
        event = fvg(bars)
        self.assertFalse(event["aligned_displacement"])
        self.assertIsNone(event["displacement_body_ratio"])
        self.assertEqual(event["displacement_body_atr"], 0)

    def test_displacement_body_interval_and_immediate_availability(self):
        for middle, direction in ((bullish()[1], "bullish"), (bearish()[1], "bearish")):
            with self.subTest(direction=direction):
                event = displacement_candidate(middle, "BTCUSDT", "15m", Config())
                self.assertEqual(event["direction"], direction)
                self.assertEqual(event["imbalance_type"], "displacement_body")
                self.assertEqual((event["lower_price"], event["upper_price"]), (100, 105))
                self.assertEqual(event["available_at_utc"], middle.available_at.isoformat())
                self.assertIsNone(event["gap_width"])

    def test_reproducible_ids_and_nonconsecutive_window(self):
        self.assertEqual(fvg(bullish())["zone_id"], fvg(bullish())["zone_id"])
        self.assertNotEqual(fvg(bullish())["zone_id"], fvg(bearish())["zone_id"])
        bars = bullish()
        bars[2] = replace(bars[2], timestamp=bars[2].timestamp + timedelta(minutes=15))
        self.assertIsNone(fvg(bars))

    def test_thresholds_inclusive_and_config_validation(self):
        bar = candle(0, 100, 104, 99, 103, atr=3)
        self.assertIsNotNone(displacement_candidate(bar, "BTCUSDT", "15m", Config()))
        self.assertIsNone(displacement_candidate(bar, "BTCUSDT", "15m", Config(1.01, .60)))
        self.assertIsNone(displacement_candidate(bar, "BTCUSDT", "15m", Config(1, .61)))
        for args in ((0, .6), (math.nan, .6), (1, 1.1)):
            with self.assertRaises(ValueError):
                Config(*args)

    def test_invalid_ohlc_cannot_form_fvg(self):
        for value in (math.nan, math.inf, -1):
            bars = bullish()
            bars[1] = replace(bars[1], close=value)
            self.assertIsNone(fvg(bars))


class EventTests(unittest.TestCase):
    def test_aligned_fvg_reuses_body_identity_without_new_creation(self):
        found = events(bullish())
        self.assertEqual([r["event_type"] for r in found], ["create", "fvg_confirmation"])
        self.assertEqual(found[0]["imbalance_type"], "displacement_body")
        self.assertEqual(found[1]["imbalance_type"], "bullish_fvg")
        self.assertEqual(found[0]["zone_id"], found[1]["zone_id"])
        self.assertEqual(found[1]["candidate_created_at_utc"], found[0]["available_at_utc"])
        self.assertGreater(found[1]["available_at_utc"], found[0]["available_at_utc"])
        self.assertEqual(sum(r["event_type"] == "create" for r in found), 1)

    def test_bearish_fvg_also_shares_body_identity(self):
        found = events(bearish())
        self.assertEqual(len(found), 2)
        self.assertEqual(found[0]["zone_id"], found[1]["zone_id"])
        self.assertEqual(found[1]["imbalance_type"], "bearish_fvg")

    def test_body_only_and_trailing_impulse_need_no_future_candle(self):
        impulse = bullish()[1]
        found = events([impulse])
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["available_at_utc"], impulse.available_at.isoformat())
        # C3 overlaps C1, so this is a body candidate with no FVG confirmation.
        bars = bullish()
        bars[2] = candle(2, 102, 106, 100, 102)
        self.assertEqual([r["imbalance_type"] for r in events(bars)], ["displacement_body"])

    def test_fvg_only_policy_produces_creation_rows(self):
        found = events(bullish(), include_body=False)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["imbalance_type"], "bullish_fvg")
        self.assertEqual(found[0]["event_type"], "create")
        self.assertEqual(found[0]["candidate_created_at_utc"], found[0]["available_at_utc"])

    def test_prefix_invariance_all_prefixes_and_changed_future(self):
        bars = bullish() + [candle(3, 106, 112, 105, 111), candle(4, 111, 114, 110, 112)]
        full = events(bars)
        for length in range(1, len(bars) + 1):
            with self.subTest(length=length):
                cutoff = bars[length - 1].available_at
                self.assertEqual(events(bars[:length]),
                                 [r for r in full if utc(r["available_at_utc"]) <= cutoff])
        changed = bars[:2] + [candle(2, 101, 110, 95, 101)]
        original_creation = events(bars[:2])
        self.assertEqual(original_creation, events(changed))
        self.assertEqual(original_creation[0], full[0])

    def test_weak_displacement_fvg_gets_independent_creation(self):
        bars = bullish()
        bars[1] = candle(1, 103, 106, 100, 104)
        found = events(bars)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["event_type"], "create")
        self.assertFalse(found[0]["aligned_displacement"])


class FileTests(unittest.TestCase):
    def test_read_only_inputs_and_output_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = paths_for(root, "BTCUSDT", "15m")
            write_input(source, source_rows(bullish()))
            original = source.read_bytes()
            stats = process_dataset("BTCUSDT", "15m", source, output, Config())
            self.assertEqual(stats["unique_candidate_count"], 1)
            self.assertEqual(stats["displacement_only_candidate_count_as_of_end"], 0)
            self.assertEqual(stats["linked_fvg_confirmation_count"], 1)
            self.assertEqual(stats["output_event_rows"], 2)
            saved = output.read_bytes()
            with self.assertRaises(FileExistsError):
                process_dataset("BTCUSDT", "15m", source, output, Config())
            self.assertEqual(saved, output.read_bytes())
            process_dataset("BTCUSDT", "15m", source, output, Config(), overwrite=True)
            self.assertEqual(saved, output.read_bytes())
            self.assertEqual(original, source.read_bytes())
            with self.assertRaises(ValueError):
                process_dataset("BTCUSDT", "15m", source, source, Config(), overwrite=True)

    def test_preflight_protects_report_and_any_existing_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for timeframe in INTERVALS:
                for symbol in ("BTCUSDT", "ETHUSDT"):
                    source, _ = paths_for(root, symbol, timeframe)
                    write_input(source, [])
            self.assertEqual(len(preflight(root, False)), 6)
            report = root / "reports" / "zone_detection" / "imbalance_summary.json"
            report.parent.mkdir(parents=True)
            report.touch()
            with self.assertRaises(FileExistsError):
                preflight(root, False)
            self.assertEqual(len(preflight(root, True)), 6)
            report.unlink()
            _, output = paths_for(root, "ETHUSDT", "4h")
            output.parent.mkdir(parents=True)
            output.touch()
            with self.assertRaises(FileExistsError):
                preflight(root, False)

    def test_missing_nonfinite_zero_negative_atr_are_not_filled(self):
        rows = source_rows([candle(i) for i in range(7)])
        for row, raw in zip(rows, (None, "nan", "inf", "0", "-1", "bad", "2")):
            row["atr_14_sma"] = raw
            row["atr_normalized_body"] = None
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            write_input(source, rows)
            stats = Counter()
            parsed = list(read_candles(source, "15m", stats))
        self.assertEqual([b.atr for b in parsed], [None] * 6 + [2])
        self.assertEqual(stats["missing_atr_rows"], 1)
        self.assertEqual(stats["invalid_atr_rows"], 5)

    def test_bad_features_disable_displacement_but_keep_geometry(self):
        rows = source_rows(bullish())
        rows[1]["body_ratio"] = "nan"
        rows[1]["atr_normalized_body"] = "999"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            write_input(source, rows)
            stats = Counter()
            found = events(read_candles(source, "15m", stats))
        self.assertEqual(len(found), 1)
        self.assertFalse(found[0]["aligned_displacement"])
        self.assertIsNone(found[0]["displacement_body_ratio"])
        self.assertEqual(found[0]["displacement_body_atr"], 2.5)
        self.assertEqual(stats["invalid_body_ratio_rows"], 1)
        self.assertEqual(stats["atr_normalized_body_mismatch_rows"], 1)

    def test_zero_range_feature_stays_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            write_input(source, source_rows([candle(0, 100, 100, 100, 100)]))
            stats = Counter()
            bar, = list(read_candles(source, "15m", stats))
        self.assertIsNone(bar.body_ratio)
        self.assertEqual(stats["zero_range_rows"], 1)
        self.assertEqual(events([bar]), [])

    def test_bad_ohlc_times_and_missing_columns_fail(self):
        mutations = (("close", "nan"), ("open", "0"), ("high", "1"),
                     ("timestamp_utc", "2020-01-01T00:00:00"),
                     ("timestamp_utc", "2020-01-01T00:01:00Z"),
                     ("available_at_utc", "2020-01-01T00:00:00Z"))
        for field, value in mutations:
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                rows = source_rows([candle(0)])
                rows[0][field] = value
                source = Path(directory) / "input.csv"
                write_input(source, rows)
                with self.assertRaises(ValueError):
                    list(read_candles(source, "15m", Counter()))
        rows = source_rows(bullish())
        for selection in ([rows[0], rows[2]], [rows[0], rows[0]], [rows[1], rows[0]]):
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "input.csv"
                write_input(source, selection)
                with self.assertRaises(ValueError):
                    list(read_candles(source, "15m", Counter()))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            with source.open("x", newline="") as handle:
                csv.writer(handle).writerow(["timestamp_utc"])
            with self.assertRaises(ValueError):
                list(read_candles(source, "15m", Counter()))

    def test_real_input_smoke_all_six(self):
        for timeframe in INTERVALS:
            for symbol in ("BTCUSDT", "ETHUSDT"):
                with self.subTest(symbol=symbol, timeframe=timeframe):
                    source, _ = paths_for(PROJECT_DIR, symbol, timeframe)
                    if not source.exists():
                        self.skipTest(f"Real input unavailable: {source}")
                    bars = list(islice(read_candles(source, timeframe, Counter()), 128))
                    found = list(detect_candidates(bars, symbol, timeframe, Config()))
                    self.assertEqual(len(bars), 128)
                    self.assertTrue(found)
                    creations = set()
                    previous = bars[0].timestamp
                    for row in found:
                        available = utc(row["available_at_utc"])
                        self.assertGreaterEqual(available, previous)
                        previous = available
                        if row["event_type"] == "create":
                            self.assertNotIn(row["zone_id"], creations)
                            creations.add(row["zone_id"])
                        else:
                            self.assertIn(row["zone_id"], creations)
                        self.assertGreater(row["upper_price"], row["lower_price"])


if __name__ == "__main__":
    unittest.main()
