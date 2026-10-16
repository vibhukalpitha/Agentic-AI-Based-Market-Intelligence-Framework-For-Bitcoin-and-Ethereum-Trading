from __future__ import annotations

import copy
import csv
import math
import sys
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from itertools import islice
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from zone_detection.reaction import (
    Candle, Config, FEATURES, REQUIRED_COLUMNS, ReactionDetector, Rejection,
    detect_rejections, paths_for, preflight, process_dataset, read_candles,
)
from zone_detection.swings import INTERVALS, utc

START = datetime(2020, 1, 1, tzinfo=timezone.utc)


def candle(index, direction="bullish", anchor=100, atr=4, timeframe="15m"):
    low, high = (anchor, anchor + 10) if direction == "bullish" else (anchor - 10, anchor)
    opening, close = (low + 6, low + 8) if direction == "bullish" else (low + 4, low + 2)
    timestamp = START + index * INTERVALS[timeframe]
    return Candle(index, timestamp, timestamp + INTERVALS[timeframe], opening, high, low, close,
                  atr, min(opening, close) - low, high - max(opening, close),
                  .6 if direction == "bullish" else .2, .2 if direction == "bullish" else .6,
                  .2, .6 if direction == "bullish" else -.6)


def rejection(index, price=100, atr=2, direction="bullish", timeframe="15m"):
    timestamp = START + index * INTERVALS[timeframe]
    return Rejection(index, timestamp, timestamp + INTERVALS[timeframe], direction,
                     price, atr, .6, .6 if direction == "bullish" else -.6, .5)


def source_rows(bars):
    return [{"timestamp_utc": b.timestamp.isoformat(), "available_at_utc": b.available_at.isoformat(),
             "open": b.open, "high": b.high, "low": b.low, "close": b.close, "atr_14_sma": b.atr,
             **{name: getattr(b, name) for name in FEATURES}} for b in bars]


def write_input(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def replay(bars):
    detector = ReactionDetector("BTCUSDT", "15m")
    history = []
    for bar in bars:
        for raw in detect_rejections(bar, detector.config):
            event = detector.consume(raw)
            if event is not None:
                history.append(event)
    return history, detector


class RejectionTests(unittest.TestCase):
    def test_bullish_and_bearish_detection_at_completion(self):
        for direction in ("bullish", "bearish"):
            with self.subTest(direction=direction):
                bar = candle(0, direction)
                found, = detect_rejections(bar, Config())
                self.assertEqual(found.direction, direction)
                self.assertEqual(found.price, 100)
                self.assertEqual(found.wick_ratio, .6)
                self.assertEqual(found.wick_atr, 1.5)
                self.assertEqual(found.available_at, START + timedelta(minutes=15))
                self.assertEqual(found.timestamp, START)

    def test_all_conditions_are_required(self):
        for changes in ({"lower_wick_ratio": .49}, {"close_location_value": .29},
                        {"atr": 25}, {"upper_wick_ratio": .49, "close_location_value": -.6}):
            with self.subTest(changes=changes):
                self.assertEqual(detect_rejections(replace(candle(0), **changes), Config()), [])

    def test_exact_thresholds_both_directions(self):
        for direction in ("bullish", "bearish"):
            bullish = direction == "bullish"
            bar = Candle(0, START, START + timedelta(minutes=15), 110, 120, 100,
                         113 if bullish else 107, 40, 10 if bullish else 7,
                         7 if bullish else 10, .5 if bullish else .35,
                         .35 if bullish else .5, .15, .3 if bullish else -.3)
            event, = detect_rejections(bar, Config())
            self.assertEqual(event.direction, direction)
            self.assertEqual(event.wick_atr, .25)

    def test_missing_invalid_atr_and_directional_features(self):
        for value in (None, 0, -1, math.nan, math.inf):
            with self.subTest(value=value):
                self.assertEqual(detect_rejections(replace(candle(0), atr=value), Config()), [])
        for name in ("lower_wick", "lower_wick_ratio", "close_location_value"):
            for value in (None, math.nan, math.inf):
                with self.subTest(name=name, value=value):
                    self.assertEqual(detect_rejections(replace(candle(0), **{name: value}), Config()), [])

    def test_zero_range_and_invalid_ohlc_do_not_reject(self):
        bar = replace(candle(0), open=100, high=100, low=100, close=100,
                      lower_wick=0, upper_wick=0, lower_wick_ratio=None,
                      upper_wick_ratio=None, body_ratio=None, close_location_value=None)
        self.assertEqual(detect_rejections(bar, Config()), [])
        for name, value in (("close", math.nan), ("high", 0), ("low", 200)):
            self.assertEqual(detect_rejections(replace(candle(0), **{name: value}), Config()), [])

    def test_body_ratio_is_not_a_new_rejection_condition(self):
        self.assertEqual(len(detect_rejections(replace(candle(0), body_ratio=None), Config())), 1)


class CooldownAndClusterTests(unittest.TestCase):
    def test_local_cooldown_preserves_first_and_does_not_reset(self):
        detector = ReactionDetector("BTCUSDT", "15m")
        detector.consume(rejection(0))
        self.assertIsNone(detector.consume(rejection(1, 100.9, atr=200)))
        self.assertIsNone(detector.consume(rejection(2, 100.9)))
        created = detector.consume(rejection(3))
        self.assertEqual(created["independent_rejection_count"], 2)
        self.assertEqual(created["observed_min_price"], 100)
        self.assertEqual(created["observed_max_price"], 100)
        self.assertEqual(created["cluster_reference_atr"], 2)
        self.assertEqual(created["formation_start_utc"], START.isoformat())
        self.assertEqual(detector.stats["raw_bullish"], 4)
        self.assertEqual(detector.stats["accepted_bullish"], 2)
        self.assertEqual(detector.stats["suppressed_bullish"], 2)

    def test_different_price_and_direction_are_not_globally_suppressed(self):
        detector = ReactionDetector("BTCUSDT", "15m")
        detector.consume(rejection(0))
        detector.consume(rejection(1, 110))
        detector.consume(rejection(2, 100, direction="bearish"))
        self.assertEqual(detector.stats["total_group_count"], 3)
        self.assertEqual(detector.stats["accepted_bullish"], 2)
        self.assertEqual(detector.stats["accepted_bearish"], 1)

    def test_cooldown_cannot_be_bypassed_by_another_matching_group(self):
        detector = ReactionDetector("BTCUSDT", "15m")
        detector.consume(rejection(0, 100, atr=3))
        detector.consume(rejection(1, 102, atr=3))
        # Group at 100 is ready; nearby group at 102 is still on cooldown.
        self.assertIsNone(detector.consume(rejection(3, 101, atr=3)))
        self.assertEqual(detector.stats["total_group_count"], 2)
        self.assertEqual(detector.stats["suppressed_bullish"], 1)
        self.assertIsNotNone(detector.consume(rejection(4, 101, atr=3)))

    def test_cooldown_cannot_be_bypassed_by_seeding_a_new_group(self):
        detector = ReactionDetector("BTCUSDT", "15m")
        detector.consume(rejection(0, 100))
        detector.consume(rejection(3, 100.8))
        # Near the area but too wide to join. Cooldown still suppresses it.
        self.assertIsNone(detector.consume(rejection(4, 101.6)))
        self.assertEqual(detector.stats["total_group_count"], 1)

    def test_zone_requires_minimum_independent_evidence(self):
        detector = ReactionDetector("BTCUSDT", "15m", Config(min_independent_rejections=3))
        self.assertIsNone(detector.consume(rejection(0)))
        self.assertIsNone(detector.consume(rejection(3)))
        event = detector.consume(rejection(6))
        self.assertEqual(event["event_type"], "create")
        self.assertEqual(event["independent_rejection_count"], 3)
        self.assertEqual(event["confirmed_at_utc"], rejection(6).available_at.isoformat())

    def test_no_chain_merging(self):
        detector = ReactionDetector("BTCUSDT", "15m")
        detector.consume(rejection(0, 100))
        detector.consume(rejection(3, 100.8))
        self.assertIsNone(detector.consume(rejection(6, 101.6)))
        self.assertEqual(detector.stats["total_group_count"], 2)
        self.assertEqual(sorted(g.count for g in detector.groups["bullish"].values()), [1, 2])

    def test_matching_ties_nearest_then_earliest(self):
        for price, first in ((101, True), (101.2, False)):
            with self.subTest(price=price):
                detector = ReactionDetector("BTCUSDT", "15m")
                detector.consume(rejection(0, 100, atr=3))
                detector.consume(rejection(1, 102, atr=3))
                expected = list(detector.groups["bullish"])[0 if first else 1]
                self.assertEqual(detector.consume(rejection(4, price, atr=3))["zone_id"], expected)

    def test_high_atr_does_not_expand_old_allowance_or_padding(self):
        detector = ReactionDetector("BTCUSDT", "15m")
        detector.consume(rejection(0, atr=2))
        self.assertIsNone(detector.consume(rejection(3, 105, atr=200)))
        event = detector.consume(rejection(6, 100.5, atr=200))
        self.assertEqual(event["cluster_reference_atr"], 2)
        self.assertEqual(event["diameter_limit"], 1)
        self.assertAlmostEqual(event["padding_price"], .2)

    def test_small_new_atr_checks_entire_existing_diameter(self):
        detector = ReactionDetector("BTCUSDT", "15m")
        detector.consume(rejection(0))
        detector.consume(rejection(3, 100.8))
        self.assertIsNone(detector.consume(rejection(6, 100.4, atr=.2)))
        self.assertEqual(detector.stats["total_group_count"], 2)

    def test_lookback_uses_latest_accepted_elapsed_time_and_preserves_history(self):
        detector = ReactionDetector("BTCUSDT", "15m")
        detector.consume(rejection(0))
        created = detector.consume(rejection(3))
        saved = copy.deepcopy(created)
        # Exactly thirty days after latest accepted evidence remains eligible.
        update = detector.consume(rejection(3 + 30 * 96))
        self.assertEqual(update["zone_id"], created["zone_id"])
        self.assertIsNone(detector.consume(rejection(183 * 96)))
        self.assertEqual(detector.stats["archived_group_count"], 1)
        self.assertEqual(created, saved)
        self.assertEqual(detector.stats["total_group_count"], 2)

    def test_suppressed_events_do_not_extend_lookback(self):
        detector = ReactionDetector("BTCUSDT", "15m", Config(min_separation_bars=100 * 96))
        detector.consume(rejection(0))
        self.assertIsNone(detector.consume(rejection(29 * 96)))
        self.assertIsNone(detector.consume(rejection(31 * 96)))
        self.assertEqual(detector.stats["suppressed_bullish"], 1)
        self.assertEqual(detector.stats["accepted_bullish"], 2)
        self.assertEqual(detector.stats["archived_group_count"], 1)

    def test_stale_expiry_entries_do_not_archive_recently_updated_group(self):
        detector = ReactionDetector("BTCUSDT", "15m", Config(lookback_15m_days=1))
        detector.consume(rejection(0))
        detector.consume(rejection(3))
        detector.expire(START + timedelta(minutes=98 * 15))
        self.assertEqual(len(detector.groups["bullish"]), 1)
        detector.expire(START + timedelta(minutes=101 * 15))
        self.assertEqual(len(detector.groups["bullish"]), 0)

    def test_timeframe_specific_elapsed_lookbacks(self):
        for timeframe, days in (("15m", 30), ("1h", 90), ("4h", 180)):
            with self.subTest(timeframe=timeframe):
                bars = int(timedelta(days=days) / INTERVALS[timeframe])
                detector = ReactionDetector("BTCUSDT", timeframe)
                detector.consume(rejection(0, timeframe=timeframe))
                self.assertIsNotNone(detector.consume(rejection(bars, timeframe=timeframe)))
                self.assertIsNone(detector.consume(rejection(2 * bars + 1, timeframe=timeframe)))
                self.assertEqual(detector.stats["archived_group_count"], 1)

    def test_price_padding_and_total_width_caps(self):
        detector = ReactionDetector("BTCUSDT", "15m")
        detector.consume(rejection(0, atr=100))
        # ATR diameter alone would allow this, but total width would exceed 5%.
        self.assertIsNone(detector.consume(rejection(3, 110, atr=100)))
        event = detector.consume(rejection(6, 100, atr=100))
        self.assertEqual((event["lower_price"], event["upper_price"]), (99, 101))
        self.assertEqual(event["padding_price"], 1)
        self.assertGreater(detector.stats["boundary_rejected_matches"], 0)
        self.assertEqual(detector.stats["total_group_count"], 2)

    def test_unrepresentable_nonzero_width_is_reported_as_rejected(self):
        detector = ReactionDetector("BTCUSDT", "15m")
        self.assertIsNone(detector.consume(rejection(0, price=1e-323)))
        self.assertEqual(detector.stats["raw_bullish"], 1)
        self.assertEqual(detector.stats["accepted_bullish"], 0)
        self.assertEqual(detector.stats["boundary_rejected_events"], 1)

    def test_separation_counts_non_rejection_candles_too(self):
        bars = [candle(i) for i in range(4)]
        bars[1] = replace(bars[1], lower_wick_ratio=.1)
        bars[2] = replace(bars[2], lower_wick_ratio=.1)
        history, detector = replay(bars)
        self.assertEqual(detector.stats["raw_bullish"], 2)
        self.assertEqual(detector.stats["suppressed_bullish"], 0)
        self.assertEqual(len(history), 1)

    def test_boundary_width_stable_ids_and_immutable_events(self):
        detector = ReactionDetector("BTCUSDT", "15m")
        detector.consume(rejection(0))
        created = detector.consume(rejection(3))
        saved = copy.deepcopy(created)
        self.assertAlmostEqual(created["lower_price"], 99.8)
        self.assertAlmostEqual(created["upper_price"], 100.2)
        updated = detector.consume(rejection(6, 100.8))
        self.assertEqual(created, saved)
        self.assertEqual(updated["zone_id"], created["zone_id"])
        self.assertEqual(updated["confirmed_at_utc"], created["confirmed_at_utc"])
        self.assertGreater(updated["available_at_utc"], created["available_at_utc"])
        self.assertEqual(updated["event_sequence"], 2)
        self.assertGreater(updated["upper_price"], created["upper_price"])
        self.assertGreater(updated["lower_price"], 0)

    def test_duplicate_or_out_of_order_rejection_fails(self):
        detector = ReactionDetector("BTCUSDT", "15m")
        detector.consume(rejection(3))
        for bad in (rejection(3), rejection(2), replace(rejection(4), atr=math.nan)):
            with self.assertRaises(ValueError):
                detector.consume(bad)

    def test_prefix_invariance_no_future_aware_deduplication(self):
        bars = [candle(i, anchor=100 + i * .05) for i in range(11)]
        full, _ = replay(bars)
        self.assertEqual([r["independent_rejection_count"] for r in full], [2, 3, 4])
        for length in range(1, len(bars) + 1):
            prefix, _ = replay(bars[:length])
            self.assertEqual(prefix, [r for r in full if utc(r["available_at_utc"]) <= bars[length - 1].available_at])
        changed_future = bars[:7] + [candle(i, anchor=110) for i in range(7, 11)]
        altered, _ = replay(changed_future)
        self.assertEqual(full[:2], altered[:2])


class FileTests(unittest.TestCase):
    def test_file_outputs_counts_read_only_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = paths_for(root, "BTCUSDT", "15m")
            write_input(source, source_rows([candle(i) for i in range(10)]))
            original = source.read_bytes()
            stats = process_dataset("BTCUSDT", "15m", source, output, Config())
            self.assertEqual(stats["raw_rejections"]["bullish"], 10)
            self.assertEqual(stats["independent_accepted_rejections"]["bullish"], 4)
            self.assertEqual(stats["suppressed_near_duplicates"]["bullish"], 6)
            self.assertEqual(stats["candidate_zones_created"], 1)
            self.assertEqual(stats["zone_update_count"], 2)
            saved = output.read_bytes()
            with self.assertRaises(FileExistsError):
                process_dataset("BTCUSDT", "15m", source, output, Config())
            self.assertEqual(output.read_bytes(), saved)
            process_dataset("BTCUSDT", "15m", source, output, Config(), overwrite=True)
            self.assertEqual(output.read_bytes(), saved)
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaises(ValueError):
                process_dataset("BTCUSDT", "15m", source, source, Config(), overwrite=True)

    def test_preflight_protects_summary_and_any_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for timeframe in INTERVALS:
                for symbol in ("BTCUSDT", "ETHUSDT"):
                    source, _ = paths_for(root, symbol, timeframe)
                    write_input(source, [])
            self.assertEqual(len(preflight(root, False)), 6)
            report = root / "reports/zone_detection/reaction_summary.json"
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

    def test_missing_invalid_features_and_atr_remain_missing(self):
        rows = source_rows([candle(i) for i in range(7)])
        for row, value in zip(rows, (None, "NaN", "inf", "0", "-1", "bad", 4)):
            row["atr_14_sma"] = value
        rows[6]["lower_wick_ratio"] = "NaN"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            write_input(path, rows)
            stats = Counter()
            bars = list(read_candles(path, "15m", stats))
        self.assertEqual(stats["missing_atr"], 1)
        self.assertEqual(stats["invalid_atr"], 5)
        self.assertEqual(stats["invalid_lower_wick_ratio"], 1)
        self.assertEqual([b.atr for b in bars[:6]], [None] * 6)
        self.assertTrue(all(detect_rejections(b, Config()) == [] for b in bars))

    def test_zero_range_and_mismatched_features_are_not_imputed(self):
        zero = replace(candle(0), open=100, high=100, low=100, close=100, lower_wick=0,
                       upper_wick=0, lower_wick_ratio=None, upper_wick_ratio=None,
                       body_ratio=None, close_location_value=None)
        rows = source_rows([zero, candle(1)])
        rows[1]["lower_wick"] = 999
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            write_input(path, rows)
            stats = Counter()
            bars = list(read_candles(path, "15m", stats))
        self.assertEqual(stats["zero_range_rows"], 1)
        self.assertEqual(stats["invalid_lower_wick"], 1)
        self.assertIsNone(bars[0].lower_wick_ratio)
        self.assertIsNone(bars[1].lower_wick)
        self.assertEqual(replay(bars)[0], [])

    def test_invalid_ohlc_timestamps_and_headers_fail(self):
        for field, value in (("close", "NaN"), ("high", "1"), ("low", "0"),
                             ("timestamp_utc", "2020-01-01T00:00:00"),
                             ("timestamp_utc", "2020-01-01T00:01:00Z"),
                             ("available_at_utc", "2020-01-01T00:00:00Z")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                rows = source_rows([candle(0)])
                rows[0][field] = value
                path = Path(directory) / "input.csv"
                write_input(path, rows)
                with self.assertRaises(ValueError):
                    list(read_candles(path, "15m", Counter()))
        rows = source_rows([candle(i) for i in range(3)])
        for selection in ([rows[0], rows[2]], [rows[0], rows[0]], [rows[1], rows[0]]):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "input.csv"
                write_input(path, selection)
                with self.assertRaises(ValueError):
                    list(read_candles(path, "15m", Counter()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            with path.open("x", newline="") as handle:
                csv.writer(handle).writerow(["timestamp_utc"])
            with self.assertRaises(ValueError):
                list(read_candles(path, "15m", Counter()))

    def test_config_invalid_values_fail(self):
        for change in ({"min_separation_bars": 0}, {"min_separation_bars": 1.5},
                       {"min_independent_rejections": 1}, {"wick_ratio_min": 1.1},
                       {"cluster_tolerance_atr": math.nan}, {"padding_atr": 0}):
            with self.assertRaises(ValueError):
                Config(**change)

    def test_real_input_smoke_all_six(self):
        for timeframe in INTERVALS:
            for symbol in ("BTCUSDT", "ETHUSDT"):
                with self.subTest(symbol=symbol, timeframe=timeframe):
                    source, _ = paths_for(PROJECT_DIR, symbol, timeframe)
                    if not source.exists():
                        self.skipTest(f"Real input unavailable: {source}")
                    stats = Counter()
                    bars = list(islice(read_candles(source, timeframe, stats), 128))
                    self.assertEqual(len(bars), 128)
                    self.assertEqual(stats["missing_atr"], 13)
                    detector = ReactionDetector(symbol, timeframe)
                    raw = [r for b in bars for r in detect_rejections(b, Config())]
                    self.assertTrue(raw)
                    history = [e for r in raw if (e := detector.consume(r)) is not None]
                    self.assertTrue(history)
                    for event in history:
                        self.assertEqual(utc(event["available_at_utc"]),
                                         utc(event["trigger_candle_time_utc"]) + INTERVALS[timeframe])
                        self.assertGreater(event["upper_price"], event["lower_price"])
