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

from zone_detection.swings import (
    Candle, INTERVALS, REQUIRED_COLUMNS, Swing, detect_swings, read_candles, utc,
)
from zone_detection.structural import (
    Config, StructuralDetector, boundaries, output_paths, preflight, process_dataset,
)

START = datetime(2020, 1, 1, tzinfo=timezone.utc)


def candles(highs, lows=None, timeframe="15m", atr=2):
    lows = lows if lows is not None else [90] * len(highs)
    interval = INTERVALS[timeframe]
    return [Candle(START + i * interval, START + (i + 1) * interval,
                   (high + low) / 2, high, low, (high + low) / 2, atr)
            for i, (high, low) in enumerate(zip(highs, lows))]


def swing(index, price=100, atr=2, kind="high", day=None, swing_id=None):
    confirmed = START + timedelta(days=index if day is None else day)
    return Swing(swing_id or f"s{index}", "BTCUSDT", "15m", kind, price,
                 confirmed - timedelta(hours=1), confirmed, atr)


def records(bars):
    return [{"timestamp_utc": c.timestamp.isoformat(), "available_at_utc": c.available_at.isoformat(),
             "open": str(c.open), "high": str(c.high), "low": str(c.low),
             "close": str(c.close), "atr_14_sma": "" if c.atr is None else str(c.atr)}
            for c in bars]


def write_input(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class SwingTests(unittest.TestCase):
    def test_strict_high_and_low(self):
        bars = candles([101, 102, 103, 110, 103, 102, 101],
                       [99, 98, 97, 90, 97, 98, 99])
        events = list(detect_swings(bars, "BTCUSDT", "15m"))
        self.assertEqual([e.swing_type for e in events], ["high", "low"])
        self.assertEqual([e.swing_price for e in events], [110, 90])
        self.assertEqual(events[0].swing_time, START + timedelta(minutes=45))

    def test_equal_extrema_do_not_qualify(self):
        bars = candles([101, 110, 103, 110, 103, 102, 101],
                       [99, 98, 97, 90, 90, 98, 99])
        self.assertEqual(list(detect_swings(bars, "BTCUSDT", "15m")), [])

    def test_right_bar_delay_and_final_pivots(self):
        bars = candles([101, 102, 103, 110, 103, 102, 101])
        self.assertEqual(list(detect_swings(bars[:6], "BTCUSDT", "15m")), [])
        event, = list(detect_swings(bars, "BTCUSDT", "15m"))
        self.assertEqual(event.confirmed_at, START + timedelta(minutes=105))
        self.assertGreater(event.confirmed_at, event.swing_time + timedelta(minutes=45))
        self.assertEqual(list(detect_swings(candles([101, 102, 103, 104, 105, 106, 110]),
                                                "BTCUSDT", "15m")), [])

    def test_confirmation_utc_and_atr_all_timeframes(self):
        for timeframe, interval in INTERVALS.items():
            with self.subTest(timeframe=timeframe):
                bars = candles([101, 102, 103, 110, 103, 102, 101], timeframe=timeframe)
                bars[3] = replace(bars[3], atr=999)
                bars[6] = replace(bars[6], atr=7)
                event, = list(detect_swings(bars, "ETHUSDT", timeframe))
                self.assertEqual(event.confirmed_at, START + 7 * interval)
                self.assertEqual(event.confirmed_at.utcoffset(), timedelta(0))
                self.assertEqual(event.atr_at_confirmation, 7)

    def test_configurable_windows(self):
        bars = candles([101, 110, 101])
        self.assertEqual(len(list(detect_swings(bars, "BTCUSDT", "15m", 1, 1))), 1)
        with self.assertRaises(ValueError):
            list(detect_swings(bars, "BTCUSDT", "15m", 0, 1))


class GroupTests(unittest.TestCase):
    def test_first_provisional_second_creates(self):
        detector = StructuralDetector("BTCUSDT", "15m")
        self.assertIsNone(detector.consume(swing(0)))
        self.assertEqual(detector.snapshots(START), [])
        event = detector.consume(swing(1, 100.5))
        self.assertEqual(event["event_type"], "create")
        self.assertEqual(event["confirmed_swing_count"], 2)
        self.assertEqual(event["confirmed_at_utc"], (START + timedelta(days=1)).isoformat())
        self.assertEqual(event["available_at_utc"], event["confirmed_at_utc"])

    def test_distant_prices_and_opposite_origins_remain_separate(self):
        detector = StructuralDetector("BTCUSDT", "15m")
        detector.consume(swing(0, 100))
        detector.consume(swing(1, 105))
        detector.consume(swing(2, 100, kind="low"))
        self.assertEqual(len(detector.groups), 3)
        self.assertEqual(detector.stats["zone_create_count"], 0)

    def test_chain_cannot_expand_group_diameter(self):
        detector = StructuralDetector("BTCUSDT", "15m")
        detector.consume(swing(0, 100))
        detector.consume(swing(1, 100.8))
        self.assertIsNone(detector.consume(swing(2, 101.6)))
        self.assertEqual([g.count for g in detector.groups.values()], [2, 1])

    def test_nearest_then_earliest_matching(self):
        detector = StructuralDetector("BTCUSDT", "15m")
        detector.consume(swing(0, 100, atr=3))
        detector.consume(swing(1, 102, atr=3))
        event = detector.consume(swing(2, 101, atr=3))
        self.assertEqual(event["cluster_id"], "group_s0")
        self.assertEqual(sum(g.count for g in detector.groups.values()), 3)
        detector = StructuralDetector("BTCUSDT", "15m")
        detector.consume(swing(0, 100, atr=3))
        detector.consume(swing(1, 102, atr=3))
        self.assertEqual(detector.consume(swing(2, 101.2, atr=3))["cluster_id"], "group_s1")

    def test_equal_time_matching_uses_group_id(self):
        detector = StructuralDetector("BTCUSDT", "15m")
        detector.consume(swing(0, 100, atr=3, swing_id="b"))
        detector.consume(swing(1, 102, atr=3, day=0, swing_id="a"))
        self.assertEqual(detector.consume(swing(2, 101, atr=3))["cluster_id"], "group_a")

    def test_lookback_exact_elapsed_boundary(self):
        detector = StructuralDetector("BTCUSDT", "15m")
        detector.consume(swing(0))
        self.assertIsNotNone(detector.consume(swing(1, day=30)))
        # The clock resets with evidence; only three sparse events span sixty days.
        self.assertIsNotNone(detector.consume(swing(2, day=60)))
        self.assertIsNone(detector.consume(swing(3, day=90 + 1 / 86400)))
        self.assertEqual(len(detector.groups), 2)

    def test_six_month_gap_preserves_old_zone(self):
        detector = StructuralDetector("BTCUSDT", "15m")
        detector.consume(swing(0))
        created = detector.consume(swing(1))
        self.assertIsNone(detector.consume(swing(2, day=183)))
        self.assertEqual(len(detector.groups), 2)
        snapshot, = detector.snapshots(START + timedelta(days=183))
        self.assertEqual(snapshot["zone_id"], created["zone_id"])
        self.assertEqual(snapshot["confirmed_swing_count"], 2)
        self.assertEqual(snapshot["candidate_status"], "candidate")
        self.assertEqual(detector.stats["groups_archived_from_matching"], 1)

    def test_timeframe_specific_lookback(self):
        for timeframe, days in (("15m", 30), ("1h", 90), ("4h", 180)):
            with self.subTest(timeframe=timeframe):
                detector = StructuralDetector("BTCUSDT", timeframe)
                detector.consume(replace(swing(0), source_timeframe=timeframe))
                self.assertIsNone(detector.consume(replace(swing(1, day=days + 1), source_timeframe=timeframe)))
                self.assertEqual(len(detector.groups), 2)

    def test_high_atr_cannot_widen_old_allowance_or_padding(self):
        detector = StructuralDetector("BTCUSDT", "15m")
        detector.consume(swing(0, 100, atr=2))
        self.assertIsNone(detector.consume(swing(1, 105, atr=200)))
        event = detector.consume(swing(2, 100.5, atr=200))
        self.assertEqual(event["cluster_reference_atr"], 2)
        self.assertEqual(event["diameter_limit"], 1)
        self.assertAlmostEqual(event["padding_price"], 0.2)
        self.assertLess(event["upper_price"] - event["lower_price"], 1)

    def test_low_atr_must_keep_whole_group_compact(self):
        detector = StructuralDetector("BTCUSDT", "15m")
        detector.consume(swing(0, 100, atr=2))
        detector.consume(swing(1, 100.8, atr=2))
        # Near the midpoint but prior spread is too wide for the new ATR.
        self.assertIsNone(detector.consume(swing(2, 100.4, atr=0.2)))
        self.assertEqual(len(detector.groups), 2)

    def test_invalid_atr_preserves_no_group(self):
        detector = StructuralDetector("BTCUSDT", "15m")
        for i, atr in enumerate((None, 0, -1, math.nan, math.inf)):
            self.assertIsNone(detector.consume(swing(i, atr=atr)))
        self.assertEqual(len(detector.groups), 0)
        self.assertEqual(detector.stats["swings_skipped_unusable_atr"], 5)

    def test_duplicate_evidence_and_reverse_time_rejected(self):
        detector = StructuralDetector("BTCUSDT", "15m")
        detector.consume(swing(1))
        with self.assertRaises(ValueError):
            detector.consume(swing(1))
        with self.assertRaises(ValueError):
            detector.consume(swing(0))
        self.assertEqual(sum(g.count for g in detector.groups.values()), 1)

    def test_zone_ids_and_historical_event_states_are_stable(self):
        detector = StructuralDetector("BTCUSDT", "15m")
        detector.consume(swing(0))
        creation = detector.consume(swing(1, 100.5))
        saved = copy.deepcopy(creation)
        update = detector.consume(swing(2, 100.8))
        self.assertEqual(creation, saved)
        self.assertEqual(update["zone_id"], creation["zone_id"])
        self.assertEqual(update["event_type"], "update")
        self.assertEqual(update["event_sequence"], 2)
        self.assertGreater(update["upper_price"], creation["upper_price"])
        self.assertEqual(update["confirmed_at_utc"], creation["confirmed_at_utc"])
        self.assertGreater(update["available_at_utc"], creation["available_at_utc"])

    def test_repeated_identical_price_is_nonzero_width(self):
        detector = StructuralDetector("BTCUSDT", "15m")
        detector.consume(swing(0))
        event = detector.consume(swing(1))
        self.assertAlmostEqual(event["lower_price"], 99.8)
        self.assertAlmostEqual(event["upper_price"], 100.2)
        self.assertGreater(event["lower_price"], 0)

    def test_price_padding_cap_and_width_rejection(self):
        lower, upper, padding = boundaries(100, 100, 10000, Config())
        self.assertEqual((lower, upper, padding), (99, 101, 1))
        detector = StructuralDetector("BTCUSDT", "15m")
        detector.consume(swing(0, 100, atr=100))
        self.assertIsNone(detector.consume(swing(1, 110, atr=100)))
        self.assertEqual(len(detector.groups), 2)
        with self.assertRaises(ValueError):
            Config(padding_atr=0)

    def test_configurable_minimum_evidence(self):
        detector = StructuralDetector("BTCUSDT", "15m", Config(min_confirmed_swings=3))
        detector.consume(swing(0))
        self.assertIsNone(detector.consume(swing(1)))
        self.assertEqual(detector.consume(swing(2))["event_type"], "create")

    def test_prefix_and_future_candles_preserve_all_earlier_events(self):
        highs = [101, 102, 103, 110, 103, 102, 101, 102, 103, 110.5, 103, 102,
                 101, 102, 103, 110.8, 103, 102, 101, 102, 103, 111, 103, 102, 101]
        bars = candles(highs)

        def run(values):
            detector = StructuralDetector("BTCUSDT", "15m")
            swings = list(detect_swings(values, "BTCUSDT", "15m"))
            events = [event for s in swings if (event := detector.consume(s)) is not None]
            return swings, events

        short_swings, short_events = run(bars[:13])
        long_swings, long_events = run(bars)
        self.assertEqual(len(short_events), 1)
        self.assertGreater(len(long_events), 1)
        self.assertEqual(short_swings, long_swings[:len(short_swings)])
        cutoff = bars[12].available_at
        self.assertEqual(short_events, [e for e in long_events if utc(e["event_at_utc"]) <= cutoff])


class InputAndIntegrationTests(unittest.TestCase):
    def test_atr_parsing_missing_invalid_and_utc_conversion(self):
        rows = records(candles([101] * 8))
        for row, value in zip(rows, ("", "NaN", "inf", "0", "-1", "bad", "2", "3")):
            row["atr_14_sma"] = value
        rows[0]["timestamp_utc"] = "2020-01-01T05:30:00+05:30"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            write_input(path, rows)
            stats = Counter()
            parsed = list(read_candles(path, "15m", stats))
        self.assertEqual(parsed[0].timestamp, START)
        self.assertEqual(stats["missing_atr_rows"], 1)
        self.assertEqual(stats["invalid_atr_rows"], 5)
        self.assertEqual(parsed[-1].atr, 3)

    def test_bad_input_timestamps_and_prices_fail(self):
        for field, value in (("timestamp_utc", "2020-01-01T00:00:00"),
                             ("timestamp_utc", "2020-01-01T00:01:00Z"),
                             ("available_at_utc", "2020-01-01T00:00:00Z"),
                             ("high", "NaN"), ("low", "0"), ("close", "999")):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                rows = records(candles([101]))
                rows[0][field] = value
                path = Path(directory) / "input.csv"
                write_input(path, rows)
                with self.assertRaises(ValueError):
                    list(read_candles(path, "15m", Counter()))

    def test_missing_duplicate_or_unordered_candles_fail(self):
        rows = records(candles([101] * 4))
        for selected in ([rows[0], rows[2]], [rows[0], rows[0]], [rows[1], rows[0]]):
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "input.csv"
                write_input(path, selected)
                with self.assertRaises(ValueError):
                    list(read_candles(path, "15m", Counter()))

    def test_missing_required_column_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            with path.open("x", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(["timestamp_utc"])
            with self.assertRaises(ValueError):
                list(read_candles(path, "15m", Counter()))

    def test_csv_history_snapshot_and_input_preservation(self):
        highs = [101, 102, 103, 110, 103, 102, 101, 102, 103, 110.5, 103, 102,
                 101, 102, 103, 110.8, 103, 102, 101]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.csv"
            write_input(source, records(candles(highs)))
            original = source.read_bytes()
            outputs = output_paths(root, "BTCUSDT", "15m")
            stats = process_dataset("BTCUSDT", "15m", source, outputs, Config())
            history = read_csv(outputs["zone_events"])
            snapshots = read_csv(outputs["snapshot"])
            self.assertEqual([e["event_type"] for e in history], ["create", "update"])
            self.assertEqual(history[0]["confirmed_swing_count"], "2")
            self.assertEqual(history[1]["confirmed_swing_count"], "3")
            self.assertEqual(history[0]["zone_id"], history[1]["zone_id"])
            self.assertEqual(snapshots[0]["snapshot_as_of_utc"], (START + timedelta(minutes=19 * 15)).isoformat())
            self.assertEqual(stats["input_rows"], 19)
            self.assertEqual(source.read_bytes(), original)
            saved = {key: path.read_bytes() for key, path in outputs.items()}
            with self.assertRaises(FileExistsError):
                process_dataset("BTCUSDT", "15m", source, outputs, Config())
            self.assertEqual(saved, {key: path.read_bytes() for key, path in outputs.items()})

    def test_preflight_protects_summary_and_all_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for timeframe in INTERVALS:
                for symbol in ("BTCUSDT", "ETHUSDT"):
                    path = root / "data" / "features" / timeframe / f"{symbol}_{timeframe}_features.csv"
                    write_input(path, records(candles([101], timeframe=timeframe)))
            jobs = preflight(root, False)
            self.assertEqual(len(jobs), 6)
            summary = root / "reports" / "zone_detection" / "structural_summary.json"
            summary.parent.mkdir(parents=True)
            with summary.open("x", encoding="utf-8") as handle:
                handle.write("existing summary")
            with self.assertRaises(FileExistsError):
                preflight(root, False)
            self.assertEqual(len(preflight(root, True)), 6)
            self.assertEqual(summary.read_text(), "existing summary")

    def test_real_rows_all_six_inputs(self):
        for timeframe in INTERVALS:
            for symbol in ("BTCUSDT", "ETHUSDT"):
                with self.subTest(symbol=symbol, timeframe=timeframe):
                    path = PROJECT_DIR / "data" / "features" / timeframe / f"{symbol}_{timeframe}_features.csv"
                    if not path.exists():
                        self.skipTest(f"Real input unavailable: {path}")
                    bars = list(islice(read_candles(path, timeframe, Counter()), 128))
                    self.assertEqual(len(bars), 128)
                    found = list(detect_swings(bars, symbol, timeframe))
                    # Independent full-window calculation for these short real prefixes.
                    expected = []
                    for i in range(3, len(bars) - 3):
                        neighbors = bars[i - 3:i] + bars[i + 1:i + 4]
                        for kind, price, valid in (
                            ("high", bars[i].high, bars[i].high > max(c.high for c in neighbors)),
                            ("low", bars[i].low, bars[i].low < min(c.low for c in neighbors)),
                        ):
                            if valid:
                                expected.append((kind, price, bars[i].timestamp, bars[i + 3].available_at, bars[i + 3].atr))
                    self.assertEqual([(s.swing_type, s.swing_price, s.swing_time, s.confirmed_at,
                                       s.atr_at_confirmation) for s in found], expected)
                    self.assertTrue(found)
                    detector = StructuralDetector(symbol, timeframe)
                    events = [event for s in found if (event := detector.consume(s)) is not None]
                    self.assertTrue(events)
                    for event in events:
                        self.assertLessEqual(utc(event["confirmed_at_utc"]), utc(event["available_at_utc"]))
                        self.assertGreater(event["upper_price"], event["lower_price"])


if __name__ == "__main__":
    unittest.main()
