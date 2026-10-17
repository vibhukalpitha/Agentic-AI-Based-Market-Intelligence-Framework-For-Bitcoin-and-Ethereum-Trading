from __future__ import annotations

import csv
import math
import random
import sys
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from itertools import islice
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from zone_detection.volume import (
    Candle, Config, REQUIRED_COLUMNS, allocate_profile, iter_observations,
    paths_for, preflight, process_dataset, read_candles, select_regions,
)
from zone_detection.swings import INTERVALS, utc

START = datetime(2020, 1, 1, tzinfo=timezone.utc)


def candle(index=0, low=100, high=104, volume=40, atr=2, timeframe="15m"):
    timestamp = START + index * INTERVALS[timeframe]
    midpoint = low + (high - low) / 2
    return Candle(timestamp, timestamp + INTERVALS[timeframe], high, low, midpoint, volume, atr, midpoint)


def series(length, timeframe="15m"):
    return [candle(i, volume=1, timeframe=timeframe) if i % 4 == 0
            else candle(i, low=101, high=101, volume=10, timeframe=timeframe) for i in range(length)]


def observations(bars, config=None, symbol="BTCUSDT", timeframe="15m"):
    stats = Counter()
    rows = list(iter_observations(bars, symbol, timeframe, config or Config(4, 2, 4), stats))
    return rows, stats


def source_rows(bars):
    return [{"timestamp_utc": b.timestamp.isoformat(), "available_at_utc": b.available_at.isoformat(),
             "high": b.high, "low": b.low, "close": b.close, "volume": b.volume,
             "atr_14_sma": b.atr, "open": b.open} for b in bars]


def write_input(path, rows, extra_columns=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS + ("open",) + extra_columns)
        writer.writeheader()
        writer.writerows(rows)


class AllocationTests(unittest.TestCase):
    def test_uniform_allocation(self):
        profile = allocate_profile([candle()], 4)
        self.assertEqual(profile.edges, (100, 101, 102, 103, 104))
        self.assertEqual(profile.bins, (10, 10, 10, 10))
        self.assertEqual(profile.total_volume, 40)
        self.assertTrue(profile.conservation_ok)

    def test_partial_overlaps(self):
        profile = allocate_profile([candle(volume=0), candle(1, low=100.5, high=102.5, volume=20)], 4)
        self.assertEqual(profile.bins, (5, 10, 5, 0))
        self.assertEqual(math.fsum(profile.bins), 20)

    def test_positive_ranges_at_exact_edges_do_not_double_count(self):
        profile = allocate_profile([candle(low=100, high=102, volume=20),
                                    candle(1, low=102, high=104, volume=40)], 4)
        self.assertEqual(profile.bins, (10, 10, 20, 20))
        self.assertTrue(profile.conservation_ok)

    def test_point_candles_at_minimum_internal_edges_and_maximum(self):
        bars = [candle(volume=0)]
        bars += [candle(i + 1, low=price, high=price, volume=volume)
                 for i, (price, volume) in enumerate(((100, 1), (101, 2), (102, 3), (104, 4)))]
        profile = allocate_profile(bars, 4)
        self.assertEqual(profile.bins, (1, 2, 3, 4))
        self.assertEqual(profile.absolute_error, 0)

    def test_conservation_for_irregular_ranges_and_volume_scales(self):
        rng = random.Random(5)
        for volume_scale in (1e-6, 1, 1e8):
            with self.subTest(volume_scale=volume_scale):
                bars = []
                for i in range(100):
                    low = 10000 + rng.random() * 3
                    high = low + rng.random() * 2
                    bars.append(candle(i, low, high, rng.random() * volume_scale))
                profile = allocate_profile(bars, 40)
                self.assertTrue(profile.conservation_ok)
                self.assertTrue(math.isclose(math.fsum(profile.bins), math.fsum(b.volume for b in bars),
                                             rel_tol=1e-12, abs_tol=1e-12))

    def test_flat_price_and_zero_volume_profiles_are_explicit(self):
        flat = allocate_profile([candle(low=100, high=100)], 4)
        self.assertEqual(flat.status, "flat_price_span")
        self.assertIsNone(flat.conservation_ok)
        self.assertEqual(flat.edges, ())
        zero = allocate_profile([candle(volume=0)], 4)
        self.assertEqual(zero.status, "zero_total_volume")
        self.assertTrue(zero.conservation_ok)
        self.assertEqual(zero.bins, (0, 0, 0, 0))

    def test_unrepresentable_grid_is_reported_without_invalid_intervals(self):
        bars = [candle(low=1, high=math.nextafter(1, math.inf))]
        profile = allocate_profile(bars, 40)
        self.assertEqual(profile.status, "unrepresentable_price_grid")
        found, stats = observations(bars, Config(1, 1, 40))
        self.assertEqual(found, [])
        self.assertEqual(stats["unrepresentable_price_grid_profiles"], 1)
        self.assertEqual(stats["conservation_skipped_profiles"], 1)

    def test_invalid_prices_and_volume_fail(self):
        valid = candle()
        for change in ({"volume": -1}, {"volume": math.inf}, {"volume": math.nan},
                       {"high": math.inf}, {"low": 0}, {"high": 99},
                       {"close": 110}, {"open": 110}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(valid, **change)
        with self.assertRaises(ValueError):
            allocate_profile([], 40)


class PeakTests(unittest.TestCase):
    def test_known_peak(self):
        region, = select_regions([1, 1, 10, 1, 1], Config())
        self.assertEqual((region["start"], region["stop"], region["peak"]), (2, 3, 2))
        self.assertEqual(region["estimated_volume"], 10)
        self.assertAlmostEqual(region["relative_bin_volume"], 10 / 2.8)

    def test_equal_plateau_is_one_area_with_leftmost_peak(self):
        region, = select_regions([1, 8, 8, 1, 0], Config())
        self.assertEqual((region["start"], region["stop"], region["peak"]), (1, 3, 1))
        self.assertEqual(region["estimated_volume"], 16)

    def test_edge_peaks_and_equal_area_ties(self):
        regions = select_regions([8, 0, 8, 0, 0], Config())
        self.assertEqual([r["peak"] for r in regions], [0, 2])
        regions = select_regions([10, 1, 0, 1, 9], Config())
        self.assertEqual([r["peak"] for r in regions], [0, 4])

    def test_valleys_separate_but_above_mean_saddles_share_one_area(self):
        regions = select_regions([0, 8, 2, 9, 0], Config())
        self.assertEqual(len(regions), 2)
        self.assertLessEqual(regions[1]["stop"], regions[0]["start"])
        connected, = select_regions([0, 8, 5, 9, 0], Config())
        self.assertEqual((connected["start"], connected["stop"], connected["peak"]), (1, 4, 3))

    def test_threshold_is_inclusive(self):
        self.assertEqual(len(select_regions([3, 1], Config())), 1)
        self.assertEqual(select_regions([3, 1], Config(peak_relative_volume_min=1.5001)), [])

    def test_ranking_area_then_peak_ratio_and_maximum_count(self):
        regions = select_regions([7, 7, 0, 10, 0, 8, 0, 9, 0, 0, 0, 0], Config())
        self.assertEqual([r["estimated_volume"] for r in regions], [14, 10, 9])
        self.assertEqual(len(regions), 3)
        tied = select_regions([6, 6, 0, 12, 0, 0, 0, 0], Config())
        self.assertEqual([r["start"] for r in tied], [3, 0])

    def test_flat_and_zero_histograms_produce_no_candidates(self):
        for bins in ([1, 1, 1, 1], [0, 0, 0, 0], [1, 1 + 1e-14, 1], []):
            self.assertEqual(select_regions(bins, Config()), [])
        for invalid in ([1, -1], [1, math.nan], [1, math.inf]):
            with self.assertRaises(ValueError):
                select_regions(invalid, Config())


class TimingTests(unittest.TestCase):
    def test_exact_rolling_windows_stride_and_no_eof_flush(self):
        bars = series(9)
        found, stats = observations(bars)
        self.assertEqual(stats["profiles_calculated"], 3)
        self.assertEqual([r["profile_start_utc"] for r in found], [bars[i].timestamp.isoformat() for i in (0, 2, 4)])
        self.assertEqual([r["profile_end_utc"] for r in found], [bars[i].timestamp.isoformat() for i in (3, 5, 7)])
        self.assertEqual([r["available_at_utc"] for r in found], [bars[i].available_at.isoformat() for i in (3, 5, 7)])
        self.assertEqual(observations(bars[:3])[0], [])

    def test_default_warmup_and_all_timeframe_completion_times(self):
        for timeframe, interval in INTERVALS.items():
            with self.subTest(timeframe=timeframe):
                bars = series(120, timeframe)
                found, stats = observations(bars, Config(), timeframe=timeframe)
                self.assertEqual(stats["profiles_calculated"], 2)
                self.assertEqual({utc(r["available_at_utc"]) for r in found}, {START + 100 * interval, START + 120 * interval})

    def test_old_extreme_drops_out_of_the_next_window(self):
        bars = [candle(0, 90, 110, 1), candle(1, volume=1),
                candle(2, 101, 101, 10), candle(3, 101, 101, 10)]
        found, _ = observations(bars, Config(3, 1, 4))
        self.assertEqual([r["bin_width"] for r in found], [5, 1])
        self.assertEqual(found[1]["profile_start_utc"], bars[1].timestamp.isoformat())

    def test_latest_available_atr_is_metadata_only(self):
        bars = series(4)
        bars[0] = replace(bars[0], atr=999)
        for atr in (None, math.nan, -1, 0, 7):
            with self.subTest(atr=atr):
                bars[-1] = replace(bars[-1], atr=atr)
                found, _ = observations(bars)
                self.assertEqual(len(found), 1)
                self.assertEqual(found[0]["reference_atr"], 7 if atr == 7 else None)

    def test_positive_bounds_disjoint_regions_base_units_and_ids(self):
        bars = series(8)
        first, _ = observations(bars)
        second, _ = observations(bars)
        self.assertEqual(first, second)
        self.assertEqual(len({r["zone_id"] for r in first}), len(first))
        for row in first:
            self.assertGreater(row["lower_price"], 0)
            self.assertGreater(row["upper_price"], row["lower_price"])
            self.assertLessEqual(row["volume_share"], 1)
            self.assertEqual(row["volume_unit"], "BTC")
            self.assertEqual(row["zone_family"], "volume_derived")
            self.assertEqual(row["method"], "uniform_ohlcv_range_allocation")
        eth, _ = observations(bars, symbol="ETHUSDT")
        self.assertEqual(eth[0]["volume_unit"], "ETH")
        self.assertNotEqual(eth[0]["zone_id"], first[0]["zone_id"])

    def test_prefix_invariance_and_changed_future(self):
        bars = series(16)
        full, _ = observations(bars)
        for length in range(1, len(bars) + 1):
            with self.subTest(length=length):
                prefix, _ = observations(bars[:length])
                self.assertEqual(prefix, [r for r in full if utc(r["available_at_utc"]) <= bars[length - 1].available_at])
        future = bars[:8] + [candle(i, 500, 510, 1e6) for i in range(8, 16)]
        altered, _ = observations(future)
        self.assertEqual(full[:3], altered[:3])

    def test_empty_profile_counts_and_conservation_skip(self):
        for bars, flat, zero, skipped in (([candle(i, 100, 100, 5) for i in range(4)], 1, 0, 1),
                                          ([candle(i, volume=0) for i in range(4)], 0, 1, 0)):
            found, stats = observations(bars)
            self.assertEqual(found, [])
            self.assertEqual(stats["no_candidate_profiles"], 1)
            self.assertEqual(stats["flat_price_span_profiles"], flat)
            self.assertEqual(stats["zero_total_volume_profiles"], zero)
            self.assertEqual(stats["conservation_skipped_profiles"], skipped)

    def test_conservation_failure_stops_observations(self):
        bars = series(4)
        broken = replace(allocate_profile(bars, 4), conservation_ok=False, absolute_error=1, relative_error=.1)
        stats = Counter()
        with patch("zone_detection.volume.allocate_profile", return_value=broken):
            with self.assertRaisesRegex(ValueError, "conservation failed"):
                list(iter_observations(bars, "BTCUSDT", "15m", Config(4, 2, 4), stats))
        self.assertEqual(stats["conservation_error_profiles"], 1)


class FileTests(unittest.TestCase):
    def test_read_only_inputs_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = paths_for(root, "BTCUSDT", "15m")
            write_input(source, source_rows(series(8)))
            original = source.read_bytes()
            stats = process_dataset("BTCUSDT", "15m", source, output, Config(4, 2, 4))
            self.assertEqual(stats["profiles_calculated"], 3)
            self.assertEqual(stats["candidate_observations"], 3)
            self.assertEqual(stats["conservation"]["conservation_error_profiles"], 0)
            saved = output.read_bytes()
            with self.assertRaises(FileExistsError):
                process_dataset("BTCUSDT", "15m", source, output, Config(4, 2, 4))
            self.assertEqual(output.read_bytes(), saved)
            process_dataset("BTCUSDT", "15m", source, output, Config(4, 2, 4), overwrite=True)
            self.assertEqual(output.read_bytes(), saved)
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaises(ValueError):
                process_dataset("BTCUSDT", "15m", source, source, Config(), overwrite=True)

    def test_preflight_protects_summary_and_later_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for timeframe in INTERVALS:
                for symbol in ("BTCUSDT", "ETHUSDT"):
                    source, _ = paths_for(root, symbol, timeframe)
                    write_input(source, [])
            self.assertEqual(len(preflight(root, False)), 6)
            report = root / "reports/zone_detection/volume_summary.json"
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

    def test_base_volume_ignores_quote_volume_and_optional_columns(self):
        rows = source_rows(series(4))
        for row in rows:
            row.update(quote_volume="999999999999", mark_close="", index_close="NaN")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            write_input(source, rows, ("quote_volume", "mark_close", "index_close"))
            bars = list(read_candles(source, "15m", Counter()))
        found, _ = observations(bars)
        self.assertEqual(found[0]["profile_total_volume"], 31)

    def test_missing_invalid_atr_is_reported_without_losing_volume(self):
        rows = source_rows(series(7))
        for row, raw in zip(rows, (None, "NaN", "inf", "0", "-1", "bad", 2)):
            row["atr_14_sma"] = raw
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            write_input(source, rows)
            stats = Counter()
            bars = list(read_candles(source, "15m", stats))
        self.assertEqual(stats["missing_atr_rows"], 1)
        self.assertEqual(stats["invalid_atr_rows"], 5)
        self.assertEqual([b.atr for b in bars], [None] * 6 + [2])
        self.assertEqual(math.fsum(b.volume for b in bars), sum(row["volume"] for row in rows))

    def test_invalid_input_values_and_timestamps_fail_with_line(self):
        for field, value in (("volume", ""), ("volume", "-1"), ("volume", "NaN"),
                             ("volume", "inf"), ("high", "99"), ("low", "0"),
                             ("close", "999"), ("open", "999"),
                             ("timestamp_utc", "2020-01-01T00:00:00"),
                             ("timestamp_utc", "2020-01-01T00:01:00Z"),
                             ("available_at_utc", "2020-01-01T00:00:00Z")):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                rows = source_rows([candle()])
                rows[0][field] = value
                source = Path(directory) / "input.csv"
                write_input(source, rows)
                with self.assertRaisesRegex(ValueError, "CSV line 2"):
                    list(read_candles(source, "15m", Counter()))
        rows = source_rows(series(3))
        for selected in ([rows[0], rows[2]], [rows[0], rows[0]], [rows[1], rows[0]]):
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "input.csv"
                write_input(source, selected)
                with self.assertRaises(ValueError):
                    list(read_candles(source, "15m", Counter()))

    def test_headers_short_history_and_invalid_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.csv"
            with source.open("x", newline="") as handle:
                csv.writer(handle).writerow(["timestamp_utc"])
            with self.assertRaises(ValueError):
                list(read_candles(source, "15m", Counter()))
            source, output = paths_for(Path(directory), "BTCUSDT", "15m")
            write_input(source, source_rows(series(3)))
            stats = process_dataset("BTCUSDT", "15m", source, output, Config())
            self.assertEqual(stats["profiles_calculated"], 0)
            with output.open(newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])
        for change in ({"profile_window_bars": 0}, {"profile_stride_bars": 0},
                       {"profile_bins": 2.5}, {"max_candidates_per_profile": 0},
                       {"peak_relative_volume_min": math.nan}, {"peak_relative_volume_min": .9}):
            with self.assertRaises(ValueError):
                Config(**change)

    def test_real_input_smoke_all_six(self):
        for timeframe in INTERVALS:
            for symbol in ("BTCUSDT", "ETHUSDT"):
                with self.subTest(symbol=symbol, timeframe=timeframe):
                    source, _ = paths_for(PROJECT_DIR, symbol, timeframe)
                    if not source.exists():
                        self.skipTest(f"Real input unavailable: {source}")
                    input_stats = Counter()
                    bars = list(islice(read_candles(source, timeframe, input_stats), 128))
                    found, stats = observations(bars, Config(), symbol, timeframe)
                    self.assertEqual(input_stats["missing_atr_rows"], 13)
                    self.assertEqual(stats["profiles_calculated"], 2)
                    self.assertTrue(found)
                    self.assertEqual(stats["conservation_error_profiles"], 0)
                    prefix, _ = observations(bars[:100], Config(), symbol, timeframe)
                    self.assertEqual(prefix, [r for r in found if utc(r["available_at_utc"]) <= bars[99].available_at])
