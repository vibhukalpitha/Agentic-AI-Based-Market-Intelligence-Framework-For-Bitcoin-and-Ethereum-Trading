from __future__ import annotations

import csv
import json
import math
import random
import sys
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from zone_detection import lifecycle, registry
from zone_detection.swings import Candle, REQUIRED_COLUMNS

START = datetime(2020, 1, 1, tzinfo=timezone.utc)
INTERVAL = timedelta(minutes=15)


def at(index):
    return START + index * INTERVAL


def bar(index, opening=104, high=105, low=103, close=104):
    return Candle(at(index), at(index + 1), opening, high, low, close, None)


def source_row(family="structural", identity="s", when=1, lower=100, upper=102,
               kind="create", sequence=1, confirmed=1, direction="bullish"):
    row = {"zone_id": identity, "symbol": "BTCUSDT", "source_timeframe": "15m",
           "zone_family": family, "lower_price": str(lower), "upper_price": str(upper),
           "available_at_utc": at(when).isoformat(), "candidate_status": "candidate"}
    if family in ("structural", "reaction"):
        origin = (("swing_low_cluster" if direction == "bullish" else "swing_high_cluster")
                  if family == "structural" else f"{direction}_rejection_cluster")
        row.update(origin_type=origin, formation_start_utc=at(0).isoformat(),
                   confirmed_at_utc=at(confirmed).isoformat(), cluster_reference_atr="2",
                   event_type=kind, event_at_utc=at(when).isoformat(), event_sequence=str(sequence))
        row["confirmed_swing_count" if family == "structural" else "independent_rejection_count"] = str(sequence + 1)
    elif family == "imbalance":
        body = kind == "create"
        reference = when - 1 if body else when - 2
        row.update(formation_start_utc=at(reference if body else when - 3).isoformat(),
                   imbalance_type="displacement_body" if body else f"{direction}_fvg",
                   direction=direction, event_type=kind,
                   candidate_created_at_utc=at(when if body else when - 1).isoformat(),
                   c1_time_utc="" if body else at(when - 3).isoformat(),
                   c2_time_utc=at(reference).isoformat(), c3_time_utc="" if body else at(when - 1).isoformat(),
                   reference_candle_time_utc=at(reference).isoformat(), reference_atr="2",
                   reference_atr_available_at_utc=at(reference + 1).isoformat(),
                   aligned_displacement="False" if body else "True")
    else:
        row.update(profile_start_utc=at(when - 4).isoformat(), profile_end_utc=at(when - 1).isoformat(),
                   profile_id=f"profile_{when}", candidate_rank="1", reference_atr="2",
                   method="uniform_ohlcv_range_allocation")
    return row


def event(row, line=2):
    family = row["zone_family"]
    path = registry.source_paths(Path("."), "BTCUSDT", "15m")[family].as_posix()
    return registry.normalize(row, family, "BTCUSDT", "15m", path, line)


def clock():
    return registry.VolumeClock(at(4), at(100), 2 * INTERVAL, 4 * INTERVAL)


def replay(bars, rows=None, config=None):
    stats = Counter()
    rows = [source_row()] if rows is None else rows
    result = list(lifecycle.iter_interactions(bars, [event(r) for r in rows],
                                            "BTCUSDT", "15m", clock(), config, stats))
    return result, stats


def select(rows, kind):
    return [r for r in rows if r["event_type"] == kind]


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def project(root, rows=None, bars=None):
    rows = [source_row()] if rows is None else rows
    bars = [bar(0), bar(1, low=101), bar(2)] if bars is None else bars
    events = []
    for family, path in registry.source_paths(root, "BTCUSDT", "15m").items():
        grouped = [r for r in rows if r["zone_family"] == family]
        write_csv(path, registry.SOURCE_COLUMNS[family], grouped)
        events.extend(event(row, i) for i, row in enumerate(grouped, 2))
    unified = root / "data/zones/unified/BTCUSDT_15m.csv"
    write_csv(unified, registry.COLUMNS, [e.record(0, set()) for e in sorted(events, key=registry.Event.order_key)])
    feature = root / "data/features/15m/BTCUSDT_15m_features.csv"
    write_csv(feature, REQUIRED_COLUMNS,
              [{"timestamp_utc": b.timestamp.isoformat(), "available_at_utc": b.available_at.isoformat(),
                "open": b.open, "high": b.high, "low": b.low, "close": b.close, "atr_14_sma": ""} for b in bars])
    report = root / "reports/zone_detection/volume_summary.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"parameters": {"profile_window_bars": 4, "profile_stride_bars": 2},
                                 "datasets": [{"symbol": "BTCUSDT", "source_timeframe": "15m",
                                               "input_candle_count": 100,
                                               "timestamp_checks": {"first_open_utc": at(0).isoformat(),
                                                                    "last_available_utc": at(100).isoformat()}}]}))
    return unified, feature


def rewrite_registry(path, change):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns, rows = reader.fieldnames, list(reader)
    change(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class ContactTests(unittest.TestCase):
    def test_creation_candle_is_not_a_touch_and_open_controls_eligibility(self):
        rows, _ = replay([bar(0, low=100), bar(1, low=101)])
        touch, = select(rows, "first_touch")
        self.assertEqual(touch["event_at_utc"], at(2).isoformat())
        self.assertEqual(touch["candle_time_utc"], at(1).isoformat())
        self.assertEqual(replay([bar(0, low=100)])[0], [])

    def test_boundary_equality_is_touch(self):
        for touching in (bar(1, low=102), bar(1, 98, 100, 97, 98)):
            with self.subTest(touching=touching):
                bars = [bar(0, 98, 99, 97, 98), touching]
                rows, _ = replay(bars)
                touch, = select(rows, "first_touch")
                self.assertEqual(touch["penetration_fraction"], 0)

    def test_continuous_contact_and_no_unchanged_bar_rows(self):
        rows, _ = replay([bar(0)] + [bar(i, low=101) for i in range(1, 5)])
        self.assertEqual(len(select(rows, "first_touch")), 1)
        self.assertFalse(select(rows, "retest"))
        self.assertFalse(select(rows, "penetration_increase"))
        self.assertEqual(len(rows), 1)

    def test_disjoint_exit_then_retest(self):
        rows, _ = replay([bar(0), bar(1, low=101), bar(2), bar(3, low=101)])
        exit_row, = select(rows, "contact_exit")
        self.assertEqual(exit_row["event_at_utc"], at(3).isoformat())
        self.assertEqual(exit_row["contact_bars"], 1)
        self.assertEqual(select(rows, "retest")[0]["episode_number"], 2)

    def test_one_candle_contact_is_not_immediately_exited(self):
        rows, stats = replay([bar(0), bar(1, low=101)])
        self.assertEqual(len(rows), 1)
        self.assertEqual(stats["open_episodes_at_end"], 1)

    def test_ambiguous_open_does_not_acquire_a_direction_later(self):
        rows, _ = replay([bar(0), bar(1, 101, 103, 100.5, 103), bar(2, low=100)])
        touch, = select(rows, "first_touch")
        self.assertIsNone(touch["approach_side"])
        self.assertIsNone(touch["penetration_fraction"])
        self.assertFalse(select(rows, "penetration_increase"))
        self.assertTrue(select(rows, "range_span"))

    def test_partial_and_clipped_full_penetration(self):
        rows, _ = replay([bar(0), bar(1, low=101.5), bar(2, low=101), bar(3, low=99)])
        self.assertEqual(select(rows, "first_touch")[0]["penetration_fraction"], .25)
        self.assertEqual([r["penetration_fraction"] for r in select(rows, "penetration_increase")], [.5, 1])
        self.assertEqual(len(select(rows, "range_span")), 1)
        self.assertFalse(select(rows, "observed_invalidation"))

    def test_penetration_from_below(self):
        rows, _ = replay([bar(0, 98, 99, 97, 98), bar(1, 98, 101, 97, 98)])
        touch, = select(rows, "first_touch")
        self.assertEqual(touch["approach_side"], "below")
        self.assertEqual(touch["penetration_fraction"], .5)

    def test_penetration_epsilon_suppresses_roundoff_only(self):
        rows, _ = replay([bar(0), bar(1, low=101), bar(2, low=101 - 1e-10), bar(3, low=100.9)])
        self.assertEqual(len(select(rows, "penetration_increase")), 1)

    def test_range_span_does_not_prove_direction_or_order(self):
        rows, _ = replay([bar(0), bar(1, 101, 104, 99, 104)])
        span, = select(rows, "range_span")
        self.assertIsNone(span["penetration_fraction"])
        self.assertIn("order_unknown", span["event_detail"])

    def test_strict_close_crossing_invalidation_and_no_reactivation(self):
        rows, _ = replay([bar(0), bar(1, 104, 105, 99, 99.5), bar(2, low=101), bar(3, low=101)])
        failure, = select(rows, "observed_invalidation")
        self.assertEqual(failure["event_at_utc"], at(2).isoformat())
        self.assertEqual(len(select(rows, "boundary_crossing")), 2)
        self.assertTrue(all(r["event_at_utc"] == at(2).isoformat() for r in rows))

    def test_wick_and_equal_close_do_not_invalidate(self):
        for close in (100, 104):
            with self.subTest(close=close):
                rows, _ = replay([bar(0), bar(1, 104, 105, 99, close)])
                self.assertFalse(select(rows, "observed_invalidation"))

    def test_gap_endpoints_do_not_fabricate_touch(self):
        rows, _ = replay([bar(0), bar(1, 98, 99, 97, 98)])
        self.assertFalse(select(rows, "first_touch"))
        self.assertFalse(select(rows, "range_span"))
        self.assertTrue(all("gap_endpoints" in r["event_detail"] for r in select(rows, "boundary_crossing")))
        self.assertEqual(len(select(rows, "observed_invalidation")), 1)

    def test_initial_close_already_beyond_is_not_a_crossing(self):
        rows, _ = replay([bar(i, 98, 99, 97, 98) for i in range(4)])
        self.assertEqual(rows, [])

    def test_structural_reaction_and_bearish_origins(self):
        for family in ("structural", "reaction"):
            for direction in ("bullish", "bearish"):
                with self.subTest(family=family, direction=direction):
                    initial = bar(0) if direction == "bullish" else bar(0, 98, 99, 97, 98)
                    crossing = bar(1, 104, 105, 99, 99) if direction == "bullish" else bar(1, 98, 104, 97, 104)
                    rows, _ = replay([initial, crossing], [source_row(family=family, direction=direction)])
                    self.assertEqual(select(rows, "observed_invalidation")[0]["direction"], direction)


class VersionAndTimeTests(unittest.TestCase):
    def test_updated_boundaries_are_not_retroactive(self):
        sources = [source_row(), source_row(when=3, kind="update", sequence=2, lower=110, upper=112)]
        rows, stats = replay([bar(0), bar(1, low=101), bar(2, low=101), bar(3, 109, 111, 108, 109)], sources)
        touches = select(rows, "first_touch")
        self.assertEqual([r["lower_price"] for r in touches], [100, 110])
        self.assertEqual([r["event_at_utc"] for r in touches], [at(2).isoformat(), at(4).isoformat()])
        end, = select(rows, "version_end")
        self.assertEqual(end["event_detail"], "update")
        self.assertIsNone(end["candle_time_utc"])
        self.assertFalse(select(rows, "contact_exit"))
        self.assertEqual(stats["source_identities"], 1)

    def test_fvg_confirmation_changes_version_not_identity(self):
        sources = [source_row("imbalance", "i", when=3),
                   source_row("imbalance", "i", when=4, kind="fvg_confirmation", lower=110, upper=112)]
        bars = [bar(0), bar(1), bar(2), bar(3, low=101), bar(4, 109, 111, 108, 109)]
        rows, stats = replay(bars, sources)
        touches = select(rows, "first_touch")
        self.assertEqual(len(touches), 2)
        self.assertEqual(len({r["registry_zone_id"] for r in rows}), 1)
        self.assertEqual([r["zone_version_at_utc"] for r in touches], [at(3).isoformat(), at(4).isoformat()])
        self.assertEqual(select(rows, "version_end")[0]["event_detail"], "fvg_confirmation")
        self.assertEqual(stats["source_identities"], 1)

    def test_new_assessment_does_not_clear_old_invalidation(self):
        sources = [source_row(), source_row(when=3, kind="update", sequence=2)]
        rows, stats = replay([bar(0), bar(1, 104, 105, 99, 99), bar(2), bar(3, low=101)], sources)
        failed, = select(rows, "observed_invalidation")
        self.assertEqual(failed["zone_version_at_utc"], at(1).isoformat())
        self.assertEqual(select(rows, "first_touch")[-1]["zone_version_at_utc"], at(3).isoformat())
        self.assertEqual(stats["source_versions"], 2)

    def test_same_timestamp_batch_order_is_deterministic(self):
        sources = [source_row(identity="a"), source_row("reaction", "b")]
        bars = [bar(0), bar(1, low=101), bar(2)]
        self.assertEqual(replay(bars, sources), replay(bars, list(reversed(sources))))

    def test_monitoring_cap_is_timer_censoring_not_invalidation(self):
        config = lifecycle.Config(monitor_15m_days=2 / 96)
        rows, _ = replay([bar(0), bar(1, low=101), bar(2, low=101), bar(3, low=101)], config=config)
        expiry, = select(rows, "monitoring_expiry")
        self.assertEqual(expiry["event_at_utc"], at(3).isoformat())
        self.assertIsNone(expiry["candle_time_utc"])
        self.assertEqual(expiry["contact_bars"], 2)
        self.assertFalse(select(rows, "contact_exit"))
        self.assertFalse(select(rows, "observed_invalidation"))

    def test_stale_expiry_entry_does_not_remove_new_version(self):
        sources = [source_row(), source_row(when=2, kind="update", sequence=2)]
        rows, _ = replay([bar(i, low=101) for i in range(5)], sources,
                         lifecycle.Config(monitor_15m_days=2 / 96))
        expiry, = select(rows, "monitoring_expiry")
        self.assertEqual(expiry["event_at_utc"], at(4).isoformat())
        self.assertEqual(expiry["zone_version_at_utc"], at(2).isoformat())

    def test_expired_history_is_not_price_scanned(self):
        sources = [source_row(identity=f"s{i}") for i in range(100)]
        rows, stats = replay([bar(i) for i in range(50)], sources,
                             lifecycle.Config(monitor_15m_days=1 / 96))
        self.assertEqual(stats["source_identities"], 100)
        self.assertEqual(stats["monitored_candle_checks"], 100)
        self.assertEqual(len(select(rows, "monitoring_expiry")), 100)
        self.assertEqual(stats["monitored_versions_at_end"], 0)

    def test_missing_duplicate_offgrid_candles_fail(self):
        for bars in ([bar(0), bar(2)], [bar(0), bar(0)],
                     [replace(bar(0), timestamp=at(0) + timedelta(minutes=1))],
                     [replace(bar(0), available_at=at(2))]):
            with self.subTest(bars=bars), self.assertRaises(ValueError):
                replay(bars)

    def test_bad_ohlc_empty_input_and_late_coverage_fail(self):
        for bars in ([], [replace(bar(0), high=math.nan)], [replace(bar(0), close=999)], [bar(2)]):
            with self.subTest(bars=bars), self.assertRaises(ValueError):
                replay(bars)

    def test_eof_emits_no_terminal_rows(self):
        rows, _ = replay([bar(0), bar(1, low=101)])
        self.assertFalse(select(rows, "monitoring_expiry"))
        self.assertFalse(select(rows, "version_end"))
        self.assertFalse(select(rows, "contact_exit"))

    def test_complete_prefixes_and_repeatability(self):
        sources = [source_row(), source_row(when=4, kind="update", sequence=2, lower=101, upper=103)]
        bars = [bar(0), bar(1, low=101), bar(2), bar(3, low=100.5), bar(4, low=102), bar(5)]
        config = lifecycle.Config(monitor_15m_days=3 / 96)
        full, _ = replay(bars, sources, config)
        for size in range(1, len(bars) + 1):
            cutoff = bars[size - 1].available_at
            prefix_sources = [r for r in sources if lifecycle.utc(r["available_at_utc"]) <= cutoff]
            self.assertEqual(replay(bars[:size], prefix_sources, config)[0],
                             [r for r in full if lifecycle.utc(r["event_at_utc"]) <= cutoff])
        self.assertEqual(replay(bars, sources, config)[0], full)
        self.assertEqual(len({(r["registry_zone_id"], r["zone_version_at_utc"], r["event_sequence"]) for r in full}), len(full))

    def test_mixed_family_prefixes_with_future_events_and_censoring(self):
        sources = [source_row(), source_row(when=5, kind="update", sequence=2),
                   source_row("reaction", "r", when=2, confirmed=2, direction="bearish"),
                   source_row("imbalance", "i", when=3),
                   source_row("imbalance", "i", when=4, kind="fvg_confirmation", lower=103, upper=105),
                   source_row("volume_derived", "v4", when=4, kind="profile_observation"),
                   source_row("volume_derived", "v8", when=8, kind="profile_observation")]
        randomizer = random.Random(7)
        bars = []
        for i in range(24):
            opening, close = randomizer.uniform(98, 106), randomizer.uniform(98, 106)
            bars.append(bar(i, opening, max(opening, close) + 1, min(opening, close) - 1, close))
        config = lifecycle.Config(monitor_15m_days=4 / 96)
        full, _ = replay(bars, sources, config)
        for size in range(1, len(bars) + 1):
            with self.subTest(size=size):
                expected = [r for r in full if lifecycle.utc(r["event_at_utc"]) <= at(size)]
                self.assertEqual(replay(bars[:size], sources, config)[0], expected)
                visible = [r for r in sources if lifecycle.utc(r["available_at_utc"]) <= at(size)]
                self.assertEqual(replay(bars[:size], visible, config)[0], expected)
        self.assertTrue(all(lifecycle.utc(r["event_at_utc"]) > lifecycle.utc(r["zone_version_at_utc"])
                            for r in full if r["candle_time_utc"] is not None))

    def test_bad_source_transitions_and_duplicate_same_time_fail(self):
        for sources in ([source_row(kind="update", sequence=2)], [source_row(), source_row()],
                        [source_row(), source_row(when=2, kind="update", sequence=3)]):
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                replay([bar(i) for i in range(4)], sources)

    def test_offgrid_event_is_not_silently_skipped(self):
        bad = replace(event(source_row()), available=at(1) - timedelta(minutes=1))
        with self.assertRaises(ValueError):
            list(lifecycle.iter_interactions([bar(0), bar(1)], [bad], "BTCUSDT", "15m"))

    def test_all_timeframes_use_their_own_interval(self):
        for timeframe, interval in lifecycle.INTERVALS.items():
            with self.subTest(timeframe=timeframe):
                e = event(source_row())
                e = replace(e, timeframe=timeframe, registry_id=f"BTCUSDT:{timeframe}:structural:s",
                            available=START + interval, confirmed=START + interval)
                bars = [replace(bar(i, low=101), timestamp=START + i * interval,
                                available_at=START + (i + 1) * interval) for i in range(2)]
                touch, = list(lifecycle.iter_interactions(bars, [e], "BTCUSDT", timeframe))
                self.assertEqual(touch["event_at_utc"], (START + 2 * interval).isoformat())

    def test_configuration_requires_positive_grid_aligned_caps(self):
        for kwargs in ({"monitor_15m_days": 0}, {"monitor_1h_days": math.inf},
                       {"monitor_15m_days": .00001}, {"penetration_epsilon": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                lifecycle.Config(**kwargs)

    def test_missing_imbalance_direction_is_not_guessed(self):
        e = replace(event(source_row("imbalance", "bullish_name", when=3)), direction=None)
        with self.assertRaises(ValueError):
            list(lifecycle.iter_interactions([bar(i) for i in range(4)], [e], "BTCUSDT", "15m"))


class VolumeTests(unittest.TestCase):
    def test_empty_next_profile_expires_observation_without_future_rows(self):
        rows, _ = replay([bar(i, low=101) for i in range(9)],
                         [source_row("volume_derived", "v", when=4, kind="profile_observation")])
        touch, = select(rows, "first_touch")
        self.assertIsNone(touch["direction"])
        end, = select(rows, "version_end")
        self.assertEqual(end["event_at_utc"], at(6).isoformat())
        self.assertEqual(end["event_detail"], "next_profile_slot")
        self.assertFalse(select(rows, "observed_invalidation"))
        self.assertFalse(select(rows, "contact_exit"))

    def test_profiles_have_separate_neutral_episodes(self):
        sources = [source_row("volume_derived", "a", when=4, kind="profile_observation"),
                   source_row("volume_derived", "b", when=6, kind="profile_observation")]
        rows, _ = replay([bar(i, low=101) for i in range(8)], sources)
        self.assertEqual([r["episode_number"] for r in select(rows, "first_touch")], [1, 1])
        self.assertFalse(select(rows, "retest"))

    def test_volume_close_crossing_is_not_directional_invalidation(self):
        bars = [bar(i) for i in range(4)] + [bar(4, 104, 105, 99, 99), bar(5, 99, 105, 98, 104)]
        rows, _ = replay(bars, [source_row("volume_derived", "v", when=4, kind="profile_observation")])
        self.assertTrue(select(rows, "boundary_crossing"))
        self.assertFalse(select(rows, "observed_invalidation"))
        self.assertTrue(all(r["direction"] is None for r in rows))

    def test_off_calendar_observations_fail(self):
        with self.assertRaises(ValueError):
            replay([bar(i) for i in range(7)], [source_row("volume_derived", "v", when=5, kind="profile_observation")])

    def test_volume_expiry_prefixes_and_offgrid_calendar(self):
        sources = [source_row("volume_derived", "v", when=4, kind="profile_observation")]
        bars = [bar(i, low=101) for i in range(8)]
        full = replay(bars, sources)[0]
        for size in range(1, len(bars) + 1):
            self.assertEqual(replay(bars[:size], sources)[0],
                             [r for r in full if lifecycle.utc(r["event_at_utc"]) <= at(size)])
        bad_clock = replace(clock(), stride=timedelta(minutes=5))
        with self.assertRaises(ValueError):
            list(lifecycle.iter_interactions(bars, [event(sources[0])], "BTCUSDT", "15m", bad_clock))


class FileTests(unittest.TestCase):
    def test_exact_source_direction_not_id_or_future_version(self):
        sources = [source_row("imbalance", "misleading_bearish_id", when=3),
                   source_row("imbalance", "misleading_bearish_id", when=4, kind="fvg_confirmation")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project(root, sources)
            events = lifecycle.load_registry_events(root, "BTCUSDT", "15m", through=at(3))
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].direction, "bullish")
            self.assertEqual(events[0].origin, "displacement_body")
            self.assertEqual(events[0].created, at(3))

    def test_each_referenced_file_is_read_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unified, _ = project(root, [source_row(), source_row(when=2, kind="update", sequence=2)])
            original, reads = Path.open, Counter()

            def counted(path, *args, **kwargs):
                reads[path] += 1
                return original(path, *args, **kwargs)

            with patch.object(Path, "open", counted):
                events = lifecycle.load_registry_events(root, "BTCUSDT", "15m")
            self.assertEqual(len(events), 2)
            self.assertEqual(reads[unified], 1)
            self.assertEqual(reads[registry.source_paths(root, "BTCUSDT", "15m")["structural"]], 1)
            self.assertEqual(sum(reads.values()), 2)

    def test_missing_malformed_or_mismatched_references_fail(self):
        changes = (("source_file", "../outside.csv"), ("source_row", "1"), ("source_row", "bad"),
                   ("source_row", "99"), ("source_zone_id", "wrong"), ("available_at_utc", at(2).isoformat()),
                   ("zone_family", "unknown"), ("lower_price", "99"), ("overlapping_family_count", "-1"))
        for key, value in changes:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                unified, _ = project(root)
                rewrite_registry(unified, lambda rows: rows[0].update({key: value}))
                with self.assertRaises(ValueError):
                    lifecycle.load_registry_events(root, "BTCUSDT", "15m")

    def test_missing_source_file_and_header_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project(root)
            source = registry.source_paths(root, "BTCUSDT", "15m")["structural"]
            source.write_text("zone_id\nwrong\n")
            with self.assertRaises(ValueError):
                lifecycle.load_registry_events(root, "BTCUSDT", "15m")
            source.unlink()
            with self.assertRaises(FileNotFoundError):
                lifecycle.load_registry_events(root, "BTCUSDT", "15m")

    def test_writer_schema_read_only_sources_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unified, feature = project(root)
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            output = root / "temporary.csv"
            result = lifecycle.process_dataset(root, "BTCUSDT", "15m", output)
            original = output.read_bytes()
            with output.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, list(lifecycle.COLUMNS))
                rows = list(reader)
            self.assertEqual(result["statistics"]["output_event_rows"], len(rows))
            self.assertEqual(result["candle_validation"]["missing_atr_rows"], 3)
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            with self.assertRaises(FileExistsError):
                lifecycle.process_dataset(root, "BTCUSDT", "15m", output)
            self.assertEqual(output.read_bytes(), original)
            lifecycle.process_dataset(root, "BTCUSDT", "15m", output, overwrite=True)
            self.assertEqual(output.read_bytes(), original)
            for path in (unified, feature, registry.source_paths(root, "BTCUSDT", "15m")["structural"]):
                with self.assertRaises(ValueError):
                    lifecycle.process_dataset(root, "BTCUSDT", "15m", path, overwrite=True)

    def test_input_errors_leave_no_success_report_or_modified_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project(root, bars=[bar(0), bar(2)])
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            with self.assertRaises(ValueError):
                lifecycle.process_dataset(root, "BTCUSDT", "15m", root / "out.csv")
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            self.assertFalse((root / "reports/zone_detection/lifecycle_summary.json").exists())

    def test_writer_cannot_overwrite_other_phase_data_or_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project(root)
            for path in (root / "data/raw/other.csv", root / "reports/other.json"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"preserve me")
                with self.assertRaises(ValueError):
                    lifecycle.process_dataset(root, "BTCUSDT", "15m", path, overwrite=True)
                self.assertEqual(path.read_bytes(), b"preserve me")

    def test_single_dataset_generation_helper_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project(root)
            result = lifecycle.generate_dataset(root, "BTCUSDT", "15m")
            output = root / "data/zones/lifecycle/BTCUSDT_15m.csv"
            self.assertEqual(result["file_size_bytes"], output.stat().st_size)
            self.assertGreaterEqual(result["runtime_seconds"], 0)
            saved = output.read_bytes()
            with self.assertRaises(FileExistsError):
                lifecycle.generate_dataset(root, "BTCUSDT", "15m")
            self.assertEqual(output.read_bytes(), saved)

    def test_summary_requires_six_audited_outputs_and_protects_existing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = []
            for symbol in ("BTCUSDT", "ETHUSDT"):
                for timeframe in lifecycle.INTERVALS:
                    output = root / "data/zones/lifecycle" / f"{symbol}_{timeframe}.csv"
                    write_csv(output, lifecycle.COLUMNS, [])
                    results.append({"symbol": symbol, "source_timeframe": timeframe,
                                    "output_path": str(output), "runtime_seconds": 0,
                                    "validation": {"status": "PASS"}})
            tests = {"successful": True, "tests_run": 1}
            with self.assertRaises(ValueError):
                lifecycle.write_summary(root, results[:1], tests)
            with self.assertRaises(ValueError):
                lifecycle.write_summary(root, results, {"successful": False})
            results[0]["validation"]["status"] = "FAIL"
            with self.assertRaises(ValueError):
                lifecycle.write_summary(root, results, tests)
            results[0]["validation"]["status"] = "PASS"
            report = lifecycle.write_summary(root, results, tests)
            saved = report.read_bytes()
            self.assertEqual(json.loads(saved)["methodology_version"], lifecycle.METHODOLOGY_VERSION)
            with self.assertRaises(FileExistsError):
                lifecycle.write_summary(root, results, tests)
            self.assertEqual(report.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
