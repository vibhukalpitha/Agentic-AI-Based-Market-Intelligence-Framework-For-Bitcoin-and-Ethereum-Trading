from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from zone_detection.registry import (
    Config, FAMILIES, SOURCE_COLUMNS, VolumeClock, normalize, overlaps, preflight,
    process_dataset, registry_rows, source_paths,
)
from zone_detection.swings import utc

START = datetime(2020, 1, 1, tzinfo=timezone.utc)
INTERVAL = timedelta(minutes=15)


def at(index):
    return START + index * INTERVAL


def common(family, source_id, when, lower=100, upper=102):
    return {"zone_id": source_id, "symbol": "BTCUSDT", "source_timeframe": "15m", "zone_family": family,
            "lower_price": str(lower), "upper_price": str(upper), "available_at_utc": at(when).isoformat(),
            "candidate_status": "candidate"}


def cluster(family="structural", source_id="s", when=4, lower=100, upper=102,
            kind="create", sequence=1, confirmed=None):
    row = common(family, source_id, when, lower, upper)
    row.update(origin_type="swing_low_cluster" if family == "structural" else "bullish_rejection_cluster",
               formation_start_utc=at(0).isoformat(), confirmed_at_utc=at(when if confirmed is None else confirmed).isoformat(),
               cluster_reference_atr="2", event_type=kind, event_at_utc=at(when).isoformat(), event_sequence=str(sequence))
    row["confirmed_swing_count" if family == "structural" else "independent_rejection_count"] = str(sequence + 1)
    return row


def imbalance(source_id="i", when=4, lower=100, upper=102, kind="create", origin="displacement_body"):
    row = common("imbalance", source_id, when, lower, upper)
    body = origin == "displacement_body"
    reference = when - 1 if body else when - 2
    row.update(formation_start_utc=at(reference if body else when - 3).isoformat(), imbalance_type=origin,
               direction="bearish" if origin == "bearish_fvg" else "bullish", event_type=kind,
               candidate_created_at_utc=at(when if kind == "create" else when - 1).isoformat(),
               c1_time_utc="" if body else at(when - 3).isoformat(), c2_time_utc=at(reference).isoformat(),
               c3_time_utc="" if body else at(when - 1).isoformat(),
               reference_candle_time_utc=at(reference).isoformat(), reference_atr="2",
               reference_atr_available_at_utc=at(reference + 1).isoformat(),
               aligned_displacement="True" if kind == "fvg_confirmation" else "False")
    return row


def volume(source_id="v", when=4, lower=100, upper=102, profile_id=None, rank=1):
    row = common("volume_derived", source_id, when, lower, upper)
    row.update(profile_start_utc=at(when - 4).isoformat(), profile_end_utc=at(when - 1).isoformat(),
               profile_id=profile_id or f"profile_{when}", candidate_rank=str(rank), reference_atr="2",
               method="uniform_ohlcv_range_allocation")
    return row


def event(row, line=2):
    family = row["zone_family"]
    return normalize(row, family, "BTCUSDT", "15m", f"data/zones/{family}/BTCUSDT_15m.csv", line)


def clock():
    return VolumeClock(at(4), at(100), 4 * INTERVAL, 4 * INTERVAL)


def build(rows, config=None):
    stats = Counter()
    result = list(registry_rows([event(row, i + 2) for i, row in enumerate(rows)], clock(), config or Config(), stats))
    return result, stats


def fixture_reports(rows_by_family):
    reports = {}
    for family in FAMILIES:
        rows = rows_by_family.get(family, [])
        dataset = {"symbol": "BTCUSDT", "source_timeframe": "15m"}
        if family in ("structural", "reaction"):
            dataset.update(candidate_zones_created=sum(r["event_type"] == "create" for r in rows),
                           zone_update_count=sum(r["event_type"] == "update" for r in rows))
        elif family == "imbalance":
            dataset.update(output_event_rows=len(rows), unique_candidate_count=len({r["zone_id"] for r in rows}),
                           linked_fvg_confirmation_count=sum(r["event_type"] == "fvg_confirmation" for r in rows),
                           bullish_fvg_count=sum(r["imbalance_type"] == "bullish_fvg" for r in rows),
                           bearish_fvg_count=sum(r["imbalance_type"] == "bearish_fvg" for r in rows),
                           displacement_body_creation_count=sum(r["imbalance_type"] == "displacement_body" for r in rows))
        else:
            dataset.update(candidate_observations=len(rows), input_candle_count=100, profiles_calculated=25,
                           no_candidate_profiles=25 - len({r["profile_id"] for r in rows}),
                           timestamp_checks={"first_open_utc": at(0).isoformat(), "last_available_utc": at(100).isoformat()})
        reports[family] = {"datasets": [dataset]}
    reports["volume_derived"]["parameters"] = {"profile_window_bars": 4, "profile_stride_bars": 4}
    return reports


def write_sources(root, rows_by_family, symbol="BTCUSDT", timeframe="15m"):
    for family, path in source_paths(root, symbol, timeframe).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SOURCE_COLUMNS[family])
            writer.writeheader()
            writer.writerows(rows_by_family.get(family, []))


class MappingTests(unittest.TestCase):
    def test_all_four_families_and_nullable_source_fields(self):
        rows, _ = build([cluster(), cluster("reaction", "r"), imbalance(), volume()])
        by_family = {r["zone_family"]: r for r in rows}
        self.assertEqual(len(rows), 4)
        for family in ("structural", "reaction"):
            self.assertEqual(by_family[family]["confirmed_at_utc"], at(4).isoformat())
            self.assertEqual(by_family[family]["source_event_sequence"], 1)
        for family in ("imbalance", "volume_derived"):
            self.assertIsNone(by_family[family]["confirmed_at_utc"])
            self.assertIsNone(by_family[family]["source_event_sequence"])
        self.assertIsNone(by_family["volume_derived"]["origin_type"])
        self.assertEqual(by_family["volume_derived"]["event_type"], "profile_observation")
        self.assertTrue(all(r["source_reference_atr"] == 2 for r in rows))

    def test_standalone_fvg_confirmed_at_c3_not_impulse_creation(self):
        row = event(imbalance(origin="bullish_fvg"))
        self.assertEqual(row.confirmed, at(4))
        self.assertEqual(row.available, at(4))
        self.assertEqual(row.formation, at(1))

    def test_shared_id_fvg_confirmation_preserves_both_intervals(self):
        rows, _ = build([imbalance(when=4, lower=100, upper=102),
                         imbalance(when=5, lower=105, upper=106, kind="fvg_confirmation", origin="bullish_fvg")])
        self.assertEqual(rows[0]["registry_zone_id"], rows[1]["registry_zone_id"])
        self.assertEqual([r["lower_price"] for r in rows], [100, 105])
        self.assertEqual([r["available_at_utc"] for r in rows], [at(4).isoformat(), at(5).isoformat()])
        self.assertIsNone(rows[0]["confirmed_at_utc"])
        self.assertEqual(rows[1]["confirmed_at_utc"], at(5).isoformat())

    def test_reaction_update_keeps_original_confirmation(self):
        rows, _ = build([cluster("reaction", "r", when=4),
                         cluster("reaction", "r", when=8, lower=100, upper=103, kind="update", sequence=2, confirmed=4)])
        self.assertEqual([r["upper_price"] for r in rows], [102, 103])
        self.assertEqual([r["confirmed_at_utc"] for r in rows], [at(4).isoformat()] * 2)
        self.assertEqual(rows[1]["source_event_sequence"], 2)

    def test_family_qualified_source_ids_cannot_collide(self):
        rows, _ = build([cluster(source_id="same"), cluster("reaction", "same"),
                         imbalance("same"), volume("same")])
        self.assertEqual({r["source_zone_id"] for r in rows}, {"same"})
        self.assertEqual(len({r["registry_zone_id"] for r in rows}), 4)

    def test_missing_atr_remains_nullable(self):
        row = imbalance(origin="bullish_fvg")
        row["reference_atr"] = ""
        found, _ = build([row])
        self.assertIsNone(found[0]["source_reference_atr"])

    def test_volume_observations_have_separate_profile_specific_ids(self):
        rows, _ = build([volume("v1", 4), volume("v2", 8)])
        self.assertNotEqual(rows[0]["registry_zone_id"], rows[1]["registry_zone_id"])
        self.assertTrue(all(r["confirmed_at_utc"] is None for r in rows))


class OverlapTests(unittest.TestCase):
    def test_strict_positive_overlap_boundary_touch_and_disjoint(self):
        first = event(cluster(lower=100, upper=102))
        self.assertTrue(overlaps(first, event(cluster(source_id="x", lower=101, upper=103))))
        self.assertFalse(overlaps(first, event(cluster(source_id="x", lower=102, upper=103))))
        self.assertFalse(overlaps(first, event(cluster(source_id="x", lower=103, upper=104))))

    def test_no_self_overlap_including_shared_id_confirmation(self):
        rows, _ = build([imbalance(), imbalance(when=5, kind="fvg_confirmation", origin="bullish_fvg")])
        self.assertEqual([r["overlapping_candidate_count"] for r in rows], [0, 0])
        self.assertEqual([r["overlapping_family_count"] for r in rows], [0, 0])

    def test_same_family_ids_count_as_candidates_not_other_families(self):
        rows, _ = build([cluster(source_id="a"), cluster(source_id="b")])
        self.assertEqual([r["overlapping_candidate_count"] for r in rows], [1, 1])
        self.assertTrue(all(r["overlapping_family_count"] == 0 and r["overlap_family_flags"] == "" for r in rows))

    def test_distinct_other_families_count_once(self):
        rows, _ = build([cluster(), cluster("reaction", "r"), imbalance(), volume()])
        for row in rows:
            self.assertEqual(row["overlapping_candidate_count"], 3)
            self.assertEqual(row["overlapping_family_count"], 3)
            self.assertNotIn(row["zone_family"], row["overlap_family_flags"].split("|"))

    def test_latest_volume_profile_only_and_no_family_inflation(self):
        rows, _ = build([cluster(upper=104), volume("v1", 4, 100, 102), volume("v2", 4, 102, 104, rank=2),
                         volume("v3", 8, 100, 104), cluster("reaction", "r", when=8, upper=104)])
        structural = next(r for r in rows if r["zone_family"] == "structural")
        self.assertEqual(structural["overlapping_candidate_count"], 2)
        self.assertEqual(structural["overlapping_family_count"], 1)
        reaction = next(r for r in rows if r["zone_family"] == "reaction")
        self.assertEqual(reaction["overlapping_candidate_count"], 2)
        self.assertEqual(reaction["overlap_family_flags"], "structural|volume_derived")

    def test_empty_profile_clears_old_volume_without_a_new_volume_row(self):
        rows, _ = build([volume(when=4), cluster(when=7), cluster("reaction", "r", when=9)])
        self.assertIn("volume_derived", rows[1]["overlap_family_flags"])
        self.assertNotIn("volume_derived", rows[2]["overlap_family_flags"])

    def test_volume_expires_after_source_coverage_plus_stride(self):
        stats = Counter()
        records = [event(volume(when=4)), event(cluster(when=8))]
        short_clock = VolumeClock(at(4), at(4), 4 * INTERVAL, 4 * INTERVAL)
        rows = list(registry_rows(records, short_clock, Config(), stats))
        self.assertEqual(rows[-1]["overlapping_candidate_count"], 0)

    def test_same_timestamp_batch_sees_other_updates_not_previous_boundaries(self):
        records = [event(cluster("reaction", "r", when=4, lower=200, upper=202)),
                   event(cluster(when=8)),
                   event(cluster("reaction", "r", when=8, kind="update", sequence=2, confirmed=4))]
        first = list(registry_rows(records, clock(), Config(), Counter()))
        second = list(registry_rows(reversed(records), clock(), Config(), Counter()))
        self.assertEqual(first, second)
        self.assertEqual(first[1]["overlapping_family_count"], 1)
        self.assertEqual(first[2]["overlapping_family_count"], 1)

    def test_future_fvg_boundaries_are_not_visible_earlier(self):
        rows, _ = build([imbalance(when=4, lower=100, upper=102), cluster(when=4, lower=105, upper=106),
                         imbalance(when=5, lower=105, upper=106, kind="fvg_confirmation", origin="bullish_fvg")])
        early = [r for r in rows if r["available_at_utc"] == at(4).isoformat()]
        self.assertTrue(all(r["overlapping_candidate_count"] == 0 for r in early))
        self.assertEqual(rows[-1]["overlap_family_flags"], "structural")

    def test_recency_excludes_search_not_history_or_validity(self):
        rows, _ = build([cluster(when=4), cluster("reaction", "r", when=5),
                         cluster("reaction", "r2", when=6)], Config(overlap_15m_days=1 / 96))
        self.assertEqual(rows[1]["overlap_family_flags"], "structural")
        self.assertEqual(rows[2]["overlapping_family_count"], 0)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["candidate_status"] == "candidate" for r in rows))

    def test_later_source_update_can_reenter_overlap_search(self):
        rows, _ = build([cluster(when=4), cluster(when=10, kind="update", sequence=2, confirmed=4),
                         cluster("reaction", "r", when=10)], Config(overlap_15m_days=1 / 96))
        reaction = next(row for row in rows if row["zone_family"] == "reaction")
        update = next(row for row in rows if row["event_type"] == "update")
        self.assertEqual(reaction["overlap_family_flags"], "structural")
        self.assertEqual(update["confirmed_at_utc"], at(4).isoformat())

    def test_prefix_invariance_and_stable_complete_batch_ordering(self):
        records = [event(cluster()), event(imbalance()), event(volume()),
                   event(imbalance(when=5, kind="fvg_confirmation", origin="bullish_fvg")),
                   event(cluster("reaction", "r", when=8)), event(volume("v2", 8)),
                   event(cluster(when=12, kind="update", sequence=2, confirmed=4, upper=103))]
        full = list(registry_rows(records, clock(), Config(), Counter()))
        for cutoff in (4, 5, 7, 8, 12):
            prefix = list(registry_rows([e for e in records if e.available <= at(cutoff)], clock(), Config(), Counter()))
            self.assertEqual(prefix, [r for r in full if utc(r["available_at_utc"]) <= at(cutoff)])
        self.assertEqual(full, list(registry_rows(reversed(records), clock(), Config(), Counter())))


class ValidationAndFileTests(unittest.TestCase):
    def test_bad_prices_timestamps_and_asset_fields_fail(self):
        for field, value in (("lower_price", "NaN"), ("upper_price", "inf"), ("lower_price", "0"),
                             ("upper_price", "99"), ("available_at_utc", "2020-01-01T01:00:00"),
                             ("available_at_utc", "2020-01-01T01:01:00Z"), ("symbol", "ETHUSDT"),
                             ("candidate_status", "active"), ("cluster_reference_atr", "0")):
            with self.subTest(field=field, value=value):
                row = cluster()
                row[field] = value
                with self.assertRaises(ValueError):
                    event(row)

    def test_bad_cluster_history_sequences_fail(self):
        bad_lists = [
            [cluster(kind="update", sequence=2)],
            [cluster(), cluster(when=5)],
            [cluster(), cluster(when=5, kind="update", sequence=3, confirmed=4)],
            [cluster(), cluster(when=5, kind="update", sequence=2, confirmed=5)],
            [cluster(), cluster(when=4, kind="update", sequence=2, confirmed=4)],
        ]
        for rows in bad_lists:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                build(rows)

    def test_fvg_confirmation_requires_original_body_once(self):
        confirmation = imbalance(when=5, kind="fvg_confirmation", origin="bullish_fvg")
        with self.assertRaises(ValueError):
            build([confirmation])
        with self.assertRaises(ValueError):
            build([imbalance(), confirmation, confirmation])
        with self.assertRaises(ValueError):
            build([imbalance(origin="bullish_fvg"), confirmation])
        row = imbalance(origin="bullish_fvg")
        row["available_at_utc"] = at(3).isoformat()
        with self.assertRaises(ValueError):
            event(row)

    def test_duplicate_volume_identity_and_bad_calendar_fail(self):
        for rows in ([volume(), volume()], [volume(when=5)],
                     [volume("v1"), volume("v2", profile_id="other_profile")]):
            with self.assertRaises(ValueError):
                build(rows)

    def test_source_read_once_snapshots_ignored_and_counts_reconciled(self):
        rows = {"structural": [cluster()], "reaction": [cluster("reaction", "r")],
                "imbalance": [imbalance(), imbalance(when=5, kind="fvg_confirmation", origin="bullish_fvg")],
                "volume_derived": [volume()]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_sources(root, rows)
            snapshot = root / "data/zones/structural/BTCUSDT_15m.csv"
            with snapshot.open("x", encoding="utf-8") as handle:
                handle.write("THIS IS A FINAL SNAPSHOT AND MUST NOT BE READ\n")
            sources = source_paths(root, "BTCUSDT", "15m")
            original = {p: p.read_bytes() for p in sources.values()}
            reads = Counter()
            original_open = Path.open

            def count_open(path, mode="r", *args, **kwargs):
                if mode == "r":
                    reads[path] += 1
                return original_open(path, mode, *args, **kwargs)

            reports = fixture_reports(rows)
            with patch.object(Path, "open", count_open):
                result = process_dataset(root, "BTCUSDT", "15m", Config(), reports)
            self.assertEqual([reads[p] for p in sources.values()], [1, 1, 1, 1])
            self.assertEqual(reads[snapshot], 0)
            self.assertEqual(result["unified_event_rows"], 5)
            self.assertEqual(result["unified_identity_count"], 4)
            self.assertEqual(result["deduplicated_non_volume_identity_count"], 3)
            self.assertEqual(result["correlated_volume_observation_identity_count"], 1)
            self.assertIsNone(result["statistically_independent_evidence_count"])
            self.assertEqual(result["count_reconciliation"]["status"], "PASS")
            self.assertTrue(all(p.read_bytes() == value for p, value in original.items()))
            output = Path(result["output_path"])
            saved = output.read_bytes()
            with self.assertRaises(FileExistsError):
                process_dataset(root, "BTCUSDT", "15m", Config(), reports)
            self.assertEqual(output.read_bytes(), saved)
            process_dataset(root, "BTCUSDT", "15m", Config(), reports, overwrite=True)
            self.assertEqual(output.read_bytes(), saved)

    def test_report_mismatch_fails_before_writing_output(self):
        rows = {"structural": [cluster()]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_sources(root, rows)
            reports = fixture_reports(rows)
            reports["structural"]["datasets"][0]["candidate_zones_created"] = 2
            with self.assertRaisesRegex(ValueError, "report count mismatch"):
                process_dataset(root, "BTCUSDT", "15m", Config(), reports)
            self.assertFalse((root / "data/zones/unified/BTCUSDT_15m.csv").exists())

    def test_missing_history_columns_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_sources(root, {})
            path = source_paths(root, "BTCUSDT", "15m")["structural"]
            with path.open("w", newline="") as handle:
                csv.writer(handle).writerow(["zone_id", "lower_price", "upper_price"])
            with self.assertRaises(ValueError):
                process_dataset(root, "BTCUSDT", "15m", Config(), fixture_reports({}))

    def test_preflight_protects_summary_and_all_six_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for timeframe in ("15m", "1h", "4h"):
                for symbol in ("BTCUSDT", "ETHUSDT"):
                    write_sources(root, {}, symbol, timeframe)
            self.assertEqual(len(preflight(root, False)), 6)
            summary = root / "reports/zone_detection/registry_summary.json"
            summary.parent.mkdir(parents=True)
            summary.touch()
            with self.assertRaises(FileExistsError):
                preflight(root, False)
            self.assertEqual(len(preflight(root, True)), 6)
            summary.unlink()
            output = root / "data/zones/unified/ETHUSDT_4h.csv"
            output.parent.mkdir(parents=True)
            output.touch()
            with self.assertRaises(FileExistsError):
                preflight(root, False)

    def test_bad_recency_and_profile_clock_fail(self):
        for value in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                Config(overlap_15m_days=value)
        with self.assertRaises(ValueError):
            VolumeClock(at(4), at(100), timedelta(0), INTERVAL)
