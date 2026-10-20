from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import tempfile
import unittest
from array import array
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dataset_construction as ds
from zone_detection import lifecycle, registry
from zone_detection.swings import Candle


START = ds.utc("2023-01-01T00:00:00Z")
INTERVAL = ds.INTERVALS["15m"]
CFG = ds.Config()


def table(count=60):
    data = {name: array("d", [1.0] * count) for name in ds.CANDLE_FIELDS}
    for name, value in (("open", 104), ("high", 105), ("low", 103), ("close", 104), ("atr_14_sma", 2)):
        data[name] = array("d", [value] * count)
    return ds.Candles(START, INTERVAL, data)


def event(candles, direction="bullish", index=0):
    return registry.Event("BTCUSDT:15m:structural:s", "s", "BTCUSDT", "15m", "structural",
                          "swing_low_cluster" if direction == "bullish" else "swing_high_cluster",
                          100.0, 102.0, candles.start, candles.available(index), candles.available(index),
                          "create", 2.0, 1, "data/zones/structural/BTCUSDT_15m_zone_events.csv", 2,
                          evidence_count=2)


def put(candles, index, o=104, h=105, l=101, c=102):
    for name, value in zip(("open", "high", "low", "close"), (o, h, l, c)):
        candles.data[name][index] = value


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def fixture(root):
    candles = table()
    put(candles, 1)
    rows = []
    for i in range(len(candles)):
        row = {name: candles.data[name][i] for name in ds.CANDLE_FIELDS}
        row.update(timestamp_utc=ds.iso(candles.available(i) - INTERVAL), available_at_utc=ds.iso(candles.available(i)))
        rows.append(row)
    source = root / "data/features/15m/BTCUSDT_15m_features.csv"
    write_csv(source, ("timestamp_utc", "available_at_utc", *ds.CANDLE_FIELDS), rows)
    e = event(candles)
    raw = {"zone_id": "s", "symbol": "BTCUSDT", "source_timeframe": "15m", "zone_family": "structural",
           "origin_type": e.origin, "lower_price": 100, "upper_price": 102,
           "formation_start_utc": ds.iso(e.formation), "confirmed_at_utc": ds.iso(e.confirmed),
           "available_at_utc": ds.iso(e.available), "candidate_status": "candidate",
           "cluster_reference_atr": 2, "event_type": "create", "event_at_utc": ds.iso(e.available),
           "event_sequence": 1, "confirmed_swing_count": 2}
    write_csv(root / e.source_file, registry.SOURCE_COLUMNS["structural"], [raw])
    write_csv(root / "data/zones/unified/BTCUSDT_15m.csv", registry.COLUMNS, [e.record(0, set())])
    bars = [Candle(candles.available(i) - INTERVAL, candles.available(i),
                  *(candles.data[n][i] for n in ("open", "high", "low", "close")), 2) for i in range(len(candles))]
    history = root / "data/zones/lifecycle/BTCUSDT_15m.csv"
    write_csv(history, lifecycle.COLUMNS, lifecycle.iter_interactions(bars, [e], "BTCUSDT", "15m"))
    report = root / "reports/zone_detection/registry_summary.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"parameters": ds.asdict(registry.Config())}), encoding="utf-8")
    return candles, e


class StrongTests(unittest.TestCase):
    def test_strong_and_exact_reaction_equality(self):
        c = table()
        put(c, 1)
        put(c, 2, h=105, l=103, c=105)
        self.assertEqual(ds.strong_outcome(event(c), c, 0, CFG), ("labeled", 1, 1, 24))

    def test_weak_without_sufficient_reaction(self):
        c = table()
        put(c, 1)
        self.assertEqual(ds.strong_outcome(event(c), c, 0, CFG)[1], 0)

    def test_late_failure_overrides_initial_reaction(self):
        c = table()
        put(c, 1)
        put(c, 2, h=105, l=103, c=105)
        put(c, 24, o=102, h=103, l=99, c=99)
        self.assertEqual(ds.strong_outcome(event(c), c, 0, CFG)[1], 0)

    def test_response_includes_touch_and_excludes_bar_25(self):
        c = table()
        put(c, 1, h=105, c=105)
        put(c, 25, o=102, h=103, l=98, c=98)
        self.assertEqual(ds.strong_outcome(event(c), c, 0, CFG)[1:], (1, 1, 24))

    def test_last_eligible_touch_is_bar_20(self):
        c = table()
        put(c, 20)
        self.assertEqual(ds.strong_outcome(event(c), c, 0, CFG)[2:], (20, 43))
        c = table()
        put(c, 21)
        self.assertEqual(ds.strong_outcome(event(c), c, 0, CFG)[0], "no_touch")

    def test_creation_candle_is_never_touch(self):
        c = table()
        put(c, 0)
        self.assertEqual(ds.strong_outcome(event(c), c, 0, CFG)[0], "no_touch")

    def test_wait_and_response_censoring_are_not_weak(self):
        c = table(10)
        self.assertEqual(ds.strong_outcome(event(c), c, 0, CFG)[0], "censored_wait")
        put(c, 1)
        self.assertEqual(ds.strong_outcome(event(c), c, 0, CFG)[:2], ("censored_response", None))

    def test_wrong_and_ambiguous_approaches(self):
        for opening, status in ((99, "wrong_approach"), (100, "ambiguous_approach"), (101, "ambiguous_approach")):
            c = table()
            put(c, 1, o=opening, l=99)
            self.assertEqual(ds.strong_outcome(event(c), c, 0, CFG)[0], status)

    def test_pre_touch_failure_including_creation_close(self):
        for index in (0, 1):
            c = table()
            put(c, index, o=99, h=99, l=98, c=99)
            self.assertEqual(ds.strong_outcome(event(c), c, 0, CFG)[0], "pre_touch_failure")

    def test_bearish_symmetry(self):
        c = table()
        for i in range(len(c)):
            put(c, i, o=98, h=99, l=97, c=98)
        put(c, 1, o=98, h=101, l=97, c=100)
        put(c, 2, o=98, h=99, l=97, c=97)
        self.assertEqual(ds.strong_outcome(event(c, "bearish"), c, 0, CFG)[1], 1)

    def test_atr_frozen_at_creation_not_touch(self):
        c = table()
        put(c, 1)
        put(c, 2, h=105, l=103, c=105)
        c.data["atr_14_sma"][1] = 100
        self.assertEqual(ds.strong_outcome(event(c), c, 0, CFG)[1], 1)


class ScenarioTests(unittest.TestCase):
    def test_real_decimal_barrier_equality_is_already_resolved(self):
        c = table()
        e = replace(event(c, "bearish"), lower=19240.7, upper=19242.7)
        c.data["atr_14_sma"][1] = 35.2
        c.data["close"][1] = 19260.3
        self.assertLess((19260.3 - 19242.7) / 35.2, 0.5)
        self.assertEqual(ds.scenario_outcome(e, c, 1, CFG)[:2], ("already_resolved", None))

    def sideways(self, count=60):
        c = table(count)
        put(c, 1)
        for i in range(2, len(c)):
            put(c, i, o=101, h=102, l=100, c=101)
        return c

    def test_initial_consolidation_then_bounce(self):
        c = self.sideways()
        put(c, 6, o=101, h=103.5, l=100, c=103.5)
        self.assertEqual(ds.scenario_outcome(event(c), c, 1, CFG), ("labeled", "bounce", 6))

    def test_initial_consolidation_then_breakout(self):
        c = self.sideways()
        put(c, 6, o=101, h=102, l=99, c=99)
        self.assertEqual(ds.scenario_outcome(event(c), c, 1, CFG), ("labeled", "breakout", 6))

    def test_first_close_wins_not_later_reversal(self):
        c = self.sideways()
        put(c, 2, h=104, c=104)
        put(c, 3, o=101, l=98, c=98)
        self.assertEqual(ds.scenario_outcome(event(c), c, 1, CFG)[1:], ("bounce", 2))

    def test_full_horizon_consolidation(self):
        c = self.sideways()
        self.assertEqual(ds.scenario_outcome(event(c), c, 1, CFG), ("labeled", "consolidation", 13))

    def test_future_only_horizon_and_exact_last_bar(self):
        c = self.sideways()
        put(c, 13, h=104, c=104)
        self.assertEqual(ds.scenario_outcome(event(c), c, 1, CFG)[1:], ("bounce", 13))
        c = self.sideways()
        put(c, 14, h=104, c=104)
        self.assertEqual(ds.scenario_outcome(event(c), c, 1, CFG)[1], "consolidation")

    def test_touch_already_resolved_is_excluded(self):
        c = self.sideways()
        put(c, 1, h=104, c=104)
        self.assertEqual(ds.scenario_outcome(event(c), c, 1, CFG)[:2], ("already_resolved", None))

    def test_short_horizon_is_censored_not_consolidation(self):
        c = self.sideways(8)
        self.assertEqual(ds.scenario_outcome(event(c), c, 1, CFG)[:2], ("censored_response", None))

    def test_early_direction_before_eof_is_labelable(self):
        c = self.sideways(4)
        put(c, 3, h=104, c=104)
        self.assertEqual(ds.scenario_outcome(event(c), c, 1, CFG)[1:], ("bounce", 3))

    def test_insufficient_sideways_closes_unresolved(self):
        c = self.sideways()
        for i in range(2, 14):
            put(c, i, o=103, h=103, l=102, c=103)
        self.assertEqual(ds.scenario_outcome(event(c), c, 1, CFG)[:2], ("unresolved", None))

    def test_excess_wick_span_not_consolidation_or_intrabar_outcome(self):
        c = self.sideways()
        put(c, 5, o=101, h=110, l=90, c=101)
        self.assertEqual(ds.scenario_outcome(event(c), c, 1, CFG)[:2], ("unresolved", None))

    def test_bearish_threshold_equality(self):
        c = self.sideways()
        put(c, 2, o=100, h=101, l=98.5, c=98.5)
        self.assertEqual(ds.scenario_outcome(event(c, "bearish"), c, 1, CFG)[1], "bounce")


class TimeAndIdentityTests(unittest.TestCase):
    def test_b_atr_and_features_known_at_touch_not_future(self):
        c = table()
        put(c, 1)
        put(c, 2, h=104, c=104)
        e = event(c)
        c.data["atr_14_sma"][2] = 100
        rows = list(ds.iter_samples([e], c, {(e.registry_id, e.available): 1}, CFG))
        b = next(r for r in rows if r["model"] == "scenario")
        self.assertEqual((b["decision_atr"], b["label"], b["decision_at_utc"]), (2, "bounce", ds.iso(c.available(1))))

    def test_b_wait_limit_and_missing_atr(self):
        c = table()
        e = event(c)
        for touch, atr, status in ((21, 2, "touch_after_wait"), (1, math.nan, "invalid_decision_atr")):
            put(c, touch)
            c.data["atr_14_sma"][touch] = atr
            rows = list(ds.iter_samples([e], c, {(e.registry_id, e.available): touch}, CFG))
            b = next(r for r in rows if r["model"] == "scenario")
            self.assertEqual(b["label_status"], status)
            self.assertIsNone(b["label"])

    def test_a_frozen_target_ignores_later_boundary_update(self):
        c = table()
        put(c, 3)
        put(c, 4, h=105, l=103, c=105)
        e = event(c)
        update = replace(e, available=c.available(1), kind="update", sequence=2, source_row=3, lower=90, upper=95)
        a = [r for r in ds.iter_samples([e, update], c, {}, CFG) if r["model"] == "strong_zone"]
        self.assertEqual(len(a), 1)
        self.assertEqual((a[0]["label"], a[0]["lower_price"], a[0]["upper_price"]), (1, 100, 102))

    def test_same_time_order_is_deterministic(self):
        c = table()
        e = event(c)
        other = replace(e, registry_id=e.registry_id + "2", source_id="s2", source_row=3)
        self.assertEqual(list(ds.iter_samples([e, other], c, {}, CFG)), list(ds.iter_samples([other, e], c, {}, CFG)))

    def test_b_observations_ordered_by_touch_not_version_creation(self):
        c = table()
        e = event(c)
        other = replace(e, registry_id=e.registry_id + "2", source_id="s2", source_row=3, available=c.available(1))
        put(c, 3)
        put(c, 5)
        touches = {(e.registry_id, e.available): 5, (other.registry_id, other.available): 3}
        rows = [r for r in ds.iter_samples([e, other], c, touches, CFG) if r["model"] == "scenario"]
        self.assertEqual([r["decision_at_utc"] for r in rows], [ds.iso(c.available(3)), ds.iso(c.available(5))])

    def test_split_boundary_purge_and_embargo(self):
        creation = ds.utc("2023-12-01T00:00:00Z")
        decision = ds.utc("2023-12-31T00:00:00Z")
        self.assertEqual(ds.split_status(decision, ds.VALIDATION_START, creation, CFG)[1], "purged_label_window")
        self.assertEqual(ds.split_status(ds.VALIDATION_START, ds.VALIDATION_START, ds.VALIDATION_START, CFG)[1], "embargo")
        after = ds.VALIDATION_START + ds.timedelta(hours=176)
        self.assertEqual(ds.split_status(after, after, ds.VALIDATION_START, CFG), ("validation", "eligible"))

    def test_identity_cannot_move_between_partitions(self):
        after = ds.TEST_START + ds.timedelta(days=20)
        self.assertEqual(ds.split_status(after, after, START, CFG)[1], "identity_other_partition")

    def test_test_period_end_excluded(self):
        self.assertIsNone(ds.partition(ds.TEST_END)[0])

    def test_updates_keep_one_a_identity_and_two_b_versions(self):
        c = table()
        e = event(c)
        update = replace(e, available=c.available(2), kind="update", sequence=2, source_row=3, evidence_count=3, lower=99)
        rows = list(ds.iter_samples([e, update], c, {}, CFG))
        self.assertEqual([r["model"] for r in rows].count("strong_zone"), 1)
        self.assertEqual([r["model"] for r in rows].count("scenario"), 2)
        self.assertEqual({r["registry_zone_id"] for r in rows}, {e.registry_id})

    def test_body_confirmation_direction_and_frozen_geometry(self):
        c = table()
        e = replace(event(c), family="imbalance", origin="displacement_body", direction="bullish")
        confirm = replace(e, origin="bullish_fvg", kind="fvg_confirmation", available=c.available(1), lower=103, upper=104)
        rows = list(ds.iter_samples([e, confirm], c, {}, CFG))
        original = next(r for r in rows if r["model"] == "strong_zone")
        self.assertEqual((original["lower_price"], original["upper_price"]), (100, 102))
        self.assertEqual(lifecycle.direction_for(e), "bullish")

    def test_features_invariant_to_changed_future(self):
        c = table()
        put(c, 1)
        e = event(c)
        row = next(ds.iter_samples([e], c, {}, CFG))
        for field in ds.CANDLE_FIELDS:
            c.data[field][10] = 123
        changed = next(ds.iter_samples([e], c, {}, CFG))
        self.assertEqual({k: row[k] for k in ds.MODEL_FEATURES["strong_zone"]},
                         {k: changed[k] for k in ds.MODEL_FEATURES["strong_zone"]})

    def test_missing_atr_never_imputed(self):
        for atr in (math.nan, 0, -1):
            c = table()
            c.data["atr_14_sma"][0] = atr
            row = next(ds.iter_samples([event(c)], c, {}, CFG))
            self.assertEqual(row["label_status"], "invalid_decision_atr")
            self.assertIsNone(row["label"])

    def test_neutral_volume_not_supervised(self):
        c = table()
        e = replace(event(c), family="volume_derived", origin=None)
        self.assertEqual(list(ds.iter_samples([e], c, {}, CFG)), [])

    def test_duplicate_version_rejected(self):
        c = table()
        e = event(c)
        with self.assertRaises(ValueError):
            ds.version_index([e, e], c)

    def test_feature_allowlist_excludes_audit_and_targets(self):
        forbidden = {"label", "touch_at_utc", "observation_end_utc", "registry_zone_id", "source_row", "split"}
        for features in ds.MODEL_FEATURES.values():
            self.assertFalse(set(features) & forbidden)

    def test_feature_allowlists_match_approved_phase8_report(self):
        path = Path(__file__).resolve().parents[1] / "reports/phase8_summary.json"
        approved = json.loads(path.read_text())["model_features"]
        for model, count in (("strong_zone", 14), ("scenario", 16)):
            features = ds.MODEL_FEATURES[model]
            self.assertEqual(list(features), approved[model])
            self.assertEqual(len(features), count)
            self.assertEqual(len(set(features)), count)


class OverlapTests(unittest.TestCase):
    def other(self, e, identity="r", family="reaction", **kwargs):
        if family == "reaction":
            kwargs.setdefault("origin", "bullish_rejection_cluster")
        return replace(e, registry_id=f"BTCUSDT:15m:{family}:{identity}", source_id=identity,
                       family=family, **kwargs)

    def counts(self, features):
        return tuple(features[name] for name in ds.OVERLAP_FEATURES)

    def test_future_zone_and_exact_availability(self):
        c = table()
        e = event(c)
        future = self.other(e, available=c.available(3))
        index = ds.OverlapEvidence([future, e])
        self.assertEqual(self.counts(index.features(e, c.available(2))), (0, 0))
        self.assertEqual(self.counts(index.features(e, c.available(3))), (1, 1))

    def test_complete_same_time_batch_is_order_independent(self):
        c = table()
        e = event(c)
        r = self.other(e)
        for records in ([e, r], [r, e]):
            self.assertEqual(self.counts(ds.OverlapEvidence(records).features(e, e.available)), (1, 1))

    def test_updates_use_latest_known_geometry(self):
        c = table()
        e = event(c)
        r = self.other(e)
        update = replace(r, lower=200, upper=202, available=c.available(3), kind="update", sequence=2)
        index = ds.OverlapEvidence([update, r, e])
        self.assertEqual(self.counts(index.features(e, c.available(2))), (1, 1))
        self.assertEqual(self.counts(index.features(e, c.available(3))), (0, 0))

    def test_future_fvg_confirmation_boundaries_not_used_early(self):
        c = table()
        e = event(c)
        body = self.other(e, family="imbalance", origin="displacement_body", direction="bullish", lower=90, upper=92)
        confirm = replace(body, lower=100, upper=102, origin="bullish_fvg", kind="fvg_confirmation", available=c.available(1))
        index = ds.OverlapEvidence([confirm, body, e])
        self.assertEqual(self.counts(index.features(e, c.available(0))), (0, 0))
        self.assertEqual(self.counts(index.features(e, c.available(1))), (1, 1))

    def test_same_family_and_self_are_excluded_from_both_counts(self):
        c = table()
        e = event(c)
        same_family = self.other(e, family="structural")
        self.assertEqual(self.counts(ds.OverlapEvidence([e, same_family]).features(e, e.available)), (0, 0))

    def test_distinct_candidates_and_families_are_different_counts(self):
        c = table()
        e = event(c)
        records = [e, self.other(e, "r1"), self.other(e, "r2"), self.other(e, "i", family="imbalance")]
        self.assertEqual(self.counts(ds.OverlapEvidence(records).features(e, e.available)), (2, 3))

    def test_boundary_touch_not_positive_width_overlap(self):
        c = table()
        e = event(c)
        r = self.other(e, lower=e.upper, upper=e.upper + 2)
        self.assertEqual(self.counts(ds.OverlapEvidence([e, r]).features(e, e.available)), (0, 0))

    def test_volume_expires_at_empty_profile_and_after_coverage(self):
        c = table()
        e = event(c)
        v = self.other(e, family="volume_derived", origin=None)
        for last in (c.available(0), c.available(20)):
            clock = registry.VolumeClock(c.available(0), last, 2 * INTERVAL, INTERVAL)
            index = ds.OverlapEvidence([e, v], clock=clock)
            self.assertEqual(self.counts(index.features(e, c.available(1))), (1, 1))
            self.assertEqual(self.counts(index.features(e, c.available(2))), (0, 0))

    def test_volume_family_counted_once_and_old_profile_replaced(self):
        c = table()
        e = event(c)
        a = self.other(e, "v1", family="volume_derived", origin=None)
        b = self.other(e, "v2", family="volume_derived", origin=None)
        later = self.other(e, "v3", family="volume_derived", origin=None, available=c.available(2))
        clock = registry.VolumeClock(c.available(0), c.available(20), 2 * INTERVAL, INTERVAL)
        index = ds.OverlapEvidence([e, a, b, later], clock=clock)
        self.assertEqual(self.counts(index.features(e, c.available(1))), (1, 2))
        self.assertEqual(self.counts(index.features(e, c.available(2))), (1, 1))

    def test_missing_volume_calendar_fails_instead_of_guessing(self):
        c = table()
        e = event(c)
        v = self.other(e, family="volume_derived", origin=None)
        with self.assertRaises(ValueError):
            ds.OverlapEvidence([e, v]).features(e, e.available)

    def test_recency_is_bounded_and_equality_included(self):
        c = table()
        e = event(c)
        r = self.other(e)
        config = registry.Config(overlap_15m_days=1 / 96)
        index = ds.OverlapEvidence([e, r], config=config)
        self.assertEqual(self.counts(index.features(e, c.available(1))), (1, 1))
        self.assertEqual(self.counts(index.features(e, c.available(2))), (0, 0))

    def test_known_retirement_excludes_only_that_version(self):
        c = table()
        e = event(c)
        r = self.other(e)
        update = replace(r, available=c.available(3), kind="update", sequence=2)
        retired = {(r.registry_id, r.available): c.available(2)}
        index = ds.OverlapEvidence([e, r, update], retired=retired)
        self.assertEqual(self.counts(index.features(e, c.available(1))), (1, 1))
        self.assertEqual(self.counts(index.features(e, c.available(2))), (0, 0))
        self.assertEqual(self.counts(index.features(e, c.available(3))), (1, 1))

    def test_frozen_sample_excludes_own_updated_identity(self):
        c = table()
        e = event(c)
        update = replace(e, available=c.available(2), kind="update", sequence=2)
        index = ds.OverlapEvidence([e, update])
        self.assertEqual(self.counts(index.features(e, update.available)), (0, 0))

    def test_all_predictors_invariant_to_future_events_and_retirements(self):
        c = table()
        put(c, 1)
        e = event(c)
        future = self.other(e, available=c.available(10))
        a = next(ds.iter_samples([e], c, {}, CFG))
        b = next(ds.iter_samples([e, future], c, {}, CFG,
                                 retired={(future.registry_id, future.available): c.available(11)}))
        self.assertEqual({k: a[k] for k in ds.MODEL_FEATURES["strong_zone"]},
                         {k: b[k] for k in ds.MODEL_FEATURES["strong_zone"]})

    def test_b_uses_touch_decision_not_creation_overlap(self):
        c = table()
        put(c, 2)
        e = event(c)
        r = self.other(e, available=c.available(2))
        rows = list(ds.iter_samples([e, r], c, {(e.registry_id, e.available): 2}, CFG))
        a = next(row for row in rows if row["model"] == "strong_zone" and row["registry_zone_id"] == e.registry_id)
        b = next(row for row in rows if row["model"] == "scenario" and row["registry_zone_id"] == e.registry_id)
        self.assertEqual(self.counts(a), (0, 0))
        self.assertEqual(self.counts(b), (1, 1))


class FileTests(unittest.TestCase):
    def test_real_joins_read_only_and_existing_output_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            original = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*.csv")}
            result = ds.process_dataset(root, "BTCUSDT", "15m")
            self.assertEqual(result["validation"]["status"], "PASS")
            self.assertEqual(result["validation"]["outputs"]["strong_zone"]["rows"], 1)
            self.assertEqual(result["validation"]["outputs"]["scenario"]["rows"], 1)
            for p, digest in original.items():
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), digest)
            with self.assertRaises(FileExistsError):
                ds.process_dataset(root, "BTCUSDT", "15m")
            before = {p: p.read_bytes() for p in ds.output_paths(root, "BTCUSDT", "15m").values()}
            ds.process_dataset(root, "BTCUSDT", "15m", overwrite=True)
            for p, content in before.items():
                self.assertEqual(p.read_bytes(), content)
            for p, digest in original.items():
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), digest)

    def test_deterministic_serialization(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            outputs = []
            for name in (a, b):
                root = Path(name)
                fixture(root)
                ds.process_dataset(root, "BTCUSDT", "15m")
                outputs.append([p.read_bytes() for p in ds.output_paths(root, "BTCUSDT", "15m").values()])
            self.assertEqual(*outputs)

    def test_missing_duplicate_unordered_and_incomplete_candles_rejected(self):
        for change in ("gap", "duplicate", "unordered", "incomplete", "nonfinite"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                fixture(root)
                path = root / "data/features/15m/BTCUSDT_15m_features.csv"
                with path.open(newline="") as f:
                    reader = csv.DictReader(f)
                    fields, rows = reader.fieldnames, list(reader)
                if change == "gap": rows.pop(2)
                elif change == "duplicate": rows[2] = rows[1]
                elif change == "unordered": rows[1], rows[2] = rows[2], rows[1]
                elif change == "incomplete": rows[2]["available_at_utc"] = rows[2]["timestamp_utc"]
                else: rows[2]["close"] = "inf"
                write_csv(path, fields, rows)
                with self.assertRaises(ValueError): ds.read_candles(path, "15m")

    def test_invalid_source_join_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            path = root / "data/zones/unified/BTCUSDT_15m.csv"
            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            rows[0]["lower_price"] = "99"
            write_csv(path, registry.COLUMNS, rows)
            with self.assertRaises(ValueError): ds.process_dataset(root, "BTCUSDT", "15m")

    def test_touch_cannot_use_version_created_at_same_close(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            c, e = fixture(root)
            path = root / "data/zones/lifecycle/BTCUSDT_15m.csv"
            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            rows[0]["candle_time_utc"] = ds.iso(c.start)
            rows[0]["event_at_utc"] = ds.iso(e.available)
            write_csv(path, lifecycle.COLUMNS, rows[:1])
            with self.assertRaises(ValueError):
                ds.read_touches(path, {(e.registry_id, e.available): e}, {}, c)

    def test_summary_requires_six_audits_and_passing_tests(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                ds.write_summary(Path(temp), [], {"successful": True, "tests_run": 1})


if __name__ == "__main__":
    unittest.main()
