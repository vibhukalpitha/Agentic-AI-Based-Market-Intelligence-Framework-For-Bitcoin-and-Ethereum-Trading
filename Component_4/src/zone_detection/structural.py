"""Chronological ATR-aware structural candidate groups and auditable zone events.

Run from the repository root:
    python -B Component_4/src/zone_detection/structural.py

Swing audit and zone event CSVs preserve confirmed evidence and history. Event
rows are immutable versions appended in confirmation order during each batch run.
--overwrite explicitly regenerates these derived files; this is not a resumable
incremental writer. Matching expiry only archives the matching index, not zones.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import heapq
import json
import sys
import unittest
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zone_detection.swings import (
    INTERVALS, SWING_COLUMNS, Swing, check_header, detect_swings, iso,
    positive, read_candles,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
SYMBOLS = ("BTCUSDT", "ETHUSDT")
CLUSTER_METHOD = "chronological_atr_bounded_diameter_v1"
ZONE_COLUMNS = (
    "zone_id", "cluster_id", "symbol", "source_timeframe", "zone_family",
    "origin_type", "lower_price", "upper_price", "formation_start_utc",
    "confirmed_at_utc", "available_at_utc", "confirmed_swing_count",
    "cluster_reference_atr", "cluster_method", "candidate_status",
    "observed_min_price", "observed_max_price", "minimum_member_atr",
    "diameter_limit", "padding_price", "latest_evidence_at_utc",
)
EVENT_COLUMNS = ("event_type", "event_at_utc", "event_sequence", "evidence_swing_id") + ZONE_COLUMNS

# Static methodology text for the report, separate from CLI execution.
RULES = {
    "swings": "Strict unique extrema; high before low for same pivot; confirming candle availability and ATR",
    "reference_atr": "Frozen from first swing confirmation ATR",
    "matching_tolerance": "cluster_tolerance_atr * min(first_reference_atr, new_confirmation_atr)",
    "compactness": "Updated diameter <= cluster_tolerance_atr * minimum confirmation ATR of all members",
    "ties": "Nearest existing midpoint, earliest first confirmation, lexicographic cluster ID",
    "eligibility": "Elapsed confirmation time since latest evidence <= timeframe lookback; expiry is not invalidation",
    "boundaries": "[min_price-padding, max_price+padding]; padding=min(padding_atr*minimum_member_atr, padding_price_cap_fraction*min_price)",
    "width_cap": "positive finite boundaries and width <= max_zone_width_fraction * observed midpoint",
    "invalid_atr": "Keep confirmed swing in audit output; exclude it from grouping; never fill ATR",
    "availability": "confirmed_at_utc is original zone creation; available_at_utc is availability of this version",
    "history": "Zone event rows are immutable states at event_at_utc",
    "scope": "Historical candidates only; no lifecycle or present-day validity assessment",
}


@dataclass(frozen=True)
class Config:
    left_bars: int = 3
    right_bars: int = 3
    cluster_tolerance_atr: float = 0.5
    min_confirmed_swings: int = 2
    lookback_15m_days: float = 30
    lookback_1h_days: float = 90
    lookback_4h_days: float = 180
    padding_atr: float = 0.1
    padding_price_cap_fraction: float = 0.01
    max_zone_width_fraction: float = 0.05

    def __post_init__(self) -> None:
        if self.left_bars < 1 or self.right_bars < 1 or self.min_confirmed_swings < 2:
            raise ValueError("Require positive swing windows and at least two swings per zone")
        for name, value in asdict(self).items():
            if not positive(value):
                raise ValueError(f"{name} must be finite and positive")
        if self.padding_price_cap_fraction >= 1 or self.max_zone_width_fraction >= 1:
            raise ValueError("Price padding and width fractions must be less than one")

    def lookback(self, timeframe: str) -> timedelta:
        return timedelta(days=getattr(self, f"lookback_{timeframe}_days"))


@dataclass
class Group:
    cluster_id: str
    symbol: str
    timeframe: str
    kind: str
    minimum: float
    maximum: float
    reference_atr: float
    minimum_atr: float
    diameter_limit: float
    formation_start: datetime
    first_confirmation: datetime
    latest_confirmation: datetime
    count: int = 1
    zone_confirmed_at: datetime | None = None


def boundaries(minimum: float, maximum: float, minimum_atr: float,
               config: Config) -> tuple[float, float, float]:
    padding = min(config.padding_atr * minimum_atr,
                  config.padding_price_cap_fraction * minimum)
    lower, upper = minimum - padding, maximum + padding
    midpoint = minimum + (maximum - minimum) / 2
    if (not all(positive(value) for value in (lower, upper, padding))
            or upper <= lower
            or upper - lower > config.max_zone_width_fraction * midpoint):
        raise ValueError("Zone boundaries violate positive width or price width cap")
    return lower, upper, padding


class StructuralDetector:
    """Price-indexed eligible groups with an elapsed-time expiry heap.

    First-evidence ATR stays frozen; the diameter cap can only shrink with ATR.
    Each unique swing joins at most one group. Expiry does not invalidate zones.
    """

    def __init__(self, symbol: str, timeframe: str, config: Config | None = None):
        self.symbol, self.timeframe = symbol, timeframe
        self.config = config or Config()
        self.lookback = self.config.lookback(timeframe)
        self.groups: dict[str, Group] = {}
        self.active: dict[str, Group] = {}
        self.price_index: dict[str, list[tuple[float, str]]] = {"high": [], "low": []}
        self.expiry: list[tuple[datetime, str]] = []
        self.seen: set[str] = set()
        self.latest_event: datetime | None = None
        self.stats: Counter = Counter()

    def _remove_price(self, group: Group) -> None:
        index = self.price_index[group.kind]
        index.pop(bisect.bisect_left(index, (group.minimum, group.cluster_id)))

    def _expire(self, now: datetime) -> None:
        cutoff = now - self.lookback
        while self.expiry and self.expiry[0][0] < cutoff:
            timestamp, group_id = heapq.heappop(self.expiry)
            group = self.active.get(group_id)
            if group is not None and group.latest_confirmation == timestamp:
                self._remove_price(group)
                del self.active[group_id]
                self.stats["groups_archived_from_matching"] += 1

    def archive_matching_as_of(self, now: datetime) -> None:
        self._expire(now)

    def state(self, group: Group) -> dict:
        lower, upper, padding = boundaries(group.minimum, group.maximum,
                                            group.minimum_atr, self.config)
        available_at = iso(group.latest_confirmation)
        return {
            "zone_id": group.cluster_id.replace("group_", "zone_", 1),
            "cluster_id": group.cluster_id, "symbol": group.symbol,
            "source_timeframe": group.timeframe, "zone_family": "structural",
            "origin_type": f"swing_{group.kind}_cluster",
            "lower_price": lower, "upper_price": upper,
            "formation_start_utc": iso(group.formation_start),
            "confirmed_at_utc": iso(group.zone_confirmed_at), "available_at_utc": available_at,
            "confirmed_swing_count": group.count, "cluster_reference_atr": group.reference_atr,
            "cluster_method": CLUSTER_METHOD, "candidate_status": "candidate",
            "observed_min_price": group.minimum, "observed_max_price": group.maximum,
            "minimum_member_atr": group.minimum_atr, "diameter_limit": group.diameter_limit,
            "padding_price": padding, "latest_evidence_at_utc": available_at,
        }

    def consume(self, swing: Swing) -> dict | None:
        if swing.symbol != self.symbol or swing.source_timeframe != self.timeframe:
            raise ValueError("Swing belongs to a different asset/timeframe")
        if swing.swing_type not in self.price_index or not positive(swing.swing_price):
            raise ValueError("Invalid swing type or price")
        if (swing.confirmed_at.tzinfo is None or swing.swing_time.tzinfo is None
                or swing.confirmed_at <= swing.swing_time):
            raise ValueError("Invalid swing confirmation timestamps")
        if self.latest_event is not None and swing.confirmed_at < self.latest_event:
            raise ValueError("Swings must arrive in confirmation order")
        if swing.swing_id in self.seen:
            raise ValueError(f"Duplicate swing evidence: {swing.swing_id}")
        self.seen.add(swing.swing_id)
        self.latest_event = swing.confirmed_at
        self._expire(swing.confirmed_at)
        if not positive(swing.atr_at_confirmation):
            self.stats["swings_skipped_unusable_atr"] += 1
            return None
        atr, price = swing.atr_at_confirmation, swing.swing_price
        index = self.price_index[swing.swing_type]
        radius = self.config.cluster_tolerance_atr * atr
        start = bisect.bisect_left(index, (price - radius, ""))
        stop = bisect.bisect_right(index, (price + radius, chr(0x10FFFF)))
        matches = []
        for _, group_id in index[start:stop]:
            group = self.active[group_id]
            self.stats["matching_groups_examined"] += 1
            minimum, maximum = min(group.minimum, price), max(group.maximum, price)
            limit = min(group.diameter_limit,
                        self.config.cluster_tolerance_atr * min(group.reference_atr, atr))
            if maximum - minimum > limit:
                continue
            try:
                boundaries(minimum, maximum, min(group.minimum_atr, atr), self.config)
            except ValueError:
                continue
            midpoint = group.minimum + (group.maximum - group.minimum) / 2
            matches.append((abs(price - midpoint), group.first_confirmation, group_id))

        if matches:
            group = self.active[min(matches)[2]]
            self._remove_price(group)
            group.minimum, group.maximum = min(group.minimum, price), max(group.maximum, price)
            group.minimum_atr = min(group.minimum_atr, atr)
            group.diameter_limit = min(group.diameter_limit,
                                      self.config.cluster_tolerance_atr * group.minimum_atr)
            group.formation_start = min(group.formation_start, swing.swing_time)
            group.latest_confirmation = swing.confirmed_at
            group.count += 1
        else:
            # Even a provisional group's eventual padding must be representable.
            try:
                boundaries(price, price, atr, self.config)
            except ValueError:
                self.stats["swings_skipped_invalid_boundaries"] += 1
                return None
            group_id = f"group_{swing.swing_id}"
            group = Group(group_id, self.symbol, self.timeframe, swing.swing_type,
                          price, price, atr, atr, self.config.cluster_tolerance_atr * atr,
                          swing.swing_time, swing.confirmed_at, swing.confirmed_at)
            self.groups[group_id] = group
            self.active[group_id] = group

        bisect.insort(index, (group.minimum, group.cluster_id))
        heapq.heappush(self.expiry, (group.latest_confirmation, group.cluster_id))
        self.stats["swings_grouped"] += 1
        if group.count < self.config.min_confirmed_swings:
            return None
        event_type = "update"
        if group.zone_confirmed_at is None:
            group.zone_confirmed_at = swing.confirmed_at
            event_type = "create"
        self.stats[f"zone_{event_type}_count"] += 1
        return {
            "event_type": event_type, "event_at_utc": iso(swing.confirmed_at),
            "event_sequence": group.count - self.config.min_confirmed_swings + 1,
            "evidence_swing_id": swing.swing_id, **self.state(group),
        }

    def snapshots(self, as_of: datetime) -> list[dict]:
        return [dict(self.state(group), snapshot_as_of_utc=iso(as_of))
                for group in self.groups.values() if group.zone_confirmed_at is not None]


def output_paths(root: Path, symbol: str, timeframe: str) -> dict[str, Path]:
    folder = root / "data" / "zones" / "structural"
    stem = f"{symbol}_{timeframe}"
    return {"swings": folder / f"{stem}_swings.csv",
            "zone_events": folder / f"{stem}_zone_events.csv"}


def preflight(root: Path, overwrite: bool) -> list[tuple[str, str, Path, dict]]:
    jobs = []
    targets = [root / "reports" / "zone_detection" / "structural_summary.json"]
    for timeframe in INTERVALS:
        for symbol in SYMBOLS:
            source = root / "data" / "features" / timeframe / f"{symbol}_{timeframe}_features.csv"
            check_header(source)
            outputs = output_paths(root, symbol, timeframe)
            targets.extend(outputs.values())
            jobs.append((symbol, timeframe, source, outputs))
    existing = [str(path) for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Outputs already exist; use --overwrite to regenerate: " + ", ".join(existing))
    return jobs


def process_dataset(symbol: str, timeframe: str, source: Path, outputs: dict[str, Path],
                    config: Config, overwrite: bool = False) -> dict:
    stats = Counter()
    detector = StructuralDetector(symbol, timeframe, config)
    first_open = last_available = None

    def tracked_candles():
        nonlocal first_open, last_available
        for candle in read_candles(source, timeframe, stats):
            if first_open is None:
                first_open = candle.timestamp
            last_available = candle.available_at
            yield candle

    with ExitStack() as stack:
        writers = {}
        for name, columns in (("swings", SWING_COLUMNS), ("zone_events", EVENT_COLUMNS)):
            path = outputs[name]
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(path.open("w" if overwrite else "x", newline="", encoding="utf-8"))
            writers[name] = csv.DictWriter(handle, fieldnames=columns)
            writers[name].writeheader()
        for swing in detect_swings(tracked_candles(), symbol, timeframe,
                                   config.left_bars, config.right_bars):
            stats[f"confirmed_swing_{swing.swing_type}_count"] += 1
            writers["swings"].writerow(swing.record())
            event = detector.consume(swing)
            if event is not None:
                writers["zone_events"].writerow(event)
        if last_available is None:
            raise ValueError(f"Empty feature input: {source}")
        detector.archive_matching_as_of(last_available)

    warnings = [f"{name}: {stats[name]}; no ATR imputation was used"
                for name in ("missing_atr_rows", "invalid_atr_rows") if stats[name]]
    warnings += [f"{name}: {detector.stats[name]}; swings remain in the swing audit CSV"
                 for name in ("swings_skipped_unusable_atr", "swings_skipped_invalid_boundaries")
                 if detector.stats[name]]
    return {
        "symbol": symbol, "source_timeframe": timeframe, "input_path": str(source),
        **{name: stats[name] for name in ("input_rows", "confirmed_swing_high_count",
                                        "confirmed_swing_low_count", "missing_atr_rows", "invalid_atr_rows")},
        "provisional_cluster_count": sum(g.zone_confirmed_at is None for g in detector.groups.values()),
        "total_cluster_count": len(detector.groups),
        "candidate_zones_created": detector.stats["zone_create_count"],
        "zone_update_count": detector.stats["zone_update_count"],
        "matching_eligible_groups_at_end": len(detector.active),
        **{name: detector.stats[name] for name in ("matching_groups_examined",
             "groups_archived_from_matching", "swings_grouped", "swings_skipped_unusable_atr")},
        "lookback_days": config.lookback(timeframe).total_seconds() / 86400,
        "timestamp_checks": {"status": "PASS", "rows_checked": stats["timestamp_rows_checked"],
                             "first_open_utc": iso(first_open), "last_available_utc": iso(last_available),
                             "checks": ["aware UTC grid", "exact interval spacing", "close-time availability"]},
        "price_boundary_checks": {"status": "PASS", "ohlc_rows_checked": stats["ohlc_rows_checked"],
                                  "event_states_checked": detector.stats["zone_create_count"] + detector.stats["zone_update_count"],
                                  "rejected_swings": detector.stats["swings_skipped_invalid_boundaries"]},
        "output_paths": {name: str(path) for name, path in outputs.items()},
        "warnings": warnings,
        "anomalies": {"invalid_atr_rows": stats["invalid_atr_rows"],
                      "invalid_boundaries_skipped": detector.stats["swings_skipped_invalid_boundaries"]},
    }


def run_tests(root: Path) -> dict:
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    return {"tests_run": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped),
            "successful": result.testsRun > 0 and result.wasSuccessful()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing Phase 2 outputs and summary")
    for name, value in asdict(Config()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    args = vars(parser.parse_args(argv))
    overwrite = args.pop("overwrite")
    try:
        config = Config(**args)
        jobs = preflight(PROJECT_DIR, overwrite)
        tests = run_tests(PROJECT_DIR)
        if not tests["successful"]:
            print("FAIL: tests did not pass; no zone outputs generated", file=sys.stderr)
            return 1
        results = []
        for symbol, timeframe, source, outputs in jobs:
            result = process_dataset(symbol, timeframe, source, outputs, config, overwrite)
            results.append(result)
            print(f"PASS {symbol} {timeframe}: {result['input_rows']} candles, "
                  f"{result['confirmed_swing_high_count']} highs, {result['confirmed_swing_low_count']} lows, "
                  f"{result['candidate_zones_created']} zones, {result['zone_update_count']} updates", flush=True)
        summary = {
            "generated_at_utc": iso(datetime.now(timezone.utc)), "phase": 2,
            "parameters": asdict(config), "parameter_status": "provisional research assumptions; not optimized",
            "rules": RULES.copy(),
            "schemas": {"swings": SWING_COLUMNS, "zone_events": EVENT_COLUMNS},
            "tests": tests, "datasets": results,
            "warnings": [f"{r['symbol']} {r['source_timeframe']}: {w}" for r in results for w in r["warnings"]],
        }
        report = PROJECT_DIR / "reports" / "zone_detection" / "structural_summary.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("w" if overwrite else "x", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(f"Summary: {report}")
        return 0
    except (OSError, ValueError, OverflowError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
