"""Unified chronological candidate history with as-of geometric overlap metadata.

No source detector is rerun. Structural snapshots and swing CSVs are not inputs.
Overlap-search eligibility is NOT zone validity or survival. Volume observations
are correlated OHLCV estimates, not independent reactions or trade-level volume.

Run: python -B Component_4/src/zone_detection/registry.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import unittest
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from itertools import groupby
from pathlib import Path
from typing import Iterable, Iterator

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zone_detection.swings import INTERVALS, iso, positive, utc


PROJECT_DIR = Path(__file__).resolve().parents[2]
SYMBOLS = ("BTCUSDT", "ETHUSDT")
FOLDERS = {"structural": "structural", "imbalance": "imbalance",
           "reaction": "reaction", "volume_derived": "volume"}
FAMILIES = tuple(FOLDERS)
PRIORITY = {"create": 0, "update": 1, "fvg_confirmation": 2, "profile_observation": 3}
COMMON_SOURCE = ("zone_id", "symbol", "source_timeframe", "zone_family",
                 "lower_price", "upper_price", "available_at_utc", "candidate_status")
CLUSTER_SOURCE = ("origin_type", "formation_start_utc", "confirmed_at_utc", "cluster_reference_atr",
                  "event_type", "event_at_utc", "event_sequence")
SOURCE_COLUMNS = {
    "structural": COMMON_SOURCE + CLUSTER_SOURCE + ("confirmed_swing_count",),
    "reaction": COMMON_SOURCE + CLUSTER_SOURCE + ("independent_rejection_count",),
    "imbalance": COMMON_SOURCE + (
        "formation_start_utc", "imbalance_type", "direction", "event_type", "candidate_created_at_utc",
        "c1_time_utc", "c2_time_utc", "c3_time_utc", "reference_candle_time_utc", "reference_atr",
        "reference_atr_available_at_utc", "aligned_displacement"),
    "volume_derived": COMMON_SOURCE + ("profile_start_utc", "profile_end_utc", "profile_id",
                                       "candidate_rank", "reference_atr", "method"),
}
COLUMNS = (
    "registry_zone_id", "source_zone_id", "symbol", "source_timeframe", "zone_family", "origin_type",
    "lower_price", "upper_price", "formation_start_utc", "confirmed_at_utc", "available_at_utc",
    "event_type", "candidate_status", "source_reference_atr", "overlapping_candidate_count",
    "overlapping_family_count", "overlap_family_flags", "source_event_sequence", "source_file", "source_row",
)


@dataclass(frozen=True)
class Config:
    overlap_15m_days: float = 30
    overlap_1h_days: float = 90
    overlap_4h_days: float = 180

    def __post_init__(self) -> None:
        if not all(positive(value) for value in asdict(self).values()):
            raise ValueError("Overlap recency settings must be finite and positive")

    def recency(self, timeframe: str) -> timedelta:
        return timedelta(days=getattr(self, f"overlap_{timeframe}_days"))


@dataclass(frozen=True)
class VolumeClock:
    first_profile_at: datetime
    last_input_available: datetime
    stride: timedelta
    window: timedelta

    def __post_init__(self) -> None:
        if (self.first_profile_at.tzinfo is None or self.last_input_available.tzinfo is None
                or self.stride <= timedelta(0) or self.window <= timedelta(0)):
            raise ValueError("Invalid volume profile calendar")

    def latest_profile_at(self, now: datetime) -> datetime | None:
        completed = min(now, self.last_input_available)
        if completed < self.first_profile_at:
            return None
        return self.first_profile_at + ((completed - self.first_profile_at) // self.stride) * self.stride


@dataclass(frozen=True, slots=True)
class Event:
    registry_id: str
    source_id: str
    symbol: str
    timeframe: str
    family: str
    origin: str | None
    lower: float
    upper: float
    formation: datetime
    confirmed: datetime | None
    available: datetime
    kind: str
    atr: float | None
    sequence: int | None
    source_file: str
    source_row: int
    created: datetime | None = None
    direction: str | None = None
    evidence_count: int | None = None
    profile_id: str | None = None

    def order_key(self) -> tuple:
        return (self.available, PRIORITY[self.kind], self.source_id,
                self.sequence or 0, FAMILIES.index(self.family), self.source_row)

    def record(self, candidate_count: int, families: set[str]) -> dict:
        return {
            "registry_zone_id": self.registry_id, "source_zone_id": self.source_id,
            "symbol": self.symbol, "source_timeframe": self.timeframe, "zone_family": self.family,
            "origin_type": self.origin, "lower_price": self.lower, "upper_price": self.upper,
            "formation_start_utc": iso(self.formation),
            "confirmed_at_utc": iso(self.confirmed) if self.confirmed is not None else None,
            "available_at_utc": iso(self.available), "event_type": self.kind, "candidate_status": "candidate",
            "source_reference_atr": self.atr, "overlapping_candidate_count": candidate_count,
            "overlapping_family_count": len(families),
            "overlap_family_flags": "|".join(family for family in FAMILIES if family in families),
            "source_event_sequence": self.sequence, "source_file": self.source_file, "source_row": self.source_row,
        }


def integer(value: str) -> int:
    result = int(value)
    if result < 1:
        raise ValueError("Positive source sequence/count/rank required")
    return result


def normalize(row: dict[str, str], family: str, symbol: str, timeframe: str,
              source_file: str, source_row: int) -> Event:
    """Map source meanings explicitly, without assigning false confirmations."""
    if (row["symbol"] != symbol or row["source_timeframe"] != timeframe
            or row["zone_family"] != family or row["candidate_status"] != "candidate"):
        raise ValueError("Source asset/timeframe/family/status mismatch")
    source_id = row["zone_id"]
    if not source_id or source_id != source_id.strip():
        raise ValueError("Empty or padded source zone ID")
    lower, upper = float(row["lower_price"]), float(row["upper_price"])
    if not positive(lower) or not positive(upper) or upper <= lower:
        raise ValueError("Price interval must be finite, positive and have positive width")
    available = utc(row["available_at_utc"])
    interval = INTERVALS[timeframe]
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    if (available - epoch) % interval != timedelta(0):
        raise ValueError("Availability is off the completed-candle UTC grid")
    confirmed = sequence = created = direction = evidence = profile_id = None
    if family in ("structural", "reaction"):
        kind, origin = row["event_type"], row["origin_type"]
        allowed = (("swing_high_cluster", "swing_low_cluster") if family == "structural"
                   else ("bullish_rejection_cluster", "bearish_rejection_cluster"))
        if kind not in ("create", "update") or origin not in allowed:
            raise ValueError("Invalid cluster origin/event type")
        formation, confirmed = utc(row["formation_start_utc"]), utc(row["confirmed_at_utc"])
        if utc(row["event_at_utc"]) != available:
            raise ValueError("Cluster event_at_utc differs from availability")
        sequence = integer(row["event_sequence"])
        evidence = integer(row["confirmed_swing_count"] if family == "structural" else row["independent_rejection_count"])
        if evidence < 2:
            raise ValueError("Cluster candidate requires at least two source evidence events")
        raw_atr = row["cluster_reference_atr"]
    elif family == "imbalance":
        kind, origin, direction = row["event_type"], row["imbalance_type"], row["direction"]
        if kind not in ("create", "fvg_confirmation") or direction not in ("bullish", "bearish"):
            raise ValueError("Invalid imbalance event or direction")
        if origin not in ("displacement_body", f"{direction}_fvg"):
            raise ValueError("Invalid imbalance origin")
        formation, created = utc(row["formation_start_utc"]), utc(row["candidate_created_at_utc"])
        reference = utc(row["reference_candle_time_utc"])
        if (utc(row["c2_time_utc"]) != reference
                or utc(row["reference_atr_available_at_utc"]) != reference + interval):
            raise ValueError("Invalid impulse-candle reference timing")
        if origin == "displacement_body":
            if (kind != "create" or row["c1_time_utc"] or row["c3_time_utc"]
                    or formation != reference or available != reference + interval):
                raise ValueError("Invalid displacement creation timing/type")
        else:
            c1, c3 = utc(row["c1_time_utc"]), utc(row["c3_time_utc"])
            if formation != c1 or reference != c1 + interval or c3 != reference + interval or available != c3 + interval:
                raise ValueError("FVG must be confirmed at C3 completion")
            confirmed = available
            if kind == "fvg_confirmation" and (row["aligned_displacement"] != "True" or created != reference + interval):
                raise ValueError("Shared-ID confirmation requires prior aligned C2 displacement")
        if created > available or (kind == "create" and created != available):
            raise ValueError("Invalid original imbalance creation timestamp")
        raw_atr = row["reference_atr"]
    elif family == "volume_derived":
        kind, origin = "profile_observation", None
        formation = utc(row["profile_start_utc"])
        if utc(row["profile_end_utc"]) + interval != available:
            raise ValueError("Invalid profile-end opening/completion timing")
        profile_id = row["profile_id"]
        if not profile_id or row["method"] != "uniform_ohlcv_range_allocation":
            raise ValueError("Missing profile identity or unexpected volume method")
        integer(row["candidate_rank"])
        raw_atr = row["reference_atr"]
    else:
        raise ValueError(f"Unknown source family: {family}")
    if (formation >= available or (formation - epoch) % interval != timedelta(0)
            or (confirmed is not None and not formation <= confirmed <= available)):
        raise ValueError("Invalid formation/confirmation chronology")
    atr = float(raw_atr) if raw_atr.strip() else None
    if atr is not None and not positive(atr):
        raise ValueError("Nonempty source ATR must be finite and positive")
    registry_id = f"{symbol}:{timeframe}:{family}:{source_id}"
    return Event(registry_id, source_id, symbol, timeframe, family, origin, lower, upper,
                 formation, confirmed, available, kind, atr, sequence, source_file, source_row,
                 created, direction, evidence, profile_id)


def source_paths(root: Path, symbol: str, timeframe: str) -> dict[str, Path]:
    return {family: root / "data" / "zones" / folder /
            (f"{symbol}_{timeframe}_zone_events.csv" if family == "structural" else f"{symbol}_{timeframe}.csv")
            for family, folder in FOLDERS.items()}


def read_events(path: Path, family: str, symbol: str, timeframe: str, root: Path) -> Iterator[Event]:
    """Exactly one CSV pass; headers and rows validated in the same open handle."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)) or set(SOURCE_COLUMNS[family]) - set(fields):
            raise ValueError(f"{path}: missing required history columns or duplicate header")
        for line, row in enumerate(reader, 2):
            try:
                yield normalize(row, family, symbol, timeframe, path.relative_to(root).as_posix(), line)
            except (ValueError, TypeError, AttributeError, KeyError) as exc:
                raise ValueError(f"{path}, CSV line {line}: {exc}") from exc


def validate_transition(previous: Event | None, event: Event) -> None:
    if previous is None:
        if event.kind not in ("create", "profile_observation") or (event.sequence is not None and event.sequence != 1):
            raise ValueError("History must start with creation/observation and source sequence 1 when provided")
        if event.family in ("structural", "reaction") and event.confirmed != event.available:
            raise ValueError("Cluster creation confirmation must equal creation availability")
        return
    if event.available <= previous.available or event.kind in ("create", "profile_observation"):
        raise ValueError("Duplicate creation/observation or non-increasing identity event time")
    if event.family in ("structural", "reaction"):
        if (event.kind != "update" or event.sequence != previous.sequence + 1
                or event.evidence_count != previous.evidence_count + 1
                or event.origin != previous.origin or event.formation != previous.formation
                or event.confirmed != previous.confirmed or event.atr != previous.atr):
            raise ValueError("Malformed cluster update sequence or rewritten original confirmation/reference")
    elif (event.family != "imbalance" or event.kind != "fvg_confirmation"
          or previous.kind != "create" or previous.origin != "displacement_body"
          or event.created != previous.created or event.direction != previous.direction):
        raise ValueError("FVG confirmation must follow the same original displacement identity")


def overlaps(first: Event, second: Event) -> bool:
    return min(first.upper, second.upper) > max(first.lower, second.lower)


def registry_rows(events: Iterable[Event], clock: VolumeClock, config: Config,
                  stats: Counter) -> Iterator[dict]:
    """Two-phase same-time batches: install every version, then compute overlaps.

    Latest-state history validates all identities; the much smaller eligible
    dictionary alone is searched. Expiry never changes a historical row. At a
    timestamp, another identity's final version for that timestamp is visible.
    Multiple events for the SAME identity at one timestamp are rejected.
    """
    ordered = sorted(events, key=Event.order_key)
    history: dict[str, Event] = {}
    eligible: dict[str, Event] = {}
    expiry: deque[tuple[datetime, str]] = deque()
    volume_ids: set[str] = set()
    profile_times: dict[str, datetime] = {}
    previous_slot = None
    for now, batch_iterator in groupby(ordered, key=lambda e: e.available):
        batch = list(batch_iterator)
        if len({e.profile_id for e in batch if e.family == "volume_derived"}) > 1:
            raise ValueError("Multiple volume profile IDs at one completed profile time")
        cutoff = now - config.recency(batch[0].timeframe)
        while expiry and expiry[0][0] < cutoff:
            available, key = expiry.popleft()
            if key in eligible and eligible[key].available == available:
                del eligible[key]
                stats["non_volume_eligibility_expirations"] += 1
        slot = clock.latest_profile_at(now)
        if slot is not None and now - slot >= clock.stride:
            slot = None  # Conservative freshness cap after volume input coverage ends.
        if slot != previous_slot:
            for key in volume_ids:
                eligible.pop(key, None)
            stats["volume_observations_replaced_or_expired"] += len(volume_ids)
            volume_ids.clear()
            previous_slot = slot
        for event in batch:
            try:
                validate_transition(history.get(event.registry_id), event)
                if event.family == "volume_derived":
                    if event.available != slot or event.formation + clock.window != event.available:
                        raise ValueError("Volume observation does not match the Phase 5 profile calendar/window")
                    if event.profile_id in profile_times and profile_times[event.profile_id] != now:
                        raise ValueError("Volume profile ID reused at a different availability")
                    profile_times[event.profile_id] = now
                    volume_ids.add(event.registry_id)
                else:
                    expiry.append((now, event.registry_id))
                history[event.registry_id] = event
                eligible[event.registry_id] = event
            except ValueError as exc:
                raise ValueError(f"{event.source_file}, CSV line {event.source_row}: {exc}") from exc
        stats["max_overlap_eligible_identities"] = max(stats["max_overlap_eligible_identities"], len(eligible))
        stats["timestamp_batches"] += 1
        for event in batch:
            candidate_count = 0
            families = set()
            for other in eligible.values():
                if other.registry_id != event.registry_id and overlaps(event, other):
                    candidate_count += 1
                    if other.family != event.family:
                        families.add(other.family)
            stats["overlap_comparisons"] += len(eligible) - 1
            stats["events_with_geometric_overlap"] += candidate_count > 0
            stats["events_with_other_family_overlap"] += bool(families)
            stats["max_overlapping_candidates"] = max(stats["max_overlapping_candidates"], candidate_count)
            stats["max_other_overlapping_families"] = max(stats["max_other_overlapping_families"], len(families))
            stats["unified_event_rows"] += 1
            yield event.record(candidate_count, families)


def source_metrics(events: list[Event]) -> dict:
    rows, kinds, origins = Counter(), Counter(), Counter()
    identities = {family: set() for family in FAMILIES}
    nulls = {family: Counter() for family in FAMILIES}
    for event in events:
        rows[event.family] += 1
        kinds[(event.family, event.kind)] += 1
        origins[(event.family, event.origin)] += 1
        identities[event.family].add(event.source_id)
        for name, value in (("origin_type", event.origin), ("confirmed_at_utc", event.confirmed),
                            ("source_reference_atr", event.atr), ("source_event_sequence", event.sequence)):
            nulls[event.family][name] += value is None
    return {
        "rows": {family: rows[family] for family in FAMILIES},
        "identities": {family: len(ids) for family, ids in identities.items()},
        "updates": {family: kinds[(family, "update")] for family in ("structural", "reaction")},
        "fvg_confirmations": kinds[("imbalance", "fvg_confirmation")],
        "bullish_fvgs": origins[("imbalance", "bullish_fvg")],
        "bearish_fvgs": origins[("imbalance", "bearish_fvg")],
        "body_creations": origins[("imbalance", "displacement_body")],
        "volume_profiles_with_observations": len({e.profile_id for e in events if e.family == "volume_derived"}),
        "nulls": {family: dict(values) for family, values in nulls.items()},
    }


def report_dataset(reports: dict, family: str, symbol: str, timeframe: str) -> dict:
    matches = [d for d in reports[family]["datasets"]
               if d["symbol"] == symbol and d.get("source_timeframe", d.get("timeframe")) == timeframe]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {family} report entry for {symbol} {timeframe}")
    return matches[0]


def profile_clock(reports: dict, symbol: str, timeframe: str) -> VolumeClock:
    report = reports["volume_derived"]
    dataset = report_dataset(reports, "volume_derived", symbol, timeframe)
    interval = INTERVALS[timeframe]
    start = utc(dataset["timestamp_checks"]["first_open_utc"])
    end = utc(dataset["timestamp_checks"]["last_available_utc"])
    window, stride = report["parameters"]["profile_window_bars"], report["parameters"]["profile_stride_bars"]
    if (type(window) is not int or type(stride) is not int or window < 1 or stride < 1
            or end - start != dataset["input_candle_count"] * interval):
        raise ValueError("Invalid Phase 5 input coverage/window/stride metadata")
    return VolumeClock(start + window * interval, end, stride * interval, window * interval)


def reconcile(metrics: dict, reports: dict, symbol: str, timeframe: str) -> dict:
    checks = {}

    def check(name, actual, expected):
        if actual != expected:
            raise ValueError(f"Source report count mismatch: {name}: {actual} != {expected}")
        checks[name] = {"actual": actual, "expected": expected}

    for family in FAMILIES:
        d = report_dataset(reports, family, symbol, timeframe)
        if family in ("structural", "reaction"):
            expected_rows = d["candidate_zones_created"] + d["zone_update_count"]
            expected_ids = d["candidate_zones_created"]
            check(f"{family}_updates", metrics["updates"][family], d["zone_update_count"])
        elif family == "imbalance":
            expected_rows, expected_ids = d["output_event_rows"], d["unique_candidate_count"]
            for metric, field in (("fvg_confirmations", "linked_fvg_confirmation_count"),
                                  ("bullish_fvgs", "bullish_fvg_count"), ("bearish_fvgs", "bearish_fvg_count"),
                                  ("body_creations", "displacement_body_creation_count")):
                check(metric, metrics[metric], d[field])
        else:
            expected_rows = expected_ids = d["candidate_observations"]
            check("volume_profiles_with_observations", metrics["volume_profiles_with_observations"],
                  d["profiles_calculated"] - d["no_candidate_profiles"])
        check(f"{family}_event_rows", metrics["rows"][family], expected_rows)
        check(f"{family}_source_identities", metrics["identities"][family], expected_ids)
    return {"status": "PASS", "checks": checks}


def load_reports(root: Path) -> dict:
    reports = {}
    for family, folder in FOLDERS.items():
        path = root / "reports" / "zone_detection" / f"{folder}_summary.json"
        with path.open("r", encoding="utf-8") as handle:
            reports[family] = json.load(handle)
    return reports


def preflight(root: Path, overwrite: bool) -> list[tuple[str, str, Path]]:
    jobs = [(symbol, timeframe, root / "data" / "zones" / "unified" / f"{symbol}_{timeframe}.csv")
            for timeframe in INTERVALS for symbol in SYMBOLS]
    targets = [job[2] for job in jobs] + [root / "reports" / "zone_detection" / "registry_summary.json"]
    if not overwrite and any(path.exists() for path in targets):
        raise FileExistsError("Registry outputs exist; use --overwrite to regenerate")
    for symbol, timeframe, _ in jobs:
        for path in source_paths(root, symbol, timeframe).values():
            if not path.is_file():
                raise FileNotFoundError(path)
    return jobs


def process_dataset(root: Path, symbol: str, timeframe: str, config: Config,
                    reports: dict, overwrite: bool = False) -> dict:
    paths = source_paths(root, symbol, timeframe)
    output = root / "data" / "zones" / "unified" / f"{symbol}_{timeframe}.csv"
    if output.exists() and not overwrite:
        raise FileExistsError(f"Registry output exists: {output}; use --overwrite")
    events = [e for family, path in paths.items() for e in read_events(path, family, symbol, timeframe, root)]
    metrics = source_metrics(events)
    reconciliation = reconcile(metrics, reports, symbol, timeframe)
    clock = profile_clock(reports, symbol, timeframe)
    stats = Counter()
    cross_family_ids = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if overwrite else "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in registry_rows(events, clock, config, stats):
            writer.writerow(row)
            if row["overlapping_family_count"]:
                cross_family_ids.add(row["registry_zone_id"])
    if stats["unified_event_rows"] != sum(metrics["rows"].values()):
        raise ValueError("Unified/source event count mismatch")
    return {
        "symbol": symbol, "source_timeframe": timeframe,
        "input_paths": {family: str(path) for family, path in paths.items()}, "output_path": str(output),
        "source_event_rows_by_family": metrics["rows"], "unique_source_candidates_by_family": metrics["identities"],
        "unified_event_rows": stats["unified_event_rows"], "unified_identity_count": sum(metrics["identities"].values()),
        "deduplicated_non_volume_identity_count": sum(v for family, v in metrics["identities"].items() if family != "volume_derived"),
        "correlated_volume_observation_identity_count": metrics["identities"]["volume_derived"],
        "statistically_independent_evidence_count": None,
        "fvg_confirmation_mapping_count": metrics["fvg_confirmations"], "cluster_update_counts": metrics["updates"],
        "volume_profile_observation_count": metrics["rows"]["volume_derived"],
        "count_reconciliation": reconciliation,
        "timestamp_validation": {"status": "PASS", "event_rows_checked": len(events),
                                 "timestamp_batches": stats["timestamp_batches"],
                                 "first_available_utc": iso(min(e.available for e in events)) if events else None,
                                 "last_available_utc": iso(max(e.available for e in events)) if events else None},
        "boundary_validation": {"status": "PASS", "event_rows_checked": len(events), "invalid_intervals": 0},
        "identity_and_sequence_checks": {"status": "PASS", "duplicate_creation_or_observation_ids": 0,
                                         "malformed_sequences": 0, "id_qualification": "symbol:timeframe:family:source_zone_id"},
        "overlap_metadata": {key: stats[key] for key in (
            "events_with_geometric_overlap", "events_with_other_family_overlap", "max_overlapping_candidates",
            "max_other_overlapping_families", "max_overlap_eligible_identities", "overlap_comparisons",
            "non_volume_eligibility_expirations", "volume_observations_replaced_or_expired")},
        "identities_with_cross_family_overlap": len(cross_family_ids),
        "null_source_field_counts_by_family": metrics["nulls"],
        "warnings": ["Nullable confirmation/origin/sequence fields retain source-specific meanings; null is not a failed detection",
                     "Identity counts are not counts of independent market reactions; volume observations are rolling and correlated"],
    }


def run_tests(root: Path) -> dict:
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    return {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
            "skipped": len(result.skipped), "successful": result.testsRun > 0 and result.wasSuccessful()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate only Phase 6 outputs and summary")
    for name, default in asdict(Config()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    args = vars(parser.parse_args(argv))
    overwrite = args.pop("overwrite")
    try:
        config = Config(**args)
        jobs = preflight(PROJECT_DIR, overwrite)
        reports = load_reports(PROJECT_DIR)
        tests = run_tests(PROJECT_DIR)
        if not tests["successful"]:
            print("FAIL: tests did not pass; no registry outputs generated", file=sys.stderr)
            return 1
        datasets = []
        for symbol, timeframe, _ in jobs:
            result = process_dataset(PROJECT_DIR, symbol, timeframe, config, reports, overwrite)
            datasets.append(result)
            print(f"PASS {symbol} {timeframe}: {result['unified_event_rows']} events; "
                  f"{result['deduplicated_non_volume_identity_count']} non-volume identities + "
                  f"{result['correlated_volume_observation_identity_count']} correlated volume observations", flush=True)
        report = PROJECT_DIR / "reports" / "zone_detection" / "registry_summary.json"
        summary = {
            "phase": 6, "generated_at_utc": iso(datetime.now(timezone.utc)), "parameters": asdict(config),
            "parameter_status": "Provisional overlap-search recency assumptions; not lifecycle or trading rules",
            "source_mapping": {
                "structural": "*_zone_events.csv only; original cluster confirmation and event boundaries; frozen cluster_reference_atr",
                "imbalance": "imbalance_type -> origin_type; body confirmation null; FVG confirmation at C3 close; shared source ID retained; reference_atr is C2 ATR",
                "reaction": "Create/update history; original independent-rejection cluster confirmation; frozen cluster_reference_atr",
                "volume_derived": "profile_observation; profile_start -> formation; confirmation/origin/sequence null; final-candle reference_atr; no directional role",
            },
            "schema": COLUMNS,
            "rules": {
                "ordering": "availability, create/update/fvg_confirmation/profile_observation priority, source ID, source sequence (null sorts as 0), family, CSV row",
                "same_timestamp": "Install ALL simultaneous versions before querying; each row compares its interval with the final simultaneous versions of OTHER identities; duplicate same-ID same-time events fail",
                "non_volume_eligibility": "Latest source event within timeframe recency, including equality; expiry excludes overlap searches only; later updates can re-enter searches without a validity claim",
                "volume_eligibility": "Only the latest scheduled completed profile from Phase 5's validated input calendar; empty profiles clear older evidence; after coverage ends apply at most one-stride freshness",
                "volume_calendar": "Phase 5 first input opening + window*interval, then stride*interval; only calendar metadata is used, never future prices or future candidate presence",
                "overlap": "min(upper_A,upper_B) > max(lower_A,lower_B); strict positive width, self excluded by qualified identity",
                "candidate_count": "Distinct matching OTHER identities, including same-family IDs; latest-profile volume areas remain correlated observations, not independent evidence",
                "family_count_and_flags": "OTHER detector families only, excluding the row's own family; volume_derived appears at most once; flags are pipe-separated in structural/imbalance/reaction/volume_derived order",
                "nulls": "Volume confirmation/origin and absent original source sequences are null; displacement creation confirmation is null; no zero/fake timestamps",
                "traceability": "source_file is project-relative; source_row is the 1-based CSV line including header; source_event_sequence is original only, null when not supplied",
                "history": "Every original event remains an immutable registry row; candidate_status is candidate, never active/valid; no blind interval merging",
            },
            "source_report_paths": {family: str(PROJECT_DIR / "reports/zone_detection" / f"{folder}_summary.json") for family, folder in FOLDERS.items()},
            "tests": tests, "datasets": datasets, "output_summary_path": str(report),
            "limitations": [
                "Geometric overlap is descriptive metadata, not confidence, confluence strength, statistical independence or prediction",
                "No lifecycle, survival, invalidation, mitigation, market reaction labels or cross-timeframe inference",
                "One asset/timeframe is read once into normalized events for sorting; overlap searches scan only the bounded recency/profile eligible dictionary, not all historical identities",
                "Prefix invariance requires complete same-timestamp batches across all families; adding omitted events at an already processed time is not appending future data",
                "Phase 5 input coverage and window/stride reports are required to recognize empty profiles without changing detectors or importing snapshots",
                "Input/report/sequence failure returns nonzero and may leave partial new CSVs; no new success summary is written",
            ],
            "warnings": [f"{d['symbol']} {d['source_timeframe']}: {w}" for d in datasets for w in d["warnings"]],
        }
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("w" if overwrite else "x", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(f"Summary: {report}")
        return 0
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
