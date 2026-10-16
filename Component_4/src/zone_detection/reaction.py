"""Completed-candle rejection evidence and chronological reaction zone histories.

Run: python -B Component_4/src/zone_detection/reaction.py

CSV rows are immutable create/update events, not active-zone snapshots. Matching
expiry never invalidates a zone. --overwrite explicitly regenerates a batch;
this script does not resume or append to files from an earlier run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import unittest
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zone_detection.structural import boundaries
from zone_detection.swings import INTERVALS, iso, positive, utc


PROJECT_DIR = Path(__file__).resolve().parents[2]
SYMBOLS = ("BTCUSDT", "ETHUSDT")
FEATURES = ("lower_wick", "upper_wick", "lower_wick_ratio", "upper_wick_ratio",
            "body_ratio", "close_location_value")
REQUIRED_COLUMNS = ("timestamp_utc", "available_at_utc", "open", "high", "low", "close",
                    "atr_14_sma") + FEATURES
COLUMNS = (
    "zone_id", "symbol", "source_timeframe", "zone_family", "origin_type",
    "lower_price", "upper_price", "formation_start_utc", "confirmed_at_utc",
    "available_at_utc", "independent_rejection_count", "cluster_reference_atr",
    "event_type", "event_at_utc", "trigger_candle_time_utc", "trigger_rejection_price",
    "trigger_wick_ratio", "trigger_close_location_value", "trigger_wick_atr",
    "candidate_status", "observed_min_price", "observed_max_price", "minimum_member_atr",
    "diameter_limit", "padding_price", "trigger_atr", "event_sequence",
)


@dataclass(frozen=True)
class Config:
    wick_ratio_min: float = 0.50
    close_location_min: float = 0.30
    wick_atr_min: float = 0.25
    min_separation_bars: int = 3
    cluster_tolerance_atr: float = 0.5
    min_independent_rejections: int = 2
    lookback_15m_days: float = 30
    lookback_1h_days: float = 90
    lookback_4h_days: float = 180
    padding_atr: float = 0.1
    padding_price_cap_fraction: float = 0.01
    max_zone_width_fraction: float = 0.05

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not positive(value):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("min_separation_bars", "min_independent_rejections"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.min_independent_rejections < 2:
            raise ValueError("A reaction zone requires at least two independent rejections")
        if self.wick_ratio_min > 1 or self.close_location_min > 1:
            raise ValueError("Ratio and close-location thresholds must be <= 1")
        if self.padding_price_cap_fraction >= 1 or self.max_zone_width_fraction >= 1:
            raise ValueError("Padding and width fractions must be < 1")

    def lookback(self, timeframe: str) -> timedelta:
        return timedelta(days=getattr(self, f"lookback_{timeframe}_days"))


@dataclass(frozen=True)
class Candle:
    bar_index: int
    timestamp: datetime
    available_at: datetime
    open: float
    high: float
    low: float
    close: float
    atr: float | None
    lower_wick: float | None
    upper_wick: float | None
    lower_wick_ratio: float | None
    upper_wick_ratio: float | None
    body_ratio: float | None
    close_location_value: float | None


@dataclass(frozen=True)
class Rejection:
    bar_index: int
    timestamp: datetime
    available_at: datetime
    direction: str
    price: float
    atr: float
    wick_ratio: float
    close_location_value: float
    wick_atr: float


def check_header(path: Path) -> None:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        fields = csv.DictReader(handle).fieldnames
    if not fields or len(fields) != len(set(fields)):
        raise ValueError(f"{path}: missing header or duplicate columns")
    missing = set(REQUIRED_COLUMNS) - set(fields)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")


def feature_number(raw: str, name: str, stats: Counter) -> float | None:
    if not raw.strip():
        stats[f"missing_{name}"] += 1
        return None
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value):
        stats[f"invalid_{name}"] += 1
        return None
    return value


def read_candles(path: Path, timeframe: str, stats: Counter) -> Iterator[Candle]:
    """Stream one file. Bad OHLC/time fails; unusable features stay missing.

    Feature formulas are checked at the precision of Phase 1 CSV values. Stored
    values are used for detection, never replaced by recomputed values. body_ratio
    is audited but is not an additional rejection threshold.
    """
    check_header(path)
    interval = INTERVALS[timeframe]
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    previous = None
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            try:
                timestamp, available = utc(row["timestamp_utc"]), utc(row["available_at_utc"])
                if (timestamp - epoch) % interval != timedelta(0) or available != timestamp + interval:
                    raise ValueError("Off-grid opening timestamp or incorrect close-time availability")
                if previous is not None and timestamp - previous != interval:
                    raise ValueError("Missing, duplicate or unordered candle timestamp")
                opening, high, low, close = (float(row[k]) for k in ("open", "high", "low", "close"))
                if (not all(positive(p) for p in (opening, high, low, close))
                        or high < max(opening, low, close) or low > min(opening, close)):
                    raise ValueError("OHLC must be finite, positive and internally consistent")
                atr = feature_number(row["atr_14_sma"], "atr", stats)
                if atr is not None and atr <= 0:
                    stats["invalid_atr"] += 1
                    atr = None
                candle_range = high - low
                lower, upper = min(opening, close) - low, high - max(opening, close)
                expected = {
                    "lower_wick": lower, "upper_wick": upper,
                    "lower_wick_ratio": lower / candle_range if candle_range else None,
                    "upper_wick_ratio": upper / candle_range if candle_range else None,
                    "body_ratio": abs(close - opening) / candle_range if candle_range else None,
                    "close_location_value": (2 * close - high - low) / candle_range if candle_range else None,
                }
                values = {}
                for name in FEATURES:
                    value = feature_number(row[name], name, stats)
                    target = expected[name]
                    if value is not None and (target is None or not math.isclose(
                            value, target, rel_tol=1e-8, abs_tol=1e-9)):
                        stats[f"invalid_{name}"] += 1
                        value = None
                    if value is not None:
                        minimum, maximum = (-1, 1) if name == "close_location_value" else (0, 1)
                        if value < minimum or ((name.endswith("ratio") or name == "close_location_value")
                                               and value > maximum):
                            stats[f"invalid_{name}"] += 1
                            value = None
                    values[name] = value
                if candle_range == 0:
                    stats["zero_range_rows"] += 1
            except (ValueError, TypeError, AttributeError, KeyError) as exc:
                raise ValueError(f"{path}, CSV line {index + 2}: {exc}") from exc
            previous = timestamp
            stats["input_rows"] += 1
            yield Candle(index, timestamp, available, opening, high, low, close, atr, **values)


def detect_rejections(candle: Candle, config: Config) -> list[Rejection]:
    """Current completed candle only; bullish/bearish refers to rejection origin."""
    if (not all(positive(p) for p in (candle.open, candle.high, candle.low, candle.close))
            or candle.high <= candle.low or candle.high < max(candle.open, candle.close)
            or candle.low > min(candle.open, candle.close) or not positive(candle.atr)):
        return []
    location = candle.close_location_value
    if location is None or not math.isfinite(location) or not -1 <= location <= 1:
        return []
    events = []
    for direction, wick, ratio, anchor, location_ok in (
        ("bullish", candle.lower_wick, candle.lower_wick_ratio, candle.low,
         location >= config.close_location_min),
        ("bearish", candle.upper_wick, candle.upper_wick_ratio, candle.high,
         location <= -config.close_location_min),
    ):
        if (not positive(wick) or ratio is None or not math.isfinite(ratio)
                or not config.wick_ratio_min <= ratio <= 1 or not location_ok):
            continue
        wick_atr = wick / candle.atr
        if math.isfinite(wick_atr) and wick_atr >= config.wick_atr_min:
            events.append(Rejection(candle.bar_index, candle.timestamp, candle.available_at,
                                    direction, anchor, candle.atr, ratio, location, wick_atr))
    return events


@dataclass
class Group:
    zone_id: str
    direction: str
    minimum: float
    maximum: float
    reference_atr: float
    minimum_atr: float
    formation_start: datetime
    first_available: datetime
    latest_available: datetime
    latest_bar_index: int
    count: int = 1
    confirmed_at: datetime | None = None


class ReactionDetector:
    """Local cooldown followed by bounded chronological grouping.

    Only elapsed-time-eligible groups are retained for matching. The expiry
    queue is chronological; stale entries from later accepted evidence are
    ignored. Archived zone history already exists in the output event stream.
    """

    def __init__(self, symbol: str, timeframe: str, config: Config | None = None):
        self.symbol, self.timeframe = symbol, timeframe
        self.config = config or Config()
        self.lookback = self.config.lookback(timeframe)
        self.groups: dict[str, dict[str, Group]] = {"bullish": {}, "bearish": {}}
        self.expiry: deque[tuple[datetime, str, str]] = deque()
        self.stats: Counter = Counter()
        self.latest_event: Rejection | None = None

    def expire(self, now: datetime) -> None:
        cutoff = now - self.lookback
        while self.expiry and self.expiry[0][0] < cutoff:
            available, direction, zone_id = self.expiry.popleft()
            group = self.groups[direction].get(zone_id)
            if group is not None and group.latest_available == available:
                del self.groups[direction][zone_id]
                self.stats["archived_group_count"] += 1

    def consume(self, event: Rejection) -> dict | None:
        if (event.direction not in self.groups or not positive(event.price) or not positive(event.atr)
                or event.timestamp.tzinfo is None or event.available_at.tzinfo is None
                or event.available_at != event.timestamp + INTERVALS[self.timeframe]
                or event.bar_index < 0):
            raise ValueError("Invalid rejection evidence")
        if self.latest_event is not None and (
                event.available_at <= self.latest_event.available_at
                or event.bar_index <= self.latest_event.bar_index):
            raise ValueError("Rejection candles must arrive once, in strict availability order")
        self.latest_event = event
        self.stats[f"raw_{event.direction}"] += 1
        self.expire(event.available_at)
        groups = self.groups[event.direction]
        nearby = []
        for group in groups.values():
            tolerance = self.config.cluster_tolerance_atr * min(group.reference_atr, event.atr)
            distance_to_area = max(group.minimum - event.price, event.price - group.maximum, 0)
            if distance_to_area > tolerance:
                continue
            # Any nearby group on cooldown suppresses the raw event. It must
            # not escape deduplication by choosing another group or seeding one.
            if event.bar_index - group.latest_bar_index < self.config.min_separation_bars:
                self.stats[f"suppressed_{event.direction}"] += 1
                return None
            nearby.append(group)
        matches = []
        for group in nearby:
            minimum, maximum = min(group.minimum, event.price), max(group.maximum, event.price)
            minimum_atr = min(group.minimum_atr, event.atr)
            if maximum - minimum > self.config.cluster_tolerance_atr * minimum_atr:
                continue
            try:
                boundaries(minimum, maximum, minimum_atr, self.config)
            except ValueError:
                self.stats["boundary_rejected_matches"] += 1
                continue
            midpoint = group.minimum + (group.maximum - group.minimum) / 2
            matches.append((abs(event.price - midpoint), group.first_available, group.zone_id))

        if matches:
            group = groups[min(matches)[2]]
            group.minimum = min(group.minimum, event.price)
            group.maximum = max(group.maximum, event.price)
            group.minimum_atr = min(group.minimum_atr, event.atr)
            group.latest_available, group.latest_bar_index = event.available_at, event.bar_index
            group.count += 1
        else:
            try:
                boundaries(event.price, event.price, event.atr, self.config)
            except ValueError:
                self.stats["boundary_rejected_events"] += 1
                return None
            zone_id = f"{self.symbol}_{self.timeframe}_reaction_{event.timestamp.strftime('%Y%m%dT%H%M%SZ')}_{event.direction}"
            group = Group(zone_id, event.direction, event.price, event.price, event.atr, event.atr,
                          event.timestamp, event.available_at, event.available_at, event.bar_index)
            groups[zone_id] = group
            self.stats["total_group_count"] += 1
        self.expiry.append((event.available_at, event.direction, group.zone_id))
        self.stats[f"accepted_{event.direction}"] += 1
        if group.count < self.config.min_independent_rejections:
            return None
        event_type = "update"
        if group.confirmed_at is None:
            group.confirmed_at = event.available_at
            event_type = "create"
        self.stats[f"zone_{event_type}_count"] += 1
        lower, upper, padding = boundaries(group.minimum, group.maximum, group.minimum_atr, self.config)
        return {
            "zone_id": group.zone_id, "symbol": self.symbol, "source_timeframe": self.timeframe,
            "zone_family": "reaction", "origin_type": f"{event.direction}_rejection_cluster",
            "lower_price": lower, "upper_price": upper, "formation_start_utc": iso(group.formation_start),
            "confirmed_at_utc": iso(group.confirmed_at), "available_at_utc": iso(event.available_at),
            "independent_rejection_count": group.count, "cluster_reference_atr": group.reference_atr,
            "event_type": event_type, "event_at_utc": iso(event.available_at),
            "trigger_candle_time_utc": iso(event.timestamp), "trigger_rejection_price": event.price,
            "trigger_wick_ratio": event.wick_ratio, "trigger_close_location_value": event.close_location_value,
            "trigger_wick_atr": event.wick_atr, "candidate_status": "candidate",
            "observed_min_price": group.minimum, "observed_max_price": group.maximum,
            "minimum_member_atr": group.minimum_atr,
            "diameter_limit": self.config.cluster_tolerance_atr * group.minimum_atr,
            "padding_price": padding, "trigger_atr": event.atr,
            "event_sequence": group.count - self.config.min_independent_rejections + 1,
        }


def paths_for(root: Path, symbol: str, timeframe: str) -> tuple[Path, Path]:
    return (root / "data" / "features" / timeframe / f"{symbol}_{timeframe}_features.csv",
            root / "data" / "zones" / "reaction" / f"{symbol}_{timeframe}.csv")


def preflight(root: Path, overwrite: bool) -> list[tuple[str, str, Path, Path]]:
    jobs = [(symbol, timeframe, *paths_for(root, symbol, timeframe))
            for timeframe in INTERVALS for symbol in SYMBOLS]
    targets = [job[3] for job in jobs] + [root / "reports" / "zone_detection" / "reaction_summary.json"]
    existing = [str(p) for p in targets if p.exists()]
    if existing and not overwrite:
        raise FileExistsError("Outputs exist; use --overwrite to regenerate: " + ", ".join(existing))
    for _, _, source, _ in jobs:
        check_header(source)
    return jobs


def process_dataset(symbol: str, timeframe: str, source: Path, output: Path,
                    config: Config, overwrite: bool = False) -> dict:
    if source.resolve() == output.resolve():
        raise ValueError("Input and output paths must differ")
    check_header(source)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}; use --overwrite to regenerate")
    input_stats = Counter()
    detector = ReactionDetector(symbol, timeframe, config)
    first_open = last_available = previous_event = None
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if overwrite else "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for candle in read_candles(source, timeframe, input_stats):
            if first_open is None:
                first_open = candle.timestamp
            last_available = candle.available_at
            for rejection in detect_rejections(candle, config):
                event = detector.consume(rejection)
                if event is not None:
                    available = utc(event["available_at_utc"])
                    if (available != candle.available_at or utc(event["confirmed_at_utc"]) > available
                            or (previous_event is not None and available <= previous_event)):
                        raise ValueError("Invalid output event chronology or availability")
                    previous_event = available
                    writer.writerow(event)
    if first_open is None:
        raise ValueError(f"Empty feature file: {source}")
    detector.expire(last_available)
    stats = detector.stats
    directions = ("bullish", "bearish")
    missing = {name: input_stats[f"missing_{name}"] for name in ("atr",) + FEATURES}
    invalid = {name: input_stats[f"invalid_{name}"] for name in ("atr",) + FEATURES}
    warnings = [f"missing_{name}: {count}; no imputation" for name, count in missing.items() if count]
    warnings += [f"invalid_{name}: {count}; unusable value excluded" for name, count in invalid.items() if count]
    if input_stats["zero_range_rows"]:
        warnings.append(f"zero_range_rows: {input_stats['zero_range_rows']}; no rejection qualification")
    if stats["boundary_rejected_events"] or stats["boundary_rejected_matches"]:
        warnings.append("Some events or matching attempts failed the boundary safeguards; see boundary_checks")
    return {
        "symbol": symbol, "source_timeframe": timeframe, "input_path": str(source),
        "output_path": str(output), "input_candle_count": input_stats["input_rows"],
        "raw_rejections": {d: stats[f"raw_{d}"] for d in directions},
        "independent_accepted_rejections": {d: stats[f"accepted_{d}"] for d in directions},
        "suppressed_near_duplicates": {d: stats[f"suppressed_{d}"] for d in directions},
        "candidate_zones_created": stats["zone_create_count"], "zone_update_count": stats["zone_update_count"],
        "output_event_rows": stats["zone_create_count"] + stats["zone_update_count"],
        "total_group_count": stats["total_group_count"],
        "grouping_eligible_at_end": sum(len(groups) for groups in detector.groups.values()),
        "archived_group_count": stats["archived_group_count"],
        "provisional_group_count_including_archived": stats["total_group_count"] - stats["zone_create_count"],
        "missing_feature_values": missing, "invalid_feature_values": invalid,
        "zero_range_rows": input_stats["zero_range_rows"],
        "timestamp_checks": {"status": "PASS", "input_rows_checked": input_stats["input_rows"],
                             "invalid_rows": 0, "first_open_utc": iso(first_open),
                             "last_available_utc": iso(last_available),
                             "event_rows_checked": stats["zone_create_count"] + stats["zone_update_count"]},
        "boundary_checks": {"status": "PASS", "invalid_ohlc_rows": 0,
                            "rejected_matching_attempts": stats["boundary_rejected_matches"],
                            "rejected_events": stats["boundary_rejected_events"]},
        "warnings": warnings,
    }


def run_tests(root: Path) -> dict:
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    return {"tests_run": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped),
            "successful": result.testsRun > 0 and result.wasSuccessful()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate only Phase 4 outputs and summary")
    for name, default in asdict(Config()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    args = vars(parser.parse_args(argv))
    overwrite = args.pop("overwrite")
    try:
        config = Config(**args)
        jobs = preflight(PROJECT_DIR, overwrite)
        tests = run_tests(PROJECT_DIR)
        if not tests["successful"]:
            print("FAIL: tests did not pass; no reaction outputs generated", file=sys.stderr)
            return 1
        datasets = []
        for symbol, timeframe, source, output in jobs:
            result = process_dataset(symbol, timeframe, source, output, config, overwrite)
            datasets.append(result)
            print(f"PASS {symbol} {timeframe}: {result['input_candle_count']} candles; "
                  f"{sum(result['raw_rejections'].values())} raw / "
                  f"{sum(result['independent_accepted_rejections'].values())} accepted / "
                  f"{sum(result['suppressed_near_duplicates'].values())} suppressed; "
                  f"{result['candidate_zones_created']} zones, {result['zone_update_count']} updates", flush=True)
        report = PROJECT_DIR / "reports" / "zone_detection" / "reaction_summary.json"
        summary = {
            "phase": 4, "generated_at_utc": iso(datetime.now(timezone.utc)), "parameters": asdict(config),
            "parameter_status": "Provisional research assumptions, not optimized trading rules",
            "rules": {
                "bullish": "lower_wick_ratio >= wick_ratio_min AND CLV >= close_location_min AND lower_wick/ATR >= wick_atr_min; anchor=low",
                "bearish": "upper_wick_ratio >= wick_ratio_min AND CLV <= -close_location_min AND upper_wick/ATR >= wick_atr_min; anchor=high",
                "raw_counts": "Completed candles satisfying all conditions, including valid ATR, before cooldown/grouping",
                "features": "Stored features audited against OHLC (rel_tol=1e-8, abs_tol=1e-9); body_ratio audited only, not an extra threshold; no imputation",
                "nearby_area": "Distance from new anchor to observed group [min,max] <= cluster_tolerance_atr * min(frozen_reference_ATR, new_ATR)",
                "cooldown": "Suppress if ANY eligible nearby same-direction group has new_bar_index-last_accepted_bar_index < min_separation_bars; no other group or new group can bypass this",
                "suppression": "Preserve the first accepted evidence; suppressed candles do not change group state, cooldown or lookback clocks",
                "separation_example": "At separation=3: accept i, suppress nearby i+1 and i+2, permit i+3; all input candles count, not just rejection candles",
                "reference_atr": "Frozen at the first accepted rejection; subsequent ATR cannot increase its matching allowance",
                "compactness": "Updated anchor diameter <= cluster_tolerance_atr * minimum ATR across all accepted members",
                "matching_ties": "Nearest existing midpoint, earliest first availability, lexicographic zone ID",
                "time_eligibility": "Elapsed UTC time since latest accepted evidence <= timeframe lookback; expiry archives matching only, not zone validity",
                "boundaries": "Reuse Phase 2 helper: [min_anchor-padding,max_anchor+padding], padding=min(padding_atr*minimum_member_ATR,padding_price_cap_fraction*min_anchor); width <= max_zone_width_fraction*midpoint",
                "event_timestamps": "formation_start=first accepted candle opening; confirmed_at=original creation close; available_at=event_at=trigger candle close for this version",
                "event_history": "Append immutable create/update rows within each batch; stable IDs; --overwrite explicitly regenerates files",
            },
            "schema": COLUMNS, "tests": tests, "datasets": datasets, "output_summary_path": str(report),
            "limitations": [
                "Cooldown approximates independent visits; it does not establish that price exited and re-entered an area",
                "Repeated rejections three or more bars apart can still belong to one prolonged visit",
                "Nearby group cooldown can suppress evidence that might represent a separate nearby area; thresholds depend on ATR and provisional grouping",
                "Only zone creation/update events are saved; raw/provisional evidence is reproducible from immutable inputs and parameters, not stored in extra audit files",
                "No active-zone snapshot, lifecycle, future reaction labels, strength prediction, merging or scenario generation",
                "Malformed OHLC/timestamps fail with a CSV line reference; failure may leave partial generated outputs and writes no new success summary",
            ],
            "warnings": [f"{d['symbol']} {d['source_timeframe']}: {w}" for d in datasets for w in d["warnings"]],
        }
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
