"""Version-aware, completed-candle interaction evidence; no predictive labels.

Evaluate [t,T) using versions known at t, then expire/install versions at T.
Contact episodes and event sequences are version-scoped, not new identities.
An update censors an open episode and begins a fresh assessment; it cannot
rewrite or reactivate the old version. A new version's first_touch is not a
new independent zone or proof of a market retest across versions.

Entry side uses the contact candle's open; an open inside/on the interval is
ambiguous. Penetration is the episode maximum reported so far. Range spans
and close crossings do not establish intrabar order or actual traded paths.
One-close invalidation and 30/90/180-day monitoring caps are provisional rules.
Volume areas are neutral OHLCV proxies, replaced at the next profile slot.

Only change events are saved. event_detail distinguishes boundary/direction,
gap endpoint crossings and censoring reasons. Timer rows have no candle time.
zone_version_at_utc is source-event availability; event_at_utc is observation
availability; candle_time_utc is the observed candle's opening. Sequence and
episode numbers restart for each version. contact_bars excludes exit candles;
penetration stays null throughout an episode whose entry side is unknown.
There is no automatic generation on import. generate_dataset runs one dataset
without overwriting; write_summary requires six successfully validated results.
"""

from __future__ import annotations

import csv
import heapq
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import Iterable, Iterator

from zone_detection import registry
from zone_detection.swings import Candle, INTERVALS, iso, positive, read_candles, utc


COLUMNS = (
    "registry_zone_id", "zone_version_at_utc", "symbol", "source_timeframe",
    "event_sequence", "event_at_utc", "candle_time_utc", "event_type",
    "episode_number", "lower_price", "upper_price", "direction", "approach_side",
    "penetration_fraction", "contact_bars", "event_detail",
)
METHODOLOGY_VERSION = "completed_candle_versioned_interactions_v1"


@dataclass(frozen=True)
class Config:
    monitor_15m_days: float = 30
    monitor_1h_days: float = 90
    monitor_4h_days: float = 180
    penetration_epsilon: float = 1e-9

    def __post_init__(self) -> None:
        for timeframe, interval in INTERVALS.items():
            days = getattr(self, f"monitor_{timeframe}_days")
            if not positive(days):
                raise ValueError("Monitoring days must be finite and positive")
            duration = self.duration(timeframe)
            if duration < interval or duration % interval:
                raise ValueError("Monitoring caps must be whole candle intervals")
        if not math.isfinite(self.penetration_epsilon) or not 0 <= self.penetration_epsilon < 1:
            raise ValueError("Penetration epsilon must be finite and in [0, 1)")

    def duration(self, timeframe: str) -> timedelta:
        return timedelta(days=getattr(self, f"monitor_{timeframe}_days"))


def load_registry_events(root: Path, symbol: str, timeframe: str,
                         through: datetime | None = None) -> list[registry.Event]:
    """Read one unified history and exact source rows, once per referenced file.

    through is an inclusive, complete availability cutoff for prefix audits.
    Source paths must be the expected family files, never arbitrary paths.
    Reusing normalize/record checks all exported source fields and recovers
    direction, original creation time, evidence counts and profile identity.
    """
    expected = registry.source_paths(root, symbol, timeframe)
    path = root / "data" / "zones" / "unified" / f"{symbol}_{timeframe}.csv"
    requests: dict[Path, dict[int, dict]] = {}
    previous = None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)) or set(registry.COLUMNS) - set(fields):
            raise ValueError(f"{path}: missing or duplicate registry columns")
        for line, row in enumerate(reader, 2):
            try:
                available = utc(row["available_at_utc"])
                if previous is not None and available < previous:
                    raise ValueError("Registry history is not chronological")
                if through is not None and available > through:
                    break
                previous = available
                family = row["zone_family"]
                source = expected[family]
                if row["source_file"] != source.relative_to(root).as_posix():
                    raise ValueError("Unexpected source file reference")
                source_line = registry.integer(row["source_row"])
                references = requests.setdefault(source, {})
                if source_line < 2 or source_line in references:
                    raise ValueError("Invalid or duplicated source row reference")
                references[source_line] = row
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                raise ValueError(f"{path}, CSV line {line}: {exc}") from exc
    events = []
    for source, references in requests.items():
        family = next(k for k, v in expected.items() if v == source)
        with source.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            if not fields or len(fields) != len(set(fields)) or set(registry.SOURCE_COLUMNS[family]) - set(fields):
                raise ValueError(f"{source}: missing or duplicate source columns")
            for line, row in enumerate(reader, 2):
                if line not in references:
                    continue
                exported = references.pop(line)
                try:
                    event = registry.normalize(row, family, symbol, timeframe,
                                               source.relative_to(root).as_posix(), line)
                    count = int(exported["overlapping_candidate_count"])
                    family_count = int(exported["overlapping_family_count"])
                    flags = set(filter(None, exported["overlap_family_flags"].split("|")))
                    if count < family_count or family_count != len(flags) or flags - (set(registry.FAMILIES) - {family}):
                        raise ValueError("Invalid overlap metadata")
                    projected = event.record(count, flags)
                    if any(exported[k] != ("" if v is None else str(v)) for k, v in projected.items()):
                        raise ValueError("Registry/source event mismatch")
                except (ValueError, TypeError, KeyError, AttributeError) as exc:
                    raise ValueError(f"{source}, CSV line {line}: {exc}") from exc
                events.append(event)
                if not references:
                    break
        if references:
            raise ValueError(f"{source}: referenced CSV rows do not exist: {sorted(references)}")
    return sorted(events, key=registry.Event.order_key)


def direction_for(event: registry.Event) -> str | None:
    if event.family == "volume_derived":
        return None
    if event.family == "imbalance":
        if event.direction not in ("bullish", "bearish"):
            raise ValueError("Explicit original imbalance direction required")
        if event.origin not in ("displacement_body", f"{event.direction}_fvg"):
            raise ValueError("Imbalance origin/direction mismatch")
        return event.direction
    directions = ({"swing_low_cluster": "bullish", "swing_high_cluster": "bearish"}
                  if event.family == "structural" else
                  {"bullish_rejection_cluster": "bullish", "bearish_rejection_cluster": "bearish"})
    if event.origin not in directions:
        raise ValueError("Unknown directional origin")
    return directions[event.origin]


@dataclass
class ContactState:
    event: registry.Event
    direction: str | None
    sequence: int = 0
    episode: int = 0
    in_contact: bool = False
    contact_bars: int = 0
    approach: str | None = None
    penetration: float | None = None
    spanned: bool = False
    invalidated: bool = False

    def record(self, when: datetime, kind: str, detail: str,
               candle: Candle | None = None) -> dict:
        self.sequence += 1
        e = self.event
        return {
            "registry_zone_id": e.registry_id, "zone_version_at_utc": iso(e.available),
            "symbol": e.symbol, "source_timeframe": e.timeframe,
            "event_sequence": self.sequence, "event_at_utc": iso(when),
            "candle_time_utc": iso(candle.timestamp) if candle is not None else None,
            "event_type": kind, "episode_number": self.episode,
            "lower_price": e.lower, "upper_price": e.upper, "direction": self.direction,
            "approach_side": self.approach, "penetration_fraction": self.penetration,
            "contact_bars": self.contact_bars, "event_detail": detail,
        }


def observe_candle(state: ContactState, candle: Candle, previous_close: float | None,
                   config: Config) -> list[dict]:
    """Emit contact changes first, then crossing facts, then rule invalidation.

    First touch/exit/retest apply within one boundary version. An open outside
    supplies the episode's entry side; inside/boundary opens remain ambiguous.
    A crossing compares previous close with current close, including gap
    endpoints, but never asserts that intermediate prices traded.
    """
    lower, upper = state.event.lower, state.event.upper
    now = candle.available_at
    rows = []
    touching = candle.high >= lower and candle.low <= upper
    if touching:
        first_bar = not state.in_contact
        if first_bar:
            state.episode += 1
            state.in_contact, state.contact_bars, state.spanned = True, 0, False
            state.approach = "above" if candle.open > upper else "below" if candle.open < lower else None
            state.penetration = None
        state.contact_bars += 1
        depth = None
        if state.approach == "above":
            depth = (upper - candle.low) / (upper - lower)
        elif state.approach == "below":
            depth = (candle.high - lower) / (upper - lower)
        if depth is not None:
            depth = min(1.0, max(0.0, depth))
        if first_bar:
            state.penetration = depth
            rows.append(state.record(now, "first_touch" if state.episode == 1 else "retest", "range_intersection", candle))
        elif depth is not None and depth - state.penetration > config.penetration_epsilon:
            state.penetration = depth
            rows.append(state.record(now, "penetration_increase", "episode_maximum", candle))
        if not state.spanned and candle.low <= lower and candle.high >= upper:
            state.spanned = True
            rows.append(state.record(now, "range_span", "both_boundaries_intrabar_order_unknown", candle))
    elif state.in_contact:
        rows.append(state.record(now, "contact_exit", "above" if candle.low > upper else "below", candle))
        state.in_contact = False
    if previous_close is not None:
        for boundary, price in (("lower", lower), ("upper", upper)):
            movement = None
            if previous_close >= price and candle.close < price:
                movement = "down"
            elif previous_close <= price and candle.close > price:
                movement = "up"
            if movement is None:
                continue
            gap = (previous_close >= price and candle.open < price) or (previous_close <= price and candle.open > price)
            detail = f"{boundary}_{movement}" + ("_gap_endpoints" if gap else "")
            rows.append(state.record(now, "boundary_crossing", detail, candle))
            if ((state.direction == "bullish" and boundary == "lower" and movement == "down")
                    or (state.direction == "bearish" and boundary == "upper" and movement == "up")):
                state.invalidated = True
        if state.invalidated:
            rows.append(state.record(now, "observed_invalidation", "provisional_one_close_rule", candle))
    return rows


def iter_interactions(candles: Iterable[Candle], events: Iterable[registry.Event],
                      symbol: str, timeframe: str, clock: registry.VolumeClock | None = None,
                      config: Config | None = None, stats: Counter | None = None) -> Iterator[dict]:
    """Replay complete candle prefixes; only monitored states are price-scanned.

    The history dictionary validates identities, not market eligibility. Heap
    entries from replaced/invalidated versions are ignored. Timers are emitted
    only through observed candle completion; EOF never invents terminal rows.
    """
    config = config or Config()
    stats = stats if stats is not None else Counter()
    interval = INTERVALS[timeframe]
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    if clock is not None and (clock.stride % interval or clock.window % interval
                              or (clock.first_profile_at - epoch) % interval
                              or (clock.last_input_available - epoch) % interval):
        raise ValueError("Volume calendar must align with completed candle times")
    ordered = iter(sorted(events, key=registry.Event.order_key))
    pending = next(ordered, None)
    history: dict[str, registry.Event] = {}
    monitored: dict[str, ContactState] = {}
    expiry = []
    profile_times = {}
    previous = None

    def install(now):
        nonlocal pending
        batch = []
        while pending is not None and pending.available == now:
            batch.append(pending)
            pending = next(ordered, None)
        if len({e.registry_id for e in batch}) != len(batch):
            raise ValueError("Multiple versions of one identity at the same time")
        if len({e.profile_id for e in batch if e.family == "volume_derived"}) > 1:
            raise ValueError("Multiple profile identities at the same time")
        for e in batch:
            if (e.symbol != symbol or e.timeframe != timeframe or e.family not in registry.FAMILIES
                    or e.registry_id != f"{symbol}:{timeframe}:{e.family}:{e.source_id}"
                    or not positive(e.lower) or not positive(e.upper) or e.upper <= e.lower
                    or e.available.tzinfo is None or (e.available - epoch) % interval):
                raise ValueError("Invalid registry event asset, identity, price or time")
            registry.validate_transition(history.get(e.registry_id), e)
            direction = direction_for(e)
            deadline = now + config.duration(timeframe)
            if e.family == "volume_derived":
                if (clock is None or e.available != clock.latest_profile_at(now)
                        or now > clock.last_input_available or e.formation + clock.window != now
                        or not e.profile_id):
                    raise ValueError("Volume event does not match the Phase 5 calendar")
                if e.profile_id in profile_times and profile_times[e.profile_id] != now:
                    raise ValueError("Volume profile ID reused at a different time")
                profile_times[e.profile_id] = now
                deadline = now + clock.stride
            old = monitored.pop(e.registry_id, None)
            if old is not None and old.in_contact:
                yield old.record(now, "version_end", e.kind)
            if e.registry_id not in history:
                stats["source_identities"] += 1
            history[e.registry_id] = e
            monitored[e.registry_id] = ContactState(e, direction)
            heapq.heappush(expiry, (deadline, e.registry_id, now))
            stats["source_versions"] += 1
        stats["peak_monitored_versions"] = max(stats["peak_monitored_versions"], len(monitored))

    for candle in candles:
        if (candle.timestamp.tzinfo is None or candle.available_at != candle.timestamp + interval
                or (candle.timestamp - epoch) % interval
                or (previous is not None and candle.timestamp != previous.available_at)):
            raise ValueError("Missing, duplicate, off-grid or invalid candle timestamps")
        if (not all(positive(p) for p in (candle.open, candle.high, candle.low, candle.close))
                or candle.high < max(candle.open, candle.low, candle.close)
                or candle.low > min(candle.open, candle.close)):
            raise ValueError("Invalid candle OHLC")
        if previous is None:
            if pending is not None and pending.available < candle.timestamp:
                raise ValueError("Candle coverage starts after source event history")
            yield from install(candle.timestamp)
        stats["input_candles"] += 1
        stats["monitored_candle_checks"] += len(monitored)
        invalidated = []
        for key, state in monitored.items():
            for row in observe_candle(state, candle, previous.close if previous is not None else None, config):
                stats[row["event_type"]] += 1
                yield row
            if state.invalidated:
                invalidated.append(key)
        for key in invalidated:
            del monitored[key]
        now = candle.available_at
        if pending is not None and pending.available < now:
            raise ValueError("Registry event falls between completed candle times")
        while expiry and expiry[0][0] <= now:
            deadline, key, version = heapq.heappop(expiry)
            state = monitored.get(key)
            if state is not None and state.event.available == version:
                kind = "version_end" if state.event.family == "volume_derived" else "monitoring_expiry"
                detail = "next_profile_slot" if state.event.family == "volume_derived" else "observation_cap_not_invalidation"
                stats[kind] += 1
                yield state.record(deadline, kind, detail)
                del monitored[key]
        for row in install(now):
            stats[row["event_type"]] += 1
            yield row
        previous = candle
    if previous is None:
        raise ValueError("Empty candle input")
    stats["monitored_versions_at_end"] = len(monitored)
    stats["open_episodes_at_end"] = sum(s.in_contact for s in monitored.values())


def process_dataset(root: Path, symbol: str, timeframe: str, output: Path,
                    config: Config | None = None, overwrite: bool = False) -> dict:
    """Explicit single-dataset writer; failures may leave a partial new output.

    No other dataset or JSON summary is invoked automatically.
    """
    if symbol not in ("BTCUSDT", "ETHUSDT") or timeframe not in INTERVALS:
        raise ValueError("Unsupported symbol/timeframe")
    source = root / "data" / "features" / timeframe / f"{symbol}_{timeframe}_features.csv"
    unified = root / "data" / "zones" / "unified" / f"{symbol}_{timeframe}.csv"
    report = root / "reports" / "zone_detection" / "volume_summary.json"
    inputs = [source, unified, report, *registry.source_paths(root, symbol, timeframe).values()]
    target = output.resolve()
    historical_output = (root / "data" / "zones" / "lifecycle" / f"{symbol}_{timeframe}.csv").resolve()
    if output.suffix.lower() != ".csv" or (target.is_relative_to((root / "data").resolve()) and target != historical_output):
        raise ValueError("Output must be a CSV; data/ writes are restricted to this lifecycle dataset")
    if target in {p.resolve() for p in inputs}:
        raise ValueError("Output must not overwrite any source input")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}; explicit overwrite required")
    config = config or Config()
    with report.open(encoding="utf-8-sig") as handle:
        clock = registry.profile_clock({"volume_derived": json.load(handle)}, symbol, timeframe)
    events = load_registry_events(root, symbol, timeframe)
    stats, input_stats = Counter(), Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if overwrite else "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in iter_interactions(read_candles(source, timeframe, input_stats), events,
                                     symbol, timeframe, clock, config, stats):
            writer.writerow(row)
            stats["output_event_rows"] += 1
    return {"symbol": symbol, "source_timeframe": timeframe, "input_path": str(source),
            "registry_path": str(unified), "output_path": str(output),
            "statistics": dict(stats), "candle_validation": dict(input_stats)}


def generate_dataset(root: Path, symbol: str, timeframe: str) -> dict:
    """Run one approved historical dataset; existing files are always protected."""
    output = root / "data" / "zones" / "lifecycle" / f"{symbol}_{timeframe}.csv"
    started = perf_counter()
    result = process_dataset(root, symbol, timeframe, output)
    result.update(runtime_seconds=perf_counter() - started, file_size_bytes=output.stat().st_size)
    return result


def write_summary(root: Path, results: list[dict], tests: dict) -> Path:
    """Save one concise run report only after every dataset has passed its audit."""
    expected = {(symbol, tf) for symbol in ("BTCUSDT", "ETHUSDT") for tf in INTERVALS}
    actual = {(r["symbol"], r["source_timeframe"]) for r in results}
    if len(results) != len(expected) or actual != expected:
        raise ValueError("Summary requires exactly the six historical datasets")
    if not tests.get("successful") or not tests.get("tests_run"):
        raise ValueError("Passing regression test results required")
    if any(r["validation"]["status"] != "PASS" or not Path(r["output_path"]).is_file() for r in results):
        raise ValueError("All historical outputs must exist and pass validation")
    summary = {
        "phase": 7, "methodology_version": METHODOLOGY_VERSION,
        "generated_at_utc": iso(datetime.now(timezone.utc)), "parameters": asdict(Config()),
        "schema": COLUMNS, "tests": tests, "datasets": results,
        "total_generation_seconds": sum(r["runtime_seconds"] for r in results),
        "runtime_measurement": "Generation wall time includes source enrichment, replay and CSV writing; separate audits/tests excluded",
        "memory_measurement": "Generation process peak working set on Windows, including Python/input loading; fresh process per dataset",
        "warnings": [
            "Directional hypotheses and one-close invalidation are provisional, not verified market validity",
            "Episodes and first_touch counts are version-scoped, not independent zone counts",
            "Monitoring expiry, source updates and EOF censor observations; they are not price outcomes",
            "Unknown approach/penetration and timer candle times intentionally remain null",
            "Volume areas are neutral correlated OHLCV estimates, replaced at every scheduled profile slot including empty slots",
            "Range spans and gap endpoint crossings do not establish intrabar order or execution paths",
            "No strength or scenario labels are created; use the registry to retain candidates with no interaction rows",
            "Interaction rows are known at event_at_utc; later interaction evidence must not become creation-time model inputs",
        ],
    }
    report = root / "reports" / "zone_detection" / "lifecycle_summary.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report
