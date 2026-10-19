from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import unittest
from array import array
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

from zone_detection import lifecycle, registry
from zone_detection.swings import INTERVALS, iso, positive, utc


ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ("BTCUSDT", "ETHUSDT")
METHOD = "frozen_geometry_close_outcomes_v1"
CANDLE_FIELDS = ("open", "high", "low", "close", "atr_14_sma", "normalized_atr",
                 "body_ratio", "close_location_value", "relative_volume_20", "rolling_mean_return_20")
COMMON_FEATURES = ("symbol", "source_timeframe", "zone_family", "origin_type", "direction",
                   "width_atr", "signed_distance_atr", "normalized_atr", "body_ratio",
                   "close_location_value", "relative_volume_20", "rolling_mean_return_20")
MODEL_FEATURES = {"strong_zone": COMMON_FEATURES,
                  "scenario": COMMON_FEATURES + ("version_age_bars", "touch_penetration")}
COLUMNS = (
    "sample_id", "model", "registry_zone_id", "zone_version_at_utc", "source_file", "source_row",
    "symbol", "source_timeframe", "zone_family", "origin_type", "direction",
    "lower_price", "upper_price", "decision_at_utc", "decision_atr", "touch_at_utc",
    "observation_end_utc", "label", "label_status", "split", "split_status",
    "width_atr", "signed_distance_atr", "normalized_atr", "body_ratio", "close_location_value",
    "relative_volume_20", "rolling_mean_return_20", "version_age_bars", "touch_penetration",
)
TRAIN_START = utc("2020-01-01T00:00:00Z")
VALIDATION_START = utc("2024-01-01T00:00:00Z")
TEST_START = utc("2025-01-01T00:00:00Z")
TEST_END = utc("2025-09-01T00:00:00Z")


@dataclass(frozen=True)
class Config:
    wait_bars: int = 20
    defense_bars: int = 24
    scenario_bars: int = 12
    strong_atr: float = 1.50
    bounce_atr: float = 0.75
    failure_atr: float = 0.50
    sideways_padding_atr: float = 0.25
    sideways_range_atr: float = 1.00
    sideways_closes: int = 9
    embargo_hours: int = 176

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not positive(value):
                raise ValueError(f"{name} must be positive and finite")
        for name in ("wait_bars", "defense_bars", "scenario_bars", "sideways_closes", "embargo_hours"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")
        if self.sideways_closes > self.scenario_bars:
            raise ValueError("Sideways count exceeds horizon")


@dataclass
class Candles:
    start: datetime
    interval: timedelta
    data: dict[str, array]

    def __len__(self):
        return len(self.data["close"])

    def available(self, index):
        return self.start + (index + 1) * self.interval

    def index(self, available):
        elapsed = available - self.start
        index = elapsed // self.interval - 1
        if elapsed % self.interval or not 0 <= index < len(self):
            raise ValueError("Timestamp does not identify a completed input candle")
        return index


def read_candles(path, timeframe):
    """Compact numeric columns for ONE dataset; fail on gaps instead of bridging them."""
    data = {name: array("d") for name in CANDLE_FIELDS}
    interval, start = INTERVALS[timeframe], None
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp_utc", "available_at_utc", *CANDLE_FIELDS}
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames) or required - set(reader.fieldnames):
            raise ValueError("Missing or duplicate feature columns")
        for row in reader:
            opening, available = utc(row["timestamp_utc"]), utc(row["available_at_utc"])
            start = opening if start is None else start
            if ((opening - epoch) % interval or opening != start + len(data["close"]) * interval
                    or available != opening + interval):
                raise ValueError("Duplicate, missing, unordered or incomplete candle")
            for name in data:
                value = float(row[name]) if row[name] else math.nan
                if row[name] and not math.isfinite(value):
                    raise ValueError(f"Nonfinite feature: {name}")
                data[name].append(value)
            o, h, l, c = (data[name][-1] for name in ("open", "high", "low", "close"))
            if not all(positive(v) for v in (o, h, l, c)) or h < max(o, l, c) or l > min(o, c):
                raise ValueError("Invalid candle OHLC")
    if start is None:
        raise ValueError("Empty candle file")
    return Candles(start, interval, data)


def version_index(events, candles):
    """Validate transitions and retain exact original identities and version times."""
    versions, latest, first, next_version = {}, {}, {}, {}
    for event in events:
        candles.index(event.available)
        key = (event.registry_id, event.available)
        if key in versions:
            raise ValueError("Duplicate zone version")
        previous = latest.get(event.registry_id)
        registry.validate_transition(previous, event)
        if previous is not None:
            if event.available <= previous.available:
                raise ValueError("Unordered zone version")
            next_version[(previous.registry_id, previous.available)] = event.available
        first.setdefault(event.registry_id, event)
        latest[event.registry_id] = event
        versions[key] = event
    return versions, first, next_version


def read_touches(path, versions, next_version, candles):
    """Stream the entire lifecycle once; retain only first-touch indices per version.

    Validate sequence, source geometry, direction and knowledge at candle opening
    even for rows that are not selected as supervised examples.
    """
    touches, sequences, counts = {}, {}, Counter()
    previous_time = None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(lifecycle.COLUMNS):
            raise ValueError("Unexpected lifecycle schema")
        for row in reader:
            when = utc(row["event_at_utc"])
            key = (row["registry_zone_id"], utc(row["zone_version_at_utc"]))
            event = versions[key]
            sequence = int(row["event_sequence"])
            if (sequence != sequences.get(key, 0) + 1 or when < event.available
                    or (previous_time is not None and when < previous_time)
                    or row["symbol"] != event.symbol or row["source_timeframe"] != event.timeframe
                    or float(row["lower_price"]) != event.lower or float(row["upper_price"]) != event.upper
                    or row["direction"] != (lifecycle.direction_for(event) or "")):
                raise ValueError("Lifecycle sequence, geometry, direction or time mismatch")
            sequences[key], previous_time = sequence, when
            counts[row["event_type"]] += 1
            if not row["candle_time_utc"]:
                if row["event_type"] not in ("version_end", "monitoring_expiry"):
                    raise ValueError("Price event has no candle")
                continue
            index = candles.index(when)
            opening = utc(row["candle_time_utc"])
            if (opening != when - candles.interval or event.available > opening
                    or next_version.get(key, when + candles.interval) <= opening):
                raise ValueError("Zone version unavailable at evaluated candle opening")
            if row["event_type"] == "first_touch":
                if key in touches or int(row["episode_number"]) != 1:
                    raise ValueError("Duplicate or invalid first touch")
                o, h, l = (candles.data[n][index] for n in ("open", "high", "low"))
                side = "above" if o > event.upper else "below" if o < event.lower else ""
                if h < event.lower or l > event.upper or row["approach_side"] != side:
                    raise ValueError("Invalid lifecycle first-touch geometry")
                touches[key] = index
    return touches, dict(counts)


def threshold_hits(event, close, atr, favorable_atr, failure_atr):
    """Compare PRICE barriers directly: normalized subtraction can lose equality."""
    if lifecycle.direction_for(event) == "bullish":
        return close >= event.upper + favorable_atr * atr, close <= event.lower - failure_atr * atr
    return close <= event.lower - favorable_atr * atr, close >= event.upper + failure_atr * atr


def approach_status(event, candles, index):
    opening = candles.data["open"][index]
    if event.lower <= opening <= event.upper:
        return "ambiguous_approach"
    expected = opening > event.upper if lifecycle.direction_for(event) == "bullish" else opening < event.lower
    return None if expected else "wrong_approach"


def strong_outcome(event, candles, decision, config):
    """First frozen-geometry contact; exactly 24 bars INCLUDING the touch bar."""
    atr = candles.data["atr_14_sma"][decision]
    if threshold_hits(event, candles.data["close"][decision], atr, config.strong_atr, config.failure_atr)[1]:
        return "pre_touch_failure", None, None, decision
    last = min(decision + config.wait_bars, len(candles) - 1)
    for index in range(decision + 1, last + 1):
        touching = candles.data["high"][index] >= event.lower and candles.data["low"][index] <= event.upper
        if touching:
            status = approach_status(event, candles, index)
            if status:
                return status, None, index, index
            end = index + config.defense_bars - 1
            if end >= len(candles):
                return "censored_response", None, index, len(candles) - 1
            strong, failure = False, False
            for j in range(index, end + 1):
                favorable, adverse = threshold_hits(event, candles.data["close"][j], atr, config.strong_atr, config.failure_atr)
                strong |= favorable
                failure |= adverse
            return "labeled", int(strong and not failure), index, end
        if threshold_hits(event, candles.data["close"][index], atr, config.strong_atr, config.failure_atr)[1]:
            return "pre_touch_failure", None, None, index
    status = "no_touch" if last == decision + config.wait_bars else "censored_wait"
    return status, None, None, last


def scenario_outcome(event, candles, touch, config):
    """First future qualifying close wins; sideways requires a FULL horizon."""
    atr = candles.data["atr_14_sma"][touch]
    favorable, adverse = threshold_hits(event, candles.data["close"][touch], atr, config.bounce_atr, config.failure_atr)
    if favorable or adverse:
        return "already_resolved", None, touch
    end = min(touch + config.scenario_bars, len(candles) - 1)
    for index in range(touch + 1, end + 1):
        favorable, adverse = threshold_hits(event, candles.data["close"][index], atr, config.bounce_atr, config.failure_atr)
        if favorable:
            return "labeled", "bounce", index
        if adverse:
            return "labeled", "breakout", index
    if end < touch + config.scenario_bars:
        return "censored_response", None, end
    indices = range(touch + 1, end + 1)
    inside = sum(event.lower - config.sideways_padding_atr * atr <= candles.data["close"][i]
                 <= event.upper + config.sideways_padding_atr * atr for i in indices)
    span = max(candles.data["high"][i] for i in indices) - min(candles.data["low"][i] for i in indices)
    sideways = inside >= config.sideways_closes and span <= event.upper - event.lower + config.sideways_range_atr * atr
    return ("labeled", "consolidation", end) if sideways else ("unresolved", None, end)


def partition(when):
    for name, start, end in (("train", TRAIN_START, VALIDATION_START),
                             ("validation", VALIDATION_START, TEST_START), ("test", TEST_START, TEST_END)):
        if start <= when < end:
            return name, start, end
    return None, None, None


def split_status(decision, end, first_available, config):
    """Identity ownership is creation-based and identical for BOTH model datasets."""
    name, start, boundary = partition(decision)
    if name is None:
        return None, "outside_period"
    if partition(first_available)[0] != name:
        return name, "identity_other_partition"
    if name != "train" and decision < start + timedelta(hours=config.embargo_hours):
        return name, "embargo"
    if end >= boundary:
        return name, "purged_label_window"
    return name, "eligible"


def sample(event, model, candles, decision, touch, status, label, end, first_available, config):
    row = dict.fromkeys(COLUMNS)
    row.update(sample_id=f"{model}:{event.registry_id}:{iso(event.available)}", model=model,
               registry_zone_id=event.registry_id, zone_version_at_utc=iso(event.available),
               source_file=event.source_file, source_row=event.source_row, symbol=event.symbol,
               source_timeframe=event.timeframe, zone_family=event.family, origin_type=event.origin,
               direction=lifecycle.direction_for(event), lower_price=event.lower, upper_price=event.upper,
               label_status=status, label=label, split_status="no_decision")
    if touch is not None:
        row["touch_at_utc"] = iso(candles.available(touch))
    if decision is None:
        return row
    now = candles.available(decision)
    observed = candles.available(end)
    row.update(decision_at_utc=iso(now), observation_end_utc=iso(observed))
    row["split"], row["split_status"] = split_status(now, observed, first_available, config)
    atr = candles.data["atr_14_sma"][decision]
    if not positive(atr):
        return row
    row.update(decision_atr=atr, width_atr=(event.upper - event.lower) / atr,
               signed_distance_atr=(candles.data["close"][decision] - (event.lower + event.upper) / 2) / atr)
    for name in CANDLE_FIELDS[5:]:
        value = candles.data[name][decision]
        row[name] = value if math.isfinite(value) else None
    if model == "scenario":
        row["version_age_bars"] = (now - event.available) // candles.interval
        depth = ((event.upper - candles.data["low"][touch]) if row["direction"] == "bullish"
                 else (candles.data["high"][touch] - event.lower)) / (event.upper - event.lower)
        row["touch_penetration"] = max(0.0, min(1.0, depth))
    return row


def iter_samples(events, candles, touches, config):
    """Each identity supplies ONE A row; each directional version ONE B audit row."""
    first = {}
    events = sorted(events, key=registry.Event.order_key)
    for event in events:
        if lifecycle.direction_for(event) is None:
            continue
        original = first.setdefault(event.registry_id, event)
        decision = candles.index(event.available)
        if event is original:
            status, label, touch, end = ("invalid_decision_atr", None, None, decision)
            if positive(candles.data["atr_14_sma"][decision]):
                status, label, touch, end = strong_outcome(event, candles, decision, config)
            yield sample(event, "strong_zone", candles, decision, touch, status, label, end, original.available, config)
    # Sort B by observed decision time, not the earlier version-creation order.
    ordered = sorted(events, key=lambda e: (touches.get((e.registry_id, e.available), candles.index(e.available)), e.order_key()))
    for event in ordered:
        if lifecycle.direction_for(event) is None:
            continue
        original = first[event.registry_id]
        decision = candles.index(event.available)
        touch = touches.get((event.registry_id, event.available))
        status, label, end = "no_observed_first_touch", None, decision
        b_decision = None
        if touch is not None:
            if not decision < touch <= decision + config.wait_bars:
                status = "touch_after_wait"
            else:
                b_decision, end = touch, touch
                status = approach_status(event, candles, touch)
                if status is None:
                    status = "invalid_decision_atr"
                    if positive(candles.data["atr_14_sma"][touch]):
                        status, label, end = scenario_outcome(event, candles, touch, config)
        yield sample(event, "scenario", candles, b_decision, touch, status, label, end, original.available, config)


def output_paths(root, symbol, timeframe):
    return {model: root / "data" / "ml" / model / f"{symbol}_{timeframe}.csv" for model in MODEL_FEATURES}


def validate_outputs(paths, candles, versions, first, config):
    """Read back serialized outputs; verify every target, split and predictor value."""
    counts = {}
    for model, path in paths.items():
        seen, families = set(), {}
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(COLUMNS):
                raise ValueError("Output schema mismatch")
            for row in reader:
                if row["sample_id"] in seen:
                    raise ValueError("Duplicate sample")
                seen.add(row["sample_id"])
                key = (row["registry_zone_id"], utc(row["zone_version_at_utc"]))
                event = versions[key]
                decision = candles.index(utc(row["decision_at_utc"])) if row["decision_at_utc"] else None
                touch = candles.index(utc(row["touch_at_utc"])) if row["touch_at_utc"] else None
                label = None
                if decision is not None and positive(candles.data["atr_14_sma"][decision]):
                    if model == "strong_zone":
                        status, label, expected_touch, end = strong_outcome(event, candles, decision, config)
                        if touch != expected_touch or event is not first[event.registry_id]:
                            raise ValueError("Original identity/touch mismatch")
                    elif row["label_status"] in ("labeled", "unresolved", "censored_response", "already_resolved"):
                        status, label, end = scenario_outcome(event, candles, touch, config)
                    else:
                        status, end = row["label_status"], decision
                else:
                    status = row["label_status"]
                    end = decision if decision is not None else candles.index(event.available)
                expected = sample(event, model, candles, decision, touch, status, label, end,
                                  first[event.registry_id].available, config)
                if any(row[k] != ("" if v is None else str(v)) for k, v in expected.items()):
                    raise ValueError("Serialized target/feature/geometry/split validation failed")
                group = families.setdefault(event.family, {"rows": 0, "labels": Counter(), "statuses": Counter(),
                                                          "splits": Counter(), "eligible_labels": Counter(), "null_features": Counter()})
                group["rows"] += 1
                group["statuses"][row["label_status"]] += 1
                group["splits"][row["split_status"]] += 1
                if row["label"]:
                    group["labels"][row["label"]] += 1
                    if row["split_status"] == "eligible":
                        group["eligible_labels"][f'{row["split"]}:{row["label"]}'] += 1
                        for feature in MODEL_FEATURES[model]:
                            group["null_features"][feature] += not bool(row[feature])
        expected_count = sum(e.family != "volume_derived" for e in (first.values() if model == "strong_zone" else versions.values()))
        if len(seen) != expected_count:
            raise ValueError("Candidate identity/version count reconciliation failed")
        counts[model] = {"rows": len(seen), "bytes": path.stat().st_size, "families": families}
    return {"status": "PASS", "outputs": counts}


def process_dataset(root, symbol, timeframe, config=None, overwrite=False):
    """Existing outputs protected by default; explicit replacement is Phase-8-only."""
    if symbol not in SYMBOLS or timeframe not in INTERVALS:
        raise ValueError("Unsupported dataset")
    started, config = perf_counter(), config or Config()
    paths = output_paths(root, symbol, timeframe)
    if not overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError("Phase 8 output exists; regeneration requires a separately approved operation")
    source = root / "data" / "features" / timeframe / f"{symbol}_{timeframe}_features.csv"
    history = root / "data" / "zones" / "lifecycle" / f"{symbol}_{timeframe}.csv"
    candles = read_candles(source, timeframe)
    events = lifecycle.load_registry_events(root, symbol, timeframe)
    versions, first, next_version = version_index(events, candles)
    touches, lifecycle_counts = read_touches(history, versions, next_version, candles)
    with ExitStack() as stack:
        writers = {}
        for model, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(path.open("w" if overwrite else "x", newline="", encoding="utf-8"))
            writers[model] = csv.DictWriter(handle, fieldnames=COLUMNS)
            writers[model].writeheader()
        for row in iter_samples(events, candles, touches, config):
            writers[row["model"]].writerow(row)
    validation = validate_outputs(paths, candles, versions, first, config)
    return {"symbol": symbol, "timeframe": timeframe, "input_candles": len(candles),
            "registry_events": len(events), "registry_identities": len(first),
            "neutral_volume_versions_excluded": sum(e.family == "volume_derived" for e in events),
            "lifecycle_event_counts": lifecycle_counts,
            "inputs": [str(source), str(history), str(root / "data/zones/unified" / f"{symbol}_{timeframe}.csv")],
            "outputs": {k: str(v) for k, v in paths.items()}, "validation": validation,
            "runtime_seconds": perf_counter() - started}


def run_tests():
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=1, stream=sys.stderr).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("Regression tests failed; generation stopped")
    return {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
            "skipped": len(result.skipped), "successful": result.wasSuccessful()}


def write_summary(root, results, tests, config=None, verification=None):
    expected = {(s, tf) for s in SYMBOLS for tf in INTERVALS}
    if (len(results) != 6 or {(r["symbol"], r["timeframe"]) for r in results} != expected
            or not tests.get("successful") or not tests.get("tests_run")
            or any(r["validation"]["status"] != "PASS" for r in results)
            or any(not Path(p).is_file() for r in results for p in r["outputs"].values())):
        raise ValueError("Six validated datasets and actual passing tests required")
    summary = {"phase": 8, "methodology_version": METHOD, "parameters": asdict(config or Config()),
               "model_features": MODEL_FEATURES, "schema": COLUMNS, "tests": tests, "datasets": results,
               "runtime_seconds": sum(r["runtime_seconds"] for r in results),
               "runtime_measurement": "Final generation plus built-in read-back validation; independent audits, tests and discarded initial runs excluded",
               "verification": verification,
               "training_filter": "label_status == labeled AND split_status == eligible; features ONLY from model_features",
               "targets": {
                   "strong_zone": "First creation only. Wait next 20 bars for first frozen-range intersection; exclude wrong/ambiguous approach or adverse close before touch (including decision close). Evaluate 24 bars including touch. 1 iff any 1.50 ATR favorable close AND no 0.50 ATR adverse close. Otherwise 0 only with complete valid response.",
                   "scenario": "First lifecycle touch per version within 20 bars. Decision at touch close; exclude already resolved. Next 12 bars only: first 0.75 ATR favorable close -> bounce, first 0.50 ATR adverse close -> breakout. Else full-horizon consolidation requires >=9 closes in zone +/-0.25 ATR and total range <= width+1.00 ATR. Otherwise unresolved/censored.",
                   "thresholds": "Bullish favorable measured above upper, adverse below lower; bearish favorable below lower, adverse above upper. Equality qualifies. No wick-order inference. ATR frozen separately at each model's decision.",
               },
               "split_policy": "Decision-time years: train 2020-2023, validation 2024, test Jan-Aug 2025. Creation partition owns identity across both tasks. Purge label endpoints >= split end. Embargo first 176h of validation/test.",
               "warnings": [
                   "All thresholds provisional; no models fitted or preprocessing learned",
                   "A is conditional on eligible observed touch; exclusions are not weak labels",
                   "Frozen geometry outcomes are not live-zone survival; ignore later updates/invalidation for response labels",
                   "B uses lifecycle-observed first touches per version, not independent identities or retests",
                   "No observed B first touch can mean replacement, invalidation, expiry, no contact or EOF; it is not a negative",
                   "Neutral estimated volume profiles are excluded as targets; no optional volume predictors added",
                   "Close-based outcomes ignore wick-hit ordering; ambiguous approaches are excluded",
                   "176h embargo is conservative for 4h horizons and unnecessarily large for shorter timeframes",
                   "Purging plus identity ownership reduces correlated leakage but does not remove all nearby-zone dependence",
                   "Missing predictor values remain null for later training-only preprocessing; missing/nonpositive decision ATR excludes labels",
                   "Audit timestamps, labels, source IDs and statuses are NOT predictors",
               ]}
    path = root / "reports" / "phase8_summary.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", choices=SYMBOLS)
    parser.add_argument("--timeframe", choices=INTERVALS)
    args = parser.parse_args(argv)
    if bool(args.symbol) != bool(args.timeframe):
        parser.error("Provide both --symbol and --timeframe, or neither")
    jobs = [(args.symbol, args.timeframe)] if args.symbol else [(s, tf) for s in SYMBOLS for tf in INTERVALS]
    if any(p.exists() for s, tf in jobs for p in output_paths(ROOT, s, tf).values()):
        raise FileExistsError("Existing outputs protected; no files written")
    if not args.symbol and (ROOT / "reports/phase8_summary.json").exists():
        raise FileExistsError("Existing summary protected")
    tests = run_tests()
    results = []
    for symbol, timeframe in jobs:
        result = process_dataset(ROOT, symbol, timeframe)
        results.append(result)
        print(json.dumps(result), flush=True)
    if not args.symbol:
        print(write_summary(ROOT, results, tests))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
