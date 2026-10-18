"""Strict three-candle FVG geometry and completed-candle displacement evidence.

Thresholds are provisional research assumptions. An OHLC candle body is a price
interval proxy, not an observed liquidity void. No strength or outcome is inferred.

Body candidates are created at their own close. If the next completed candle
confirms an aligned FVG, a confirmation event reuses that candidate ID. It does
not create another independent candidate or rewrite the earlier body interval.
Thus this single CSV is a creation/confirmation event history, not an active-zone
snapshot. --fvg-only omits optional body candidates and produces creation rows only.
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
from typing import Iterable, Iterator

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zone_detection.swings import INTERVALS, iso, positive, utc


PROJECT_DIR = Path(__file__).resolve().parents[2]
SYMBOLS = ("BTCUSDT", "ETHUSDT")
REQUIRED_COLUMNS = (
    "timestamp_utc", "available_at_utc", "open", "high", "low", "close",
    "atr_14_sma", "body_ratio", "atr_normalized_body",
)
COLUMNS = (
    "zone_id", "symbol", "source_timeframe", "zone_family", "imbalance_type",
    "direction", "lower_price", "upper_price", "formation_start_utc",
    "available_at_utc", "reference_candle_time_utc", "gap_width", "gap_width_atr",
    "displacement_body_atr", "displacement_body_ratio", "aligned_displacement",
    "candidate_status", "c1_time_utc", "c2_time_utc", "c3_time_utc",
    "reference_atr", "reference_atr_available_at_utc", "zone_width",
    "event_type", "candidate_created_at_utc",
)


@dataclass(frozen=True)
class Config:
    displacement_body_atr_min: float = 1.0
    displacement_body_ratio_min: float = 0.60

    def __post_init__(self) -> None:
        if not positive(self.displacement_body_atr_min):
            raise ValueError("displacement_body_atr_min must be finite and positive")
        if not positive(self.displacement_body_ratio_min) or self.displacement_body_ratio_min > 1:
            raise ValueError("displacement_body_ratio_min must be in (0, 1]")


@dataclass(frozen=True)
class FeatureCandle:
    timestamp: datetime
    available_at: datetime
    open: float
    high: float
    low: float
    close: float
    atr: float | None
    body_ratio: float | None


def check_header(path: Path) -> None:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        fields = csv.DictReader(handle).fieldnames
    if not fields or len(fields) != len(set(fields)):
        raise ValueError(f"{path}: missing header or duplicate column names")
    missing = set(REQUIRED_COLUMNS) - set(fields)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")


def feature_number(value: str, name: str, stats: Counter) -> float | None:
    if not value.strip():
        stats[f"missing_{name}_rows"] += 1
        return None
    try:
        number = float(value)
    except ValueError:
        number = math.nan
    if not math.isfinite(number):
        stats[f"invalid_{name}_rows"] += 1
        return None
    return number


def body_atr(candle: FeatureCandle) -> float | None:
    if not positive(candle.atr):
        return None
    ratio = abs(candle.close - candle.open) / candle.atr
    return ratio if math.isfinite(ratio) else None


def read_candles(path: Path, timeframe: str, stats: Counter) -> Iterator[FeatureCandle]:
    """Stream validated input. Bad OHLC/time fails; bad evidence stays missing.

    body_ratio is reused after checking its OHLC definition. atr_normalized_body
    is audited against the signed OHLC/ATR formula at the CSV's stored precision.
    Displacement uses the explicitly requested absolute-body/ATR calculation.
    Optional mark/index columns are neither required nor modified.
    """
    check_header(path)
    interval = INTERVALS[timeframe]
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    previous = None
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for line, row in enumerate(csv.DictReader(handle), 2):
            try:
                timestamp, available = utc(row["timestamp_utc"]), utc(row["available_at_utc"])
                if (timestamp - epoch) % interval != timedelta(0):
                    raise ValueError("opening timestamp is off the UTC interval grid")
                if available != timestamp + interval:
                    raise ValueError("availability is not candle close time")
                if previous is not None and timestamp - previous != interval:
                    raise ValueError("missing, duplicate or unordered candle timestamp")
                opening, high, low, close = (float(row[k]) for k in ("open", "high", "low", "close"))
                if (not all(positive(p) for p in (opening, high, low, close))
                        or high < max(opening, low, close) or low > min(opening, close)):
                    raise ValueError("OHLC must be finite, positive and internally consistent")
                atr = feature_number(row["atr_14_sma"], "atr", stats)
                if atr is not None and not positive(atr):
                    stats["invalid_atr_rows"] += 1
                    atr = None
                ratio = feature_number(row["body_ratio"], "body_ratio", stats)
                if high == low:
                    stats["zero_range_rows"] += 1
                    if ratio is not None:
                        stats["invalid_body_ratio_rows"] += 1
                    ratio = None
                elif ratio is not None and (
                    not 0 <= ratio <= 1
                    or not math.isclose(ratio, abs(close - opening) / (high - low),
                                        rel_tol=1e-8, abs_tol=1e-10)
                ):
                    stats["invalid_body_ratio_rows"] += 1
                    ratio = None
                source_body_atr = feature_number(row["atr_normalized_body"], "atr_normalized_body", stats)
                if (source_body_atr is not None and atr is not None
                        and not math.isclose(source_body_atr, (close - opening) / atr,
                                             rel_tol=1e-8, abs_tol=1e-10)):
                    stats["atr_normalized_body_mismatch_rows"] += 1
                candle = FeatureCandle(timestamp, available, opening, high, low, close, atr, ratio)
                if atr is not None and body_atr(candle) is None:
                    stats["nonfinite_derived_body_atr_rows"] += 1
            except (ValueError, TypeError, AttributeError, KeyError) as exc:
                raise ValueError(f"{path}, CSV line {line}: {exc}") from exc
            previous = timestamp
            stats["input_rows"] += 1
            yield candle


def displacement_direction(candle: FeatureCandle, config: Config) -> str | None:
    ratio = body_atr(candle)
    if (ratio is None or candle.body_ratio is None or not math.isfinite(candle.body_ratio)
            or not 0 <= candle.body_ratio <= 1 or candle.high <= candle.low
            or ratio < config.displacement_body_atr_min
            or candle.body_ratio < config.displacement_body_ratio_min
            or candle.close == candle.open):
        return None
    return "bullish" if candle.close > candle.open else "bearish"


def candidate_id(symbol: str, timeframe: str, candle: FeatureCandle,
                 kind: str, direction: str) -> str:
    return f"{symbol}_{timeframe}_imbalance_{candle.timestamp.strftime('%Y%m%dT%H%M%SZ')}_{kind}_{direction}"


def valid_prices(candle: FeatureCandle) -> bool:
    return (all(positive(p) for p in (candle.open, candle.high, candle.low, candle.close))
            and candle.high >= max(candle.open, candle.low, candle.close)
            and candle.low <= min(candle.open, candle.close))


def fvg_candidate(c1: FeatureCandle, c2: FeatureCandle, c3: FeatureCandle,
                  symbol: str, timeframe: str, config: Config) -> dict | None:
    """One strict FVG, available at C3 close; all normalized evidence uses C2 ATR."""
    interval = INTERVALS[timeframe]
    if (not all(valid_prices(c) for c in (c1, c2, c3))
            or c2.timestamp - c1.timestamp != interval or c3.timestamp - c2.timestamp != interval
            or any(c.available_at != c.timestamp + interval for c in (c1, c2, c3))):
        return None
    if c3.low > c1.high:
        direction, lower, upper = "bullish", c1.high, c3.low
    elif c3.high < c1.low:
        direction, lower, upper = "bearish", c3.high, c1.low
    else:
        return None
    gap = upper - lower
    reference_atr = c2.atr if positive(c2.atr) else None
    normalized = gap / reference_atr if reference_atr is not None else None
    if normalized is not None and not math.isfinite(normalized):
        normalized = None
    kind = f"{direction}_fvg"
    return dict(zip(COLUMNS, (
        candidate_id(symbol, timeframe, c2, "fvg", direction), symbol, timeframe,
        "imbalance", kind, direction, lower, upper, iso(c1.timestamp), iso(c3.available_at),
        iso(c2.timestamp), gap, normalized, body_atr(c2), c2.body_ratio,
        displacement_direction(c2, config) == direction, "candidate", iso(c1.timestamp),
        iso(c2.timestamp), iso(c3.timestamp), reference_atr,
        iso(c2.available_at), gap, "create", iso(c3.available_at),
    )))


def displacement_candidate(candle: FeatureCandle, symbol: str, timeframe: str,
                           config: Config) -> dict | None:
    """Pure C2-close body proxy; no later candle is read or assumed here."""
    if not valid_prices(candle):
        return None
    direction = displacement_direction(candle, config)
    if direction is None:
        return None
    lower, upper = min(candle.open, candle.close), max(candle.open, candle.close)
    return dict(zip(COLUMNS, (
        candidate_id(symbol, timeframe, candle, "body", direction), symbol, timeframe,
        "imbalance", "displacement_body", direction, lower, upper, iso(candle.timestamp),
        iso(candle.available_at), iso(candle.timestamp), None, None, body_atr(candle),
        candle.body_ratio, False, "candidate", None, iso(candle.timestamp), None,
        candle.atr, iso(candle.available_at), upper - lower, "create", iso(candle.available_at),
    )))


def detect_candidates(candles: Iterable[FeatureCandle], symbol: str, timeframe: str,
                      config: Config, include_displacement_body: bool = True) -> Iterator[dict]:
    """Emit chronological immutable events with at most one ID per impulse.

    At a common close time, confirm the previous candle's FVG before creating
    the current candle's body proxy. No look-ahead or EOF decisions are needed.
    Earlier body rows never assert that a future FVG will or will not occur.
    """
    window: deque[FeatureCandle] = deque(maxlen=3)
    for candle in candles:
        window.append(candle)
        if len(window) == 3:
            c1, c2, c3 = window
            fvg = fvg_candidate(c1, c2, c3, symbol, timeframe, config)
            if fvg is not None:
                if include_displacement_body and fvg["aligned_displacement"]:
                    fvg["zone_id"] = candidate_id(symbol, timeframe, c2, "body", fvg["direction"])
                    fvg["event_type"] = "fvg_confirmation"
                    fvg["candidate_created_at_utc"] = iso(c2.available_at)
                yield fvg
        if include_displacement_body:
            body = displacement_candidate(candle, symbol, timeframe, config)
            if body is not None:
                yield body


def paths_for(root: Path, symbol: str, timeframe: str) -> tuple[Path, Path]:
    return (root / "data" / "features" / timeframe / f"{symbol}_{timeframe}_features.csv",
            root / "data" / "zones" / "imbalance" / f"{symbol}_{timeframe}.csv")


def preflight(root: Path, overwrite: bool) -> list[tuple[str, str, Path, Path]]:
    jobs = [(symbol, timeframe, *paths_for(root, symbol, timeframe))
            for timeframe in INTERVALS for symbol in SYMBOLS]
    targets = [job[3] for job in jobs] + [root / "reports" / "zone_detection" / "imbalance_summary.json"]
    existing = [str(p) for p in targets if p.exists()]
    if existing and not overwrite:
        raise FileExistsError("Outputs exist; use --overwrite to regenerate: " + ", ".join(existing))
    for _, _, source, _ in jobs:
        check_header(source)
    return jobs


def process_dataset(symbol: str, timeframe: str, source: Path, output: Path,
                    config: Config, overwrite: bool = False,
                    include_displacement_body: bool = True) -> dict:
    if source.resolve() == output.resolve():
        raise ValueError("Input and output paths must differ")
    check_header(source)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}; use --overwrite to regenerate")
    stats = Counter()
    first_open = last_available = None

    def tracked_candles():
        nonlocal first_open, last_available
        for candle in read_candles(source, timeframe, stats):
            if first_open is None:
                first_open = candle.timestamp
            last_available = candle.available_at
            yield candle

    output.parent.mkdir(parents=True, exist_ok=True)
    previous_event_time = None
    with output.open("w" if overwrite else "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for event in detect_candidates(tracked_candles(), symbol, timeframe,
                                       config, include_displacement_body):
            available = utc(event["available_at_utc"])
            if previous_event_time is not None and available < previous_event_time:
                raise ValueError("Candidate events are not chronological")
            if (not positive(event["lower_price"]) or not positive(event["upper_price"])
                    or event["upper_price"] <= event["lower_price"]):
                raise ValueError("Invalid candidate price boundaries")
            if utc(event["candidate_created_at_utc"]) > available or available > last_available:
                raise ValueError("Invalid candidate availability")
            previous_event_time = available
            writer.writerow(event)
            stats.update(("output_event_rows", event["imbalance_type"] + "_count",
                          event["event_type"] + "_count"))
            if event["imbalance_type"] != "displacement_body" and event["aligned_displacement"]:
                stats["fvg_with_aligned_displacement_count"] += 1
    if first_open is None:
        raise ValueError(f"Empty input: {source}")
    warnings = [f"{name}: {stats[name]}; no input values were filled or changed"
                for name in ("missing_atr_rows", "invalid_atr_rows", "missing_body_ratio_rows",
                             "invalid_body_ratio_rows", "zero_range_rows", "missing_atr_normalized_body_rows",
                             "invalid_atr_normalized_body_rows", "atr_normalized_body_mismatch_rows",
                             "nonfinite_derived_body_atr_rows") if stats[name]]
    return {
        "symbol": symbol, "source_timeframe": timeframe, "input_path": str(source),
        "output_path": str(output), "input_candle_count": stats["input_rows"],
        **{name: stats[name] for name in ("bullish_fvg_count", "bearish_fvg_count",
                                        "fvg_with_aligned_displacement_count")},
        "displacement_body_creation_count": stats["displacement_body_count"],
        "displacement_only_candidate_count_as_of_end": stats["displacement_body_count"] - stats["fvg_confirmation_count"],
        "linked_fvg_confirmation_count": stats["fvg_confirmation_count"],
        "unique_candidate_count": stats["create_count"], "output_event_rows": stats["output_event_rows"],
        "missing_atr_rows": stats["missing_atr_rows"], "invalid_atr_rows": stats["invalid_atr_rows"],
        "invalid_data_counts": {key: stats[key] for key in (
            "invalid_body_ratio_rows", "invalid_atr_normalized_body_rows",
            "atr_normalized_body_mismatch_rows", "nonfinite_derived_body_atr_rows")},
        "zero_range_rows": stats["zero_range_rows"],
        "timestamp_checks": {"status": "PASS", "input_rows_checked": stats["input_rows"],
                             "event_rows_checked": stats["output_event_rows"],
                             "first_open_utc": iso(first_open), "last_available_utc": iso(last_available),
                             "invalid_timestamp_rows": 0,
                             "checks": ["aware UTC interval grid", "consecutive candles", "close-time availability", "chronological events"]},
        "price_boundary_checks": {"status": "PASS", "invalid_ohlc_rows": 0,
                                  "input_rows_checked": stats["input_rows"],
                                  "event_rows_checked": stats["output_event_rows"]},
        "warnings": warnings,
    }


FIELD_MEANINGS = {
    "zone_id": "Deterministic independent-candidate identity; a later aligned FVG confirmation reuses the body ID",
    "symbol/source_timeframe": "Input asset and candle interval; no cross-asset or cross-timeframe detection",
    "zone_family/imbalance_type/direction": "imbalance; bullish_fvg, bearish_fvg or displacement_body; bullish or bearish",
    "lower_price/upper_price": "Interval known at this event; exact FVG gap or impulse body without padding",
    "formation_start_utc": "C1 opening for FVG; impulse opening for displacement body",
    "available_at_utc": "Availability of this row: C3 close for FVG, impulse close for body",
    "candidate_created_at_utc": "Original independent-candidate creation time; unchanged by FVG confirmation",
    "reference_candle_time_utc": "Impulse candle opening (C2 for FVG)",
    "c1_time_utc/c2_time_utc/c3_time_utc": "Three candle openings for FVG; only impulse/c2 populated for body",
    "reference_atr/reference_atr_available_at_utc": "Impulse candle ATR and its close time; never C3 ATR for FVG evidence",
    "gap_width/gap_width_atr": "FVG upper-lower and gap/C2 ATR; blank for body; normalized value blank for unusable ATR",
    "zone_width": "upper_price-lower_price for every row, including displacement body",
    "displacement_body_atr/displacement_body_ratio": "Absolute impulse body/impulse ATR, and validated Phase 1 body_ratio; valid weak/opposed values retained",
    "aligned_displacement": "FVG direction matches strong impulse; false on body creation because no FVG is known then",
    "candidate_status": "candidate; no fill, mitigation, lifecycle or validity conclusion",
    "event_type": "create or fvg_confirmation; confirmation adds FVG evidence without a second independent creation",
}


def run_tests(root: Path) -> dict:
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    return {"tests_run": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped),
            "successful": result.testsRun > 0 and result.wasSuccessful()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate only Phase 3 outputs and summary")
    parser.add_argument("--fvg-only", action="store_true", help="Omit optional body candidates; emit FVG creation events only")
    parser.add_argument("--displacement-body-atr-min", type=float, default=1.0)
    parser.add_argument("--displacement-body-ratio-min", type=float, default=0.60)
    args = parser.parse_args(argv)
    try:
        config = Config(args.displacement_body_atr_min, args.displacement_body_ratio_min)
        jobs = preflight(PROJECT_DIR, args.overwrite)
        tests = run_tests(PROJECT_DIR)
        if not tests["successful"]:
            print("FAIL: regression tests did not pass; no outputs generated", file=sys.stderr)
            return 1
        datasets = []
        for symbol, timeframe, source, output in jobs:
            result = process_dataset(symbol, timeframe, source, output, config,
                                     args.overwrite, not args.fvg_only)
            datasets.append(result)
            print(f"PASS {symbol} {timeframe}: {result['input_candle_count']} candles; "
                  f"{result['bullish_fvg_count']} bullish / {result['bearish_fvg_count']} bearish FVGs; "
                  f"{result['displacement_only_candidate_count_as_of_end']} body-only as of end", flush=True)
        report = PROJECT_DIR / "reports" / "zone_detection" / "imbalance_summary.json"
        summary = {
            "phase": 3, "generated_at_utc": iso(datetime.now(timezone.utc)),
            "parameters": {**asdict(config), "include_displacement_body": not args.fvg_only},
            "parameter_status": "Provisional research assumptions, not validated trading rules",
            "rules": {
                "bullish_fvg": "C3.low > C1.high; [C1.high, C3.low] at C3 close",
                "bearish_fvg": "C3.high < C1.low; [C3.high, C1.low] at C3 close",
                "displacement": "Directionally aligned C2 body with abs(close-open)/C2 ATR >= body_atr_min and C2 body_ratio >= body_ratio_min",
                "normalization": "Both displacement and FVG width use C2 ATR; atr_normalized_body is audited against the signed formula; body_ratio is reused",
                "precision": "Feature consistency comparison rel_tol=1e-8, abs_tol=1e-10; geometry and threshold comparisons remain strict/inclusive respectively",
                "missing_data": "No imputation; unusable evidence cannot qualify displacement but valid strict FVG geometry remains eligible",
                "input_errors": "Invalid OHLC or nonconsecutive/malformed timestamps fail the run with a CSV line reference",
                "event_policy": "Body at own close; later aligned FVG confirmation shares ID without rewriting prior events; --fvg-only emits creations only",
                "counting": "Count create rows or distinct zone_id for independent candidates, not every event row",
                "ordering": "Chronological availability; at one close, prior-middle FVG precedes current-candle body",
            },
            "schema": COLUMNS, "field_meanings": FIELD_MEANINGS,
            "limitations": [
                "OHLC bodies are provisional interval proxies, not verified liquidity voids",
                "FVG geometry is an OHLC pattern, not evidence of market inefficiency or future price response",
                "To preserve immediate displacement knowledge and avoid duplicate independent IDs, default CSVs include FVG confirmation rows as well as creations",
                "Body-only counts are as-of-input-end counts, not a historical feature; the final impulse may acquire FVG evidence after the dataset ends",
                "No fill, mitigation, invalidation, zone merging, strength, outcome or scenario inference",
                "A failed input validation can leave partial generated files; failures return nonzero and no new success summary is written",
            ],
            "tests": tests, "datasets": datasets, "output_summary_path": str(report),
            "warnings": [f"{d['symbol']} {d['source_timeframe']}: {w}" for d in datasets for w in d["warnings"]],
        }
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("w" if args.overwrite else "x", encoding="utf-8") as handle:
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
