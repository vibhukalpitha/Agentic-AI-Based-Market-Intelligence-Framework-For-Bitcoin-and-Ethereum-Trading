"""OHLCV-based estimated volume-at-price proxy, not trade-level Volume Profile.

Uniform low-to-high allocation is an assumption about unknown execution prices.
Rows are independent as-of-time rolling observations, not durable tracked zones,
actual liquidity, order concentrations, or forecasts. Base volume units are BTC
or ETH. Earlier phase inputs and outputs are read-only.

Run: python -B Component_4/src/zone_detection/volume.py
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sys
import unittest
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zone_detection.swings import INTERVALS, iso, positive, utc


PROJECT_DIR = Path(__file__).resolve().parents[2]
VOLUME_UNITS = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}
METHOD = "uniform_ohlcv_range_allocation"
CONSERVATION_REL_TOL = 1e-9
CONSERVATION_ABS_TOL = 1e-10
REQUIRED_COLUMNS = ("timestamp_utc", "available_at_utc", "high", "low", "close", "volume", "atr_14_sma")
COLUMNS = (
    "zone_id", "symbol", "source_timeframe", "zone_family", "lower_price", "upper_price",
    "profile_start_utc", "available_at_utc", "estimated_volume", "profile_total_volume",
    "volume_share", "relative_bin_volume", "reference_atr", "bin_width", "candidate_status",
    "method", "profile_end_utc", "profile_id", "candidate_rank", "peak_bin_index",
    "number_of_bins_in_zone", "volume_unit",
)


@dataclass(frozen=True)
class Config:
    profile_window_bars: int = 100
    profile_stride_bars: int = 20
    profile_bins: int = 40
    peak_relative_volume_min: float = 1.5
    max_candidates_per_profile: int = 3

    def __post_init__(self) -> None:
        for name in ("profile_window_bars", "profile_stride_bars", "profile_bins", "max_candidates_per_profile"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.peak_relative_volume_min) or self.peak_relative_volume_min < 1:
            raise ValueError("peak_relative_volume_min must be finite and >= 1")


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    available_at: datetime
    high: float
    low: float
    close: float
    volume: float
    atr: float | None
    open: float | None = None

    def __post_init__(self) -> None:
        prices = (self.high, self.low, self.close)
        if (not all(positive(p) for p in prices) or not self.low <= self.close <= self.high
                or (self.open is not None and (not positive(self.open) or not self.low <= self.open <= self.high))):
            raise ValueError("Prices must be finite, positive and OHLC-consistent")
        if not math.isfinite(self.volume) or self.volume < 0:
            raise ValueError("Base volume must be finite and nonnegative")
        if self.timestamp.tzinfo is None or self.available_at.tzinfo is None or self.available_at <= self.timestamp:
            raise ValueError("Aware opening and later completion timestamps required")


@dataclass(frozen=True)
class Profile:
    edges: tuple[float, ...]
    bins: tuple[float, ...]
    total_volume: float
    status: str
    absolute_error: float | None = None
    relative_error: float | None = None
    conservation_ok: bool | None = None


def check_header(path: Path) -> None:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        fields = csv.DictReader(handle).fieldnames
    if not fields or len(fields) != len(set(fields)):
        raise ValueError(f"{path}: empty header or duplicate columns")
    missing = set(REQUIRED_COLUMNS) - set(fields)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")


def read_candles(path: Path, timeframe: str, stats: Counter) -> Iterator[Candle]:
    """Strict HLC/volume/time validation; also validate open when it is present.

    ATR is metadata, not an allocation or selection input. Unusable ATR remains
    missing and is counted; it never causes volume imputation or window removal.
    """
    check_header(path)
    previous = None
    interval = INTERVALS[timeframe]
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for line, row in enumerate(csv.DictReader(handle), 2):
            try:
                timestamp, available = utc(row["timestamp_utc"]), utc(row["available_at_utc"])
                if ((timestamp - epoch) % interval != timedelta(0) or available != timestamp + interval
                        or (previous is not None and timestamp - previous != interval)):
                    raise ValueError("Off-grid, nonconsecutive or incorrect candle availability timestamps")
                raw_atr = row["atr_14_sma"].strip()
                if not raw_atr:
                    atr = None
                    stats["missing_atr_rows"] += 1
                else:
                    try:
                        atr = float(raw_atr)
                    except ValueError:
                        atr = None
                    if not positive(atr):
                        atr = None
                        stats["invalid_atr_rows"] += 1
                candle = Candle(timestamp, available, float(row["high"]), float(row["low"]),
                                float(row["close"]), float(row["volume"]), atr,
                                float(row["open"]) if "open" in row else None)
            except (ValueError, TypeError, AttributeError, KeyError) as exc:
                raise ValueError(f"{path}, CSV line {line}: {exc}") from exc
            previous = timestamp
            stats["input_rows"] += 1
            stats["zero_range_candles"] += candle.high == candle.low
            yield candle


def allocate_profile(candles: Sequence[Candle], number_of_bins: int) -> Profile:
    """Allocate exactly the requested overlap fractions, without renormalizing.

    Bins are [left,right), except the final bin includes the profile maximum.
    A point candle on an internal edge belongs to the bin on its right. Interval
    endpoints have zero measure and never contribute to two bins at an edge.
    """
    if not candles or type(number_of_bins) is not int or number_of_bins < 1:
        raise ValueError("A nonempty window and positive integer bin count are required")
    low, high = min(c.low for c in candles), max(c.high for c in candles)
    total = math.fsum(c.volume for c in candles)
    if not math.isfinite(total):
        raise ValueError("Profile total volume is not finite")
    if high == low:
        return Profile((), (), total, "flat_price_span")
    span = high - low
    edges = tuple(low + span * (i / number_of_bins) for i in range(number_of_bins)) + (high,)
    if any(right <= left for left, right in zip(edges, edges[1:])):
        return Profile((), (), total, "unrepresentable_price_grid")
    estimated = [0.0] * number_of_bins
    for candle in candles:
        if candle.high == candle.low:
            index = min(number_of_bins - 1, bisect.bisect_right(edges, candle.low) - 1)
            estimated[index] += candle.volume
            continue
        first = bisect.bisect_right(edges, candle.low) - 1
        stop = bisect.bisect_left(edges, candle.high)
        for index in range(first, stop):
            overlap = min(candle.high, edges[index + 1]) - max(candle.low, edges[index])
            if overlap > 0:
                estimated[index] += candle.volume * (overlap / (candle.high - candle.low))
    allocated = math.fsum(estimated)
    error = abs(allocated - total)
    valid = math.isclose(allocated, total, rel_tol=CONSERVATION_REL_TOL, abs_tol=CONSERVATION_ABS_TOL)
    return Profile(edges, tuple(estimated), total, "ok" if total else "zero_total_volume",
                   error, error / total if total else 0.0, valid)


def flat_histogram(bins: Sequence[float]) -> bool:
    return bool(bins) and all(math.isclose(value, bins[0], rel_tol=1e-12, abs_tol=0) for value in bins)


def select_regions(bins: Sequence[float], config: Config) -> list[dict]:
    """One area per connected run strictly above the histogram mean.

    Its highest bin is a local maximum (only the existing neighbor matters at
    an edge). Equal peak heights, including plateaus, choose the leftmost bin.
    Several peaks connected entirely above the mean represent one area. A bin
    at or below the mean always separates areas. No smoothing is performed.
    """
    if any(not math.isfinite(value) or value < 0 for value in bins):
        raise ValueError("Estimated histogram volumes must be finite and nonnegative")
    if not bins or flat_histogram(bins):
        return []
    mean = math.fsum(bins) / len(bins)
    if mean <= 0:
        return []
    regions = []
    index = 0
    while index < len(bins):
        if bins[index] <= mean:
            index += 1
            continue
        start = index
        while index < len(bins) and bins[index] > mean:
            index += 1
        stop = index  # Exclusive: these are disjoint connected components.
        peak = max(range(start, stop), key=lambda i: bins[i])
        relative = bins[peak] / mean
        if relative >= config.peak_relative_volume_min:
            regions.append({"start": start, "stop": stop, "peak": peak,
                            "estimated_volume": math.fsum(bins[start:stop]), "relative_bin_volume": relative})
    regions.sort(key=lambda r: (-r["estimated_volume"], -r["relative_bin_volume"], r["start"]))
    return regions[:config.max_candidates_per_profile]


def iter_observations(candles: Iterable[Candle], symbol: str, timeframe: str,
                      config: Config, stats: Counter) -> Iterator[dict]:
    """First window ends at bar W (1-based), then W+S, W+2S; no EOF flush.

    Each window uses only its most recent W completed candles. reference_atr is
    from the final candle at this observation's availability, never a later ATR.
    """
    window: deque[Candle] = deque(maxlen=config.profile_window_bars)
    for count, candle in enumerate(candles, 1):
        window.append(candle)
        if count < config.profile_window_bars or (count - config.profile_window_bars) % config.profile_stride_bars:
            continue
        stats["profiles_calculated"] += 1
        profile = allocate_profile(tuple(window), config.profile_bins)
        stats["flat_price_span_profiles"] += profile.status == "flat_price_span"
        stats["zero_total_volume_profiles"] += profile.total_volume == 0
        stats["unrepresentable_price_grid_profiles"] += profile.status == "unrepresentable_price_grid"
        if profile.conservation_ok is not None:
            stats["conservation_checked_profiles"] += 1
            stats["max_absolute_conservation_error"] = max(stats["max_absolute_conservation_error"], profile.absolute_error)
            stats["max_relative_conservation_error"] = max(stats["max_relative_conservation_error"], profile.relative_error)
            if not profile.conservation_ok:
                stats["conservation_error_profiles"] += 1
                raise ValueError(f"Estimated volume conservation failed at {iso(candle.available_at)}")
        else:
            stats["conservation_skipped_profiles"] += 1
        if profile.total_volume > 0 and flat_histogram(profile.bins):
            stats["flat_histogram_profiles"] += 1
        regions = select_regions(profile.bins, config) if profile.status == "ok" else []
        if not regions:
            stats["no_candidate_profiles"] += 1
        reference_atr = candle.atr if positive(candle.atr) else None
        stats["profiles_missing_reference_atr"] += reference_atr is None
        profile_id = f"{symbol}_{timeframe}_estimated_vap_{candle.available_at.strftime('%Y%m%dT%H%M%SZ')}"
        for rank, region in enumerate(regions, 1):
            stats["candidate_observations"] += 1
            yield {
                "zone_id": f"{profile_id}_rank_{rank:02d}", "symbol": symbol, "source_timeframe": timeframe,
                "zone_family": "volume_derived", "lower_price": profile.edges[region["start"]],
                "upper_price": profile.edges[region["stop"]], "profile_start_utc": iso(window[0].timestamp),
                "available_at_utc": iso(candle.available_at), "estimated_volume": region["estimated_volume"],
                "profile_total_volume": profile.total_volume,
                "volume_share": region["estimated_volume"] / profile.total_volume,
                "relative_bin_volume": region["relative_bin_volume"], "reference_atr": reference_atr,
                "bin_width": (profile.edges[-1] - profile.edges[0]) / config.profile_bins,
                "candidate_status": "candidate", "method": METHOD, "profile_end_utc": iso(candle.timestamp),
                "profile_id": profile_id, "candidate_rank": rank, "peak_bin_index": region["peak"],
                "number_of_bins_in_zone": region["stop"] - region["start"], "volume_unit": VOLUME_UNITS[symbol],
            }


def paths_for(root: Path, symbol: str, timeframe: str) -> tuple[Path, Path]:
    return (root / "data" / "features" / timeframe / f"{symbol}_{timeframe}_features.csv",
            root / "data" / "zones" / "volume" / f"{symbol}_{timeframe}.csv")


def preflight(root: Path, overwrite: bool) -> list[tuple[str, str, Path, Path]]:
    jobs = [(symbol, timeframe, *paths_for(root, symbol, timeframe))
            for timeframe in INTERVALS for symbol in VOLUME_UNITS]
    targets = [job[3] for job in jobs] + [root / "reports" / "zone_detection" / "volume_summary.json"]
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
    stats = Counter()
    first_open = last_available = previous_output = None

    def tracked_candles():
        nonlocal first_open, last_available
        for candle in read_candles(source, timeframe, stats):
            if first_open is None:
                first_open = candle.timestamp
            last_available = candle.available_at
            yield candle

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if overwrite else "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for observation in iter_observations(tracked_candles(), symbol, timeframe, config, stats):
            available = utc(observation["available_at_utc"])
            if ((previous_output is not None and available < previous_output)
                    or available != last_available
                    or utc(observation["profile_end_utc"]) + INTERVALS[timeframe] != available
                    or utc(observation["profile_start_utc"]) + config.profile_window_bars * INTERVALS[timeframe] != available):
                raise ValueError("Invalid output window timestamps")
            if (not positive(observation["lower_price"]) or not positive(observation["upper_price"])
                    or observation["upper_price"] <= observation["lower_price"]
                    or not 0 < observation["volume_share"] <= 1 + CONSERVATION_REL_TOL):
                raise ValueError("Invalid output interval or estimated volume share")
            writer.writerow(observation)
            previous_output = available
    if first_open is None:
        raise ValueError(f"Empty feature file: {source}")
    expected = max(0, (stats["input_rows"] - config.profile_window_bars) // config.profile_stride_bars + 1)
    if stats["profiles_calculated"] != expected:
        raise ValueError("Unexpected rolling profile count")
    warning_keys = ("missing_atr_rows", "invalid_atr_rows", "profiles_missing_reference_atr",
                    "flat_price_span_profiles", "zero_total_volume_profiles", "unrepresentable_price_grid_profiles")
    return {
        "symbol": symbol, "source_timeframe": timeframe, "input_path": str(source), "output_path": str(output),
        "volume_unit": VOLUME_UNITS[symbol], "input_candle_count": stats["input_rows"],
        "profiles_calculated": stats["profiles_calculated"], "expected_profile_count": expected,
        "candidate_observations": stats["candidate_observations"],
        "no_candidate_profiles": stats["no_candidate_profiles"],
        "flat_price_span_profiles": stats["flat_price_span_profiles"],
        "zero_total_volume_profiles": stats["zero_total_volume_profiles"],
        "flat_histogram_profiles": stats["flat_histogram_profiles"],
        "unrepresentable_price_grid_profiles": stats["unrepresentable_price_grid_profiles"],
        "zero_range_candles": stats["zero_range_candles"],
        "missing_atr_rows": stats["missing_atr_rows"], "invalid_atr_rows": stats["invalid_atr_rows"],
        "profiles_missing_reference_atr": stats["profiles_missing_reference_atr"],
        "conservation": {key: stats[key] for key in (
            "conservation_checked_profiles", "conservation_skipped_profiles", "conservation_error_profiles",
            "max_absolute_conservation_error", "max_relative_conservation_error")},
        "invalid_input_counts": {"timestamps": 0, "prices": 0, "base_volume": 0},
        "timestamp_checks": {"status": "PASS", "input_rows_checked": stats["input_rows"],
                             "observation_rows_checked": stats["candidate_observations"],
                             "first_open_utc": iso(first_open), "last_available_utc": iso(last_available)},
        "boundary_checks": {"status": "PASS", "observation_rows_checked": stats["candidate_observations"]},
        "warnings": [f"{key}: {stats[key]}" for key in warning_keys if stats[key]],
    }


def run_tests(root: Path) -> dict:
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    return {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
            "skipped": len(result.skipped), "successful": result.testsRun > 0 and result.wasSuccessful()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate only Phase 5 CSVs and summary")
    for name, default in asdict(Config()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    args = vars(parser.parse_args(argv))
    overwrite = args.pop("overwrite")
    try:
        config = Config(**args)
        jobs = preflight(PROJECT_DIR, overwrite)
        tests = run_tests(PROJECT_DIR)
        if not tests["successful"]:
            print("FAIL: tests did not pass; no volume observations generated", file=sys.stderr)
            return 1
        datasets = []
        for symbol, timeframe, source, output in jobs:
            result = process_dataset(symbol, timeframe, source, output, config, overwrite)
            datasets.append(result)
            print(f"PASS {symbol} {timeframe}: {result['input_candle_count']} candles; "
                  f"{result['profiles_calculated']} profiles; {result['candidate_observations']} estimated-volume observations", flush=True)
        report = PROJECT_DIR / "reports" / "zone_detection" / "volume_summary.json"
        summary = {
            "phase": 5, "method": METHOD, "method_name": "OHLCV-based estimated volume-at-price proxy",
            "generated_at_utc": iso(datetime.now(timezone.utc)), "parameters": asdict(config),
            "parameter_status": "Provisional research assumptions, not optimized trading settings",
            "rules": {
                "allocation": "Base volume * overlap_width/(high-low); no residual correction or renormalization",
                "point_candles": "Whole volume into [left,right); exact internal edge goes right, profile maximum goes into the last bin",
                "windows": "First profile after W completed candles, then every S candles; latest W candles only, no partial window or EOF flush",
                "availability": "Final included candle available_at_utc; no later candles or later ATR",
                "peaks": "One region per contiguous run strictly above the histogram mean; its maximum bin must reach peak_relative_volume_min; equal maxima use leftmost bin; edge peaks allowed",
                "ranking": "Estimated area volume descending, peak/mean descending, lower bin index ascending; at most max_candidates_per_profile",
                "flat_histogram": "All bins equal within relative tolerance 1e-12 yields no candidates",
                "conservation": {"rel_tol": CONSERVATION_REL_TOL, "abs_tol_base_units": CONSERVATION_ABS_TOL,
                                 "policy": "Check every allocatable profile; fail on mismatch; flat-price or unrepresentable grids skipped explicitly"},
                "degenerate_profiles": "No observations for zero price span, zero total volume, unrepresentable grids, or no qualifying area; reason counters may overlap",
                "invalid_inputs": "Malformed/nonconsecutive time, invalid prices or missing/negative/nonfinite volume fail with CSV line; no input imputation",
                "reference_atr": "Final candle ATR is metadata only; missing/invalid values are blank, not used to reject or rank areas",
            },
            "schema": COLUMNS,
            "field_meanings": {
                "zone_id/profile_id/candidate_rank": "Deterministic asset/timeframe/profile-availability identity with 1-based rank; no cross-profile tracking",
                "profile_start_utc": "First included candle opening timestamp",
                "profile_end_utc": "Last included candle opening timestamp (not its completion)",
                "available_at_utc": "Last included candle completion; actual decision time",
                "estimated_volume/profile_total_volume/volume_unit": "Area allocation estimate and input window base-volume total, in BTC or ETH",
                "volume_share": "Estimated area volume / total input base volume",
                "relative_bin_volume": "Selected peak bin estimated volume / whole-profile mean bin estimated volume; not area/mean",
                "reference_atr": "ATR from the last included completed candle",
                "bin_width": "Nominal (window maximum high - minimum low) / profile_bins; actual float edges bound intervals",
                "peak_bin_index/number_of_bins_in_zone": "0-based selected peak index and count of contiguous included bins",
                "lower_price/upper_price": "Observed profile bin edges; positive interval width without fixed-dollar padding",
                "zone_family/candidate_status/method": "volume_derived / candidate / uniform_ohlcv_range_allocation",
            },
            "limitations": [
                "OHLCV-based estimate, not true trade-level exchange Volume Profile or actual execution-price volume",
                "Uniform range allocation is an unverified assumption; OHLCV does not reveal intrabar trade-price distribution",
                "Does not identify actual liquidity, institutional orders or verified high-volume nodes",
                "Changing rolling bins and overlapping windows produce correlated observations, not independent reactions or confirmed durable zones",
                "Bins at or below the mean separate areas; distinct peaks connected above the mean share one area",
                "No directional trading role, strength, lifecycle, future outcomes, cross-profile merging or scenario inference",
                "Input or conservation failure returns nonzero and can leave partial generated CSVs; no new success summary is written",
            ],
            "tests": tests, "datasets": datasets, "output_summary_path": str(report),
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
