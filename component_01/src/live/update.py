"""Live data - one update: fetch, check, build in a workspace, verify, publish.

Run (from inside component_01):
  python -m src.live.update                  the full update
  python -m src.live.update --fetch-only     only fetch and store new candles
  python -m src.live.update --no-fetch       rebuild from the candles already stored

What happens:
 1. FETCH   new closed candles for the 10 series from the Binance REST API
            (src/live/binance_rest.py) and store them in data/live/candles/
            after checking them (src/live/store.py).
 2. PREPARE a private workspace, data/live/workspace/: the frozen candles plus
            the live ones, and copies of the ten saved models and their reports.
 3. BUILD   the same BUILD steps as `python -m src.run_all`, run inside the
            workspace (config.WORKSPACE_ENV). Nothing is retrained: scaling,
            models, the stability model and the archetype base rates all come
            from the 2021-2022 training period, so they are identical.
 4. VERIFY  the reproduction check: everything the workspace computed for the
            period up to 30 Sep 2026 must equal the saved project files. This
            proves, on every update, that the live path is the same pipeline
            that produced the committed results, and that new candles changed
            nothing in the past (all labels use past candles only).
            If anything differs, nothing is published.
 5. PUBLISH data/live/output/latest_<symbol>.json, plus a dated copy in
            data/live/output/history/.
 6. CHARTS  the historical-similarity charts (src/figures/similarity_charts.py)
            for the latest Daily candle -> data/live/output/similarity/.

It never trains a model, never re-runs an experiment, never writes data/parquet/,
models/ or results/, and never runs Git. Everything it writes is in data/live/
(git-ignored).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from src import config
from src.live import binance_rest, store

WORKSPACE = store.LIVE_DIR / "workspace"
OUTPUT_DIR = store.LIVE_DIR / "output"
LOG_DIR = store.LIVE_DIR / "logs"
FROZEN_DATA = config.DATA_DIR       # the project's own data/ (read only here)
# Two runs of identical code on identical inputs give identical numbers; the
# tolerance only absorbs representation noise far below any reported digit.
TOLERANCE = 1e-9


# ---------------------------------------------------------------- 1. fetch
def fetch_all(client) -> list[dict]:
    now = binance_rest.server_time(client)
    print(f"Binance server time: {now} UTC")
    reports = []
    for symbol in config.SYMBOLS:
        for interval in config.INTERVALS:
            step = config.INTERVAL_STEP[interval]
            frozen_last = store.frozen_last_open(symbol, interval)
            existing = store.load_live(symbol, interval)
            start = (existing["open_time"].max() if not existing.empty else frozen_last) + step
            new = binance_rest.fetch_closed(client, symbol, interval, start, now)
            merged, added = store.merge(existing, new)
            facts = store.check(merged, frozen_last, step)
            store.save(symbol, interval, merged)
            reports.append({"symbol": symbol, "interval": interval, "new": added, **facts})
            print(f"  {symbol} {interval:>3}: {added:5d} new closed candles; stored "
                  f"{facts['candles']:5d} ({facts['first']} -> {facts['last']}); gaps {facts['gaps']}"
                  + ("" if facts["continuous_start"] in (True, None) else
                     "; NOT continuous with the frozen data"))
    print(f"  request weight used in the last minute: {client.used_weight}")
    return reports


# ---------------------------------------------------------------- 2. prepare
def prepare_workspace() -> None:
    """Fresh copies every time, so the workspace can never drift from the project."""
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    (WORKSPACE / "data" / "parquet").mkdir(parents=True)
    shutil.copytree(config.MODELS_DIR, WORKSPACE / "models")
    (WORKSPACE / "results").mkdir()
    for report in config.RESULTS_DIR.glob("*_hmm_report.json"):
        shutil.copy2(report, WORKSPACE / "results" / report.name)
    for symbol in config.SYMBOLS:
        for interval in config.INTERVALS:
            store.combined(symbol, interval).to_parquet(
                WORKSPACE / "data" / "parquet" / f"{symbol}_{interval}.parquet", index=False)


# ---------------------------------------------------------------- 3. build
def build(log_path) -> bool:
    env = {**os.environ, config.WORKSPACE_ENV: str(WORKSPACE), "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run([sys.executable, "-W", "ignore", "-m", "src.run_all"],
                            cwd=config.ROOT_DIR, env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    # run_all returns 2 when files in its watched folders changed. Inside the
    # workspace that is expected (new candles), so only 1 means failure.
    return result.returncode in (0, 2)


# ---------------------------------------------------------------- 4. verify
def compare_prefix(frozen: pd.DataFrame, work: pd.DataFrame, key: str,
                   exact: list[str], numeric: list[str]) -> tuple[bool, str]:
    """The workspace must contain the frozen rows unchanged, at the start."""
    work = work[work[key] <= frozen[key].max()].reset_index(drop=True)
    frozen = frozen.reset_index(drop=True)
    if len(work) != len(frozen) or not (work[key].values == frozen[key].values).all():
        return False, f"rows differ ({len(frozen)} saved, {len(work)} in the workspace)"
    for column in exact:
        mismatches = int((work[column].astype(str) != frozen[column].astype(str)).sum())
        if mismatches:
            return False, f"{column}: {mismatches} rows differ"
    worst = 0.0
    for column in numeric:
        a, b = frozen[column].to_numpy(float), work[column].to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        if (np.isnan(a) != np.isnan(b)).any():
            return False, f"{column}: missing values differ"
        diff = np.abs(a[~both_nan] - b[~both_nan])
        worst = max(worst, float(diff.max()) if diff.size else 0.0)
    if worst > TOLERANCE:
        return False, f"largest difference {worst:.2e} exceeds {TOLERANCE:.0e}"
    return True, f"{len(frozen):,} rows identical (largest difference {worst:.1e})"


def reproduction_check() -> list[tuple[str, bool, str]]:
    def pair(sub: str):
        return pd.read_parquet(FROZEN_DATA / sub), pd.read_parquet(WORKSPACE / "data" / sub)

    checks = []
    for symbol in config.SYMBOLS:
        for interval in config.INTERVALS:
            f, w = pair(f"features/{symbol}_{interval}_features.parquet")
            checks.append((f"{symbol} {interval} features",
                           *compare_prefix(f, w, "open_time", [], config.FEATURE_COLUMNS)))
            f, w = pair(f"regimes/{symbol}_{interval}_regimes.parquet")
            probabilities = [c for c in f.columns if c.startswith("p_")]
            checks.append((f"{symbol} {interval} regime labels",
                           *compare_prefix(f, w, "open_time", ["state", "regime"],
                                           ["confidence", *probabilities])))
        for tag in ("all", "ex1d"):
            f, w = pair(f"fusion/{symbol}_alignment_{tag}.parquet")
            checks.append((f"{symbol} alignment ({tag})",
                           *compare_prefix(f, w, "time", ["dominant"],
                                           ["alignment", "confidence_score"])))
        f, w = pair(f"stability/{symbol}_stability.parquet")
        checks.append((f"{symbol} stability score",
                       *compare_prefix(f, w, "date", ["regime"], ["B0", "B1", "S1", "S2", "delivered"])))
        f, w = pair(f"explain/{symbol}_archetypes.parquet")
        checks.append((f"{symbol} archetypes",
                       *compare_prefix(f, w, "date", ["archetype", "n_agree"], [])))
    return checks


# ---------------------------------------------------------------- 5. publish
def publish() -> list[dict]:
    (OUTPUT_DIR / "history").mkdir(parents=True, exist_ok=True)
    records = []
    for symbol in config.SYMBOLS:
        record = json.loads((WORKSPACE / "data" / "output" / f"latest_{symbol}.json")
                            .read_text(encoding="utf-8"))
        text = json.dumps(record, indent=2)
        (OUTPUT_DIR / f"latest_{symbol}.json").write_text(text, encoding="utf-8")
        stamp = record["as_of"].replace("-", "").replace(":", "")[:15]
        (OUTPUT_DIR / "history" / f"{symbol}_{stamp}.json").write_text(text, encoding="utf-8")
        records.append(record)
    return records


def similarity_charts() -> bool:
    """Draw the historical-similarity charts for the latest live Daily candle, in the
    workspace, and copy them to data/live/output/similarity/."""
    env = {**os.environ, config.WORKSPACE_ENV: str(WORKSPACE), "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run([sys.executable, "-W", "ignore", "-m", "src.figures.similarity_charts"],
                            cwd=config.ROOT_DIR, env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(result.stdout[-2000:] + result.stderr[-2000:])
        return False
    target = OUTPUT_DIR / "similarity"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(WORKSPACE / "results" / "similarity", target)
    for line in result.stdout.splitlines()[:-1]:
        print("   " + line)
    return True


def headline(record: dict) -> str:
    stability = record.get("stability") or {}
    explanation = record.get("explanation") or {}
    return (f"{record['symbol']} at {record['as_of']}: Daily {record['timeframes']['1d']['regime']}, "
            f"dominant {record['fusion']['dominant_regime']}, alignment "
            f"{record['fusion']['alignment_score']}, "
            f"P(Daily regime holds 3 days) {stability.get('probability_regime_persists')}, "
            f"archetype {explanation.get('archetype')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fetch-only", action="store_true", help="only fetch and store candles")
    parser.add_argument("--no-fetch", action="store_true", help="rebuild from stored candles")
    args = parser.parse_args()
    if os.environ.get(config.WORKSPACE_ENV):
        print(f"{config.WORKSPACE_ENV} is set; run the live update from a normal shell.")
        return 1
    started = time.perf_counter()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    if not args.no_fetch:
        print("1. Fetching new closed candles from the Binance REST API")
        reports = fetch_all(binance_rest.BinanceClient())
        (LOG_DIR / f"fetch_{stamp}.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
        if args.fetch_only:
            return 0

    print("2. Preparing the workspace (frozen + live candles, copies of the saved models)")
    prepare_workspace()
    print("3. Building in the workspace (the run_all BUILD steps; about 3 minutes)")
    log_path = LOG_DIR / f"build_{stamp}.log"
    if not build(log_path):
        print(f"   BUILD FAILED. Nothing published. See {log_path}")
        return 1
    print("4. Reproduction check: is everything up to 30 Sep 2026 identical to the saved files?")
    checks = reproduction_check()
    for name, ok, detail in checks:
        print(f"   {'ok  ' if ok else 'FAIL'} {name:32s} {detail}")
    if not all(ok for _, ok, _ in checks):
        print("   REPRODUCTION CHECK FAILED. Nothing published.")
        return 1
    print("5. Publishing the latest output records")
    for record in publish():
        print("   " + headline(record))
    print("6. Historical similarity charts for the latest Daily candle")
    if not similarity_charts():
        print("   SIMILARITY CHARTS FAILED (the records above were published).")
        return 1
    print(f"   Charts: {(OUTPUT_DIR / 'similarity').relative_to(config.ROOT_DIR)}")
    print(f"\nDone in {(time.perf_counter() - started) / 60:.1f} minutes. "
          f"Records: {OUTPUT_DIR.relative_to(config.ROOT_DIR)}\\latest_<symbol>.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
