"""Run the whole pipeline with one command, in the right order.

Run (from inside component_01):
  python -m src.run_all                 BUILD: the steps that produce the component's output
  python -m src.run_all --reproduce     BUILD + every experiment, chart and check + all tests
  python -m src.run_all --list          print the steps without running anything

WHY. The component is about thirty separate commands that must run in a fixed
order (features before labels, labels before fusion, fusion before the
stability score, ...). This module only automates that order. Every step is the
same module, with the same arguments, that can be run and inspected by hand;
each runs as its own process, exactly as if typed.

What it never does:
  - download data (hours; the data is frozen and verified)
  - TRAIN a model: labels always come from the saved models (train --relabel).
    Retraining would replace the ten models and invalidate every result.
  - the multi-hour BIC and convergence checks (one-off, already done)
  - any Git command

Safety. Before the first step, a fingerprint (SHA-256) is taken of every file in
results/, docs/ and models/; after the last step the fingerprints are compared
and every changed, added or removed file is listed. On the frozen data a run
must change nothing: the experiments re-run with their fixed settings and seeds
and reproduce their committed results byte for byte. The run stops at the first
failing step. Each step's full output goes to data/logs/run_all_<time>.log
(git-ignored); the screen shows one line per step.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from src import config

LOG_DIR = config.DATA_DIR / "logs"
OUTPUT_DIR = config.DATA_DIR / "output"      # latest output records (git-ignored)
WATCHED_DIRS = [config.RESULTS_DIR, config.ROOT_DIR / "docs", config.MODELS_DIR]


@dataclass(frozen=True)
class Step:
    name: str
    module: str
    args: tuple[str, ...] = ()
    save_output_as: str | None = None    # write the step's printed JSON to data/output/
    frozen_data_only: bool = False       # skipped in the live workspace (see below)

    def command(self) -> list[str]:
        return [sys.executable, "-W", "ignore", "-m", self.module, *self.args]


# ---- BUILD: what produces the component's output (later reused for live data) ----
# The raw-data check compares the files with the FROZEN inventory (exact row
# counts to Sep 2026). In the live workspace the frozen part is a fresh copy and
# the live candles get the same hard checks in src/live/store.py before storing,
# so that one step is skipped there, and says so.
BUILD_STEPS = [
    Step("Check raw data (M1)", "src.data.validate", frozen_data_only=True),
    Step("Features (M2)", "src.features.indicators"),
    *[Step(f"Regime labels {i}, saved models (M3)", "src.regime.train",
           ("--interval", i, "--relabel")) for i in config.INTERVALS],
    Step("Fusion (M4)", "src.fusion.alignment"),
    Step("Propagation (M5)", "src.fusion.propagation"),
    Step("Stability score (M6)", "src.stability.stability"),
    Step("Archetype base rates (N4)", "src.explain.archetypes"),
    *[Step(f"Latest output record {s}", "src.output.snapshot", ("--symbol", s),
           save_output_as=f"latest_{s}.json") for s in config.SYMBOLS],
]

# ---- REPRODUCE: every experiment, chart and check, with fixed settings and seeds ----
REPRODUCE_STEPS = [
    Step("Regime definitions document", "src.regime.describe"),
    Step("Model quality checks", "src.regime.model_quality"),
    Step("Exploratory data analysis", "src.eda.explore"),
    Step("H1 test", "src.experiments.h1_precursor"),
    Step("H2 design check (training)", "src.experiments.h2_propagation", ("--period", "train")),
    Step("H2 test", "src.experiments.h2_propagation", ("--period", "test")),
    Step("H2 placebo control", "src.experiments.h2_placebo"),
    Step("H2b design check (training)", "src.experiments.h2b_failed_propagation", ("--period", "train")),
    Step("H2b test", "src.experiments.h2b_failed_propagation", ("--period", "test")),
    Step("M6 evaluation (training)", "src.experiments.m6_evaluation", ("--period", "train")),
    Step("M6 evaluation (test)", "src.experiments.m6_evaluation", ("--period", "test")),
    Step("Novelty 3 weights (training fit)", "src.fusion.adaptive_weights"),
    Step("Novelty 3 check (training)", "src.experiments.n3_adaptive_weights", ("--period", "train")),
    Step("Novelty 3 test", "src.experiments.n3_adaptive_weights", ("--period", "test")),
    Step("Novelty 4 check (training)", "src.experiments.n4_archetypes", ("--period", "train")),
    Step("Novelty 4 test", "src.experiments.n4_archetypes", ("--period", "test")),
    Step("Similarity check (M7)", "src.similarity.dtw_match"),
    *[Step(f"Sample output record {s}", "src.output.snapshot", ("--symbol", s, "--save"))
      for s in config.SYMBOLS],
    Step("Verify output records", "src.output.verify_outputs"),
    Step("BIC summary (from saved checks)", "src.regime.bic_check", ("--summary",)),
    Step("Model-quality charts", "src.figures.model_quality_charts"),
    Step("What the nulls rule out", "src.experiments.null_bounds"),
    Step("All tests", "pytest", ("-q",)),
]


def fingerprint(folders: list[Path]) -> dict[str, str]:
    """SHA-256 of every file under the folders, keyed by path relative to component_01."""
    prints = {}
    for folder in folders:
        if not folder.exists():
            continue
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            prints[path.relative_to(config.ROOT_DIR).as_posix()] = hashlib.sha256(
                path.read_bytes()).hexdigest()
    return prints


def compare(before: dict[str, str], after: dict[str, str]) -> dict[str, list[str]]:
    return {"changed": sorted(k for k in before.keys() & after.keys() if before[k] != after[k]),
            "added": sorted(after.keys() - before.keys()),
            "removed": sorted(before.keys() - after.keys())}


def run_step(step: Step, log) -> tuple[bool, float, str]:
    """Run one step as its own process; return (succeeded, seconds, its output)."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}
    started = time.perf_counter()
    result = subprocess.run(step.command(), cwd=config.ROOT_DIR, env=env,
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    seconds = time.perf_counter() - started
    output = result.stdout + (("\n--- stderr ---\n" + result.stderr) if result.stderr.strip() else "")
    log.write(f"\n{'=' * 78}\n{step.name}\n$ {' '.join(step.command()[1:])}\n"
              f"exit code {result.returncode}, {seconds:.1f} s\n{'=' * 78}\n{output}\n")
    log.flush()
    ok = result.returncode == 0
    if ok and step.save_output_as:
        record = json.loads(result.stdout)          # fails loudly if the output is not one record
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / step.save_output_as).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return ok, seconds, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reproduce", action="store_true",
                        help="also run every experiment, chart and check, then all tests")
    parser.add_argument("--list", action="store_true", help="print the steps and exit")
    args = parser.parse_args()
    steps = BUILD_STEPS + (REPRODUCE_STEPS if args.reproduce else [])
    mode = "REPRODUCE" if args.reproduce else "BUILD"

    if args.list:
        for k, step in enumerate(steps, 1):
            print(f"{k:2d}. {step.name:42s} python -m {step.module} {' '.join(step.args)}")
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"run_all_{time.strftime('%Y%m%d_%H%M%S')}.log"
    print(f"run_all: {mode}, {len(steps)} steps. Full output: {log_path.relative_to(config.ROOT_DIR)}")
    print("Not run here (separate, manual): download, model training, BIC and convergence checks.\n")
    before = fingerprint(WATCHED_DIRS)
    failed, total_started = None, time.perf_counter()

    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"run_all {mode} started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        in_workspace = bool(os.environ.get(config.WORKSPACE_ENV))
        for k, step in enumerate(steps, 1):
            print(f"[{k:2d}/{len(steps)}] {step.name:42s} ", end="", flush=True)
            if in_workspace and step.frozen_data_only:
                print("skipped (live workspace: candles checked in src/live/store.py)")
                log.write(f"\n{step.name}: skipped in the live workspace\n")
                continue
            ok, seconds, output = run_step(step, log)
            print(f"{'ok' if ok else 'FAILED':6s} {seconds:7.1f} s", flush=True)
            if not ok:
                failed = step
                print("\nLast lines of its output:")
                print("\n".join(output.strip().splitlines()[-20:]))
                break

    difference = compare(before, fingerprint(WATCHED_DIRS))
    minutes = (time.perf_counter() - total_started) / 60
    print(f"\n{'Stopped at: ' + failed.name if failed else 'All steps succeeded'} "
          f"({minutes:.1f} minutes).")
    n_changed = sum(len(v) for v in difference.values())
    print(f"Files in results/, docs/, models/: {len(before)} before; "
          f"{len(difference['changed'])} changed, {len(difference['added'])} added, "
          f"{len(difference['removed'])} removed.")
    for kind, paths in difference.items():
        for path in paths:
            print(f"  {kind}: {path}")
    if n_changed:
        print("Committed versions are safe in Git; inspect with:  git status   /  git diff")
    if not args.reproduce and not failed:
        print(f"Latest output records: {OUTPUT_DIR.relative_to(config.ROOT_DIR)}"
              f"\\latest_<symbol>.json")
    return 1 if failed else (2 if n_changed else 0)


if __name__ == "__main__":
    sys.exit(main())
