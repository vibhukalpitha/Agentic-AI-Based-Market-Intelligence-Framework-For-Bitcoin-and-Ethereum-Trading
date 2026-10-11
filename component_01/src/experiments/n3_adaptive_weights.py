"""M9 - Test of Novelty 3: do context-adaptive timeframe weights help?

Run:  python -m src.experiments.n3_adaptive_weights --period train   (in-sample check of the script)
      python -m src.experiments.n3_adaptive_weights --period test    (the test; run ONCE)

Three ways of weighting the four faster timeframes (see
src/fusion/adaptive_weights.py), all fitted on the training period:

  F  fixed weights (the proposed 0.25 / 0.20 / 0.15 / 0.10)
  L  one learned weight per timeframe
  A  one learned weight per timeframe per context (calm / active)

Outcome: does the Daily regime hold for the next 3 days.

Questions, fixed before the test-period run:

  1. PRIMARY (Novelty 3). Is A better than F, and better than L?
     Novelty 3 is supported only if both differences are clearly negative.
  2. Is L better than F - is learning the weights worth it at all?
  3. For reference: is F better than S1, the delivered Stability Score, which
     uses no cross-timeframe signal? (This repeats the M6 question with one
     signal instead of three.)

Brier differences come with 95% intervals from resampling whole weeks.
"""
import argparse
import json

import numpy as np
import pandas as pd

from src import config
from src.experiments.m6_evaluation import auc, brier, weekly_block_differences
from src.stability.stability import log_loss

SCORES = ["S1", "F", "L", "A"]
PAIRS = [("A", "F"), ("A", "L"), ("L", "F"), ("F", "S1")]


def evaluate(symbol: str, period: str, rng: np.random.Generator) -> dict:
    scores = pd.read_parquet(config.FUSION_DIR / f"{symbol}_adaptive_weights.parquet")
    reference = pd.read_parquet(config.STABILITY_DIR / f"{symbol}_stability.parquet",
                                columns=["date", "S1"])
    frame = scores.merge(reference, on="date")
    frame = frame[(frame["period"] == period) & frame["target"].notna()].reset_index(drop=True)
    target = frame["target"].to_numpy()
    with open(config.RESULTS_DIR / f"adaptive_weights_{symbol}.json") as file:
        record = json.load(file)

    metrics = [{"score": name, "brier": brier(target, frame[name].to_numpy()),
                "log_loss": log_loss(target, frame[name].to_numpy()),
                "auc": auc(target, frame[name].to_numpy())} for name in SCORES]
    by_context = []
    for label, flag in (("calm", 0.0), ("active", 1.0)):
        part = frame[frame["active"] == flag]
        by_context.append({"context": label, "days": int(len(part)),
                           "persisted_rate": float(part["target"].mean()),
                           **{name: brier(part["target"].to_numpy(), part[name].to_numpy())
                              for name in ("F", "L", "A")}})
    return {
        "days": int(len(frame)), "from": str(frame["date"].iloc[0].date()),
        "to": str(frame["date"].iloc[-1].date()),
        "active_share": float(frame["active"].mean()),
        "delivered_model": record["delivered_model"],
        "metrics": metrics, "brier_by_context": by_context,
        "differences": weekly_block_differences(frame, PAIRS, rng),
        "learned_weights": record["learned_weights"],
    }


def print_results(symbol: str, period: str, result: dict) -> None:
    print(f"\n{'=' * 74}\n{symbol}  ({period} period: {result['from']} -> {result['to']}, "
          f"{result['days']} days; active on {result['active_share']:.1%})\n{'=' * 74}")
    print(f"model chosen on training data: {result['delivered_model']}\n")
    print(pd.DataFrame(result["metrics"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nBrier by context:")
    print(pd.DataFrame(result["brier_by_context"]).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nBrier differences with 95% intervals (negative = the first is better):")
    print(pd.DataFrame(result["differences"]).T.to_string(float_format=lambda v: f"{v:+.4f}"))
    print("\nweights learned on the training period:")
    print(pd.DataFrame(result["learned_weights"]).T.to_string(float_format=lambda v: f"{v:+.3f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--period", choices=["train", "test"], required=True,
                        help="'train' is an in-sample check; 'test' is the test, run once")
    args = parser.parse_args()

    rng = np.random.default_rng(config.RANDOM_SEED)
    results = {"design": {"period": args.period, "horizon_days": config.STABILITY_HORIZON_DAYS,
                          "block_days": config.STABILITY_BLOCK_DAYS,
                          "bootstraps": config.H1_BOOTSTRAPS, "seed": config.RANDOM_SEED}}
    for symbol in config.SYMBOLS:
        results[symbol] = evaluate(symbol, args.period, rng)
        print_results(symbol, args.period, results[symbol])

    path = config.RESULTS_DIR / f"n3_adaptive_weights_{args.period}.json"
    with open(path, "w") as file:
        json.dump(results, file, indent=2)
    print(f"\nSaved: {path.name}")


if __name__ == "__main__":
    main()
