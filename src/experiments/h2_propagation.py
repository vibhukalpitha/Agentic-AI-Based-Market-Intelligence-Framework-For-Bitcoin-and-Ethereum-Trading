"""M9 - The H2 experiment.

Run:  python -m src.experiments.h2_propagation --period train   (design check)
      python -m src.experiments.h2_propagation --period test    (the test; run ONCE)

H2: regime changes that were already present at faster timeframes ("supported
from below") are more durable than changes that were not, and the deeper the
support, the more durable the change.

The design (fixed on the training period, before any test-period result):

  event       every regime flip at a timeframe (the first candle of a new regime).
  depth       how many faster timeframes, in an unbroken row from the
              next-faster one, were already in the new regime BEFORE the flip
              candle opened (see src/fusion/propagation.py).
  outcome     durable = the new regime held at least H2_DURABLE_CANDLES candles.
              Shorter runs are noise ("false breakouts").
  effect      probability that a durable flip had DEEPER support than a
              short-lived flip (0.5 = no difference; above 0.5 supports H2),
              compared only between flips INTO the same regime.
  second view probability that an unsupported flip (depth 0) is shorter-lived
              than a supported one, using the full run length. Added after the
              training-period check showed that few Daily flips are
              short-lived (8 of 51 for BTC), which leaves the yes/no outcome
              with little power. Added BEFORE any test-period result.
  headline    Daily flips, test period, BTC and ETH separately.
  secondary   the same test at 4h, 1h and 15m (replication at other levels).
  baseline    does depth add anything beyond the model's own confidence in the
              new regime and the size of the price move on the flip candle?
  robustness  durable = 2 or 5 candles; plain count instead of unbroken depth;
              direction-only labels.

Depth is measured before the flip candle opens and durability after it, so
no information from the outcome can enter the predictor.
"""
import argparse
import json

import numpy as np
import pandas as pd

from src import config
from src.experiments.stats import (bootstrap_interval, logistic_with_intervals,
                                   matched_auc, permutation_p_value)
from src.fusion import propagation


def load_events(symbol: str, interval: str, period: str, durable_candles: int,
                label_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Flips at one timeframe in one period, with the durable / short-lived outcome.

    A flip in the last run of the data is dropped unless it has already lasted
    long enough to be called durable; otherwise its outcome is unknown.
    """
    if label_map is None:
        table = pd.read_parquet(config.PROPAGATION_DIR / f"{symbol}_flips.parquet")
    else:
        table = propagation.build_flips(symbol, label_map)
    events = table[(table["interval"] == interval) & (table["period"] == period)].copy()
    known = ~events["censored"] | (events["run_length"] >= durable_candles)
    events = events[known].reset_index(drop=True)
    events["durable"] = events["run_length"] >= durable_candles
    return events


def compare_depth(events: pd.DataFrame, column: str, rng: np.random.Generator) -> dict:
    """The H2 comparison for one definition of depth."""
    durable = events["durable"].to_numpy()
    strata = events["to_regime"].to_numpy()
    # matched_auc measures "lower"; negating depth turns it into "deeper".
    values = -events[column].to_numpy(dtype=float)
    effect, n_durable, n_short = matched_auc(values, durable, strata)
    low, high = bootstrap_interval(values, durable, strata, rng, config.H1_BOOTSTRAPS)
    supported = events[column] >= 1
    return {
        "measure": column,
        "flips": int(len(events)),
        "durable": int(durable.sum()),
        "short_lived": int((~durable).sum()),
        "durable_matched": n_durable, "short_matched": n_short,
        "durable_rate_supported": float(durable[supported].mean()) if supported.any() else np.nan,
        "durable_rate_unsupported": float(durable[~supported].mean()) if (~supported).any() else np.nan,
        "effect_auc": float(effect), "ci_low": low, "ci_high": high,
        "p_matched": permutation_p_value(values, durable, strata, rng, config.H1_PERMUTATIONS),
    }


def compare_run_length(events: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Second view of H2 that uses the full run length, not just a yes/no:
    the probability that an UNSUPPORTED flip (depth 0) is shorter-lived than a
    supported one (depth >= 1), among flips into the same regime.
    Above 0.5 supports H2."""
    unsupported = (events["depth"] == 0).to_numpy()
    strata = events["to_regime"].to_numpy()
    values = events["run_length"].to_numpy(dtype=float)
    effect, n_unsupported, n_supported = matched_auc(values, unsupported, strata)
    low, high = bootstrap_interval(values, unsupported, strata, rng, config.H1_BOOTSTRAPS)
    return {
        "unsupported_matched": n_unsupported, "supported_matched": n_supported,
        "median_run_unsupported": float(np.median(values[unsupported])) if unsupported.any() else np.nan,
        "median_run_supported": float(np.median(values[~unsupported])) if (~unsupported).any() else np.nan,
        "effect_auc": float(effect), "ci_low": low, "ci_high": high,
        "p_matched": permutation_p_value(values, unsupported, strata, rng, config.H1_PERMUTATIONS),
    }


def by_depth(events: pd.DataFrame) -> list[dict]:
    """Durable rate and typical run length at each depth - the plain-reading table."""
    rows = []
    for depth, part in events.groupby("depth"):
        rows.append({"depth": int(depth), "flips": int(len(part)),
                     "durable_rate": float(part["durable"].mean()),
                     "median_run_candles": float(part["run_length"].median())})
    return rows


def beyond_baselines(events: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Does depth still matter once confidence and move size are in the model?
    A positive depth coefficient whose interval excludes zero supports H2."""
    x = np.column_stack([events["confidence"], events["flip_return"].abs(), events["depth"]])
    return logistic_with_intervals(
        x.astype(float), events["durable"].to_numpy(dtype=float),
        ["confidence", "move_size", "depth"], rng, config.H1_BOOTSTRAPS // 4)


def analyse(symbol: str, period: str, rng: np.random.Generator) -> dict:
    results = {}
    for interval in [config.H2_HEADLINE_INTERVAL] + config.H2_SECONDARY_INTERVALS:
        events = load_events(symbol, interval, period, config.H2_DURABLE_CANDLES)
        results[interval] = {
            "comparison": compare_depth(events, "depth", rng),
            "run_length": compare_run_length(events, rng),
            "by_depth": by_depth(events),
            "beyond_baselines": beyond_baselines(events, rng),
        }

    headline = config.H2_HEADLINE_INTERVAL
    robustness = {}
    events = load_events(symbol, headline, period, config.H2_DURABLE_CANDLES)
    robustness["plain_count"] = compare_depth(events, "support_count", rng)
    for candles in config.H2_ROBUSTNESS_DURABLE:
        robustness[f"durable_{candles}_candles"] = compare_depth(
            load_events(symbol, headline, period, candles), "depth", rng)
    robustness["direction_only"] = compare_depth(
        load_events(symbol, headline, period, config.H2_DURABLE_CANDLES,
                    label_map=config.DIRECTION_OF_REGIME), "depth", rng)
    results["robustness_" + headline] = robustness
    return results


def print_results(symbol: str, period: str, results: dict) -> None:
    columns = ["measure", "flips", "durable", "short_lived", "durable_rate_supported",
               "durable_rate_unsupported", "effect_auc", "ci_low", "ci_high", "p_matched"]
    print(f"\n{'=' * 78}\n{symbol}  ({period} period)\n{'=' * 78}")
    for interval in [config.H2_HEADLINE_INTERVAL] + config.H2_SECONDARY_INTERVALS:
        block = results[interval]
        tag = "HEADLINE" if interval == config.H2_HEADLINE_INTERVAL else "secondary"
        print(f"\n[{interval}  {tag}]  effect_auc above 0.5 supports H2")
        print(pd.DataFrame([block["comparison"]])[columns].to_string(
            index=False, float_format=lambda v: f"{v:.3f}"))
        print(pd.DataFrame(block["by_depth"]).to_string(
            index=False, float_format=lambda v: f"{v:.3f}"))
        print("run length, unsupported vs supported (effect above 0.5 supports H2):")
        print(pd.DataFrame([block["run_length"]]).to_string(
            index=False, float_format=lambda v: f"{v:.3f}"))
        joint = block["beyond_baselines"]
        if "note" in joint:
            print(f"joint model: not fitted ({joint['note']})")
        else:
            print("joint model (standardised; positive depth supports H2):")
            print(pd.DataFrame(joint).T.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\n[robustness, {config.H2_HEADLINE_INTERVAL}]")
    table = pd.DataFrame(results["robustness_" + config.H2_HEADLINE_INTERVAL]).T[columns[1:]]
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--period", choices=["train", "test"], required=True,
                        help="'train' checks the design; 'test' is the real test, run once")
    args = parser.parse_args()

    rng = np.random.default_rng(config.RANDOM_SEED)
    everything = {"design": {
        "period": args.period, "durable_candles": config.H2_DURABLE_CANDLES,
        "headline_interval": config.H2_HEADLINE_INTERVAL,
        "secondary_intervals": config.H2_SECONDARY_INTERVALS,
        "propagation_window": config.PROPAGATION_WINDOW_CANDLES,
        "train_end": config.TRAIN_END, "permutations": config.H1_PERMUTATIONS,
        "bootstraps": config.H1_BOOTSTRAPS, "seed": config.RANDOM_SEED}}
    for symbol in config.SYMBOLS:
        everything[symbol] = analyse(symbol, args.period, rng)
        print_results(symbol, args.period, everything[symbol])

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"h2_results_{args.period}.json"
    with open(path, "w") as file:
        json.dump(everything, file, indent=2)
    print(f"\nSaved: {path.name}")


if __name__ == "__main__":
    main()
