"""M9 - H2b: does a failed propagation mark a regime change as noise?

Run:  python -m src.experiments.h2b_failed_propagation --period train   (design check)
      python -m src.experiments.h2b_failed_propagation --period test    (the test; run ONCE)

Claim under test (third case of Novelty 2): when a regime flip at one
timeframe is NOT taken up by the next slower timeframe, the flip was noise.

The trap. A flip that lasts longer gives the slower timeframe more time to
adopt it, so "adopted flips last longer" is true by construction. To remove
this, the test looks only at flips that are STILL ALIVE when the adoption
window closes, and compares what happens AFTER that moment.

Design (fixed on the training period, before any test-period result):

  events      flips at a timeframe where the next slower timeframe was NOT
              already in the new regime (so adoption was genuinely open),
              and whose regime is still running when the adoption window
              (PROPAGATION_WINDOW_CANDLES slower candles) closes.
  groups      adopted  = the slower timeframe showed the regime in the window
              failed   = it did not
  outcome     candles the regime goes on to last AFTER the window closes.
  effect      probability that a failed flip has a SHORTER remaining life
              than an adopted one, compared only between flips into the same
              regime. 0.5 = no difference; above 0.5 supports the claim.
  placebo     the same comparison with "adoption" measured PLACEBO_LAG_DAYS
              earlier. Only the part of the effect that exceeds the placebo
              can be credited to propagation.
  primary     4h flips (adoption by Daily), BTC and ETH separately.
  secondary   1h, 15m and 5m flips.
"""
import argparse
import json

import numpy as np
import pandas as pd

from src import config
from src.experiments.stats import matched_auc, permutation_p_value
from src.fusion import propagation


def load_survivors(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Flips with open adoption that were still alive when the window closed."""
    table = pd.read_parquet(config.PROPAGATION_DIR / f"{symbol}_flips.parquet")
    events = table[(table["interval"] == interval) & (table["period"] == period)
                   & ~table["slower_already"] & table["adoption_known"]].copy()
    events["open_adoption_flips"] = len(events)
    # An unfinished final run has an unknown length, so it cannot be compared.
    events = events[~events["censored"]]
    events = events[events["run_length"] > events["candles_at_window_end"]]
    events["remaining"] = events["run_length"] - events["candles_at_window_end"]
    events["failed"] = ~events["adopted_above"]

    # Placebo: was the slower timeframe showing this regime a month earlier?
    slower = propagation.load_labels(
        symbol, config.INTERVALS[config.INTERVALS.index(interval) - 1])
    earlier = (events["known_time"] - pd.Timedelta(days=config.PLACEBO_LAG_DAYS)).to_numpy()
    _, placebo_adopted, _, _ = propagation.adopted_above(
        slower, earlier, events["to_regime"].to_numpy(), config.PROPAGATION_WINDOW_CANDLES)
    events["placebo_failed"] = ~placebo_adopted
    return events.reset_index(drop=True)


def compare(events: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Real effect, placebo effect, and real minus placebo with a 95% interval."""
    values = events["remaining"].to_numpy(dtype=float)
    strata = events["to_regime"].to_numpy()
    failed = events["failed"].to_numpy()
    placebo = events["placebo_failed"].to_numpy()

    real_effect, n_failed, n_adopted = matched_auc(values, failed, strata)
    placebo_effect, _, _ = matched_auc(values, placebo, strata)

    # Resample flips inside each regime; both effects use the SAME resample.
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    draws = np.empty((config.H1_BOOTSTRAPS, 2))
    for b in range(config.H1_BOOTSTRAPS):
        pick = np.concatenate([rng.choice(g, len(g)) for g in groups])
        draws[b, 0], _, _ = matched_auc(values[pick], failed[pick], strata[pick])
        draws[b, 1], _, _ = matched_auc(values[pick], placebo[pick], strata[pick])
    difference = draws[:, 0] - draws[:, 1]
    return {
        "open_adoption_flips": int(events["open_adoption_flips"].iloc[0]) if len(events) else 0,
        "survivors": int(len(events)),
        "failed": int(failed.sum()), "adopted": int((~failed).sum()),
        "failed_matched": n_failed, "adopted_matched": n_adopted,
        "median_remaining_failed": float(np.median(values[failed])) if failed.any() else np.nan,
        "median_remaining_adopted": float(np.median(values[~failed])) if (~failed).any() else np.nan,
        "effect_auc": float(real_effect),
        "ci_low": float(np.nanpercentile(draws[:, 0], 2.5)),
        "ci_high": float(np.nanpercentile(draws[:, 0], 97.5)),
        "p_matched": permutation_p_value(values, failed, strata, rng, config.H1_PERMUTATIONS),
        "placebo_effect": float(placebo_effect),
        "real_minus_placebo": float(real_effect - placebo_effect),
        "difference_ci_low": float(np.nanpercentile(difference, 2.5)),
        "difference_ci_high": float(np.nanpercentile(difference, 97.5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--period", choices=["train", "test"], required=True,
                        help="'train' checks the design; 'test' is the real test, run once")
    args = parser.parse_args()

    rng = np.random.default_rng(config.RANDOM_SEED)
    results = {"design": {
        "period": args.period, "primary_interval": config.H2B_PRIMARY_INTERVAL,
        "secondary_intervals": config.H2B_SECONDARY_INTERVALS,
        "adoption_window_candles": config.PROPAGATION_WINDOW_CANDLES,
        "placebo_lag_days": config.PLACEBO_LAG_DAYS, "train_end": config.TRAIN_END,
        "permutations": config.H1_PERMUTATIONS, "bootstraps": config.H1_BOOTSTRAPS,
        "seed": config.RANDOM_SEED}}
    rows = []
    for symbol in config.SYMBOLS:
        results[symbol] = {}
        for interval in [config.H2B_PRIMARY_INTERVAL] + config.H2B_SECONDARY_INTERVALS:
            outcome = compare(load_survivors(symbol, interval, args.period), rng)
            results[symbol][interval] = outcome
            rows.append({"symbol": symbol, "interval": interval, **outcome})

    table = pd.DataFrame(rows)
    table["exceeds_placebo"] = table["difference_ci_low"] > 0
    print(f"H2b ({args.period} period). Effect above 0.5 = failed flips die sooner. "
          f"Primary level: {config.H2B_PRIMARY_INTERVAL}.\n")
    print(table[["symbol", "interval", "open_adoption_flips", "survivors", "failed", "adopted",
                 "median_remaining_failed", "median_remaining_adopted"]].to_string(index=False))
    print()
    print(table[["symbol", "interval", "effect_auc", "ci_low", "ci_high", "p_matched",
                 "placebo_effect", "real_minus_placebo", "difference_ci_low",
                 "difference_ci_high", "exceeds_placebo"]].to_string(
                     index=False, float_format=lambda v: f"{v:.3f}"))

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"h2b_results_{args.period}.json"
    with open(path, "w") as file:
        json.dump(results, file, indent=2)
    print(f"\nSaved: {path.name}")


if __name__ == "__main__":
    main()
