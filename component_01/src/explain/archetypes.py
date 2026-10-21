"""Novelty 4 - Conflict archetypes with historical base rates.

Run:  python -m src.explain.archetypes      (base rates from the training period only)

At every Daily close the four faster timeframes (4h, 1h, 15m, 5m) either share
the Daily regime or do not. That pattern is sorted into one of four named
archetypes by a fixed rule:

  Aligned        at least 3 of the 4 are in the Daily regime
  Partly backed  exactly 2 are in the Daily regime
  Opposed        at least 3 of the 4 agree on ONE OTHER regime (Daily stands alone)
  Fragmented     at most 1 backs Daily, and no 3 agree on any single other regime
                 (the remaining case; two of the others may still agree)

The rule was fixed from how often each pattern occurs in the training period,
so that every archetype has enough days. No outcome was used to draw the lines.

Each archetype then gets a BASE RATE: the share of training days with that
pattern on which the Daily regime changed within STABILITY_HORIZON_DAYS. The
explanation (M8) quotes it as evidence: "this pattern was followed by a regime
change N% of the time".

Scope: the base rate is a statistic about this component's own categories.
It says how often the present regime ended, never what price did or would do.
"""
import json

import numpy as np
import pandas as pd

from src import config
from src.fusion import propagation
from src.stability.stability import persistence_target

ARCHETYPES = ["Aligned", "Partly backed", "Fragmented", "Opposed"]
DEFINITIONS = {
    "Aligned": "at least three of the four faster timeframes are in the same regime as Daily",
    "Partly backed": "exactly two of the four faster timeframes are in the same regime as Daily",
    "Fragmented": "at most one faster timeframe backs Daily, and no three agree on any "
                  "single other regime",
    "Opposed": "at least three of the four faster timeframes agree on one regime that "
               "differs from Daily's",
}
FASTER = config.INTERVALS[1:]
ONE_DAY = pd.Timedelta(days=1)


def classify(daily: np.ndarray, faster: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(archetype, number agreeing with Daily, opposing regime or None) per day.

    daily has one regime per day; faster has one column per faster timeframe.
    """
    agrees = faster == daily[:, None]
    n_agree = agrees.sum(axis=1)
    opposing = np.full(len(daily), None, dtype=object)
    for regime in config.REGIMES:
        holders = ((faster == regime) & ~agrees).sum(axis=1)
        opposing[holders >= config.ARCHETYPE_MAJORITY] = regime
    has_opposing = np.array([value is not None for value in opposing])

    archetype = np.where(n_agree >= config.ARCHETYPE_MAJORITY, "Aligned",
                 np.where(n_agree == 2, "Partly backed",
                  np.where(has_opposing, "Opposed", "Fragmented")))
    return archetype, n_agree, opposing


def build_table(symbol: str) -> pd.DataFrame:
    """One row per day: Daily regime, archetype, and whether the regime changed
    within the horizon (NaN where the future is not in the data yet)."""
    horizon = config.STABILITY_HORIZON_DAYS
    daily = propagation.load_labels(symbol, "1d")
    regime = daily["regime"].to_numpy()
    close = daily["known_time"].to_numpy()
    faster = np.column_stack([
        propagation.labels_as_of(propagation.load_labels(symbol, i), close) for i in FASTER])

    table = pd.DataFrame({"regime": regime, "changed": 1.0 - persistence_target(regime, horizon)},
                         index=pd.DatetimeIndex(daily["open_time"], name="date"))
    known = np.array([all(value is not None for value in row) for row in faster])
    table, faster, regime = table[known], faster[known], regime[known]
    table["archetype"], table["n_agree"], table["opposing_regime"] = classify(regime, faster)

    train_end = pd.Timestamp(config.TRAIN_END)
    table["period"] = np.where(table.index + horizon * ONE_DAY <= train_end, "train",
                               np.where(table.index > train_end, "test", "gap"))
    return table


def base_rates(train: pd.DataFrame) -> dict:
    """Training-period base rate of each archetype, and of all days together."""
    rates = {name: {"days": int((train["archetype"] == name).sum()),
                    "change_rate": float(train.loc[train["archetype"] == name, "changed"].mean())}
             for name in ARCHETYPES}
    return {"overall_change_rate": float(train["changed"].mean()),
            "training_days": int(len(train)), "archetypes": rates}


def run(symbol: str) -> dict:
    table = build_table(symbol)
    train = table[table["period"] == "train"]
    rates = base_rates(train)
    rates.update({"symbol": symbol, "horizon_days": config.STABILITY_HORIZON_DAYS,
                  "train_from": str(train.index[0].date()),
                  "train_to": str(train.index[-1].date()), "definitions": DEFINITIONS})

    # The archetype forecast for any day is its archetype's TRAINING base rate.
    table["base_rate"] = table["archetype"].map(
        {name: info["change_rate"] for name, info in rates["archetypes"].items()})
    table["overall_rate"] = rates["overall_change_rate"]
    config.EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
    table.reset_index().to_parquet(config.EXPLAIN_DIR / f"{symbol}_archetypes.parquet", index=False)
    with open(config.RESULTS_DIR / f"archetype_base_rates_{symbol}.json", "w") as file:
        json.dump(rates, file, indent=2)

    print(f"\n=== {symbol}: archetype base rates, training period "
          f"({rates['train_from']} -> {rates['train_to']}, {len(train)} days) ===")
    report = pd.DataFrame(rates["archetypes"]).T
    report["share_of_days"] = report["days"] / len(train)
    print(report.to_string(formatters={"days": "{:.0f}".format,
                                       "change_rate": "{:.1%}".format,
                                       "share_of_days": "{:.1%}".format}))
    print(f"all days: Daily regime changed within {config.STABILITY_HORIZON_DAYS} days on "
          f"{rates['overall_change_rate']:.1%}")
    return rates


def main() -> None:
    for symbol in config.SYMBOLS:
        run(symbol)
    print("\nSaved: data/explain/<symbol>_archetypes.parquet, "
          "results/archetype_base_rates_<symbol>.json")


if __name__ == "__main__":
    main()
