"""M3 - Quality checks for the ten regime models.

Run:  python -m src.regime.model_quality

WHY there is no "accuracy". Accuracy needs a correct answer for every candle
to compare with. The regime models are unsupervised: no candle carries a true
regime label, so accuracy cannot be computed and any figure quoted for it
would be invented. Three checks take its place:

  1. Convergence      did training settle before the iteration limit?
  2. Generalisation   does the model explain unseen data about as well as its
                      training data? Measured by the average log-likelihood
                      per candle on the training and the test period. A large
                      drop would mean the model learned its training period
                      and not the market.
  3. Meaning holds    are the regimes still what their names say on the test
                      period? The names were given from training-period
                      behaviour; here the same quantities are measured on data
                      the model never saw:
                        Bullish has the highest average RSI and a positive average return
                        Bearish has the lowest average RSI and a negative average return
                        Volatile is more volatile than Sideways

Nothing is fitted or changed. It reads the saved models and feature files.
"""
import json
import pickle

import numpy as np
import pandas as pd

from src import config
from src.regime import hmm_model


def check_model(symbol: str, interval: str) -> dict:
    with open(config.MODELS_DIR / f"{symbol}_{interval}_hmm.pkl", "rb") as file:
        model = pickle.load(file)
    features = pd.read_parquet(config.FEATURES_DIR / f"{symbol}_{interval}_features.parquet")
    regimes = pd.read_parquet(config.REGIMES_DIR / f"{symbol}_{interval}_regimes.parquet",
                              columns=["regime", "confidence", "is_train"])
    is_train = regimes["is_train"].to_numpy()
    x = model.standardise(features)

    converged, last_gain = hmm_model.convergence(model.hmm)
    train_fit = model.hmm.score(x[is_train]) / is_train.sum()
    test_fit = model.hmm.score(x[~is_train]) / (~is_train).sum()

    test = features[~is_train].assign(regime=regimes["regime"].to_numpy()[~is_train])
    profile = test.groupby("regime")[["rsi", "log_return", "atr_rel"]].mean()
    checks = {
        "bullish_highest_rsi": bool(profile["rsi"].idxmax() == "Bullish"),
        "bearish_lowest_rsi": bool(profile["rsi"].idxmin() == "Bearish"),
        "bullish_positive_return": bool(profile.loc["Bullish", "log_return"] > 0),
        "bearish_negative_return": bool(profile.loc["Bearish", "log_return"] < 0),
        "volatile_above_sideways": bool(profile.loc["Volatile", "atr_rel"]
                                        > profile.loc["Sideways", "atr_rel"]),
    }
    return {
        "symbol": symbol, "interval": interval,
        "train_candles": int(is_train.sum()), "test_candles": int((~is_train).sum()),
        "converged": converged, "iterations": int(model.hmm.monitor_.iter),
        "log_likelihood_per_candle_train": float(train_fit),
        "log_likelihood_per_candle_test": float(test_fit),
        "test_minus_train": float(test_fit - train_fit),
        "checks_passed": int(sum(checks.values())), "checks": checks,
        "test_profile": {name: {"rsi": float(row["rsi"]), "mean_return": float(row["log_return"]),
                                "volatility_vs_norm": float(np.exp(row["atr_rel"]))}
                         for name, row in profile.iterrows()},
    }


def main() -> None:
    results = [check_model(symbol, interval)
               for symbol in config.SYMBOLS for interval in config.INTERVALS]
    table = pd.DataFrame(results)
    print("1 and 2. Convergence, and fit on training against unseen data "
          "(log-likelihood per candle; higher is better):\n")
    print(table[["symbol", "interval", "converged", "iterations",
                 "log_likelihood_per_candle_train", "log_likelihood_per_candle_test",
                 "test_minus_train"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n3. Do the regimes still mean what their names say on the TEST period?\n")
    checks = pd.DataFrame([{"symbol": r["symbol"], "interval": r["interval"], **r["checks"],
                            "passed": f"{r['checks_passed']} of 5"} for r in results])
    print(checks.to_string(index=False))

    print("\nTest-period behaviour of each regime (average RSI | average return per candle | "
          "volatility as a multiple of its yearly norm):\n")
    rows = []
    for r in results:
        row = {"symbol": r["symbol"], "interval": r["interval"]}
        for name in config.REGIMES:
            p = r["test_profile"][name]
            row[name] = f"{p['rsi']:.0f} | {p['mean_return']:+.4%} | {p['volatility_vs_norm']:.2f}x"
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / "model_quality.json", "w") as file:
        json.dump(results, file, indent=2)
    total = sum(r["checks_passed"] for r in results)
    print(f"\nMeaning checks passed: {total} of {5 * len(results)}. "
          f"Models converged: {sum(r['converged'] for r in results)} of {len(results)}.")
    print("Saved: results/model_quality.json")


if __name__ == "__main__":
    main()
