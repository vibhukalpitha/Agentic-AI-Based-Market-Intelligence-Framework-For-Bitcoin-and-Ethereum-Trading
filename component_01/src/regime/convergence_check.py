"""M3 - Would the labels change if a model that stopped early were trained to the end?

Run:  python -m src.regime.convergence_check                 (BTCUSDT 5m, the one case)
      python -m src.regime.convergence_check --symbol ETHUSDT --interval 5m

WHY this exists. The BTC 5m model stopped at the 300-iteration limit (last
log-likelihood gain 0.00033, tolerance 0.0001); the other nine reached the
tolerance. "It stopped early" raises one question for the report: does it
matter? This script answers it by measurement:

  1. A COPY of the saved model continues training from exactly where it
     stopped (init_params="" keeps its current parameters as the start), on
     the same standardised training data, until it reaches the tolerance.
  2. The regime labels are recomputed with the continued model, by the same
     filtered (past-only) probabilities and the same naming rule.
  3. They are compared with the saved labels candle by candle, separately
     for the training and the test period.

What it does NOT do: it does not replace the saved model or the saved labels,
and it changes no result. Every later stage (H1, H2, M6, ...) keeps using the
saved model. Only results/convergence_check_<symbol>_<interval>.json is written.
"""
import argparse
import copy
import json
import pickle
import time

import numpy as np
import pandas as pd

from src import config
from src.regime import hmm_model, state_mapping

# Enough for the remaining gain (0.00033 per candle, falling) to drop below the
# tolerance; if it is not, the record says so rather than hiding it.
EXTRA_ITERATIONS = 2000


def agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float((a == b).mean())


def run(symbol: str, interval: str) -> dict:
    started = time.time()
    features = pd.read_parquet(config.FEATURES_DIR / f"{symbol}_{interval}_features.parquet")
    is_train = hmm_model.split_train(features, config.TRAIN_END).to_numpy()
    with open(config.MODELS_DIR / f"{symbol}_{interval}_hmm.pkl", "rb") as file:
        saved = pickle.load(file)
    x_all = saved.standardise(features)
    x_train = x_all[is_train]

    # The saved labels must be reproducible from the saved model before
    # anything is compared with them.
    stored = pd.read_parquet(config.REGIMES_DIR / f"{symbol}_{interval}_regimes.parquet")
    saved_states = hmm_model.filtered_probabilities(saved.hmm, x_all).argmax(axis=1)
    if not np.array_equal(saved_states, stored["state"].to_numpy()):
        raise ValueError("the saved labels are not reproduced by the saved model")

    # ---- 1. continue training a copy ----
    before = saved.hmm.score(x_train) / len(x_train)
    continued = copy.deepcopy(saved.hmm)
    continued.init_params = ""          # start from the saved parameters, not afresh
    continued.n_iter = EXTRA_ITERATIONS
    continued.fit(x_train)
    reached_tolerance, last_gain = hmm_model.convergence(continued)
    after = continued.score(x_train) / len(x_train)

    # ---- 2. relabel with the continued model, same rules ----
    probabilities = hmm_model.filtered_probabilities(continued, x_all)
    states = probabilities.argmax(axis=1)
    profile = state_mapping.state_profiles(features[is_train], states[is_train])
    mapping = state_mapping.map_states(profile)
    new_names = np.array([mapping[s] for s in states])
    old_names = stored["regime"].to_numpy()

    # ---- 3. compare ----
    shares = pd.DataFrame({
        "saved_train": pd.Series(old_names[is_train]).value_counts(normalize=True),
        "continued_train": pd.Series(new_names[is_train]).value_counts(normalize=True),
        "saved_test": pd.Series(old_names[~is_train]).value_counts(normalize=True),
        "continued_test": pd.Series(new_names[~is_train]).value_counts(normalize=True),
    }).reindex(config.REGIMES).fillna(0.0)
    record = {
        "symbol": symbol, "interval": interval,
        "saved_model": {"iterations": int(saved.hmm.monitor_.iter),
                        "last_log_likelihood_gain": hmm_model.convergence(saved.hmm)[1],
                        "train_log_likelihood_per_candle": before},
        "continued_model": {"extra_iterations_allowed": EXTRA_ITERATIONS,
                            "extra_iterations_used": int(continued.monitor_.iter),
                            "reached_tolerance": reached_tolerance,
                            "last_log_likelihood_gain": last_gain,
                            "train_log_likelihood_per_candle": after},
        "naming_unchanged": {str(k): v for k, v in mapping.items()}
                            == {str(k): v for k, v in saved.state_to_regime.items()},
        "label_agreement": {"train": agreement(old_names[is_train], new_names[is_train]),
                            "test": agreement(old_names[~is_train], new_names[~is_train]),
                            "all": agreement(old_names, new_names)},
        "transitions": {"saved": hmm_model.count_transitions(saved_states),
                        "continued": hmm_model.count_transitions(states)},
        "mean_confidence": {"saved": float(stored["confidence"].mean()),
                            "continued": float(probabilities.max(axis=1).mean())},
        "regime_share": json.loads(shares.to_json(orient="index")),
        "minutes": round((time.time() - started) / 60, 1),
    }
    path = config.RESULTS_DIR / f"convergence_check_{symbol}_{interval}.json"
    with open(path, "w") as file:
        json.dump(record, file, indent=2)

    c = record["continued_model"]
    print(f"{symbol} {interval}: saved model stopped after {record['saved_model']['iterations']} "
          f"iterations; the copy needed {c['extra_iterations_used']} more "
          f"(reached tolerance: {c['reached_tolerance']}, last gain {c['last_log_likelihood_gain']:.6f})")
    print(f"training log-likelihood per candle: {before:.5f} -> {after:.5f}")
    print(f"naming unchanged: {record['naming_unchanged']}")
    print("label agreement with the saved labels: "
          + ", ".join(f"{k} {v:.2%}" for k, v in record["label_agreement"].items()))
    print(f"transitions: {record['transitions']} | mean confidence: {record['mean_confidence']}")
    print(shares.to_string(float_format=lambda v: f"{v:8.1%}"))
    print(f"Saved: {path} ({record['minutes']} minutes)")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", default="BTCUSDT", choices=config.SYMBOLS)
    parser.add_argument("--interval", default="5m", choices=config.INTERVALS)
    args = parser.parse_args()
    run(args.symbol, args.interval)


if __name__ == "__main__":
    main()
