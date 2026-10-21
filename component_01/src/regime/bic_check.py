"""M3 - The state-count check (BIC) for the timeframes where it was skipped.

Run:  python -m src.regime.bic_check --symbol BTCUSDT --interval 1h
      python -m src.regime.bic_check --summary       (table of all ten models)

WHY this exists. When the models were trained, the comparison of 2 to 6
states was run for Daily and 4-hour only; for 1h, 15m and 5m it was skipped
because it means fitting 100 extra models on very large data. This script
runs the same comparison, with the same settings, for those timeframes.

What it does NOT do: it does not change, replace or retrain any of the ten
saved models, and it does not touch any existing result. The comparison
models are fitted, scored and discarded; only their scores are saved, to
results/bic/<symbol>_<interval>_bic.json.

The training data and its scaling are prepared exactly as in
src/regime/train.py (training period only, training mean and spread), so the
scores are comparable with the saved model's.

BIC rewards fit but charges a penalty per parameter; lower is better. With
hundreds of thousands of candles BIC usually keeps improving as states are
added, so the expected outcome is that four states remain a design choice.
"""
import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd

from src import config
from src.regime import hmm_model
from src.regime.train import bic_table

BIC_DIR = config.RESULTS_DIR / "bic"


def training_matrix(symbol: str, interval: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The standardised training features, prepared as in train.py, with the
    training mean and spread used for the scaling."""
    features = pd.read_parquet(config.FEATURES_DIR / f"{symbol}_{interval}_features.parquet")
    is_train = hmm_model.split_train(features, config.TRAIN_END).to_numpy()
    values = features.loc[is_train, config.FEATURE_COLUMNS].to_numpy(dtype=float)
    mean, spread = values.mean(axis=0), values.std(axis=0)
    return (values - mean) / spread, mean, spread


def run(symbol: str, interval: str) -> dict:
    started = time.time()
    x_train, mean, spread = training_matrix(symbol, interval)

    # The saved model, scored on the same data, as a reference point. Its
    # scaling must be the one computed here, or the comparison would be unfair.
    with open(config.MODELS_DIR / f"{symbol}_{interval}_hmm.pkl", "rb") as file:
        saved = pickle.load(file)
    if not (np.allclose(saved.feature_mean, mean) and np.allclose(saved.feature_std, spread)):
        raise ValueError("saved model was trained on differently scaled data")

    table = bic_table(x_train)
    record = {
        "symbol": symbol, "interval": interval, "train_candles": int(len(x_train)),
        "restarts": config.HMM_N_RESTARTS, "max_iterations": config.HMM_MAX_ITER,
        "state_candidates": config.HMM_STATE_CANDIDATES,
        "bic": table.to_dict(orient="records"),
        "saved_model_bic": float(saved.hmm.bic(x_train)),
        "best_full_covariance_states": int(table[table["covariance"] == "full"]
                                           .sort_values("bic")["states"].iloc[0]),
        "minutes": round((time.time() - started) / 60, 1),
    }
    BIC_DIR.mkdir(parents=True, exist_ok=True)
    with open(BIC_DIR / f"{symbol}_{interval}_bic.json", "w") as file:
        json.dump(record, file, indent=2)

    print(f"{symbol} {interval}: {len(x_train):,} training candles, {record['minutes']} minutes")
    print(table.pivot(index="states", columns="covariance", values="bic")
          .to_string(float_format=lambda v: f"{v:14.0f}"))
    print(f"saved 4-state model BIC: {record['saved_model_bic']:.0f} | "
          f"lowest BIC (full covariance) at {record['best_full_covariance_states']} states")
    return record


def summary() -> None:
    """One table for all ten models: the Daily and 4-hour scores come from the
    original training reports, the others from this script."""
    rows = []
    for symbol in config.SYMBOLS:
        for interval in config.INTERVALS:
            path = BIC_DIR / f"{symbol}_{interval}_bic.json"
            if path.exists():
                with open(path) as file:
                    bic = json.load(file)["bic"]
                source = "bic_check"
            else:
                with open(config.RESULTS_DIR / f"{symbol}_{interval}_hmm_report.json") as file:
                    bic = json.load(file)["bic"]
                source = "training report" if bic else "missing"
            full = {row["states"]: row["bic"] for row in bic if row["covariance"] == "full"}
            rows.append({"asset": symbol, "timeframe": interval, "source": source,
                         **{f"{k} states": full.get(k, np.nan) for k in config.HMM_STATE_CANDIDATES},
                         "lowest at": min(full, key=full.get) if full else None})
    table = pd.DataFrame(rows)
    print("BIC with full covariance (lower is better):\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:12.0f}"))
    BIC_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(BIC_DIR / "bic_summary.csv", index=False)
    print(f"\nSaved: {BIC_DIR / 'bic_summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", choices=config.SYMBOLS)
    parser.add_argument("--interval", choices=config.INTERVALS)
    parser.add_argument("--summary", action="store_true", help="print the table for all ten models")
    args = parser.parse_args()
    if args.summary:
        summary()
    elif args.symbol and args.interval:
        run(args.symbol, args.interval)
    else:
        parser.error("give --symbol and --interval, or --summary")


if __name__ == "__main__":
    main()
