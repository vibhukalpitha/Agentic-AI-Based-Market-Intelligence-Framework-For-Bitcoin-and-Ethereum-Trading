"""M3 - Train one regime model per asset for one timeframe, and report on it.

Run:  python -m src.regime.train --interval 1d

Steps, per asset:
  1. Split by time: fit on candles up to TRAIN_END, never on later ones.
  2. Standardise with training-period statistics only.
  3. Compare state counts by BIC (a check on the 4-state assumption).
  4. Fit the 4-state model from several random starts.
  5. Label every candle from filtered (past-only) probabilities.
  6. Name the states by the fixed rule in state_mapping.py.
  7. Count regime transitions - the number that decides how H1 can be tested.
  8. Save the model, the labels, a JSON record and a figure.
"""
import argparse
import json
import pickle

import matplotlib
matplotlib.use("Agg")  # write image files without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.regime import hmm_model, state_mapping

# Figure colours (validated for colour-blind separation; each regime also has
# its own row in the chart, so colour is never the only cue).
REGIME_COLOURS = {"Bullish": "#1baf7a", "Bearish": "#e34948",
                  "Sideways": "#2a78d6", "Volatile": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def bic_table(x_train: np.ndarray) -> pd.DataFrame:
    """BIC for each candidate state count and covariance type (lower is better).

    BIC rewards fit but charges a penalty per parameter, so it does not simply
    prefer the biggest model.
    """
    rows = []
    for covariance_type in ("diag", "full"):
        for n_states in config.HMM_STATE_CANDIDATES:
            model, _ = hmm_model.fit_hmm(x_train, n_states, covariance_type)
            rows.append({"covariance": covariance_type, "states": n_states,
                         "log_likelihood": model.score(x_train),
                         "bic": model.bic(x_train)})
    return pd.DataFrame(rows)


def transition_table(states: np.ndarray, is_train: np.ndarray) -> pd.DataFrame:
    """Transition counts for each candidate persistence threshold."""
    rows = []
    for persistence in config.PERSISTENCE_CANDIDATES:
        confirmed = hmm_model.confirmed_states(states, persistence)
        changed = np.concatenate(([False], confirmed[1:] != confirmed[:-1]))
        total = int(changed.sum())
        rows.append({"persistence": persistence,
                     "train": int(changed[is_train].sum()),
                     "test": int(changed[~is_train].sum()),
                     "total": total,
                     "mean_days_per_regime": len(states) / (total + 1)})
    return pd.DataFrame(rows)


def plot_regimes(regimes: pd.DataFrame, symbol: str, interval: str) -> None:
    """Price (log scale) above a strip showing which regime was active when."""
    fig, (ax_price, ax_strip) = plt.subplots(
        2, 1, figsize=(13, 6.5), sharex=True, height_ratios=[3, 1.3],
        gridspec_kw={"hspace": 0.06})
    time = regimes["open_time"]

    ax_price.plot(time, regimes["close"], color=INK, linewidth=1.1)
    ax_price.set_yscale("log")
    ax_price.set_ylabel("Close price (USDT, log scale)", color=MUTED)
    ax_price.set_title(f"{symbol} {interval}: detected regimes (filtered, past-only labels)",
                       loc="left", color=INK, fontsize=12)
    ax_price.grid(True, which="major", color=GRID, linewidth=0.6)

    for row, name in enumerate(reversed(config.REGIMES)):
        active = (regimes["regime"] == name).to_numpy()
        ax_strip.fill_between(time, row + 0.1, row + 0.9, where=active, step="post",
                              color=REGIME_COLOURS[name], linewidth=0)
    ax_strip.set_yticks(np.arange(len(config.REGIMES)) + 0.5, list(reversed(config.REGIMES)))
    ax_strip.set_ylim(0, len(config.REGIMES))
    ax_strip.tick_params(axis="y", length=0)

    train_end = pd.Timestamp(config.TRAIN_END)
    for ax in (ax_price, ax_strip):
        ax.axvline(train_end, color=MUTED, linewidth=1, linestyle="--")
        ax.tick_params(colors=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    ax_price.annotate("train | test", xy=(train_end, 1), xycoords=("data", "axes fraction"),
                      xytext=(0, -12), textcoords="offset points", ha="center",
                      color=MUTED, fontsize=9, backgroundcolor="white")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.RESULTS_DIR / f"{symbol}_{interval}_regimes.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_saved(symbol: str, interval: str) -> tuple[hmm_model.RegimeModel, pd.DataFrame]:
    """Load a model fitted earlier, with the BIC table from its saved record."""
    with open(config.MODELS_DIR / f"{symbol}_{interval}_hmm.pkl", "rb") as file:
        model = pickle.load(file)
    with open(config.RESULTS_DIR / f"{symbol}_{interval}_hmm_report.json") as file:
        bic = pd.DataFrame(json.load(file)["bic"])
    return model, bic


def train_one(symbol: str, interval: str, run_bic: bool = True, relabel: bool = False) -> None:
    """relabel=True reuses the saved fitted model and only redoes the naming,
    labels, report and figure. The fit itself is not touched."""
    features = pd.read_parquet(config.FEATURES_DIR / f"{symbol}_{interval}_features.parquet")
    is_train = hmm_model.split_train(features, config.TRAIN_END).to_numpy()
    train_values = features.loc[is_train, config.FEATURE_COLUMNS].to_numpy(dtype=float)

    model = hmm_model.RegimeModel(
        symbol=symbol, interval=interval, hmm=None,
        feature_mean=train_values.mean(axis=0), feature_std=train_values.std(axis=0),
        train_end=config.TRAIN_END, seed=-1)
    x_all = model.standardise(features)
    x_train = x_all[is_train]

    print(f"\n{'=' * 70}\n{symbol} {interval}\n{'=' * 70}")
    print(f"train: {is_train.sum():,} candles up to {config.TRAIN_END} | "
          f"test: {(~is_train).sum():,} candles after")

    bic = pd.DataFrame()
    if relabel:
        saved, bic = load_saved(symbol, interval)
        # The saved model must belong to exactly these features and this split.
        if not (np.allclose(saved.feature_mean, model.feature_mean)
                and np.allclose(saved.feature_std, model.feature_std)
                and saved.train_end == config.TRAIN_END):
            raise ValueError("saved model does not match the current features; refit it")
        model.hmm, model.seed = saved.hmm, saved.seed
    elif run_bic:
        bic = bic_table(x_train)
    if not bic.empty:
        print("\nState-count check (BIC, lower is better):")
        print(bic.pivot(index="states", columns="covariance", values="bic")
              .to_string(float_format=lambda v: f"{v:12.0f}"))

    if not relabel:
        model.hmm, model.seed = hmm_model.fit_hmm(
            x_train, config.HMM_N_STATES, config.HMM_COVARIANCE_TYPE)
    converged, last_gain = hmm_model.convergence(model.hmm)
    print(f"\nFinal model: {config.HMM_N_STATES} states, {config.HMM_COVARIANCE_TYPE} "
          f"covariance, best seed {model.seed}, converged={converged} "
          f"after {model.hmm.monitor_.iter} of {config.HMM_MAX_ITER} iterations "
          f"(last log-likelihood gain {last_gain:.5f}, tolerance {config.HMM_TOLERANCE})")
    if not converged:
        print("  WARNING: training stopped at the iteration limit before reaching the tolerance")

    probabilities = hmm_model.filtered_probabilities(model.hmm, x_all)
    states = probabilities.argmax(axis=1)

    # Names are decided from TRAINING candles only.
    profile = state_mapping.state_profiles(features[is_train], states[is_train])
    model.state_to_regime = state_mapping.map_states(profile)
    profile.insert(0, "regime", pd.Series(model.state_to_regime))
    print("\nState profiles on training data (averages, original units):")
    print(profile.to_string(float_format=lambda v: f"{v:9.4f}"))
    warnings = state_mapping.mapping_warnings(profile, model.state_to_regime)
    for warning in warnings:
        print("  WARNING:", warning)

    regime_names = np.array([model.state_to_regime[s] for s in states])
    shares = pd.DataFrame({
        "train": pd.Series(regime_names[is_train]).value_counts(normalize=True),
        "test": pd.Series(regime_names[~is_train]).value_counts(normalize=True),
    }).reindex(config.REGIMES).fillna(0.0)
    print("\nShare of time in each regime:")
    print(shares.to_string(float_format=lambda v: f"{v:7.1%}"))

    order = [s for name in config.REGIMES
             for s, mapped in model.state_to_regime.items() if mapped == name]
    transmat = pd.DataFrame(model.hmm.transmat_[np.ix_(order, order)],
                            index=config.REGIMES, columns=config.REGIMES)
    print("\nTransition matrix (row = today, column = tomorrow):")
    print(transmat.to_string(float_format=lambda v: f"{v:8.3f}"))

    transitions = transition_table(states, is_train)
    print("\nRegime transitions by persistence threshold (candles a new regime must hold):")
    print(transitions.to_string(index=False, float_format=lambda v: f"{v:8.1f}"))

    # ---- save everything needed to reproduce and audit this run ----
    regimes = features[config.META_COLUMNS].copy()
    regimes["is_train"] = is_train
    regimes["state"] = states
    regimes["regime"] = regime_names
    regimes["confidence"] = probabilities.max(axis=1)
    for state, name in model.state_to_regime.items():
        regimes[f"p_{name.lower()}"] = probabilities[:, state]
    config.REGIMES_DIR.mkdir(parents=True, exist_ok=True)
    regimes.to_parquet(config.REGIMES_DIR / f"{symbol}_{interval}_regimes.parquet", index=False)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.MODELS_DIR / f"{symbol}_{interval}_hmm.pkl", "wb") as file:
        pickle.dump(model, file)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "symbol": symbol, "interval": interval, "train_end": config.TRAIN_END,
        "train_candles": int(is_train.sum()), "test_candles": int((~is_train).sum()),
        "n_states": config.HMM_N_STATES, "covariance_type": config.HMM_COVARIANCE_TYPE,
        "restarts": config.HMM_N_RESTARTS, "best_seed": model.seed,
        "converged": converged,
        "iterations": int(model.hmm.monitor_.iter),
        "last_log_likelihood_gain": last_gain,
        "train_log_likelihood": float(model.hmm.score(x_train)),
        "features": config.FEATURE_COLUMNS,
        "state_to_regime": {str(k): v for k, v in model.state_to_regime.items()},
        "mapping_warnings": warnings,
        "state_profiles": json.loads(profile.to_json(orient="index")),
        "regime_share": json.loads(shares.to_json(orient="index")),
        "transition_matrix": json.loads(transmat.to_json(orient="index")),
        "bic": bic.to_dict(orient="records"),
        "transitions": transitions.to_dict(orient="records"),
    }
    with open(config.RESULTS_DIR / f"{symbol}_{interval}_hmm_report.json", "w") as file:
        json.dump(record, file, indent=2)

    plot_regimes(regimes, symbol, interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interval", default="1d", choices=config.INTERVALS)
    parser.add_argument("--symbols", nargs="+", default=config.SYMBOLS, choices=config.SYMBOLS)
    parser.add_argument("--skip-bic", action="store_true",
                        help="skip the state-count comparison (slow on 15m and 5m)")
    parser.add_argument("--relabel", action="store_true",
                        help="reuse the saved fitted model; redo naming, labels, report, figure")
    args = parser.parse_args()
    for symbol in args.symbols:
        train_one(symbol, args.interval, run_bic=not args.skip_bic, relabel=args.relabel)
    print("\nSaved: labels in", config.REGIMES_DIR, "| models in", config.MODELS_DIR,
          "| reports and figures in", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
