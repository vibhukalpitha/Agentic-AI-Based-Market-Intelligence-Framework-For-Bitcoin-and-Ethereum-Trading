"""M9 - Evaluation of the Regime Stability Score (M6).

Run:  python -m src.experiments.m6_evaluation --period train   (in-sample check of the script)
      python -m src.experiments.m6_evaluation --period test    (the evaluation; run ONCE)

Questions, fixed before the test-period run:

  1. Is the delivered score better than the two simple baselines?
       B0  a constant (the training base rate)
       B1  the Daily HMM's own persistence probability
  2. Is it calibrated - when it says 80%, does the regime persist about 80% of the time?
  3. Do the cross-timeframe signals add anything (S2 against S1)?

Measures:
  Brier score   mean squared gap between the forecast probability and what
                happened (0 or 1). Lower is better. 0.25 is a coin flip.
  log loss      like Brier but punishes confident wrong forecasts harder.
  AUC           probability that a day whose regime persisted got a higher
                score than a day whose regime changed. 0.5 = no skill.

Neighbouring days share most of their 3-day outcome, so they are not
independent. Confidence intervals therefore resample whole weeks, not days.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src import config
from src.stability.stability import log_loss

SCORES = ["B0", "B1", "S1", "S2"]
CALIBRATION_BINS = 5
SCORE_COLOUR, MUTED, GRID, INK = "#2a78d6", "#898781", "#e1e0d9", "#0b0b0b"


def brier(target: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean((probability - target) ** 2))


def auc(target: np.ndarray, probability: np.ndarray) -> float:
    """P(a persisting day scored higher than a changing day), ties count half."""
    positive = target == 1
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(probability)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def calibration_table(target: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    """Days sorted by forecast and cut into equal-sized groups: the average
    forecast in each group against how often the regime really persisted."""
    order = np.argsort(probability, kind="stable")
    rows = []
    for group in np.array_split(order, CALIBRATION_BINS):
        rows.append({"days": len(group),
                     "mean_forecast": float(probability[group].mean()),
                     "observed_rate": float(target[group].mean())})
    return pd.DataFrame(rows)


def weekly_block_differences(frame: pd.DataFrame, pairs: list[tuple[str, str]],
                             rng: np.random.Generator) -> dict:
    """Brier(first) - Brier(second) with a 95% interval from resampling whole
    weeks. Negative means the first score is better."""
    block = (np.arange(len(frame)) // config.STABILITY_BLOCK_DAYS)
    members = [np.flatnonzero(block == b) for b in np.unique(block)]
    target = frame["target"].to_numpy()
    out = {}
    for first, second in pairs:
        errors = ((frame[first].to_numpy() - target) ** 2
                  - (frame[second].to_numpy() - target) ** 2)
        draws = np.empty(config.H1_BOOTSTRAPS)
        for b in range(config.H1_BOOTSTRAPS):
            pick = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
            draws[b] = errors[pick].mean()
        out[f"{first}_minus_{second}"] = {
            "brier_difference": float(errors.mean()),
            "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5))}
    return out


def evaluate(symbol: str, period: str, rng: np.random.Generator) -> dict:
    scores = pd.read_parquet(config.STABILITY_DIR / f"{symbol}_stability.parquet")
    # Days with no known outcome yet (the last few of the data) cannot be scored.
    frame = scores[(scores["period"] == period) & scores["target"].notna()].reset_index(drop=True)
    with open(config.RESULTS_DIR / f"stability_model_{symbol}.json") as file:
        delivered = json.load(file)["delivered_model"]
    target = frame["target"].to_numpy()

    metrics = [{"score": name, "delivered": name == delivered,
                "brier": brier(target, frame[name].to_numpy()),
                "log_loss": log_loss(target, frame[name].to_numpy()),
                "auc": auc(target, frame[name].to_numpy())} for name in SCORES]
    return {
        "days": int(len(frame)), "from": str(frame["date"].iloc[0].date()),
        "to": str(frame["date"].iloc[-1].date()),
        "persisted_rate": float(target.mean()), "delivered_model": delivered,
        "metrics": metrics,
        "calibration": calibration_table(target, frame[delivered].to_numpy()).to_dict("records"),
        "differences": weekly_block_differences(
            frame, [(delivered, "B0"), (delivered, "B1"), ("S2", "S1")], rng),
    }


def print_results(symbol: str, period: str, result: dict) -> None:
    print(f"\n{'=' * 74}\n{symbol}  ({period} period: {result['from']} -> {result['to']}, "
          f"{result['days']} days)\n{'=' * 74}")
    print(f"regime persisted {config.STABILITY_HORIZON_DAYS} days on "
          f"{result['persisted_rate']:.1%} of days | delivered model: {result['delivered_model']}")
    print("\nscores (Brier and log loss: lower is better; AUC: higher is better):")
    print(pd.DataFrame(result["metrics"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\ncalibration of {result['delivered_model']} (forecast against what happened):")
    print(pd.DataFrame(result["calibration"]).to_string(index=False,
                                                         float_format=lambda v: f"{v:.3f}"))
    print("\nBrier differences with 95% intervals (negative = the first is better):")
    print(pd.DataFrame(result["differences"]).T.to_string(float_format=lambda v: f"{v:+.4f}"))


def plot_calibration(results: dict, period: str, path) -> None:
    """Reliability diagram: points on the diagonal mean the score can be read
    as a probability."""
    fig, axes = plt.subplots(1, len(config.SYMBOLS), figsize=(11, 4.8), sharey=True)
    for ax, symbol in zip(axes, config.SYMBOLS):
        table = pd.DataFrame(results[symbol]["calibration"])
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.5, linestyle="--")
        ax.plot(table["mean_forecast"], table["observed_rate"], color=SCORE_COLOUR,
                linewidth=2, marker="o", markersize=8, markeredgecolor="white",
                markeredgewidth=2)
        ax.annotate("perfect calibration", xy=(0.52, 0.52), xytext=(6, -12),
                    textcoords="offset points", color=MUTED, fontsize=9)
        ax.set_xlim(0.4, 1.0)
        ax.set_ylim(0.4, 1.0)
        ax.set_title(f"{symbol}: {results[symbol]['delivered_model']} score, {period} period",
                     loc="left", color=INK, fontsize=11)
        ax.set_xlabel("Forecast probability that the regime persists", color=MUTED)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.tick_params(colors=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    axes[0].set_ylabel("Share of days on which it did persist", color=MUTED)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--period", choices=["train", "test"], required=True,
                        help="'train' is an in-sample check; 'test' is the evaluation, run once")
    args = parser.parse_args()

    rng = np.random.default_rng(config.RANDOM_SEED)
    results = {"design": {"period": args.period, "horizon_days": config.STABILITY_HORIZON_DAYS,
                          "block_days": config.STABILITY_BLOCK_DAYS,
                          "bootstraps": config.H1_BOOTSTRAPS, "seed": config.RANDOM_SEED}}
    for symbol in config.SYMBOLS:
        results[symbol] = evaluate(symbol, args.period, rng)
        print_results(symbol, args.period, results[symbol])

    path = config.RESULTS_DIR / f"m6_evaluation_{args.period}.json"
    with open(path, "w") as file:
        json.dump(results, file, indent=2)
    figure = config.RESULTS_DIR / f"m6_calibration_{args.period}.png"
    plot_calibration(results, args.period, figure)
    print(f"\nSaved: {path.name}, {figure.name}")


if __name__ == "__main__":
    main()
