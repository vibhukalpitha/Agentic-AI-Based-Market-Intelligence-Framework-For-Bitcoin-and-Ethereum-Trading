"""M9 - Test of Novelty 4: do the conflict archetypes carry information?

Run:  python -m src.experiments.n4_archetypes --period train   (in-sample check of the script)
      python -m src.experiments.n4_archetypes --period test    (the test; run ONCE)

The archetype rules and their base rates come from the training period
(src/explain/archetypes.py). This script asks whether they hold up on other data.

Questions, fixed before the test-period run:

  1. PRIMARY. Using each day's archetype base rate (from training) as a
     forecast of "the Daily regime changes within 3 days", is the forecast
     better than one constant rate for every day? (Brier score.)
  2. Is the Daily regime really more likely to change under "Opposed" than
     under "Aligned"? (Difference in change rates on this period.)
  3. Do the archetypes keep the same order of risk as in training?

How it can fail: if every archetype changes about equally often, the forecast
is no better than the constant and the Opposed - Aligned difference includes
zero. Then the taxonomy carries no information.

For context only, the Regime Stability Score (M6) is scored on the same days.
The archetypes are meant to EXPLAIN, so they are not expected to beat it.

Intervals resample whole weeks, because neighbouring days share most of their
3-day outcome.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import config
from src.experiments.m6_evaluation import brier, weekly_block_differences
from src.explain.archetypes import ARCHETYPES

TRAIN_COLOUR, PERIOD_COLOUR = "#898781", "#2a78d6"
MUTED, GRID, INK = "#898781", "#e1e0d9", "#0b0b0b"


def contrast(frame: pd.DataFrame, high: str, low: str, rng: np.random.Generator) -> dict:
    """Change rate under `high` minus change rate under `low`, with a 95%
    interval from resampling whole weeks."""
    block = np.arange(len(frame)) // config.STABILITY_BLOCK_DAYS
    members = [np.flatnonzero(block == b) for b in np.unique(block)]
    changed = frame["changed"].to_numpy()
    is_high = (frame["archetype"] == high).to_numpy()
    is_low = (frame["archetype"] == low).to_numpy()

    def difference(rows: np.ndarray) -> float:
        if not is_high[rows].any() or not is_low[rows].any():
            return np.nan
        return changed[rows][is_high[rows]].mean() - changed[rows][is_low[rows]].mean()

    draws = np.array([
        difference(np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))]))
        for _ in range(config.H1_BOOTSTRAPS)])
    return {"difference": float(difference(np.arange(len(frame)))),
            "ci_low": float(np.nanpercentile(draws, 2.5)),
            "ci_high": float(np.nanpercentile(draws, 97.5))}


def evaluate(symbol: str, period: str, rng: np.random.Generator) -> dict:
    table = pd.read_parquet(config.EXPLAIN_DIR / f"{symbol}_archetypes.parquet")
    frame = table[(table["period"] == period) & table["changed"].notna()].reset_index(drop=True)

    rows = []
    for name in ARCHETYPES:
        part = frame[frame["archetype"] == name]
        rows.append({"archetype": name, "days": int(len(part)),
                     "share_of_days": len(part) / len(frame),
                     "training_base_rate": float(part["base_rate"].iloc[0]) if len(part) else np.nan,
                     "change_rate": float(part["changed"].mean()) if len(part) else np.nan})
    by_archetype = pd.DataFrame(rows)
    order = spearmanr(by_archetype["training_base_rate"], by_archetype["change_rate"]).statistic

    target = frame["changed"].to_numpy()
    scoring = pd.DataFrame({"target": target, "archetype": frame["base_rate"],
                            "constant": frame["overall_rate"]})
    result = {
        "days": int(len(frame)), "from": str(frame["date"].iloc[0].date()),
        "to": str(frame["date"].iloc[-1].date()), "change_rate": float(target.mean()),
        "by_archetype": by_archetype.to_dict("records"),
        "brier_constant": brier(target, scoring["constant"].to_numpy()),
        "brier_archetype": brier(target, scoring["archetype"].to_numpy()),
        "archetype_minus_constant": weekly_block_differences(
            scoring, [("archetype", "constant")], rng)["archetype_minus_constant"],
        "opposed_minus_aligned": contrast(frame, "Opposed", "Aligned", rng),
        "rank_agreement_with_training": float(order),
    }

    # Context only: the Regime Stability Score on the same days (change = 1 - persist).
    stability = pd.read_parquet(config.STABILITY_DIR / f"{symbol}_stability.parquet",
                                columns=["date", "delivered"])
    both = frame.merge(stability, on="date")
    result["context_days_with_stability_score"] = int(len(both))
    result["context_brier_stability_score"] = brier(both["changed"].to_numpy(),
                                                    1.0 - both["delivered"].to_numpy())
    result["context_brier_archetype_same_days"] = brier(both["changed"].to_numpy(),
                                                        both["base_rate"].to_numpy())
    return result


def print_results(symbol: str, period: str, result: dict) -> None:
    print(f"\n{'=' * 74}\n{symbol}  ({period} period: {result['from']} -> {result['to']}, "
          f"{result['days']} days)\n{'=' * 74}")
    print(f"Daily regime changed within {config.STABILITY_HORIZON_DAYS} days on "
          f"{result['change_rate']:.1%} of days\n")
    print(pd.DataFrame(result["by_archetype"]).to_string(
        index=False, formatters={"share_of_days": "{:.1%}".format,
                                 "training_base_rate": "{:.1%}".format,
                                 "change_rate": "{:.1%}".format}))
    d = result["archetype_minus_constant"]
    print(f"\n1. Brier: constant {result['brier_constant']:.4f} | archetype forecast "
          f"{result['brier_archetype']:.4f} | difference {d['brier_difference']:+.4f} "
          f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  (negative = archetypes better)")
    c = result["opposed_minus_aligned"]
    print(f"2. Opposed - Aligned change rate: {c['difference']:+.1%} "
          f"[{c['ci_low']:+.1%}, {c['ci_high']:+.1%}]")
    print(f"3. Rank agreement with training (1.0 = same order): "
          f"{result['rank_agreement_with_training']:+.2f}")
    print(f"context: on the {result['context_days_with_stability_score']} days that have a "
          f"stability score, Brier is {result['context_brier_stability_score']:.4f} for that "
          f"score and {result['context_brier_archetype_same_days']:.4f} for the archetype forecast")


def plot_rates(results: dict, period: str, path) -> None:
    """Change rate of each archetype: training base rate against this period."""
    fig, axes = plt.subplots(1, len(config.SYMBOLS), figsize=(12, 4.2), sharey=True)
    for ax, symbol in zip(axes, config.SYMBOLS):
        table = pd.DataFrame(results[symbol]["by_archetype"])
        rows = np.arange(len(table))
        for column, colour, shift in (("training_base_rate", TRAIN_COLOUR, -0.12),
                                      ("change_rate", PERIOD_COLOUR, 0.12)):
            ax.plot(100 * table[column], rows + shift, "o", color=colour, markersize=9,
                    markeredgecolor="white", markeredgewidth=2)
            for value, row in zip(100 * table[column], rows + shift):
                ax.annotate(f"{value:.0f}%", (value, row), xytext=(9, -3),
                            textcoords="offset points", color=INK, fontsize=9)
        ax.axvline(100 * results[symbol]["change_rate"], color=MUTED, linewidth=1, linestyle="--")
        ax.set_yticks(rows, table["archetype"])
        ax.set_ylim(len(table) - 0.5, -0.5)
        ax.set_xlim(0, 50)
        ax.set_title(f"{symbol}", loc="left", color=INK, fontsize=11)
        ax.set_xlabel(f"Days on which the Daily regime changed within "
                      f"{config.STABILITY_HORIZON_DAYS} days (%)", color=MUTED)
        ax.grid(True, axis="x", color=GRID, linewidth=0.6)
        ax.tick_params(colors=MUTED, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=TRAIN_COLOUR,
                          label="Training period (the base rate)"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=PERIOD_COLOUR,
                          label=f"{period.capitalize()} period"),
               plt.Line2D([], [], color=MUTED, linestyle="--", label=f"All days, {period} period")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.06), labelcolor=INK)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


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

    path = config.RESULTS_DIR / f"n4_archetypes_{args.period}.json"
    with open(path, "w") as file:
        json.dump(results, file, indent=2)
    figure = config.RESULTS_DIR / f"n4_archetype_rates_{args.period}.png"
    plot_rates(results, args.period, figure)
    print(f"\nSaved: {path.name}, {figure.name}")


if __name__ == "__main__":
    main()
