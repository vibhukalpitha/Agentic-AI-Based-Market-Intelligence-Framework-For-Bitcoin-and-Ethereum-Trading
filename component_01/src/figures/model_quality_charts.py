"""Charts of model quality, for the report and the viva.

Run:  python -m src.figures.model_quality_charts      (writes results/model_quality/)

WHY. The regime models are unsupervised: no candle has a true regime label, so
there is no "accuracy" to show. They are judged instead by how well they fit
data they never saw, whether the number of states is supported, and whether
each regime keeps its meaning on unseen data. The one supervised part, the
Regime Stability Score, is judged against baselines out of sample. Until now
these checks existed only as numbers in JSON files; this module draws them.

It only READS existing results (model_quality.json, bic/bic_summary.csv,
m6_evaluation_test.json) and the saved stability scores. It changes no data,
model or result. As a safety check, the ROC curve's area is recomputed from
the saved scores and must equal the AUC in m6_evaluation_test.json, which shows
the chart uses exactly the evaluated test days.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

OUT_DIR = config.RESULTS_DIR / "model_quality"
ASSET_COLOURS = {"BTCUSDT": "#2a78d6", "ETHUSDT": "#eb6834"}
TRAIN_COLOUR, TEST_COLOUR = "#898781", "#2a78d6"          # same as the Novelty 4 chart
REGIME_COLOURS = {"Bullish": "#1baf7a", "Bearish": "#e34948",   # same as the regime charts
                  "Sideways": "#2a78d6", "Volatile": "#eda100"}
# Red and green are close for colour-blind readers, so every regime also has its own shape.
REGIME_MARKERS = {"Bullish": "^", "Bearish": "v", "Sideways": "o", "Volatile": "D"}
SCORE_COLOURS = {"B0": "#c3c2b7", "B1": "#898781", "S1": "#2a78d6"}
SCORE_NAMES = {"B0": "Constant (training rate)", "B1": "HMM persistence only",
               "S1": "Stability Score (delivered)"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
NAMES = {"1d": "Daily", "4h": "4-hour", "1h": "1-hour", "15m": "15-minute", "5m": "5-minute"}


def _style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=10.5)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _save(fig, name: str) -> None:
    fig.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_quality() -> list[dict]:
    with open(config.RESULTS_DIR / "model_quality.json") as file:
        return json.load(file)


# --------------------------------------------------------------------------
# 1. Fit on training data against fit on unseen data
# --------------------------------------------------------------------------
def fit_chart(quality: list[dict]) -> pd.DataFrame:
    """Log-likelihood per candle: how probable the model finds the data. Higher
    is better. A model that has only memorised its training data would score
    much lower on the unseen test period."""
    table = pd.DataFrame([{"asset": q["symbol"], "timeframe": q["interval"],
                           "train": q["log_likelihood_per_candle_train"],
                           "test": q["log_likelihood_per_candle_test"],
                           "converged": q["converged"]} for q in quality])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for ax, symbol in zip(axes, config.SYMBOLS):
        rows = table[table["asset"] == symbol].set_index("timeframe").loc[config.INTERVALS]
        y = np.arange(len(rows))[::-1]
        for k, (interval, row) in enumerate(rows.iterrows()):
            ax.plot([row["train"], row["test"]], [y[k], y[k]], color=GRID, linewidth=2, zorder=1)
        ax.scatter(rows["train"], y, s=70, facecolor="white", edgecolor=TRAIN_COLOUR,
                   linewidth=2, zorder=2, label="training data (to 2022)")
        ax.scatter(rows["test"], y, s=70, color=TEST_COLOUR, zorder=3,
                   label="unseen test data (2023 - Sep 2026)")
        for k, (interval, row) in enumerate(rows.iterrows()):
            ax.annotate(f"{row['test'] - row['train']:+.2f}", (max(row["train"], row["test"]), y[k]),
                        xytext=(8, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=INK)
        ax.set_yticks(y, [NAMES[i] for i in rows.index])
        _style(ax, f"{symbol}", xlabel="Log-likelihood per candle (higher = better fit)")
    axes[0].set_xlim(right=axes[0].get_xlim()[1] + 0.45)   # room for the labels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper left", ncol=2,
               bbox_to_anchor=(0.07, 0.95), labelcolor=INK)
    fig.suptitle("How well each regime model fits data it never saw "
                 "(number = test minus training)", x=0.07, y=1.02, ha="left", color=INK,
                 fontsize=12)
    fig.subplots_adjust(top=0.84, wspace=0.28)
    _save(fig, "1_fit_train_vs_test.png")
    return table


# --------------------------------------------------------------------------
# 2. BIC by number of states
# --------------------------------------------------------------------------
def bic_chart() -> pd.DataFrame:
    """Small multiples, one per model, each with its own scale: the ten BIC
    ranges differ by a factor of 200, so one shared axis would flatten the
    Daily models into a line."""
    table = pd.read_csv(config.RESULTS_DIR / "bic" / "bic_summary.csv")
    states = config.HMM_STATE_CANDIDATES
    fig, axes = plt.subplots(2, 5, figsize=(15, 5.6))
    for r, symbol in enumerate(config.SYMBOLS):
        for c, interval in enumerate(config.INTERVALS):
            ax = axes[r, c]
            row = table[(table["asset"] == symbol) & (table["timeframe"] == interval)].iloc[0]
            values = np.array([row[f"{k} states"] for k in states]) / 1000
            ax.plot(states, values, color=ASSET_COLOURS[symbol], linewidth=2, marker="o",
                    markersize=5, zorder=2)
            chosen = values[states.index(config.HMM_N_STATES)]
            ax.scatter([config.HMM_N_STATES], [chosen], s=110, facecolor="white",
                       edgecolor=INK, linewidth=1.8, zorder=3)
            ax.set_xticks(states)
            _style(ax, f"{symbol[:3]} {NAMES[interval]}",
                   xlabel="states" if r == 1 else "", ylabel="BIC (thousands)" if c == 0 else "")
    fig.suptitle(f"BIC by number of states, full covariance (lower is better). Circled: the "
                 f"{config.HMM_N_STATES} states used. In all ten models BIC keeps falling to 6, "
                 "so four states is a design choice, one per named regime.",
                 x=0.05, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    _save(fig, "2_bic_by_states.png")
    return table


# --------------------------------------------------------------------------
# 3. Do the regimes keep their meaning on unseen data?
# --------------------------------------------------------------------------
def meaning_chart(quality: list[dict]) -> pd.DataFrame:
    rows = []
    for q in quality:
        for regime, profile in q["test_profile"].items():
            rows.append({"asset": q["symbol"], "timeframe": q["interval"], "regime": regime,
                         "rsi": profile["rsi"], "volatility_vs_norm": profile["volatility_vs_norm"],
                         "mean_return": profile["mean_return"]})
    table = pd.DataFrame(rows)
    order = [(s, i) for s in config.SYMBOLS for i in config.INTERVALS]
    y_of = {key: len(order) - 1 - k for k, key in enumerate(order)}
    passed = sum(q["checks_passed"] for q in quality)
    total = sum(len(q["checks"]) for q in quality)

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for regime in config.REGIMES:
        part = table[table["regime"] == regime]
        y = [y_of[(a, t)] for a, t in zip(part["asset"], part["timeframe"])]
        for ax, column in ((left, "rsi"), (right, "volatility_vs_norm")):
            ax.scatter(part[column], y, s=60, color=REGIME_COLOURS[regime],
                       marker=REGIME_MARKERS[regime], edgecolor="white", linewidth=0.8,
                       label=regime, zorder=3)
    left.axvline(50, color=MUTED, linewidth=1, zorder=1)
    right.axvline(1.0, color=MUTED, linewidth=1, zorder=1)
    left.set_yticks([y_of[key] for key in order], [f"{s[:3]} {NAMES[i]}" for s, i in order])
    for ax in (left, right):
        ax.axhline(len(config.INTERVALS) - 0.5, color=GRID, linewidth=1.2)
    _style(left, "Average RSI of each regime (50 = neutral)", xlabel="RSI")
    _style(right, "Volatility of each regime against its own past-year norm (1.0 = normal)",
           xlabel="times the yearly norm")
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9.5, loc="upper left", ncol=4,
               bbox_to_anchor=(0.07, 0.935), labelcolor=INK)
    fig.suptitle(f"On unseen test data (2023 - Sep 2026) the regimes keep their meaning: "
                 f"{passed} of {total} checks pass. Bullish has the highest RSI, Bearish the "
                 "lowest,\nand Volatile is more volatile than Sideways, in every model.",
                 x=0.07, y=1.0, ha="left", color=INK, fontsize=11)
    fig.subplots_adjust(top=0.84)
    _save(fig, "3_regime_meaning_test.png")
    return table


# --------------------------------------------------------------------------
# 4. The Regime Stability Score against its baselines
# --------------------------------------------------------------------------
def roc_points(target: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """False- and true-positive rates over every threshold, highest score first.
    Here the 'positive' outcome is: the Daily regime persisted for 3 days."""
    order = np.argsort(-score, kind="mergesort")
    target, score = target[order], score[order]
    distinct = np.r_[np.where(np.diff(score))[0], len(score) - 1]
    true_pos = np.cumsum(target)[distinct]
    false_pos = (distinct + 1) - true_pos
    return (np.r_[0, false_pos / (len(target) - target.sum())],
            np.r_[0, true_pos / target.sum()])


def stability_chart() -> dict:
    with open(config.RESULTS_DIR / "m6_evaluation_test.json") as file:
        evaluation = json.load(file)
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2),
                                      gridspec_kw={"width_ratios": [1, 1.25]})
    facts = {}
    for symbol in config.SYMBOLS:
        scores = pd.read_parquet(config.STABILITY_DIR / f"{symbol}_stability.parquet")
        test = scores[(scores["period"] == "test") & scores["target"].notna()]
        target, score = test["target"].to_numpy(), test["S1"].to_numpy()
        fpr, tpr = roc_points(target, score)
        area = float(np.trapezoid(tpr, fpr))
        reported = next(m["auc"] for m in evaluation[symbol]["metrics"] if m["score"] == "S1")
        if len(test) != evaluation[symbol]["days"] or abs(area - reported) > 1e-6:
            raise ValueError(f"{symbol}: chart data differ from the M6 evaluation")
        facts[symbol] = {"days": len(test), "auc": area}
        left.plot(fpr, tpr, color=ASSET_COLOURS[symbol], linewidth=2,
                  label=f"{symbol}: AUC {area:.3f}")
    left.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="no skill: AUC 0.5")
    left.set_aspect("equal")
    left.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
    _style(left, "ROC curve of the Stability Score, test period",
           xlabel="share of regime ENDINGS wrongly scored as stable",
           ylabel="share of regimes that HELD, scored as stable")

    width = 0.26
    x = np.arange(len(config.SYMBOLS))
    for k, name in enumerate(SCORE_NAMES):
        values = [next(m["brier"] for m in evaluation[s]["metrics"] if m["score"] == name)
                  for s in config.SYMBOLS]
        bars = right.bar(x + (k - 1) * width, values, width * 0.92, color=SCORE_COLOURS[name],
                         label=SCORE_NAMES[name], zorder=2)
        for bar, value in zip(bars, values):
            right.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value),
                           xytext=(0, 3), textcoords="offset points", ha="center",
                           fontsize=8.5, color=INK)
    right.set_xticks(x, config.SYMBOLS)
    top = max(m["brier"] for s in config.SYMBOLS for m in evaluation[s]["metrics"])
    for k, symbol in enumerate(config.SYMBOLS):
        d = evaluation[symbol]["differences"]["S1_minus_B0"]
        right.annotate(f"Score minus constant: {d['brier_difference']:+.3f}\n"
                       f"95% interval [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]",
                       (x[k], top * 1.13), ha="center", va="bottom", fontsize=8.5, color=INK)
    right.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=3,
                 bbox_to_anchor=(0.5, -0.09), labelcolor=INK)
    right.set_ylim(0, top * 1.32)
    _style(right, "Brier score, test period (lower is better)", ylabel="Brier score")
    fig.suptitle("The Regime Stability Score (will the Daily regime hold 3 more days?) on unseen "
                 f"data, {evaluation['BTCUSDT']['from']} to {evaluation['BTCUSDT']['to']}",
                 x=0.06, ha="left", color=INK, fontsize=12)
    _save(fig, "4_stability_score_test.png")
    return facts


def write_summary(fit: pd.DataFrame, quality: list[dict], roc: dict) -> None:
    passed = sum(q["checks_passed"] for q in quality)
    total = sum(len(q["checks"]) for q in quality)
    lines = ["# Model quality: charts for the report", "",
             "*Generated by `python -m src.figures.model_quality_charts` from existing result "
             "files. Nothing is retrained.*", "",
             "The regime models are unsupervised: no candle has a true regime label, so there "
             "is no accuracy. They are judged by fit on unseen data, the state-count check, and "
             "whether the regimes keep their meaning. The supervised Stability Score is judged "
             "against baselines.", "",
             "## 1. Fit on training vs unseen data (`1_fit_train_vs_test.png`)", "",
             "| Model | Converged | Training | Test | Test minus training |",
             "|---|---|---|---|---|"]
    for row in fit.itertuples():
        lines.append(f"| {row.asset} {row.timeframe} | {'yes' if row.converged else 'iteration limit*'} "
                     f"| {row.train:.2f} | {row.test:.2f} | {row.test - row.train:+.2f} |")
    lines += ["", "\\* Trained on to convergence in a copy, 0.11% of labels changed "
                  "(`results/convergence_check_BTCUSDT_5m.json`).", "",
              "The intraday models fit unseen data about as well as training data or better. "
              "The two Daily models fit unseen data worse: a limitation to report.", "",
              "## 2. Number of states (`2_bic_by_states.png`)", "",
              "BIC is lowest at six states in all ten models, so four states is a design "
              "choice (one per named regime). Numbers: `results/bic/bic_summary.csv`.", "",
              "## 3. Meaning on unseen data (`3_regime_meaning_test.png`)", "",
              f"{passed} of {total} checks pass on the test period.", "",
              "## 4. Stability Score (`4_stability_score_test.png`)", ""]
    for symbol, info in roc.items():
        lines.append(f"- {symbol}: AUC {info['auc']:.3f} over {info['days']} test days "
                     "(recomputed from the saved scores; equals `m6_evaluation_test.json`).")
    lines.append("")
    (OUT_DIR / "MODEL_QUALITY_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    quality = load_quality()
    fit = fit_chart(quality)
    bic_chart()
    meaning_chart(quality)
    roc = stability_chart()
    write_summary(fit, quality, roc)
    print(fit.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for symbol, info in roc.items():
        print(f"{symbol}: ROC area {info['auc']:.4f} over {info['days']} days "
              "(matches m6_evaluation_test.json)")
    print(f"Saved 4 charts and MODEL_QUALITY_SUMMARY.md to {OUT_DIR}")


if __name__ == "__main__":
    main()
