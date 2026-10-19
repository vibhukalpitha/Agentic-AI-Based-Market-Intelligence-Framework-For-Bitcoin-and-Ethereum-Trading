"""What the null results rule out: bounds from the intervals already computed.

Run:  python -m src.experiments.null_bounds     (reads results/, writes 3 new files)

WHY. "Not supported" can mean two very different things:
  - the effect was looked for with enough data and is ABSENT (an informative
    null: even a small effect lies outside the 95% interval), or
  - there was not enough data to tell (INCONCLUSIVE: a small effect still
    lies inside the interval).
Reporting which one applies makes a null result defensible. This is the
confidence-interval approach to equivalence testing: a smallest effect of
interest is fixed, and the interval is compared with it.

What this module is NOT: it runs no new test, recomputes no statistic and
changes no result. It only re-reads the 95% intervals in h1_results.json,
h2_placebo_test.json, h2b_results_test.json and n3_adaptive_weights_test.json
(all bootstrap percentile intervals, 2.5% to 97.5%). It was defined AFTER
those results were known and must be reported as a re-reading, not as a new
pre-registered test.

Thresholds, fixed from outside the results:
  - Effects of the form P(...) (H1, H2, H2b; 0.5 = none): the smallest effect
    of interest is Vargha & Delaney's (2000) conventional "small" effect,
    0.56, i.e. 0.06 above no effect.
  - Novelty 3 (Brier score differences, which have no conventional scale): the
    yardstick is the project's own gain from adding fixed-weight cross-timeframe
    agreement at all (S1 minus F in n3_adaptive_weights_test.json). The question
    is whether adaptive weighting could add as much again.
A 95% two-sided interval is stricter than the 90% interval usual in
equivalence tests, so "ruled out" here is conservative.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

SMALL_EFFECT = 0.06            # 0.56 - 0.5: Vargha & Delaney's "small" on the P(...) scale
STATUS_COLOURS = {"ruled out": "#2a78d6", "opposite direction": "#2a78d6",
                  "inconclusive": "#898781", "supported": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
SHORT = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}


def classify(low: float, high: float, small: float) -> str:
    """Interval [low, high] of the effect in the PREDICTED direction (0 = none).

    supported           the whole interval lies above 0
    opposite direction  the whole interval lies below 0: every effect in the
                        predicted direction is excluded
    ruled out           a small effect (or larger) is excluded: high < small
    inconclusive        a small effect is still possible
    """
    if low > 0:
        return "supported"
    if high < 0:
        return "opposite direction"
    if high < small:
        return "ruled out"
    return "inconclusive"


def _load(name: str) -> dict:
    with open(config.RESULTS_DIR / name) as file:
        return json.load(file)


def h1_rows() -> list[dict]:
    """Novelty 1. Effect = P(alignment lower before a transition); 0.5 = none."""
    h1 = _load("h1_results.json")
    wanted = {"level_standard": "H1 level (headline)", "slope_standard": "H1 slope",
              "level_no_confidence": "H1 level, no confidence",
              "level_direction_only": "H1 level, direction only"}
    rows = []
    for symbol in config.SYMBOLS:
        for row in h1[symbol]["headline_test"] + h1[symbol]["variants_test"]:
            if row["measure"] not in wanted:
                continue
            low, high = row["ci_low"] - 0.5, row["ci_high"] - 0.5
            rows.append({"novelty": "1", "test": wanted[row["measure"]], "asset": symbol,
                         "events": row["transitions_matched"],
                         "estimate": row["effect_auc"] - 0.5, "low": low, "high": high,
                         "yardstick": SMALL_EFFECT, "status": classify(low, high, SMALL_EFFECT)})
    return rows


def placebo_rows(file: str, label: str, levels: list[str]) -> list[dict]:
    """Novelty 2. Effect = real minus placebo on the P(...) scale; 0 = none."""
    results = _load(file)
    rows = []
    for symbol in config.SYMBOLS:
        for interval in levels:
            r = results[symbol][interval]
            low, high = r["difference_ci_low"], r["difference_ci_high"]
            rows.append({"novelty": "2", "test": f"{label} {interval}", "asset": symbol,
                         "events": r.get("flips", r.get("survivors")),
                         "estimate": r["real_minus_placebo"], "low": low, "high": high,
                         "yardstick": SMALL_EFFECT, "status": classify(low, high, SMALL_EFFECT)})
    return rows


def n3_rows() -> list[dict]:
    """Novelty 3. Effect = Brier IMPROVEMENT of the first model over the second
    (the stored differences are first minus second, so the sign is flipped)."""
    n3 = _load("n3_adaptive_weights_test.json")
    names = {"A_minus_F": "Adaptive vs fixed (primary)", "A_minus_L": "Adaptive vs learned (primary)",
             "L_minus_F": "Learned vs fixed"}
    rows = []
    for symbol in config.SYMBOLS:
        d = n3[symbol]["differences"]
        yardstick = -d["F_minus_S1"]["brier_difference"]   # gain from cross-timeframe agreement
        fixed = next(m["brier"] for m in n3[symbol]["metrics"] if m["score"] == "F")
        for key, label in names.items():
            low, high = -d[key]["ci_high"], -d[key]["ci_low"]
            rows.append({"novelty": "3", "test": label, "asset": symbol, "events": n3[symbol]["days"],
                         "estimate": -d[key]["brier_difference"], "low": low, "high": high,
                         "yardstick": yardstick, "largest_gain_pct_of_fixed": 100 * high / fixed,
                         "status": classify(low, high, yardstick)})
    return rows


def plot(table: pd.DataFrame) -> None:
    probability = table[table["novelty"] != "3"].reset_index(drop=True)
    brier = table[table["novelty"] == "3"].reset_index(drop=True)
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(14, 8.5), gridspec_kw={"width_ratios": [1.15, 1]})

    for ax, part, xlabel in (
            (left, probability, "effect in the predicted direction (0 = none; P(...) scale)"),
            (right, brier, "Brier improvement over the comparison model (0 = none)")):
        y = np.arange(len(part))[::-1]
        for k, row in part.iterrows():
            colour = STATUS_COLOURS[row["status"]]
            ax.plot([row["low"], row["high"]], [y[k], y[k]], color=colour, linewidth=2.5,
                    solid_capstyle="round", zorder=2)
            ax.scatter(row["estimate"], y[k], s=40, color=colour, zorder=3)
            ax.annotate(row["status"], (row["high"], y[k]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK)
        ax.axvline(0, color=INK, linewidth=1, zorder=1)
        ax.set_yticks(y, [f"N{r.novelty}  {r.test}, {SHORT[r.asset]}" for r in part.itertuples()],
                      fontsize=8.5)
        ax.set_xlabel(xlabel, color=MUTED)
        ax.grid(True, axis="x", color=GRID, linewidth=0.6)
        ax.tick_params(colors=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)

    left.set_xlim(probability["low"].min() - 0.03, probability["high"].max() + 0.17)
    left.axvspan(SMALL_EFFECT, left.get_xlim()[1], color="#f0efec", zorder=0)
    left.axvline(SMALL_EFFECT, color=MUTED, linestyle="--", linewidth=1)
    left.set_title("Novelties 1 and 2: 95% intervals. Shaded: a 'small' effect or larger "
                   "(Vargha & Delaney)", loc="left", color=INK, fontsize=10)
    for k, row in brier.iterrows():
        yy = len(brier) - 1 - k
        right.scatter(row["yardstick"], yy, marker="|", s=220, color=MUTED, zorder=4)
    right.set_xlim(right=right.get_xlim()[1] + 0.002)
    right.xaxis.set_major_locator(plt.MaxNLocator(5))
    right.set_title("Novelty 3: 95% intervals. Grey tick: the gain from adding "
                    "cross-timeframe agreement at all", loc="left", color=INK, fontsize=10)
    fig.suptitle("What the null results rule out (a re-reading of intervals already computed; "
                 "no new test). Blue = ruled out, grey = inconclusive, green = supported",
                 x=0.02, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "null_bounds.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def write_markdown(table: pd.DataFrame) -> None:
    def fmt(row, digits):
        return (f"{row['estimate']:+.{digits}f} [{row['low']:+.{digits}f}, {row['high']:+.{digits}f}]")

    lines = ["# What the null results rule out", "",
             "*Generated by `python -m src.experiments.null_bounds`. A re-reading of the 95% "
             "intervals already in the result files: no new test was run and no result changed. "
             "Defined after the results were known; report it as such.*", "",
             "**Rule.** Each interval is turned into the effect in the direction the hypothesis "
             "predicted (0 = none). *Supported*: the whole interval is above 0. *Opposite "
             "direction*: the whole interval is below 0. *Ruled out*: even a small effect lies "
             "above the interval. *Inconclusive*: a small effect is still inside it.", "",
             f"**Small effect.** Novelties 1 and 2: {0.5 + SMALL_EFFECT:.2f} on the P(...) scale "
             f"(+{SMALL_EFFECT:.2f}), Vargha & Delaney's conventional 'small'. Novelty 3: the "
             "Brier gain the project measured from adding fixed-weight cross-timeframe agreement "
             "at all.", ""]
    for novelty, title, digits in (("1", "Novelty 1 (H1)", 3), ("2", "Novelty 2 (H2 at Daily and 4h, "
                                    "H2b at 4h and 1h; real minus placebo)", 3),
                                   ("3", "Novelty 3 (Brier improvement)", 4)):
        part = table[table["novelty"] == novelty]
        lines += [f"## {title}", "", "| Test | Asset | Events | Effect [95%] | Small effect | Status |",
                  "|---|---|---|---|---|---|"]
        for _, row in part.iterrows():
            lines.append(f"| {row['test']} | {row['asset']} | {row['events']} | {fmt(row, digits)} "
                         f"| {row['yardstick']:+.{digits}f} | **{row['status']}** |")
        lines.append("")
    (config.RESULTS_DIR / "NULL_RESULTS_BOUNDS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = h1_rows()
    rows += placebo_rows("h2_placebo_test.json", "H2 Daily/4h", ["1d", "4h"])
    rows += placebo_rows("h2b_results_test.json", "H2b", ["4h", "1h"])
    rows += n3_rows()
    table = pd.DataFrame(rows)
    table.to_json(config.RESULTS_DIR / "null_bounds.json", orient="records", indent=2)
    plot(table)
    write_markdown(table)
    print(table[["novelty", "test", "asset", "estimate", "low", "high", "yardstick", "status"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\nSaved: results/null_bounds.json, results/null_bounds.png, results/NULL_RESULTS_BOUNDS.md")


if __name__ == "__main__":
    main()
