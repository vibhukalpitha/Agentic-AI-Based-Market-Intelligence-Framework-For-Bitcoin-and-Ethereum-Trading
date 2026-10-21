"""M9 - The H1 experiment.

Run:  python -m src.experiments.h1_precursor --counts-only   (sample sizes only)
      python -m src.experiments.h1_precursor                 (the full test)

H1: cross-timeframe alignment is lower, and falling, in the days before a
Daily regime transition than in comparable stable periods.

The design was fixed before any result was computed (see src/config.py):

  transition    a Daily regime change whose new regime holds H1_PERSISTENCE days.
                The "transition day" is the first day of the new regime.
  window        the H1_WINDOW_DAYS whole UTC days BEFORE the transition day.
                Nothing from the transition day itself is used.
  alignment     computed WITHOUT the Daily timeframe (Daily defines the
                transitions, so including it would be circular).
  comparison    stable days: no transition within H1_STABLE_MARGIN_DAYS either
                side. Each transition is compared only with stable days in the
                same Daily regime and on the same day of the week.
  fairness      headline transitions also need H1_STABLE_MARGIN_DAYS quiet
                days before them, so both kinds of window start from a settled
                market. All transitions are reported as a second line.
  headline      alignment LEVEL, test period (2023 onward), BTC and ETH separately.
  baselines     Daily volatility alone; number of 4-hour regime changes in the
                window. Alignment must tell us something these do not.

Every result here can come out against H1. That is a finding, not a failure.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src import config
from src.experiments.stats import (bootstrap_interval, logistic_coefficients,
                                   matched_auc, permutation_p_value)
from src.fusion import alignment
from src.regime import hmm_model

ONE_DAY = pd.Timedelta(days=1)
ALIGNMENT_VARIANTS = ["standard", "no_confidence", "direction_only"]
POINTS_PER_DAY = int(ONE_DAY / config.INTERVAL_STEP["5m"])

EVENT_COLOUR, CONTROL_COLOUR, MUTED, GRID, INK = "#2a78d6", "#898781", "#898781", "#e1e0d9", "#0b0b0b"


# --------------------------------------------------------------------------
# Daily inputs
# --------------------------------------------------------------------------
def daily_alignment(symbol: str) -> pd.DataFrame:
    """Mean alignment per UTC day (Daily timeframe excluded), in three variants.

    standard        the M4 score as saved
    no_confidence   every timeframe's confidence set to 1 (plain weighted vote)
    direction_only  regimes reduced to up / down / neutral before comparing
    """
    intervals = config.H1_ALIGNMENT_INTERVALS
    grid = pd.read_parquet(config.FUSION_DIR / f"{symbol}_alignment_ex1d.parquet")
    series = {"standard": grid["alignment"].to_numpy()}

    flat = grid.copy()
    for interval in intervals:
        flat[f"conf_{interval}"] = 1.0
    series["no_confidence"] = alignment.compute_alignment(
        flat, intervals, config.TIMEFRAME_WEIGHTS)["alignment"].to_numpy()

    direction = grid.copy()
    for interval in intervals:
        direction[f"regime_{interval}"] = direction[f"regime_{interval}"].map(
            config.DIRECTION_OF_REGIME)
    series["direction_only"] = alignment.compute_alignment(
        direction, intervals, config.TIMEFRAME_WEIGHTS,
        classes=sorted(set(config.DIRECTION_OF_REGIME.values())))["alignment"].to_numpy()

    # A 5-minute candle closing at 23:59:59.999 belongs to that day.
    by_day = pd.DataFrame(series).groupby(grid["time"].dt.floor("D"))
    daily = by_day.mean()
    daily["coverage"] = by_day.size() / POINTS_PER_DAY
    return daily


def four_hour_changes(symbol: str) -> pd.Series:
    """Number of 4-hour regime changes completed on each UTC day (naive baseline)."""
    regimes = pd.read_parquet(config.REGIMES_DIR / f"{symbol}_4h_regimes.parquet",
                              columns=["close_time", "regime"])
    changed = regimes["regime"] != regimes["regime"].shift()
    changed.iloc[0] = False
    return changed.groupby(regimes["close_time"].dt.floor("D")).sum()


def daily_table(symbol: str) -> pd.DataFrame:
    """One row per day: Daily regime, the Daily model's confidence in it, and
    Daily volatility (atr_rel), indexed by date."""
    regimes = pd.read_parquet(config.REGIMES_DIR / f"{symbol}_1d_regimes.parquet",
                              columns=["open_time", "regime", "confidence"])
    features = pd.read_parquet(config.FEATURES_DIR / f"{symbol}_1d_features.parquet",
                               columns=["open_time", "atr_rel"])
    table = regimes.merge(features, on="open_time").set_index("open_time")
    if not (table.index.to_series().diff().dropna() == ONE_DAY).all():
        raise ValueError("Daily series has a gap; day arithmetic would be wrong")
    return table


# --------------------------------------------------------------------------
# Transitions and windows
# --------------------------------------------------------------------------
def transition_starts(regimes: np.ndarray, persistence: int) -> tuple[np.ndarray, np.ndarray]:
    """(is_start, settled) for a Daily regime series.

    settled[d]  the regime in force on day d once flicker shorter than
                `persistence` days is ignored (see hmm_model.confirmed_states).
    is_start[d] True on the FIRST day of a new regime that went on to hold for
                `persistence` days. confirmed_states marks the change on the
                day it becomes certain, which is persistence - 1 days later, so
                the start is found by stepping back that many days.

    Knowing that a regime "went on to hold" uses later days. That is allowed
    here because it defines the OUTCOME being studied; the predictors (the
    alignment window) never use anything after the window ends.
    """
    settled = hmm_model.confirmed_states(regimes, persistence)
    confirmed_on = np.flatnonzero(settled[1:] != settled[:-1]) + 1
    is_start = np.zeros(len(regimes), dtype=bool)
    is_start[confirmed_on - (persistence - 1)] = True
    return is_start, settled


def build_windows(symbol: str, persistence: int) -> pd.DataFrame:
    """One row per usable target day: a transition day or a stable day.

    For target day D the window is days D-3, D-2, D-1 (with H1_WINDOW_DAYS = 3).
    """
    table = daily_table(symbol)
    align = daily_alignment(symbol).reindex(table.index)
    changes = four_hour_changes(symbol).reindex(table.index)
    is_start, settled = transition_starts(table["regime"].to_numpy(), persistence)

    margin, width = config.H1_STABLE_MARGIN_DAYS, config.H1_WINDOW_DAYS
    covered = (align["coverage"] >= config.H1_MIN_DAY_COVERAGE).to_numpy()
    test_start = pd.Timestamp(config.TRAIN_END) + ONE_DAY
    dates = table.index

    rows = []
    for d in range(margin, len(table) - margin):
        window = slice(d - width, d)
        if not covered[window].all():
            continue                                   # alignment missing in the window
        quiet_before = not is_start[d - margin:d].any()
        quiet_after = not is_start[d + 1:d + margin + 1].any()
        if is_start[d]:
            kind = "transition"
        elif quiet_before and quiet_after:
            kind = "stable"
        else:
            continue                                   # too close to a transition to be either

        # The whole window must lie inside one period, never across the split.
        if dates[d - 1] <= pd.Timestamp(config.TRAIN_END):
            period = "train"
        elif dates[d - width] >= test_start:
            period = "test"
        else:
            continue

        row = {"date": dates[d], "kind": kind, "quiet_before": quiet_before, "period": period,
               "regime": settled[d - 1],               # regime in force during the window
               "weekday": dates[d].dayofweek,
               "volatility": table["atr_rel"].iloc[d - 1],
               "daily_confidence": table["confidence"].iloc[d - 1],
               "changes_4h": changes.iloc[window].sum()}
        for variant in ALIGNMENT_VARIANTS:
            values = align[variant].iloc[window].to_numpy()
            row[f"level_{variant}"] = values.mean()
            row[f"slope_{variant}"] = values[-1] - values[0]   # last day minus first day
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def _strata(frame: pd.DataFrame) -> np.ndarray:
    """Matching group of each row: same Daily regime AND same day of week."""
    return (frame["regime"] + "_" + frame["weekday"].astype(str)).to_numpy()


def compare_measure(windows: pd.DataFrame, column: str, lower_supports_h1: bool,
                 rng: np.random.Generator) -> dict:
    """Run the matched comparison for one measure and return every number reported."""
    is_event = (windows["kind"] == "transition").to_numpy()
    strata = _strata(windows)
    raw = windows[column].to_numpy(dtype=float)
    values = raw if lower_supports_h1 else -raw       # so "above 0.5" always supports H1

    auc, n_events, n_controls = matched_auc(values, is_event, strata)
    low, high = bootstrap_interval(values, is_event, strata, rng)
    # Plain, unmatched Mann-Whitney U on everything, for comparison.
    pooled = mannwhitneyu(values[is_event], values[~is_event], alternative="less") \
        if is_event.any() and (~is_event).any() else None
    return {
        "measure": column,
        "transitions_total": int(is_event.sum()),
        "stable_total": int((~is_event).sum()),
        "transitions_matched": n_events,
        "stable_matched": n_controls,
        "median_transition": float(np.median(raw[is_event])) if is_event.any() else np.nan,
        "median_stable": float(np.median(raw[~is_event])) if (~is_event).any() else np.nan,
        "effect_auc": float(auc),
        "ci_low": float(low), "ci_high": float(high),
        "p_matched": permutation_p_value(values, is_event, strata, rng),
        "p_unmatched_mwu": float(pooled.pvalue) if pooled else np.nan,
    }


def beyond_baselines(windows: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Does alignment still matter once the two baselines are in the model?

    A logistic model predicts "transition window or stable window" from
    volatility, 4-hour changes and the alignment level, all standardised. If
    the alignment coefficient's 95% interval lies below zero, lower alignment
    goes with transitions even among windows with the same volatility and the
    same amount of 4-hour change.
    """
    columns = ["volatility", "changes_4h", "level_standard"]
    y = (windows["kind"] == "transition").to_numpy(dtype=float)
    if y.sum() < 10 or (1 - y).sum() < 10:
        return {"note": "too few windows for a joint model"}
    x = windows[columns].to_numpy(dtype=float)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    fitted = logistic_coefficients(x, y)[1:]

    events, controls = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = np.empty((config.H1_BOOTSTRAPS // 4, len(columns)))
    for b in range(len(draws)):
        pick = np.concatenate([rng.choice(events, len(events)), rng.choice(controls, len(controls))])
        draws[b] = logistic_coefficients(x[pick], y[pick])[1:]
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    return {name: {"coefficient": float(fitted[k]), "ci_low": float(low[k]), "ci_high": float(high[k])}
            for k, name in enumerate(columns)}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def counts_report(symbol: str) -> None:
    """Sample sizes only - no alignment value is shown."""
    print(f"\n=== {symbol}: sample sizes ===")
    rows = []
    for persistence in [config.H1_PERSISTENCE] + config.H1_ROBUSTNESS_PERSISTENCE:
        windows = build_windows(symbol, persistence)
        for period in ("train", "test"):
            part = windows[windows["period"] == period]
            events = part[part["kind"] == "transition"]
            headline = part[(part["kind"] == "stable") | part["quiet_before"]]
            _, matched_events, matched_stable = matched_auc(
                np.zeros(len(headline)), (headline["kind"] == "transition").to_numpy(),
                _strata(headline))
            rows.append({"persistence": persistence, "period": period,
                         "transitions": len(events),
                         "with_quiet_week_before": int(events["quiet_before"].sum()),
                         "stable_days": int((part["kind"] == "stable").sum()),
                         "matched_transitions": matched_events,
                         "matched_stable_days": matched_stable})
    print(pd.DataFrame(rows).to_string(index=False))


def analyse(symbol: str, rng: np.random.Generator) -> dict:
    """All H1 numbers for one asset."""
    results = {}

    def run(windows: pd.DataFrame, label: str, columns: list[tuple[str, bool]]) -> None:
        results[label] = [compare_measure(windows, column, lower, rng) for column, lower in columns]

    windows = build_windows(symbol, config.H1_PERSISTENCE)
    windows.to_csv(config.RESULTS_DIR / f"h1_windows_{symbol}.csv", index=False)
    headline_rows = (windows["kind"] == "stable") | windows["quiet_before"]
    test, train = windows["period"] == "test", windows["period"] == "train"

    main_measures = [("level_standard", True), ("slope_standard", True),
                     ("volatility", False), ("changes_4h", False)]
    run(windows[test & headline_rows], "headline_test", main_measures)
    run(windows[test], "all_transitions_test", [("level_standard", True), ("slope_standard", True)])
    run(windows[train & headline_rows], "in_sample_train", [("level_standard", True), ("slope_standard", True)])
    run(windows[test & headline_rows], "variants_test",
        [("level_no_confidence", True), ("level_direction_only", True)])
    for persistence in config.H1_ROBUSTNESS_PERSISTENCE:
        other = build_windows(symbol, persistence)
        keep = (other["period"] == "test") & ((other["kind"] == "stable") | other["quiet_before"])
        run(other[keep], f"persistence_{persistence}_test", [("level_standard", True)])
    results["beyond_baselines_test"] = beyond_baselines(windows[test & headline_rows], rng)

    # Positive control: proves the test can detect a signal when one exists.
    # The Daily model's own confidence on the day before a transition is lower
    # almost by construction. It is NOT evidence for H1 (it uses Daily itself).
    # A separate random stream keeps every H1 number above unchanged.
    control_rng = np.random.default_rng(config.RANDOM_SEED + 1)
    results["positive_control_test"] = [
        compare_measure(windows[test & headline_rows], "daily_confidence", True, control_rng)]

    # Exploratory only (not part of the agreed design): the headline split by
    # the Daily regime in force during the window. Small groups; no p-values.
    by_regime = []
    head = windows[test & headline_rows]
    for regime in config.REGIMES:
        part = head[head["regime"] == regime]
        is_event = (part["kind"] == "transition").to_numpy()
        auc, n_events, n_controls = matched_auc(
            part["level_standard"].to_numpy(dtype=float), is_event, _strata(part))
        by_regime.append({
            "regime": regime, "transitions_matched": n_events, "stable_matched": n_controls,
            "median_transition": float(part.loc[is_event, "level_standard"].median()),
            "median_stable": float(part.loc[~is_event, "level_standard"].median()),
            "effect_auc": float(auc)})
    results["exploratory_by_regime_test"] = by_regime
    return results


def print_results(symbol: str, results: dict) -> None:
    print(f"\n{'=' * 78}\n{symbol}\n{'=' * 78}")
    for label, rows in results.items():
        if label == "beyond_baselines_test":
            continue
        if label == "exploratory_by_regime_test":
            print(f"\n[{label}]  alignment level by Daily regime in force; "
                  "small groups, descriptive only")
            print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
            continue
        table = pd.DataFrame(rows)[["measure", "transitions_matched", "stable_matched",
                                    "median_transition", "median_stable", "effect_auc",
                                    "ci_low", "ci_high", "p_matched", "p_unmatched_mwu"]]
        print(f"\n[{label}]  effect_auc above 0.5 supports H1")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n[beyond_baselines_test]  standardised logistic coefficients; "
          "a negative level_standard supports H1")
    print(pd.DataFrame(results["beyond_baselines_test"]).T.to_string(
        float_format=lambda v: f"{v:.3f}"))


def plot_event_study(path) -> None:
    """Mean daily alignment on each day around a transition, test period,
    against the average over stable days. Day 0 is the transition day."""
    offsets = np.arange(-7, 4)
    fig, axes = plt.subplots(1, len(config.SYMBOLS), figsize=(13, 4.6), sharey=True)
    for ax, symbol in zip(axes, config.SYMBOLS):
        windows = build_windows(symbol, config.H1_PERSISTENCE)
        test = windows[windows["period"] == "test"]
        daily = daily_alignment(symbol)["standard"]
        events = test[(test["kind"] == "transition") & test["quiet_before"]]["date"]
        stable = test[test["kind"] == "stable"]["date"]
        curve = [daily.reindex(events + k * ONE_DAY).mean() for k in offsets]
        stable_level = daily.reindex(stable - ONE_DAY).mean()

        ax.axvspan(-config.H1_WINDOW_DAYS - 0.5, -0.5, color=GRID, alpha=0.6, linewidth=0)
        ax.axhline(stable_level, color=CONTROL_COLOUR, linewidth=2, linestyle="--")
        ax.plot(offsets, curve, color=EVENT_COLOUR, linewidth=2, marker="o", markersize=6)
        ax.annotate("stable days", xy=(offsets[-1], stable_level), xytext=(0, 6),
                    textcoords="offset points", ha="right", color=MUTED, fontsize=9)
        ax.set_title(f"{symbol}: mean alignment around {len(events)} transitions "
                     "(test period, 2023 onward)", loc="left", color=INK, fontsize=11)
        ax.set_xlabel("Days relative to the transition day (shaded = the 3-day window tested)",
                      color=MUTED)
        ax.set_xticks(offsets)
        ax.grid(True, axis="y", color=GRID, linewidth=0.6)
        ax.tick_params(colors=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    axes[0].set_ylabel("Mean Alignment Score (Daily excluded)", color=MUTED)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--counts-only", action="store_true",
                        help="print sample sizes and stop, without computing any result")
    args = parser.parse_args()

    if args.counts_only:
        for symbol in config.SYMBOLS:
            counts_report(symbol)
        return

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.RANDOM_SEED)
    everything = {"design": {
        "persistence": config.H1_PERSISTENCE, "window_days": config.H1_WINDOW_DAYS,
        "stable_margin_days": config.H1_STABLE_MARGIN_DAYS,
        "alignment_intervals": config.H1_ALIGNMENT_INTERVALS,
        "train_end": config.TRAIN_END, "permutations": config.H1_PERMUTATIONS,
        "bootstraps": config.H1_BOOTSTRAPS, "seed": config.RANDOM_SEED}}
    for symbol in config.SYMBOLS:
        everything[symbol] = analyse(symbol, rng)
        print_results(symbol, everything[symbol])
    with open(config.RESULTS_DIR / "h1_results.json", "w") as file:
        json.dump(everything, file, indent=2)
    plot_event_study(config.RESULTS_DIR / "h1_event_study.png")
    print("\nSaved: results/h1_results.json, results/h1_windows_<symbol>.csv, "
          "results/h1_event_study.png")


if __name__ == "__main__":
    main()
