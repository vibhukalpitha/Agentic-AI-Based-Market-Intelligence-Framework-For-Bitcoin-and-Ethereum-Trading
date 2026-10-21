"""Historical similarity (M7) shown in charts, with a similarity percentile.

Run:  python -m src.figures.similarity_charts                    (last Daily candle in the data)
      python -m src.figures.similarity_charts --date 2025-03-01  (any other Daily candle)
Writes results/similarity/ (in the live workspace: the workspace's results/,
which src/live/update.py copies to data/live/output/similarity/).

WHAT IT SHOWS, for each asset:
  1. where in history the three stretches most like the last 10 days lie
  2. the four key features over the 10 days: today's stretch beside each match
  3. how close the matches are compared with EVERY past stretch
  4. how dynamic time warping paired the days of today's stretch and the best match

THE SIMILARITY PERCENTILE. A DTW distance has no natural "100 %". The honest
summary is a rank: the share of all earlier 10-day stretches that are FURTHER
from today than this match ("closer than 99.6 % of 2,800 past stretches").
It is computed from the same distances as the matches; nothing is tuned.

It reuses src/similarity/dtw_match.py unchanged, and checks that its matches
are exactly those in the output record's similar_history block.

SCOPE - SUPPORTING EVIDENCE ONLY. The charts show WHEN similar stretches
occurred and what the regime was then. They do not show or assess what
happened AFTER any stretch; that is another component's scope. The price line
in chart 1 is context for locating the stretches, nothing more.

It only reads data and models and writes new chart files.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.similarity import dtw_match

OUT_DIR = config.RESULTS_DIR / "similarity"
TODAY_COLOUR = "#2a78d6"
MATCH_COLOURS = ["#eb6834", "#6250d6", "#e87ba4"]      # validated with today's blue
MATCH_MARKERS = ["s", "^", "D"]
INK, MUTED, GRID, SHADE = "#0b0b0b", "#898781", "#e1e0d9", "#f0efec"
SHOWN_FEATURES = {"rsi": "RSI (0-100)",
                  "sma_dist": "Distance from the 50-day average (trend)",
                  "macd_hist_norm": "MACD histogram / price (momentum)",
                  "atr_rel": "Volatility vs its past-year norm (log)"}


# ---------------------------------------------------------------- calculations
def all_distances(x: np.ndarray, end: int, window: int) -> tuple[np.ndarray, np.ndarray]:
    """DTW distance from the stretch ending at row `end` to EVERY allowed earlier
    stretch (ending before it begins), exactly as dtw_match.find_matches does."""
    start = end - window + 1
    candidate_ends = np.arange(window - 1, start)
    candidates = np.stack([x[e - window + 1:e + 1] for e in candidate_ends])
    return candidate_ends, dtw_match.dtw_distances(x[start:end + 1], candidates)


def percentile(distance: float, distances: np.ndarray) -> float:
    """Share (%) of all past stretches that are FURTHER away than this one."""
    return float(100 * np.mean(distances > distance))


def dtw_path(a: np.ndarray, b: np.ndarray) -> tuple[list[tuple[int, int]], float]:
    """The day pairing DTW chooses between two stretches, and its distance.
    Same recurrence as dtw_match.dtw_distances, with the choices traced back."""
    days = len(a)
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    total = np.full((days, days), np.inf)
    total[0, 0] = cost[0, 0]
    for i in range(days):
        for j in range(days):
            if i or j:
                total[i, j] = cost[i, j] + min(
                    total[i - 1, j - 1] if i and j else np.inf,
                    total[i - 1, j] if i else np.inf,
                    total[i, j - 1] if j else np.inf)
    path, i, j = [(days - 1, days - 1)], days - 1, days - 1
    while i or j:
        options = []
        if i and j:
            options.append((total[i - 1, j - 1], i - 1, j - 1))
        if i:
            options.append((total[i - 1, j], i - 1, j))
        if j:
            options.append((total[i, j - 1], i, j - 1))
        _, i, j = min(options)
        path.append((i, j))
    return path[::-1], float(total[-1, -1] / days)


def analyse(symbol: str, date: pd.Timestamp | None) -> dict:
    regimes, x = dtw_match.load_daily(symbol)
    features = pd.read_parquet(config.FEATURES_DIR / f"{symbol}_1d_features.parquet")
    days = regimes["open_time"]
    date = days.iloc[-1] if date is None else pd.Timestamp(date)
    end = int(np.flatnonzero(days.to_numpy() == np.datetime64(date))[0])
    window = config.SIMILARITY_WINDOW_DAYS

    candidate_ends, distances = all_distances(x, end, window)
    found = dtw_match.find_matches(x, end, window, config.SIMILARITY_MATCHES)
    record_block = dtw_match.similar_history(regimes, x, date)
    if [m["window_end"] for m in record_block["matches"]] != \
            [str(days.iloc[e].date()) for e, _ in found]:
        raise ValueError("chart matches differ from the output record's similar_history")

    matches = []
    for e, distance in found:
        matches.append({"window_start": str(days.iloc[e - window + 1].date()),
                        "window_end": str(days.iloc[e].date()),
                        "distance": round(distance, 3),
                        "similarity_percentile": round(percentile(distance, distances), 2),
                        "regime_then": regimes["regime"].iloc[e], "end_row": e})
    return {"symbol": symbol, "date": date, "end_row": end, "window": window,
            "regime_now": regimes["regime"].iloc[end], "regimes": regimes,
            "features": features, "x": x, "distances": distances, "matches": matches}


# ---------------------------------------------------------------- charts
def _style(ax, title="", xlabel="", ylabel=""):
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


def _label(m: dict, k: int) -> str:
    return (f"Match {k + 1}: {m['window_start']} to {m['window_end']}  ·  closer than "
            f"{m['similarity_percentile']:.2f}% of past stretches  ·  regime then {m['regime_then']}")


def chart_where(a: dict, path) -> None:
    f, w, days = a["features"], a["window"], a["regimes"]["open_time"]
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.plot(f["open_time"], f["close"], color=MUTED, linewidth=1, zorder=1)
    ax.set_yscale("log")
    spans = [(a["end_row"], TODAY_COLOUR, f"Today's stretch: {days.iloc[a['end_row'] - w + 1].date()} "
              f"to {a['date'].date()}  ·  regime now {a['regime_now']}")]
    spans += [(m["end_row"], MATCH_COLOURS[k], _label(m, k)) for k, m in enumerate(a["matches"])]
    for end, colour, label in spans:
        ax.axvspan(days.iloc[end - w + 1], days.iloc[end] + pd.Timedelta(days=1),
                   color=colour, alpha=0.35, label=label, zorder=2)
        ax.axvline(days.iloc[end], color=colour, linewidth=1.2, zorder=2)
    _style(ax, ylabel="Daily close (USDT, log scale)")
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.08),
              labelcolor=INK)
    fig.suptitle(f"{a['symbol']}: where the three past stretches most like the last {w} days "
                 "lie (shaded)", x=0.06, y=1.02, ha="left", color=INK, fontsize=12)
    fig.text(0.06, 0.93, "Supporting evidence for the present regime only. The price line "
             "locates the stretches; nothing about what followed any stretch is assessed.",
             color=MUTED, fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def chart_patterns(a: dict, path) -> None:
    f, w = a["features"], a["window"]
    day = np.arange(1, w + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    for ax, (column, title) in zip(axes.ravel(), SHOWN_FEATURES.items()):
        for k, m in enumerate(a["matches"]):
            values = f[column].iloc[m["end_row"] - w + 1:m["end_row"] + 1].to_numpy()
            ax.plot(day, values, color=MATCH_COLOURS[k], linewidth=1.6, marker=MATCH_MARKERS[k],
                    markersize=5, label=f"Match {k + 1} ({m['window_start']})")
        today = f[column].iloc[a["end_row"] - w + 1:a["end_row"] + 1].to_numpy()
        ax.plot(day, today, color=TODAY_COLOUR, linewidth=3, marker="o", markersize=6,
                label=f"Today's stretch (to {a['date'].date()})", zorder=5)
        ax.set_xticks(day)
        _style(ax, title, xlabel="day of the 10-day stretch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [len(handles) - 1, *range(len(handles) - 1)]          # today first, then 1, 2, 3
    fig.legend([handles[k] for k in order], [labels[k] for k in order], frameon=False, ncol=4,
               fontsize=9,
               loc="upper left", bbox_to_anchor=(0.04, 0.95), labelcolor=INK)
    fig.suptitle(f"{a['symbol']}: today's last {w} days beside the three most similar past "
                 "stretches (4 of the 9 features DTW compares)",
                 x=0.05, y=1.0, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def chart_how_close(a: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.hist(a["distances"], bins=60, color=SHADE, edgecolor=MUTED, linewidth=0.5, zorder=2)
    top = ax.get_ylim()[1]
    for k, m in enumerate(a["matches"]):
        ax.axvline(m["distance"], color=MATCH_COLOURS[k], linewidth=2.2, zorder=3,
                   label=f"Match {k + 1}: distance {m['distance']:.3f}, closer than "
                         f"{m['similarity_percentile']:.2f}% of past stretches")
    median = float(np.median(a["distances"]))
    ax.axvline(median, color=MUTED, linestyle="--", linewidth=1,
               label=f"Median of all {len(a['distances']):,} past stretches: {median:.2f}")
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK)
    ax.set_ylim(0, top * 1.25)                     # room for the legend above the bars
    _style(ax, xlabel="DTW distance to today's stretch (average per day; smaller = more similar)",
           ylabel="number of past 10-day stretches")
    fig.suptitle(f"{a['symbol']}: how close the matches are, compared with every earlier "
                 "10-day stretch", x=0.06, ha="left", color=INK, fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def chart_dtw(a: dict, path) -> None:
    """How DTW paired the days of today's stretch and the best match. The pairing
    is chosen on all nine features; RSI is drawn because it is easy to read."""
    w, f, m = a["window"], a["features"], a["matches"][0]
    start_now, start_then = a["end_row"] - w + 1, m["end_row"] - w + 1
    pairs, distance = dtw_path(a["x"][start_now:a["end_row"] + 1],
                               a["x"][start_then:m["end_row"] + 1])
    if abs(distance - m["distance"]) > 1e-3:
        raise ValueError("traced DTW path does not reproduce the match distance")
    now = f["rsi"].iloc[start_now:a["end_row"] + 1].to_numpy()
    then = f["rsi"].iloc[start_then:m["end_row"] + 1].to_numpy()
    gap = max(np.ptp(now), np.ptp(then)) + 15          # draw the two lines apart
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, j in pairs:
        ax.plot([i + 1, j + 1], [now[i] + gap, then[j]], color=MUTED, linewidth=0.9, zorder=1)
    ax.plot(np.arange(1, w + 1), now + gap, color=TODAY_COLOUR, linewidth=2.5, marker="o",
            label=f"Today's stretch (drawn {gap:.0f} RSI points higher)", zorder=3)
    ax.plot(np.arange(1, w + 1), then, color=MATCH_COLOURS[0], linewidth=2.5,
            marker=MATCH_MARKERS[0], label=f"Match 1 ({m['window_start']} to {m['window_end']})",
            zorder=3)
    shifted = sum(i != j for i, j in pairs)
    ax.set_xticks(np.arange(1, w + 1))
    ax.set_yticks([])
    _style(ax, xlabel="day of the 10-day stretch")
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, labelcolor=INK)
    reading = ("the two stretches were in step, so every day was paired with the same day"
               if shifted == 0 else
               f"{shifted} of {len(pairs)} pairings join a day with a neighbouring day "
               "(slanted lines): the shapes were slightly out of step")
    fig.suptitle(f"{a['symbol']}: how dynamic time warping paired the days of today's stretch and "
                 f"match 1 (grey lines)\nHere {reading}.",
                 x=0.06, y=1.02, ha="left", color=INK, fontsize=12)
    fig.text(0.06, -0.1, "The pairing is chosen on all nine features together; RSI is shown "
             "because it is easy to read. A vertical line = the same day in both stretches.",
             color=MUTED, fontsize=8.5)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", default=None, help="Daily candle, e.g. 2025-03-01 (default: last)")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in config.SYMBOLS:
        a = analyse(symbol, args.date)
        chart_where(a, OUT_DIR / f"{symbol}_1_where_in_history.png")
        chart_patterns(a, OUT_DIR / f"{symbol}_2_pattern_comparison.png")
        chart_how_close(a, OUT_DIR / f"{symbol}_3_how_close.png")
        chart_dtw(a, OUT_DIR / f"{symbol}_4_dtw_explained.png")
        summary = {"symbol": symbol, "date": str(a["date"].date()), "window_days": a["window"],
                   "regime_now": a["regime_now"], "past_stretches_compared": int(len(a["distances"])),
                   "median_distance": round(float(np.median(a["distances"])), 3),
                   "matches": [{k: v for k, v in m.items() if k != "end_row"} for m in a["matches"]],
                   "scope": "supporting evidence for the present regime only; nothing about what "
                            "followed a past stretch is reported"}
        (OUT_DIR / f"{symbol}_similarity.json").write_text(json.dumps(summary, indent=2),
                                                            encoding="utf-8")
        print(f"{symbol} {summary['date']} ({a['regime_now']}), {summary['past_stretches_compared']:,} "
              "past stretches compared:")
        for k, m in enumerate(summary["matches"]):
            print(f"  match {k + 1}: {m['window_start']} -> {m['window_end']}  distance {m['distance']:.3f}"
                  f"  closer than {m['similarity_percentile']:.2f}%  regime then {m['regime_then']}")
    print(f"Saved 4 charts and a summary per asset to {OUT_DIR}")


if __name__ == "__main__":
    main()
