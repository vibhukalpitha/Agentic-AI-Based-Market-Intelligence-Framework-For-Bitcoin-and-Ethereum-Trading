"""Exploratory data analysis (EDA): a description of the data before modelling.

Run:  python -m src.eda.explore        (writes charts and tables to results/eda/)

WHY. Before any model is trusted, the data it learns from should be understood:
how prices moved, how returns are spread, how volatility changed over the
years, how the features relate to each other, and when the market is active.
This module documents that. Several of its findings were the reasons for
design decisions taken earlier (shown in brackets):

  - crypto volatility fell over the years       (volatility measured relative
                                                  to its own past year)
  - returns have fat tails                       (rank-based statistics, not
                                                  tests that assume a bell curve)
  - several features overlap strongly            (full covariance in the HMM)
  - the market is much quieter at weekends       (H1 matched on day of week)

It only READS the data. It changes no data, feature, model or result.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from src import config

EDA_DIR = config.RESULTS_DIR / "eda"
COLOURS = {"BTCUSDT": "#2a78d6", "ETHUSDT": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_red", ["#2a78d6", "#f0efec", "#e34948"])
TRAIN_END = pd.Timestamp(config.TRAIN_END)
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _save(fig, name: str) -> None:
    fig.savefig(EDA_DIR / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def raw(symbol: str, interval: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(config.PARQUET_DIR / f"{symbol}_{interval}.parquet", columns=columns)


# --------------------------------------------------------------------------
# 1. Data quality
# --------------------------------------------------------------------------
def quality_row(candles: pd.DataFrame, step: pd.Timedelta) -> dict:
    """Counts that describe how complete and clean one raw file is."""
    open_time = candles["open_time"]
    length = candles["close_time"] - open_time
    return {
        "candles": int(len(candles)),
        "first": str(open_time.iloc[0].date()),
        "last": str(open_time.iloc[-1].date()),
        "gaps": int((open_time.diff().dropna() != step).sum()),
        "odd_length_candles": int(((length - step).abs() > pd.Timedelta(seconds=1)).sum()),
        "zero_volume_candles": int((candles["volume"] == 0).sum()),
        "duplicate_times": int(open_time.duplicated().sum()),
        "missing_values": int(candles.isna().sum().sum()),
    }


def data_quality() -> pd.DataFrame:
    rows = []
    for symbol in config.SYMBOLS:
        for interval in config.INTERVALS:
            rows.append({"asset": symbol, "timeframe": interval,
                         **quality_row(raw(symbol, interval), config.INTERVAL_STEP[interval])})
    table = pd.DataFrame(rows)
    table.to_csv(EDA_DIR / "data_quality.csv", index=False)
    return table


# --------------------------------------------------------------------------
# 2. Price history
# --------------------------------------------------------------------------
def price_history() -> dict:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    facts = {}
    for ax, symbol in zip(axes, config.SYMBOLS):
        daily = raw(symbol, "1d", ["open_time", "close"])
        ax.plot(daily["open_time"], daily["close"], color=COLOURS[symbol], linewidth=1.2)
        ax.set_yscale("log")
        ax.axvline(TRAIN_END, color=MUTED, linestyle="--", linewidth=1)
        ax.annotate("training | test", xy=(TRAIN_END, 1), xycoords=("data", "axes fraction"),
                    xytext=(0, -12), textcoords="offset points", ha="center",
                    color=MUTED, fontsize=9, backgroundcolor="white")
        _style(ax, f"{symbol}: Daily close (log scale)", ylabel="Price (USDT)")
        low, high = daily.loc[daily["close"].idxmin()], daily.loc[daily["close"].idxmax()]
        facts[symbol] = {"lowest_close": float(low["close"]), "lowest_on": str(low["open_time"].date()),
                         "highest_close": float(high["close"]), "highest_on": str(high["open_time"].date()),
                         "ratio_high_to_low": float(high["close"] / low["close"])}
    _save(fig, "price_history.png")
    return facts


# --------------------------------------------------------------------------
# 3. Volatility over the years
# --------------------------------------------------------------------------
def volatility_over_time() -> dict:
    """Absolute volatility (ATR / price) against relative volatility (atr_rel).
    Two panels, not two axes on one panel, because the units differ."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    facts = {}
    for symbol in config.SYMBOLS:
        f = pd.read_parquet(config.FEATURES_DIR / f"{symbol}_1d_features.parquet",
                            columns=["open_time", "atr_norm", "atr_rel"]).set_index("open_time")
        absolute = f["atr_norm"].rolling(90).median() * 100
        relative = np.exp(f["atr_rel"]).rolling(90).median()
        top.plot(absolute.index, absolute, color=COLOURS[symbol], linewidth=1.5, label=symbol)
        bottom.plot(relative.index, relative, color=COLOURS[symbol], linewidth=1.5, label=symbol)
        by_year = (f["atr_norm"] * 100).groupby(f.index.year).median()
        facts[symbol] = {"median_daily_range_pct_by_year": {int(y): round(float(v), 2)
                                                             for y, v in by_year.items()}}
    for ax in (top, bottom):
        ax.axvline(TRAIN_END, color=MUTED, linestyle="--", linewidth=1)
        ax.legend(frameon=False, labelcolor=INK)
    bottom.axhline(1.0, color=MUTED, linewidth=1)
    _style(top, "Absolute volatility: typical daily range as % of price (90-day median)",
           ylabel="% of price")
    _style(bottom, "Relative volatility: daily range against its own past-year norm "
                   "(90-day median; 1.0 = normal)", ylabel="times the yearly norm")
    _save(fig, "volatility_over_time.png")
    return facts


# --------------------------------------------------------------------------
# 4. Returns: how they are spread
# --------------------------------------------------------------------------
def return_statistics() -> pd.DataFrame:
    """Shape of the returns at each timeframe. Excess kurtosis 0 and 0.27% of
    returns beyond 3 standard deviations are what a bell curve would give."""
    rows = []
    for symbol in config.SYMBOLS:
        for interval in config.INTERVALS:
            r = pd.read_parquet(config.FEATURES_DIR / f"{symbol}_{interval}_features.parquet",
                                columns=["log_return"])["log_return"].to_numpy()
            z = (r - r.mean()) / r.std()
            rows.append({"asset": symbol, "timeframe": interval, "candles": len(r),
                         "mean_pct": 100 * r.mean(), "std_pct": 100 * r.std(),
                         "skewness": float(stats.skew(r)),
                         "excess_kurtosis": float(stats.kurtosis(r)),
                         "beyond_3_std_pct": 100 * float((np.abs(z) > 3).mean()),
                         "min_pct": 100 * r.min(), "max_pct": 100 * r.max()})
    table = pd.DataFrame(rows)
    table.to_csv(EDA_DIR / "return_statistics.csv", index=False)
    return table


def return_distribution() -> None:
    """Daily returns against a bell curve with the same mean and spread. The
    logarithmic count axis makes the extra extreme days visible."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, symbol in zip(axes, config.SYMBOLS):
        r = 100 * pd.read_parquet(config.FEATURES_DIR / f"{symbol}_1d_features.parquet",
                                  columns=["log_return"])["log_return"].to_numpy()
        bins = np.linspace(r.min(), r.max(), 80)
        ax.hist(r, bins=bins, color=COLOURS[symbol], alpha=0.85, label="observed days")
        centres = (bins[:-1] + bins[1:]) / 2
        expected = len(r) * np.diff(bins) * stats.norm.pdf(centres, r.mean(), r.std())
        ax.plot(centres, expected, color=INK, linewidth=1.5, label="bell curve, same mean and spread")
        ax.set_yscale("log")
        ax.set_ylim(0.5, 2000)
        ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper left")
        _style(ax, f"{symbol}: Daily log returns", xlabel="Daily log return (%)",
               ylabel="Days (log scale)")
    _save(fig, "return_distribution.png")


# --------------------------------------------------------------------------
# 5. How the features relate
# --------------------------------------------------------------------------
def feature_correlation() -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    facts = {}
    for ax, symbol in zip(axes, config.SYMBOLS):
        f = pd.read_parquet(config.FEATURES_DIR / f"{symbol}_1d_features.parquet",
                            columns=config.FEATURE_COLUMNS)
        corr = f.corr()
        ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color=INK)
        ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(corr)), corr.index, fontsize=8)
        ax.set_title(f"{symbol}: correlation of the 9 Daily features "
                     "(blue = opposite, red = together)", loc="left", color=INK, fontsize=10)
        ax.tick_params(colors=MUTED, length=0)
        pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
        strongest = pairs.abs().sort_values(ascending=False).head(5)
        facts[symbol] = [{"pair": f"{a} / {b}", "correlation": round(float(pairs[(a, b)]), 2)}
                         for a, b in strongest.index]
        corr.to_csv(EDA_DIR / f"feature_correlation_{symbol}_1d.csv")
    _save(fig, "feature_correlation.png")
    return facts


def feature_summary() -> pd.DataFrame:
    rows = []
    for symbol in config.SYMBOLS:
        for interval in config.INTERVALS:
            f = pd.read_parquet(config.FEATURES_DIR / f"{symbol}_{interval}_features.parquet",
                                columns=config.FEATURE_COLUMNS)
            for column in config.FEATURE_COLUMNS:
                v = f[column]
                rows.append({"asset": symbol, "timeframe": interval, "feature": column,
                             "mean": v.mean(), "std": v.std(), "p1": v.quantile(0.01),
                             "median": v.median(), "p99": v.quantile(0.99)})
    table = pd.DataFrame(rows)
    table.to_csv(EDA_DIR / "feature_summary.csv", index=False)
    return table


# --------------------------------------------------------------------------
# 6. When the market is active
# --------------------------------------------------------------------------
def activity_by_time() -> dict:
    """Average size of a 1-hour move by UTC hour and by weekday, relative to the
    overall average (1.0). Measured on raw 1-hour candles, not on regimes."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    facts = {}
    for symbol in config.SYMBOLS:
        candles = raw(symbol, "1h", ["open_time", "close"])
        candles = candles[candles["open_time"] >= pd.Timestamp("2019-01-01")]
        size = np.log(candles["close"]).diff().abs()
        overall = size.mean()
        by_hour = size.groupby(candles["open_time"].dt.hour).mean() / overall
        by_day = size.groupby(candles["open_time"].dt.dayofweek).mean() / overall
        left.plot(by_hour.index, by_hour, color=COLOURS[symbol], linewidth=2, marker="o",
                  markersize=4, label=symbol)
        right.plot(by_day.index, by_day, color=COLOURS[symbol], linewidth=2, marker="o",
                   markersize=6, label=symbol)
        weekend = by_day.loc[[5, 6]].mean()
        weekday = by_day.loc[list(range(5))].mean()
        facts[symbol] = {"quietest_hour_utc": int(by_hour.idxmin()),
                         "busiest_hour_utc": int(by_hour.idxmax()),
                         "busiest_to_quietest_hour": round(float(by_hour.max() / by_hour.min()), 2),
                         "weekend_vs_weekday_activity": round(float(weekend / weekday), 2)}
    for ax in (left, right):
        ax.axhline(1.0, color=MUTED, linewidth=1)
        ax.legend(frameon=False, labelcolor=INK)
    left.set_xticks(range(0, 24, 3))
    right.set_xticks(range(7), WEEKDAYS)
    _style(left, "Size of 1-hour moves by hour of day (UTC)", xlabel="Hour (UTC)",
           ylabel="times the overall average")
    _style(right, "Size of 1-hour moves by weekday", xlabel="")
    _save(fig, "activity_by_time.png")
    return facts


# --------------------------------------------------------------------------
# Summary document
# --------------------------------------------------------------------------
def write_summary(quality: pd.DataFrame, returns: pd.DataFrame, facts: dict) -> None:
    def table(frame: pd.DataFrame, fmt: str = "{:.2f}") -> str:
        head = "| " + " | ".join(frame.columns) + " |\n|" + "---|" * len(frame.columns) + "\n"
        body = "".join("| " + " | ".join(fmt.format(v) if isinstance(v, float) else str(v)
                                         for v in row) + " |\n" for row in frame.itertuples(index=False))
        return head + body

    vol = {s: facts["volatility"][s]["median_daily_range_pct_by_year"] for s in config.SYMBOLS}
    lines = [
        "# Exploratory data analysis",
        "",
        "*Generated by `python -m src.eda.explore`. Every number below is computed from the data.*",
        "",
        "## 1. Data quality",
        "",
        table(quality),
        "## 2. Price history (`price_history.png`)",
        "",
    ]
    for s in config.SYMBOLS:
        p = facts["prices"][s]
        lines.append(f"- {s}: lowest daily close {p['lowest_close']:,.2f} on {p['lowest_on']}, "
                     f"highest {p['highest_close']:,.2f} on {p['highest_on']} "
                     f"({p['ratio_high_to_low']:.0f} times the low).")
    lines += ["", "## 3. Volatility over the years (`volatility_over_time.png`)", "",
              "Median daily range as a percentage of price, by year:", "",
              "| Year | " + " | ".join(config.SYMBOLS) + " |", "|---|---|---|"]
    for year in sorted(vol[config.SYMBOLS[0]]):
        lines.append(f"| {year} | " + " | ".join(f"{vol[s].get(year, float('nan')):.2f}"
                                                for s in config.SYMBOLS) + " |")
    lines += ["", "Absolute volatility falls over the years; measured against its own past "
                  "year (bottom panel of the chart) it stays around 1.0. This is why the model "
                  "uses relative volatility.", "",
              "## 4. Returns (`return_distribution.png`, `return_statistics.csv`)", "",
              "Log returns over the feature period, i.e. after the indicator warm-up (Daily "
              "from 2018-11-25, 15m and 5m from 2021-01-01), so the candle counts are smaller "
              "than in the data-quality table.", "",
              table(returns[["asset", "timeframe", "candles", "std_pct", "skewness",
                             "excess_kurtosis", "beyond_3_std_pct"]]),
              "A bell curve would give an excess kurtosis of 0 and 0.27% of returns beyond 3 "
              "standard deviations. Both are much higher at every timeframe: returns have fat "
              "tails. This is why the tests use rank-based statistics.", "",
              "## 5. Feature relationships (`feature_correlation.png`)", "",
              "The five strongest correlations between Daily features:", ""]
    for s in config.SYMBOLS:
        lines.append(f"- {s}: " + "; ".join(f"{p['pair']} {p['correlation']:+.2f}"
                                           for p in facts["correlation"][s]))
    lines += ["", "The trend features (rsi, ema_dist, sma_dist, bb_pctb) largely measure the same "
                  "thing. The HMM uses full covariance, so it models these overlaps instead of "
                  "counting them as independent evidence.", "",
              "## 6. When the market is active (`activity_by_time.png`)", ""]
    for s in config.SYMBOLS:
        a = facts["activity"][s]
        lines.append(f"- {s}: quietest hour {a['quietest_hour_utc']:02d}:00 UTC, busiest "
                     f"{a['busiest_hour_utc']:02d}:00 UTC ({a['busiest_to_quietest_hour']} times "
                     f"larger moves); weekend moves are {a['weekend_vs_weekday_activity']} times "
                     f"the size of weekday moves.")
    lines += ["", "This is why the H1 test compared like with like on day of week, and measured "
                  "alignment over whole days.", ""]
    (EDA_DIR / "EDA_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    quality = data_quality()
    returns = return_statistics()
    facts = {"prices": price_history(), "volatility": volatility_over_time(),
             "correlation": feature_correlation(), "activity": activity_by_time()}
    return_distribution()
    feature_summary()
    with open(EDA_DIR / "eda_facts.json", "w") as file:
        json.dump(facts, file, indent=2)
    write_summary(quality, returns, facts)

    print(quality.to_string(index=False))
    print()
    print(returns[["asset", "timeframe", "std_pct", "skewness", "excess_kurtosis",
                   "beyond_3_std_pct"]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print()
    print(json.dumps(facts, indent=1))
    print(f"\nSaved to {EDA_DIR}")


if __name__ == "__main__":
    main()
