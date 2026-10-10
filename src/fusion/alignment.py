"""M4 - Regime fusion: dominant regime, Alignment Score, Confidence Score.

Run:  python -m src.fusion.alignment

This is DECISION-LEVEL (late) fusion. Each timeframe has already made its own
decision (a regime and a confidence). Here those decisions are compared, so the
information "which timeframe disagreed" is kept. Combining the raw features of
all timeframes into one model would destroy it.

    weight_i x confidence_i  = how strongly timeframe i votes
    dominant regime          = the regime with the largest total vote
    Alignment Score          = 100 x (votes for the dominant regime) / (all votes)
    Confidence Score         = Alignment Score x mean confidence of the agreeing timeframes

No look-ahead: at each moment a timeframe contributes its last candle that has
already CLOSED. A candle that is still forming is never used.
"""
import numpy as np
import pandas as pd

from src import config


def load_regime_grid(symbol: str, intervals: list[str]) -> pd.DataFrame:
    """One row per closed candle of the fastest timeframe, holding every
    timeframe's most recent CLOSED regime and confidence.

    merge_asof(direction="backward") picks, for each moment, the latest slower
    candle whose close_time is not after that moment.
    """
    ordered = [i for i in config.INTERVALS if i in intervals]   # slow -> fast
    fastest = ordered[-1]

    def load(interval: str) -> pd.DataFrame:
        regimes = pd.read_parquet(
            config.REGIMES_DIR / f"{symbol}_{interval}_regimes.parquet",
            columns=["open_time", "close_time", "regime", "confidence"])
        # Binance records a few outage candles with a close_time BEFORE their
        # open_time (e.g. 1h, 2020-12-21 14:00 "closing" at 13:47). A candle
        # cannot be known before it opens, so its time is never earlier than that.
        regimes["close_time"] = regimes[["open_time", "close_time"]].max(axis=1)
        if not regimes["close_time"].is_monotonic_increasing:
            raise ValueError(f"{symbol} {interval}: candle times are not in order")
        return regimes.drop(columns="open_time").rename(
            columns={"regime": f"regime_{interval}", "confidence": f"conf_{interval}"})

    grid = load(fastest).rename(columns={"close_time": "time"})
    for interval in ordered[:-1]:
        grid = pd.merge_asof(grid, load(interval), left_on="time", right_on="close_time",
                             direction="backward").drop(columns="close_time")
    # Before a slower timeframe has its first closed candle there is nothing
    # to compare, so those early rows are removed.
    return grid.dropna().reset_index(drop=True)


def compute_alignment(grid: pd.DataFrame, intervals: list[str],
                      weights: dict[str, float],
                      classes: list[str] | None = None) -> pd.DataFrame:
    """Add dominant regime, Alignment Score (0-100) and Confidence Score (0-100).

    `classes` lists the possible labels. It defaults to the four regimes; the
    H1 robustness check passes direction labels (up / down / neutral) instead.
    """
    classes = config.REGIMES if classes is None else classes
    labels = np.column_stack([grid[f"regime_{i}"].to_numpy() for i in intervals])
    confidence = np.column_stack([grid[f"conf_{i}"].to_numpy(dtype=float) for i in intervals])
    votes = confidence * np.array([weights[i] for i in intervals])

    # Total vote for each regime at each moment: shape (moments, regimes).
    regime_votes = np.column_stack([
        np.where(labels == regime, votes, 0.0).sum(axis=1) for regime in classes])
    dominant_index = regime_votes.argmax(axis=1)
    dominant = np.array(classes)[dominant_index]

    agrees = labels == dominant[:, None]
    alignment = 100.0 * regime_votes.max(axis=1) / votes.sum(axis=1)
    mean_agreeing_confidence = (confidence * agrees).sum(axis=1) / agrees.sum(axis=1)

    out = grid.copy()
    out["dominant"] = dominant
    out["n_agreeing"] = agrees.sum(axis=1)
    out["alignment"] = alignment
    out["confidence_score"] = alignment * mean_agreeing_confidence
    return out


def build_alignment(symbol: str, intervals: list[str], tag: str) -> pd.DataFrame:
    """Compute and save one alignment series; print its sanity report."""
    grid = load_regime_grid(symbol, intervals)
    result = compute_alignment(grid, intervals, config.TIMEFRAME_WEIGHTS)
    config.FUSION_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(config.FUSION_DIR / f"{symbol}_alignment_{tag}.parquet", index=False)

    is_train = result["time"] < pd.Timestamp(config.TRAIN_END) + pd.Timedelta(days=1)
    print(f"\n=== {symbol} alignment [{tag}] using {', '.join(intervals)} ===")
    print(f"rows {len(result):,} | {result['time'].iloc[0]:%Y-%m-%d} -> "
          f"{result['time'].iloc[-1]:%Y-%m-%d}")
    print(f"alignment  mean {result['alignment'].mean():5.1f} | "
          f"train {result.loc[is_train, 'alignment'].mean():5.1f} | "
          f"test {result.loc[~is_train, 'alignment'].mean():5.1f} | "
          f"min {result['alignment'].min():5.1f} | max {result['alignment'].max():5.1f}")
    print("timeframes agreeing with the dominant regime (share of time):")
    print((result["n_agreeing"].value_counts(normalize=True).sort_index()
           .rename_axis("n_agreeing")).to_string(float_format=lambda v: f"{v:6.1%}"))
    print("dominant regime (share of time):")
    print(result["dominant"].value_counts(normalize=True).reindex(config.REGIMES)
          .fillna(0.0).to_string(float_format=lambda v: f"{v:6.1%}"))
    print("how often each pair of timeframes names the same regime:")
    pairs = pd.DataFrame(
        {a: {b: (result[f"regime_{a}"] == result[f"regime_{b}"]).mean() for b in intervals}
         for a in intervals}).loc[intervals, intervals]
    print(pairs.to_string(float_format=lambda v: f"{v:6.1%}"))
    return result


def main() -> None:
    for symbol in config.SYMBOLS:
        build_alignment(symbol, config.INTERVALS, tag="all")
        build_alignment(symbol, config.H1_ALIGNMENT_INTERVALS, tag="ex1d")
    print("\nAlignment series written to", config.FUSION_DIR)


if __name__ == "__main__":
    main()
