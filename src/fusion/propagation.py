"""M5 - Regime propagation between timeframes (Novelty 2).

Run:  python -m src.fusion.propagation

For every regime flip at every timeframe this module records WHERE the change
stood in the hierarchy when it happened:

  support depth   How many faster timeframes, counted in an unbroken row from
                  the next-faster one downward, were ALREADY in the new regime
                  before the flip candle opened.
  bottom-up       depth >= 1: the change was already present below and climbed up.
  top-down        depth  = 0: this timeframe moved first; nothing below it had.
  adopted above   Did the next SLOWER timeframe show the same regime within
                  PROPAGATION_WINDOW_CANDLES of its own candles? If not, the
                  propagation failed at this level.

WHY "before the flip candle opened": faster timeframes are refreshed more
often, so by the time a slow candle closes they have always reacted to the
same price move. Agreement measured at the close would be true by
construction. Agreement that existed before the slow candle even began is not.

Timeframe order (slow -> fast): Daily, 4h, 1h, 15m, 5m.
"""
import numpy as np
import pandas as pd

from src import config


def load_labels(symbol: str, interval: str,
                label_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Regime labels of one timeframe with the time each label became known.

    label_map optionally renames the labels (the direction-only check maps the
    four regimes to up / down / neutral).
    """
    labels = pd.read_parquet(
        config.REGIMES_DIR / f"{symbol}_{interval}_regimes.parquet",
        columns=["open_time", "close_time", "regime", "confidence", "close"])
    # A candle is known when it closes, and never before it opens (one faulty
    # Binance record has a close_time earlier than its open_time).
    labels["known_time"] = labels[["open_time", "close_time"]].max(axis=1)
    if not labels["known_time"].is_monotonic_increasing:
        raise ValueError(f"{symbol} {interval}: candle times are not in order")
    if label_map is not None:
        labels["regime"] = labels["regime"].map(label_map)
    return labels.drop(columns="close_time")


def find_flips(labels: pd.DataFrame) -> pd.DataFrame:
    """One row per regime flip: the first candle of each new run of labels.

    run_length is how many candles the new regime then lasted. For the final
    run of the data the true length is unknown, so `censored` is True.
    """
    regime = labels["regime"].to_numpy()
    start = np.flatnonzero(regime[1:] != regime[:-1]) + 1
    end = np.r_[start[1:], len(regime)]
    close = labels["close"].to_numpy(dtype=float)
    return pd.DataFrame({
        "open_time": labels["open_time"].to_numpy()[start],
        "known_time": labels["known_time"].to_numpy()[start],
        "from_regime": regime[start - 1],
        "to_regime": regime[start],
        "run_length": end - start,
        "censored": end == len(regime),
        "confidence": labels["confidence"].to_numpy(dtype=float)[start],
        "flip_return": np.log(close[start] / close[start - 1]),
    })


def labels_as_of(labels: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    """The most recent label that was already KNOWN at each given time
    (None where nothing was known yet)."""
    position = np.searchsorted(labels["known_time"].to_numpy(), times, side="right") - 1
    regime = labels["regime"].to_numpy()
    out = np.full(len(times), None, dtype=object)
    out[position >= 0] = regime[position[position >= 0]]
    return out


def support_depth(target: np.ndarray, faster_labels: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(unbroken depth, plain count) of faster timeframes already in `target`.

    faster_labels is ordered from the next-faster timeframe downward. The
    unbroken depth stops at the first timeframe that disagrees; this is the
    cascade idea (5m -> 15m -> 1h -> ...). The plain count ignores order.
    """
    if not faster_labels:
        zeros = np.zeros(len(target), dtype=int)
        return zeros, zeros
    agrees = np.column_stack([labels == target for labels in faster_labels])
    return np.cumprod(agrees, axis=1).sum(axis=1), agrees.sum(axis=1)


def adopted_above(slower: pd.DataFrame, known_times: np.ndarray, target: np.ndarray,
                  window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each flip: (slower timeframe was already in the regime,
                       it showed the regime within `window` of its next candles,
                       the full window exists in the data).

    The "next candles" are the slower candles that become known AFTER the flip
    is known, so the candle still forming at the flip is the first of them.
    """
    regime = slower["regime"].to_numpy()
    position = np.searchsorted(slower["known_time"].to_numpy(), known_times, side="right") - 1
    already = np.zeros(len(target), dtype=bool)
    has_past = position >= 0
    already[has_past] = regime[position[has_past]] == target[has_past]

    adopted = np.zeros(len(target), dtype=bool)
    for step in range(1, window + 1):
        index = position + step
        inside = index < len(regime)
        adopted[inside] |= regime[index[inside]] == target[inside]
    complete = position + window < len(regime)
    return already, adopted, complete


def build_flips(symbol: str, label_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Flip table for every timeframe of one asset."""
    labels = {i: load_labels(symbol, i, label_map) for i in config.INTERVALS}
    # Every timeframe must have at least one known label before a flip is usable.
    first_complete = max(frame["known_time"].iloc[0] for frame in labels.values())

    tables = []
    for level, interval in enumerate(config.INTERVALS):
        flips = find_flips(labels[interval])
        flips = flips[flips["open_time"] >= first_complete].reset_index(drop=True)
        target = flips["to_regime"].to_numpy()
        faster = config.INTERVALS[level + 1:]

        # Support BEFORE the flip candle opened (the measure used by H2) ...
        at_open = [labels_as_of(labels[i], flips["open_time"].to_numpy()) for i in faster]
        flips["depth"], flips["support_count"] = support_depth(target, at_open)
        # ... and at its close, kept only to show how much is "by construction".
        at_close = [labels_as_of(labels[i], flips["known_time"].to_numpy()) for i in faster]
        flips["depth_at_close"], _ = support_depth(target, at_close)
        flips["faster_levels"] = len(faster)
        flips["direction"] = np.where(flips["depth"] >= 1, "bottom-up", "top-down")

        if level > 0:
            already, adopted, complete = adopted_above(
                labels[config.INTERVALS[level - 1]], flips["known_time"].to_numpy(),
                target, config.PROPAGATION_WINDOW_CANDLES)
            flips["slower_already"], flips["adopted_above"], flips["adoption_known"] = \
                already, adopted, complete
        else:
            flips["slower_already"] = flips["adopted_above"] = flips["adoption_known"] = False

        flips.insert(0, "interval", interval)
        tables.append(flips)

    table = pd.concat(tables, ignore_index=True)
    train_end = pd.Timestamp(config.TRAIN_END) + pd.Timedelta(days=1)
    table.insert(1, "period", np.where(table["open_time"] < train_end, "train", "test"))
    return table


def save_and_report(symbol: str) -> None:
    table = build_flips(symbol)
    config.PROPAGATION_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(config.PROPAGATION_DIR / f"{symbol}_flips.parquet", index=False)

    # The report shows the PREDICTOR side only (how flips are distributed).
    # How long flips lasted is the H2 outcome and is reported by the H2 script.
    print(f"\n=== {symbol}: regime flips, {table['open_time'].min():%Y-%m-%d} -> "
          f"{table['open_time'].max():%Y-%m-%d} ===")
    rows = []
    for interval in config.INTERVALS:
        part = table[table["interval"] == interval]
        row = {"interval": interval,
               "flips_train": int((part["period"] == "train").sum()),
               "flips_test": int((part["period"] == "test").sum()),
               "faster_levels": int(part["faster_levels"].iloc[0])}
        if row["faster_levels"]:
            row["bottom_up_share"] = (part["direction"] == "bottom-up").mean()
            for depth in range(row["faster_levels"] + 1):
                row[f"depth_{depth}"] = (part["depth"] == depth).mean()
            row["supported_at_close"] = (part["depth_at_close"] >= 1).mean()
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False, na_rep="-",
                                       float_format=lambda v: f"{v:.1%}"))


def main() -> None:
    for symbol in config.SYMBOLS:
        save_and_report(symbol)
    print("\nFlip tables written to", config.PROPAGATION_DIR)


if __name__ == "__main__":
    main()
