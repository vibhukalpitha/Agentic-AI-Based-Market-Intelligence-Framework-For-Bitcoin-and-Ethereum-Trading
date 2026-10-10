"""Output of the Market Regime Engine for the Fusion Intelligence Engine.

Run:  python -m src.output.snapshot --symbol BTCUSDT
      python -m src.output.snapshot --symbol ETHUSDT --as-of 2025-03-01T12:00

One call returns one record: everything this component knows about an asset
at one moment. The field-by-field description is in docs/OUTPUT_FORMAT.md.

No look-ahead: for any `as_of` moment the record uses only candles that had
already closed by then, so a record for a past moment is exactly what the
engine would have reported at that moment. tests/test_snapshot.py checks this
by deleting the future and comparing.

Scope: the record describes the PRESENT state and how settled it is. It never
contains a price forecast or future scenarios (another component's job).
"""
import argparse
import json

import numpy as np
import pandas as pd

from src import config
from src.fusion import alignment, propagation

SCHEMA_VERSION = "0.1"
PROBABILITY_COLUMNS = [f"p_{name.lower()}" for name in config.REGIMES]

# Facts a consumer of this record must know to use it correctly.
CAVEATS = [
    "confidence is the model's own probability for its current regime; it is above 0.99 "
    "about 75% of the time and should not be read as a calibrated accuracy.",
    "each timeframe has its own independent model, so a regime name is not guaranteed to "
    "mean exactly the same thing at every timeframe.",
    "alignment_score did not anticipate Daily regime transitions in testing (H1 not "
    "supported); use it as a description of agreement, not as a warning signal.",
    "propagation.direction: 'bottom-up' flips were somewhat more durable than 'top-down' "
    "flips at the 1h and 15m timeframes in testing. At the Daily timeframe the difference "
    "could not be separated from a background effect, so do not rely on it there.",
]


def _iso(time: pd.Timestamp) -> str:
    return pd.Timestamp(time).isoformat(timespec="milliseconds") + "Z"


def load_all_labels(symbol: str) -> dict[str, pd.DataFrame]:
    """Saved regime labels of all five timeframes."""
    return {interval: propagation.load_labels(symbol, interval, extra_columns=PROBABILITY_COLUMNS)
            for interval in config.INTERVALS}


def _position(labels: pd.DataFrame, as_of: pd.Timestamp) -> int:
    """Row of the last candle already closed at `as_of` (-1 if none)."""
    return int(np.searchsorted(labels["known_time"].to_numpy(), np.datetime64(as_of),
                               side="right")) - 1


def _run_start(regime: np.ndarray, position: int) -> int:
    """First row of the unbroken run of the same regime that contains `position`."""
    start = position
    while start > 0 and regime[start - 1] == regime[position]:
        start -= 1
    return start


def timeframe_state(labels: pd.DataFrame, as_of: pd.Timestamp) -> dict | None:
    """Regime, confidence and age at one timeframe, from its last closed candle."""
    position = _position(labels, as_of)
    if position < 0:
        return None
    row = labels.iloc[position]
    regime = labels["regime"].to_numpy()
    return {
        "regime": row["regime"],
        "confidence": round(float(row["confidence"]), 4),
        "probabilities": {name: round(float(row[column]), 4)
                          for name, column in zip(config.REGIMES, PROBABILITY_COLUMNS)},
        "candles_in_regime": position - _run_start(regime, position) + 1,
        "last_closed_candle": _iso(row["known_time"]),
    }


def adoption_status(slower: pd.DataFrame, flip_known: pd.Timestamp, as_of: pd.Timestamp,
                    target: str) -> str:
    """Has the next slower timeframe taken up a flip, judged at `as_of`?

      already_there  it was in that regime before the flip
      adopted        it showed the regime within PROPAGATION_WINDOW_CANDLES of its candles
      pending        the window has not finished yet
      failed         the window finished without adoption
    """
    regime = slower["regime"].to_numpy()
    at_flip, now = _position(slower, flip_known), _position(slower, as_of)
    if at_flip >= 0 and regime[at_flip] == target:
        return "already_there"
    last_in_window = at_flip + config.PROPAGATION_WINDOW_CANDLES
    seen = regime[at_flip + 1:min(last_in_window, now) + 1]
    if (seen == target).any():
        return "adopted"
    return "pending" if now < last_in_window else "failed"


def latest_flip(labels: dict[str, pd.DataFrame], interval: str, as_of: pd.Timestamp) -> dict | None:
    """The flip that started the current regime at one timeframe, and where it
    stood in the hierarchy (see src/fusion/propagation.py)."""
    frame = labels[interval]
    position = _position(frame, as_of)
    if position < 0:
        return None
    regime = frame["regime"].to_numpy()
    start = _run_start(regime, position)
    if start == 0:
        return None                      # the regime began before the data did
    flip = frame.iloc[start]
    target = np.array([flip["regime"]])
    level = config.INTERVALS.index(interval)

    faster = config.INTERVALS[level + 1:]
    before_open = [propagation.labels_as_of(labels[i], np.array([flip["open_time"]],
                                                                dtype="datetime64[ns]"))
                   for i in faster]
    depth, _ = propagation.support_depth(target, before_open)
    record = {
        "from_regime": regime[start - 1],
        "to_regime": flip["regime"],
        "flip_candle_open": _iso(flip["open_time"]),
        "support_depth": int(depth[0]),
        "faster_timeframes": len(faster),
        "direction": ("not_applicable" if not faster
                      else "bottom-up" if depth[0] >= 1 else "top-down"),
        "adopted_by_slower": ("not_applicable" if level == 0 else adoption_status(
            labels[config.INTERVALS[level - 1]], flip["known_time"], as_of, flip["regime"])),
    }
    return record


def fusion_state(states: dict[str, dict]) -> dict:
    """Dominant regime and scores from the five timeframe states (M4 formula)."""
    intervals = list(states)
    grid = pd.DataFrame([{**{f"regime_{i}": states[i]["regime"] for i in intervals},
                          **{f"conf_{i}": states[i]["confidence"] for i in intervals}}])
    result = alignment.compute_alignment(grid, intervals, config.TIMEFRAME_WEIGHTS).iloc[0]
    agreeing = [i for i in intervals if states[i]["regime"] == result["dominant"]]
    return {
        "dominant_regime": result["dominant"],
        "alignment_score": round(float(result["alignment"]), 2),
        "confidence_score": round(float(result["confidence_score"]), 2),
        "timeframes_agreeing": agreeing,
        "timeframes_disagreeing": [i for i in intervals if i not in agreeing],
        "weights": {i: config.TIMEFRAME_WEIGHTS[i] for i in intervals},
    }


def build_snapshot(symbol: str, as_of: pd.Timestamp | None = None,
                   labels: dict[str, pd.DataFrame] | None = None) -> dict:
    """The full record for one asset at one moment (default: the latest data)."""
    labels = load_all_labels(symbol) if labels is None else labels
    if as_of is None:
        as_of = max(frame["known_time"].iloc[-1] for frame in labels.values())
    as_of = pd.Timestamp(as_of)

    states = {i: timeframe_state(labels[i], as_of) for i in config.INTERVALS}
    missing = [i for i, state in states.items() if state is None]
    if missing:
        raise ValueError(f"no closed candle yet at {as_of} for: {', '.join(missing)}")

    return {
        "schema_version": SCHEMA_VERSION,
        "component": "market_regime_engine",
        "symbol": symbol,
        "as_of": _iso(as_of),
        "timeframes": states,
        "fusion": fusion_state(states),
        "propagation": {i: latest_flip(labels, i, as_of) for i in config.INTERVALS},
        "stability": None,       # Regime Stability Score - added with module M6
        "explanation": None,     # plain-language text with base rates - added with M8
        "caveats": CAVEATS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", required=True, choices=config.SYMBOLS)
    parser.add_argument("--as-of", default=None,
                        help="UTC moment, e.g. 2025-03-01T12:00 (default: latest data)")
    parser.add_argument("--save", action="store_true",
                        help="also write results/sample_output_<symbol>.json")
    args = parser.parse_args()

    snapshot = build_snapshot(args.symbol, args.as_of)
    text = json.dumps(snapshot, indent=2)
    print(text)
    if args.save:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (config.RESULTS_DIR / f"sample_output_{args.symbol}.json").write_text(text)


if __name__ == "__main__":
    main()
