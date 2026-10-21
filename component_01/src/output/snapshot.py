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
from src.explain import explainer
from src.fusion import alignment, propagation
from src.regime import describe
from src.similarity import dtw_match

# 0.6: times are written to the microsecond (they were rounded to milliseconds),
# and a moment later than the newest data is refused instead of answered with
# the newest data. No field was added or removed.
SCHEMA_VERSION = "0.6"
PROBABILITY_COLUMNS = [f"p_{name.lower()}" for name in config.REGIMES]

# Facts a consumer of this record must know to use it correctly.
CAVEATS = [
    "confidence is the model's own probability for its current regime; it is above 0.99 "
    "about 75% of the time and should not be read as a calibrated accuracy.",
    "each timeframe has its own independent model, so a regime name is not guaranteed to "
    "mean exactly the same thing at every timeframe. 'Volatile' is the more active of the "
    "two non-directional states, not always the most volatile one; check regime_profile.",
    "stability.probability_regime_persists ranks days usefully (in testing a day whose "
    "regime persisted got the higher score 73% of the time for BTC and 67% for ETH), but "
    "it is too confident at the top: when it said above 90%, the regime persisted about "
    "86-93% of the time for BTC and about 80% for ETH.",
    "explanation.base_rate_regime_change is a training-period figure. In testing the order "
    "of the patterns held (the Daily regime changed 16-18 percentage points more often "
    "under 'Opposed' than under 'Aligned'), but every rate was higher than in training, "
    "most for ETH. Read base rates as relative, not exact. Part of the difference between "
    "patterns arises simply because faster timeframes react sooner than Daily.",
    "similar_history is supporting evidence for the present regime label only. It says "
    "when the last days looked most alike and what the regime was then; it deliberately "
    "says nothing about what followed those past stretches.",
    "alignment_score did not anticipate Daily regime transitions in testing (H1 not "
    "supported); use it as a description of agreement, not as a warning signal.",
    "propagation.direction: 'bottom-up' flips were somewhat more durable than 'top-down' "
    "flips at the 1h and 15m timeframes in testing. At the Daily timeframe the difference "
    "could not be separated from a background effect, so do not rely on it there.",
]


def _iso(time: pd.Timestamp) -> str:
    """UTC time to the microsecond. Binance candles from 2025 close at ...59.999999;
    rounding to milliseconds (...59.999) would put a written time one microsecond
    BEFORE the candle closed, so the time could not be fed back in as `as_of`
    and give the same record."""
    return pd.Timestamp(time).isoformat(timespec="microseconds") + "Z"


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


def timeframe_state(labels: pd.DataFrame, as_of: pd.Timestamp,
                    profiles: dict[str, dict] | None = None) -> dict | None:
    """Regime, confidence and age at one timeframe, from its last closed candle.

    profiles (from src.regime.describe) adds what the current regime name means
    in THIS model: its typical RSI and its volatility relative to the yearly norm.
    """
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
        "regime_profile": None if profiles is None else profiles[row["regime"]],
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


def load_stability(symbol: str) -> pd.DataFrame | None:
    """Saved Regime Stability Scores (M6), indexed by Daily candle date."""
    path = config.STABILITY_DIR / f"{symbol}_stability.parquet"
    return pd.read_parquet(path).set_index("date") if path.exists() else None


def stability_state(table: pd.DataFrame | None, daily: pd.DataFrame,
                    as_of: pd.Timestamp) -> dict | None:
    """The Regime Stability Score of the last Daily candle closed at `as_of`.

    The score for a day is computed from signals known at that day's close by
    a model fitted on the training period only, so it uses no future data.
    Returns None where no score exists (before 2021, or signals missing).
    """
    position = _position(daily, as_of)
    if table is None or position < 0:
        return None
    date = daily["open_time"].iloc[position]
    if date not in table.index:
        return None
    row = table.loc[date]
    return {
        "probability_regime_persists": round(float(row["delivered"]), 4),
        "horizon_days": config.STABILITY_HORIZON_DAYS,
        "regime": row["regime"],
        "daily_candle": _iso(date),
        # True when the day lies in the period the model was fitted on, where
        # the score is not an out-of-sample estimate.
        "in_training_period": bool(row["period"] != "test"),
        "baseline_always": round(float(row["B0"]), 4),
    }


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
                   labels: dict[str, pd.DataFrame] | None = None,
                   profiles: dict[str, dict] | None = None,
                   stability: pd.DataFrame | None = None,
                   archetypes: tuple[pd.DataFrame | None, dict | None] | None = None,
                   similarity: tuple[pd.DataFrame, np.ndarray] | None = None) -> dict:
    """The full record for one asset at one moment (default: the latest data).

    labels, profiles, stability and archetypes are read from the saved files
    when not given; tests pass them in directly.
    """
    if stability is None and labels is None:
        stability = load_stability(symbol)
    if archetypes is None:
        archetypes = explainer.load_archetypes(symbol) if labels is None else (None, None)
    if similarity is None and labels is None:
        similarity = dtw_match.load_daily(symbol)
    labels = load_all_labels(symbol) if labels is None else labels
    if profiles is None:
        profiles = {i: describe.load_profiles(symbol, i) for i in config.INTERVALS}
    newest = max(frame["known_time"].iloc[-1] for frame in labels.values())
    as_of = newest if as_of is None else pd.Timestamp(as_of)
    # A moment for which candles are missing would silently be described with
    # older data. The next fastest candle closes one step after the newest one;
    # from then on the data no longer covers the moment, so it is refused.
    fastest_step = config.INTERVAL_STEP[config.INTERVALS[-1]]
    if as_of >= newest + fastest_step:
        raise ValueError(
            f"the data ends at {_iso(newest)} (newest closed candle); a record for "
            f"{_iso(as_of)} would miss later candles. For later moments run the live "
            "update: python -m src.live.update")

    states = {i: timeframe_state(labels[i], as_of, profiles.get(i)) for i in config.INTERVALS}
    missing = [i for i, state in states.items() if state is None]
    if missing:
        raise ValueError(f"no closed candle yet at {as_of} for: {', '.join(missing)}")

    propagation_block = {i: latest_flip(labels, i, as_of) for i in config.INTERVALS}
    stability_block = stability_state(stability, labels["1d"], as_of)

    # The archetype is that of the last Daily candle closed at `as_of`, and the
    # explanation describes the faster timeframes as they stood at that close.
    archetype_table, base_rates = archetypes
    daily_row = labels["1d"].iloc[_position(labels["1d"], as_of)]
    last_daily = daily_row["open_time"]
    archetype_row = (archetype_table.loc[last_daily]
                     if archetype_table is not None and last_daily in archetype_table.index
                     else None)
    close = np.array([daily_row["known_time"]], dtype="datetime64[ns]")
    faster_at_close = {i: propagation.labels_as_of(labels[i], close)[0]
                       for i in config.INTERVALS[1:]}
    if any(regime is None for regime in faster_at_close.values()):
        faster_at_close = None
    similar_block = (dtw_match.similar_history(*similarity, last_daily)
                     if similarity is not None else None)

    return {
        "schema_version": SCHEMA_VERSION,
        "component": "market_regime_engine",
        "symbol": symbol,
        "as_of": _iso(as_of),
        "timeframes": states,
        "fusion": fusion_state(states),
        "propagation": propagation_block,
        "stability": stability_block,
        "explanation": explainer.build(symbol, states, stability_block, propagation_block,
                                       archetype_row, base_rates, faster_at_close,
                                       str(pd.Timestamp(last_daily).date()), similar_block),
        "similar_history": similar_block,
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
