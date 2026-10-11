"""Tests for the Fusion Engine output. Run:  python -m pytest"""
import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.output import snapshot

START = pd.Timestamp("2024-01-01")
ALMOST = pd.Timedelta(milliseconds=1)


def make_labels(interval: str, days: int, seed: int) -> pd.DataFrame:
    """Random regime labels for one timeframe, in the saved-file layout."""
    rng = np.random.default_rng(seed)
    step = config.INTERVAL_STEP[interval]
    count = int(pd.Timedelta(days=days) / step)
    open_time = pd.DatetimeIndex([START + k * step for k in range(count)])
    # Regimes change every few candles so the test meets flips of every kind.
    regime = np.repeat(rng.choice(config.REGIMES, count // 3 + 1), 3)[:count]
    frame = pd.DataFrame({
        "open_time": open_time, "known_time": open_time + step - ALMOST,
        "regime": regime, "confidence": rng.uniform(0.5, 1.0, count),
        "close": 100.0 + np.arange(count),
    })
    for column in snapshot.PROBABILITY_COLUMNS:
        frame[column] = 0.25
    return frame


@pytest.fixture(scope="module")
def labels() -> dict[str, pd.DataFrame]:
    return {interval: make_labels(interval, days=12, seed=k)
            for k, interval in enumerate(config.INTERVALS)}


def test_snapshot_for_a_past_moment_ignores_everything_after_it(labels):
    """Delete all candles that closed after `as_of`; the record must not change."""
    for as_of in (START + pd.Timedelta(days=6, hours=13, minutes=7),
                  START + pd.Timedelta(days=9) - ALMOST):
        past_only = {i: frame[frame["known_time"] <= as_of].reset_index(drop=True)
                     for i, frame in labels.items()}
        assert (snapshot.build_snapshot("TEST", as_of, labels)
                == snapshot.build_snapshot("TEST", as_of, past_only))


def test_snapshot_is_valid_json_with_the_documented_blocks(labels):
    record = snapshot.build_snapshot("TEST", START + pd.Timedelta(days=8), labels)
    assert json.loads(json.dumps(record)) == record
    assert set(record) == {"schema_version", "component", "symbol", "as_of", "timeframes",
                           "fusion", "propagation", "stability", "explanation", "caveats"}
    assert list(record["timeframes"]) == config.INTERVALS
    assert 0 <= record["fusion"]["alignment_score"] <= 100
    assert (sorted(record["fusion"]["timeframes_agreeing"]
                   + record["fusion"]["timeframes_disagreeing"]) == sorted(config.INTERVALS))


def test_unfinished_daily_candle_is_not_reported(labels):
    """At 12:00 on day 5 the day-5 Daily candle is still forming."""
    as_of = START + pd.Timedelta(days=5, hours=12)
    state = snapshot.build_snapshot("TEST", as_of, labels)["timeframes"]["1d"]
    assert state["last_closed_candle"].startswith("2024-01-05T23:59:59")
    assert state["regime"] == labels["1d"]["regime"].iloc[4]


def test_regime_profile_explains_the_label_for_that_model(labels):
    """The record carries what the current regime name means in each model."""
    profile = {name: {"typical_rsi": 50.0, "volatility_vs_norm": 1.0,
                      "share_of_training_time": 0.25, "is_most_volatile_state": name == "Bearish"}
               for name in config.REGIMES}
    as_of = START + pd.Timedelta(days=8)
    record = snapshot.build_snapshot("TEST", as_of, labels,
                                     profiles={"1d": profile, "4h": None, "1h": None,
                                               "15m": None, "5m": None})
    daily = record["timeframes"]["1d"]
    assert daily["regime_profile"] == profile[daily["regime"]]
    assert record["timeframes"]["4h"]["regime_profile"] is None


def test_stability_comes_from_the_last_closed_daily_candle(labels):
    """At 12:00 on day 6 the day-6 Daily candle is still forming, so the record
    must carry day 5's score, never day 6's."""
    days = labels["1d"]["open_time"]
    table = pd.DataFrame({"delivered": np.linspace(0.5, 0.9, len(days)), "B0": 0.8,
                          "regime": labels["1d"]["regime"].to_numpy(),
                          "period": "test"}, index=pd.DatetimeIndex(days, name="date"))
    as_of = START + pd.Timedelta(days=6, hours=12)
    block = snapshot.build_snapshot("TEST", as_of, labels, stability=table)["stability"]
    assert block["daily_candle"].startswith("2024-01-06")          # the 6th day = index 5
    assert block["probability_regime_persists"] == round(float(table["delivered"].iloc[5]), 4)
    assert block["horizon_days"] == config.STABILITY_HORIZON_DAYS
    assert block["in_training_period"] is False

    # No score for that day -> the block is empty, not an error.
    without = snapshot.build_snapshot("TEST", as_of, labels, stability=table.drop(days.iloc[5]))
    assert without["stability"] is None
    assert snapshot.build_snapshot("TEST", as_of, labels)["stability"] is None


def test_candles_in_regime_counts_the_current_run():
    frame = pd.DataFrame({
        "open_time": pd.date_range(START, periods=6, freq="1D"),
        "known_time": pd.date_range(START, periods=6, freq="1D") + pd.Timedelta(days=1) - ALMOST,
        "regime": ["Bullish", "Bearish", "Bearish", "Bearish", "Sideways", "Sideways"],
        "confidence": 0.9, "close": 1.0,
        **{column: 0.25 for column in snapshot.PROBABILITY_COLUMNS}})
    assert snapshot.timeframe_state(frame, START + pd.Timedelta(days=4))["candles_in_regime"] == 3
    assert snapshot.timeframe_state(frame, START + pd.Timedelta(days=6))["candles_in_regime"] == 2
    assert snapshot.timeframe_state(frame, START) is None        # nothing closed yet


def test_adoption_status_moves_from_pending_to_adopted_or_failed():
    hours = pd.date_range(START, periods=8, freq="4h")
    slower = pd.DataFrame({"known_time": hours + pd.Timedelta(hours=4) - ALMOST,
                           "regime": ["A", "A", "A", "B", "A", "A", "A", "A"]})
    flip_known = START + pd.Timedelta(hours=1) - ALMOST      # during slower candle 0

    def status(hours_later: float, target: str) -> str:
        return snapshot.adoption_status(slower, flip_known,
                                        START + pd.Timedelta(hours=hours_later), target)

    # Window = slower candles 0, 1, 2 (PROPAGATION_WINDOW_CANDLES = 3). None is "B".
    assert status(2, "B") == "pending"       # no slower candle has even closed yet
    assert status(5, "B") == "pending"
    assert status(12, "B") == "failed"
    assert status(40, "B") == "failed"       # candle 3 is "B" but lies outside the window
    assert status(12, "A") == "adopted"

    # A flip that happens when the slower timeframe is ALREADY in that regime.
    later_flip = START + pd.Timedelta(hours=9)               # slower candle 1 ("A") has closed
    assert snapshot.adoption_status(slower, later_flip, later_flip, "A") == "already_there"
