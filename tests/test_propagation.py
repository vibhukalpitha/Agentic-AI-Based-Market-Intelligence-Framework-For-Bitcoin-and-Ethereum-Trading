"""Tests for M5 (propagation). Run:  python -m pytest"""
import numpy as np
import pandas as pd

from src.fusion import propagation

HOUR = pd.Timedelta(hours=1)
ALMOST = pd.Timedelta(milliseconds=1)
START = pd.Timestamp("2024-01-01")


def make_labels(regimes: list[str], step: pd.Timedelta) -> pd.DataFrame:
    """Labels for consecutive candles of length `step`, starting at START."""
    open_time = pd.DatetimeIndex([START + k * step for k in range(len(regimes))])
    return pd.DataFrame({
        "open_time": open_time,
        "known_time": open_time + step - ALMOST,
        "regime": regimes,
        "confidence": 0.9,
        "close": 100.0 + np.arange(len(regimes)),
    })


def test_find_flips_records_runs_and_marks_the_last_as_unfinished():
    flips = propagation.find_flips(make_labels(["A", "A", "B", "B", "B", "A"], HOUR))
    assert list(flips["to_regime"]) == ["B", "A"]
    assert list(flips["from_regime"]) == ["A", "B"]
    assert list(flips["run_length"]) == [3, 1]
    assert list(flips["censored"]) == [False, True]
    assert flips["open_time"].iloc[0] == START + 2 * HOUR


def test_labels_as_of_uses_only_candles_already_closed():
    labels = make_labels(["A", "B", "C"], HOUR)
    times = np.array([START, START + HOUR, START + HOUR + 30 * pd.Timedelta(minutes=1),
                      START + 2 * HOUR], dtype="datetime64[ns]")
    # At 00:00 nothing has closed. At 01:00 only candle 0 has. At 01:30 candle 1
    # is still forming, so the answer is still "A".
    assert list(propagation.labels_as_of(labels, times)) == [None, "A", "A", "B"]


def test_support_depth_stops_at_the_first_disagreement():
    target = np.array(["Up", "Up", "Up"])
    next_faster = np.array(["Up", "Down", "Up"])
    fastest = np.array(["Up", "Up", "Down"])
    depth, count = propagation.support_depth(target, [next_faster, fastest])
    assert list(depth) == [2, 0, 1]     # row 2: fastest agrees but the chain is broken
    assert list(count) == [2, 1, 1]


def test_depth_is_measured_before_the_flip_candle_opens():
    """4-hour flips to B at 04:00. The 1-hour series turns B only at 05:00,
    inside the flip candle. That is not support from below: depth must be 0
    before the open, even though it is 1 by the close."""
    slow = make_labels(["A", "B", "B"], 4 * HOUR)
    fast = make_labels(["A"] * 5 + ["B"] * 7, HOUR)
    flip = propagation.find_flips(slow).iloc[[0]]
    target = flip["to_regime"].to_numpy()

    before, _ = propagation.support_depth(
        target, [propagation.labels_as_of(fast, flip["open_time"].to_numpy())])
    after, _ = propagation.support_depth(
        target, [propagation.labels_as_of(fast, flip["known_time"].to_numpy())])
    assert (before[0], after[0]) == (0, 1)


def test_adopted_above_looks_only_at_the_window():
    # Slower timeframe (4h candles): A A B A A ...  A 1h flip to B is known at 00:59:59.999.
    slower = make_labels(["A", "A", "B", "A", "A", "A"], 4 * HOUR)
    known = np.array([START + HOUR - ALMOST], dtype="datetime64[ns]")
    target = np.array(["B"])
    # Next slower candles to become known are indices 0, 1, 2, ...
    _, adopted_2, _ = propagation.adopted_above(slower, known, target, window=2)
    _, adopted_3, complete = propagation.adopted_above(slower, known, target, window=3)
    assert (adopted_2[0], adopted_3[0], complete[0]) == (False, True, True)

    # A flip near the end of the data has no full window to judge.
    late = np.array([START + 22 * HOUR], dtype="datetime64[ns]")
    already, _, complete = propagation.adopted_above(slower, late, target, window=3)
    assert (already[0], complete[0]) == (False, False)
