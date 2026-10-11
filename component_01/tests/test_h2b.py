"""Tests for H2b (failed propagation). Run:  python -m pytest"""
import numpy as np
import pandas as pd

from src import config
from src.experiments import h2b_failed_propagation as h2b
from src.fusion import propagation

START = pd.Timestamp("2024-06-01")
DAY = pd.Timedelta(days=1)


def test_only_flips_alive_at_the_end_of_the_window_are_compared(tmp_path, monkeypatch):
    """A flip counts only if (a) the slower timeframe was not already in the
    regime, (b) its run is finished so its length is known, and (c) it was
    still running when the adoption window closed. The outcome is what
    remained AFTER the window."""
    monkeypatch.setattr(config, "PROPAGATION_DIR", tmp_path)
    flips = pd.DataFrame({
        "interval": "4h", "period": "test", "to_regime": "Bullish",
        "known_time": [START + k * DAY for k in range(6)],
        "slower_already":        [False, False, False, True, False, False],
        "adoption_known":        [True,  True,  True,  True, True,  False],
        "censored":              [False, False, True,  False, False, False],
        "run_length":            [30,    10,    40,    50,    25,    60],
        "candles_at_window_end": [18,    18,    18,    18,    18,    -1],
        "adopted_above":         [True,  True,  True,  True,  False, False],
    })
    flips.to_parquet(tmp_path / "TEST_flips.parquet")
    slower = pd.DataFrame({
        "known_time": pd.date_range(START - 60 * DAY, periods=120, freq="1D"),
        "regime": "Bearish"})
    monkeypatch.setattr(propagation, "load_labels", lambda symbol, interval: slower)

    survivors = h2b.load_survivors("TEST", "4h", "test")
    # Row 0: alive, 12 candles remain.   Row 1: died before the window closed (10 <= 18).
    # Row 2: unfinished run.             Row 3: slower timeframe was already there.
    # Row 4: alive, 7 candles remain.    Row 5: window not complete in the data.
    assert list(survivors["remaining"]) == [12, 7]
    assert list(survivors["failed"]) == [False, True]
    assert survivors["open_adoption_flips"].iloc[0] == 4       # rows 0, 1, 2, 4
    assert survivors["placebo_failed"].all()                   # slower was never Bullish


def test_effect_and_placebo_are_measured_on_the_same_flips():
    rng = np.random.default_rng(config.RANDOM_SEED)
    count = 400
    failed = rng.random(count) < 0.6
    events = pd.DataFrame({
        "to_regime": rng.choice(["Bullish", "Bearish"], count),
        "failed": failed,
        "placebo_failed": rng.random(count) < 0.6,             # unrelated to the outcome
        # Failed flips have clearly shorter remaining lives.
        "remaining": np.where(failed, rng.integers(1, 6, count), rng.integers(6, 30, count)),
        "open_adoption_flips": count,
    })
    result = h2b.compare(events, rng)
    assert result["effect_auc"] > 0.95
    assert abs(result["placebo_effect"] - 0.5) < 0.1
    assert result["difference_ci_low"] > 0                     # the real effect exceeds the placebo
    assert result["failed"] + result["adopted"] == count
