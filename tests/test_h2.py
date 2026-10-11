"""Tests for the H2 experiment and its placebo control. Run:  python -m pytest"""
import numpy as np
import pandas as pd

from src import config
from src.experiments import h2_placebo
from src.experiments import h2_propagation as h2
from src.fusion import propagation

DAY = pd.Timedelta(days=1)
START = pd.Timestamp("2024-01-01")


def test_durable_outcome_and_unfinished_runs(tmp_path, monkeypatch):
    """A flip is durable if its regime held >= H2_DURABLE_CANDLES. A flip in the
    last, unfinished run is kept only if it is already long enough to be durable;
    otherwise its outcome is unknown and it must be dropped."""
    monkeypatch.setattr(config, "PROPAGATION_DIR", tmp_path)
    pd.DataFrame({
        "interval": ["1d", "1d", "1d", "1d", "4h"],
        "period": ["test", "test", "test", "test", "test"],
        "run_length": [1, 3, 2, 8, 5],
        "censored": [False, False, True, True, False],
        "to_regime": "Bullish", "depth": 0,
    }).to_parquet(tmp_path / "TEST_flips.parquet")

    events = h2.load_events("TEST", "1d", "test", durable_candles=3)
    assert list(events["run_length"]) == [1, 3, 8]        # the unfinished run of 2 is dropped
    assert list(events["durable"]) == [False, True, True]
    assert len(h2.load_events("TEST", "1d", "train", durable_candles=3)) == 0


def make_events(depth_if_durable: int, depth_if_short: int) -> pd.DataFrame:
    """200 flips into two regimes; durable and short-lived flips get the given depths."""
    durable = np.tile([True, True, True, False], 50)
    return pd.DataFrame({
        "durable": durable,
        "to_regime": np.repeat(["Bullish", "Bearish"], 100),
        "depth": np.where(durable, depth_if_durable, depth_if_short),
        "support_count": 0, "run_length": np.where(durable, 9, 1),
    })


def test_effect_is_above_half_only_when_durable_flips_had_deeper_support():
    rng = np.random.default_rng(config.RANDOM_SEED)
    supports = h2.compare_depth(make_events(depth_if_durable=2, depth_if_short=0), "depth", rng)
    against = h2.compare_depth(make_events(depth_if_durable=0, depth_if_short=2), "depth", rng)
    nothing = h2.compare_depth(make_events(depth_if_durable=1, depth_if_short=1), "depth", rng)
    assert supports["effect_auc"] == 1.0 and supports["p_matched"] < 0.01
    assert against["effect_auc"] == 0.0 and against["p_matched"] > 0.99
    assert nothing["effect_auc"] == 0.5
    assert supports["durable"] == 150 and supports["short_lived"] == 50


def test_run_length_view_compares_unsupported_with_supported_flips():
    rng = np.random.default_rng(config.RANDOM_SEED)
    # Unsupported flips (depth 0) are the short-lived ones here, so they are shorter.
    result = h2.compare_run_length(make_events(depth_if_durable=2, depth_if_short=0), rng)
    assert result["effect_auc"] == 1.0
    assert (result["median_run_unsupported"], result["median_run_supported"]) == (1.0, 9.0)


def test_placebo_depth_reads_the_faster_timeframe_thirty_days_earlier():
    """The faster timeframe is 'A' for 40 days, then 'B'. A Daily flip to 'B' on
    day 50 is really supported (the faster timeframe is already 'B'), but 30 days
    earlier the faster timeframe was 'A', so the placebo depth must be 0."""
    hours = pd.date_range(START, periods=60 * 6, freq="4h")
    faster = pd.DataFrame({
        "known_time": hours + pd.Timedelta(hours=4) - pd.Timedelta(milliseconds=1),
        "regime": np.where(hours < START + 40 * DAY, "A", "B")})
    labels = {interval: faster for interval in config.INTERVALS}
    events = pd.DataFrame({"open_time": [START + 50 * DAY], "to_regime": ["B"]})

    assert h2_placebo.PLACEBO_LAG_DAYS == 30
    assert list(h2_placebo.placebo_depth(events, labels, "1d")) == [0]

    # The real measure, taken when the flip candle opens, sees all four faster
    # timeframes already in 'B'.
    at_open = [propagation.labels_as_of(faster, events["open_time"].to_numpy())
               for _ in config.INTERVALS[1:]]
    real_depth, _ = propagation.support_depth(events["to_regime"].to_numpy(), at_open)
    assert list(real_depth) == [4]
