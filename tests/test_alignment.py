"""Tests for M4. Run:  python -m pytest"""
import numpy as np
import pandas as pd

from src import config
from src.fusion import alignment

INTERVALS = ["1d", "4h", "1h"]
WEIGHTS = {"1d": 0.5, "4h": 0.3, "1h": 0.2}


def make_grid(regimes: list[str], confidences: list[float]) -> pd.DataFrame:
    row = {"time": pd.Timestamp("2024-01-01")}
    for interval, regime, conf in zip(INTERVALS, regimes, confidences):
        row[f"regime_{interval}"] = regime
        row[f"conf_{interval}"] = conf
    return pd.DataFrame([row])


def test_full_agreement_scores_100():
    result = alignment.compute_alignment(
        make_grid(["Bullish"] * 3, [0.9, 0.8, 0.7]), INTERVALS, WEIGHTS)
    assert result["dominant"].iloc[0] == "Bullish"
    np.testing.assert_allclose(result["alignment"].iloc[0], 100.0)
    np.testing.assert_allclose(result["confidence_score"].iloc[0], 100.0 * 0.8)


def test_alignment_matches_hand_calculation():
    # votes: 1d 0.5*0.8 = 0.40 Bullish | 4h 0.3*1.0 = 0.30 Bearish | 1h 0.2*0.5 = 0.10 Bullish
    # Bullish 0.50 of 0.80 total -> 62.5 ; agreeing confidences 0.8 and 0.5 -> mean 0.65
    result = alignment.compute_alignment(
        make_grid(["Bullish", "Bearish", "Bullish"], [0.8, 1.0, 0.5]), INTERVALS, WEIGHTS)
    assert result["dominant"].iloc[0] == "Bullish"
    assert result["n_agreeing"].iloc[0] == 2
    np.testing.assert_allclose(result["alignment"].iloc[0], 62.5)
    np.testing.assert_allclose(result["confidence_score"].iloc[0], 62.5 * 0.65)


def test_confident_fast_timeframes_can_outvote_an_unsure_slow_one():
    # 1d 0.5*0.3 = 0.15 Bullish | 4h 0.3*0.9 = 0.27 + 1h 0.2*0.9 = 0.18 -> 0.45 Bearish
    result = alignment.compute_alignment(
        make_grid(["Bullish", "Bearish", "Bearish"], [0.3, 0.9, 0.9]), INTERVALS, WEIGHTS)
    assert result["dominant"].iloc[0] == "Bearish"


def test_unfinished_candle_is_never_used(tmp_path, monkeypatch):
    """At 12:00 on day 2 the day-2 Daily candle is still forming, so the grid
    must still show day 1's Daily regime."""
    monkeypatch.setattr(config, "REGIMES_DIR", tmp_path)
    day = pd.Timedelta(days=1)
    hour = pd.Timedelta(hours=1)
    almost = pd.Timedelta(milliseconds=1)
    start = pd.Timestamp("2024-01-01")

    pd.DataFrame({
        "close_time": [start + day - almost, start + 2 * day - almost],
        "regime": ["Bullish", "Bearish"], "confidence": [0.9, 0.9],
    }).to_parquet(tmp_path / "TEST_1d_regimes.parquet")
    hourly_close = [start + (h + 1) * hour - almost for h in range(48)]
    pd.DataFrame({
        "close_time": hourly_close, "regime": "Sideways", "confidence": 0.8,
    }).to_parquet(tmp_path / "TEST_1h_regimes.parquet")

    grid = alignment.load_regime_grid("TEST", ["1d", "1h"]).set_index("time")
    assert grid.index[0] == start + day - almost          # nothing before day 1 closes
    assert grid.loc[start + day + 12 * hour - almost, "regime_1d"] == "Bullish"
    assert grid.loc[start + 2 * day - almost, "regime_1d"] == "Bearish"
