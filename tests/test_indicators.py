"""Tests for M2. Run:  python -m pytest

Each test protects one claim that will be made in the report or the viva.
"""
import numpy as np
import pandas as pd

from src import config
from src.features import indicators


def make_candles(rows: int = 600, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Synthetic random-walk candles (no real data needed for unit tests)."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, rows)))
    open_ = np.concatenate(([100.0], close[:-1]))
    spread = np.abs(rng.normal(0, 0.004, rows)) * close
    open_time = pd.date_range("2024-01-01", periods=rows, freq="1h")
    return pd.DataFrame({
        "open_time": open_time,
        "open": open_,
        "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread,
        "close": close,
        "close_time": open_time + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
    })


def test_wilder_smooth_matches_hand_calculation():
    # period 3: seed = mean(1, 2, 3) = 2; then 2 + (6 - 2)/3 = 3.3333; then + (9 - 3.3333)/3
    result = indicators.wilder_smooth(np.array([1.0, 2.0, 3.0, 6.0, 9.0]), 3)
    assert np.isnan(result[:2]).all()
    np.testing.assert_allclose(result[2:], [2.0, 10 / 3, 10 / 3 + (9 - 10 / 3) / 3])


def test_ema_matches_hand_calculation():
    # period 3: alpha = 0.5; seed = 2; then 2 + 0.5 * (6 - 2) = 4
    result = indicators.ema(np.array([1.0, 2.0, 3.0, 6.0]), 3)
    np.testing.assert_allclose(result[2:], [2.0, 4.0])


def test_rsi_extremes():
    rising = np.arange(1.0, 60.0)
    assert indicators.rsi(rising, 14)[-1] == 100.0          # only gains
    assert indicators.rsi(rising[::-1].copy(), 14)[-1] == 0.0  # only losses
    assert indicators.rsi(np.full(60, 5.0), 14)[-1] == 50.0    # flat = neutral


def test_flat_market_produces_no_nan_or_inf():
    """Exchange-outage candles (open = high = low = close) must not break anything."""
    candles = make_candles()
    for column in ("open", "high", "low", "close"):
        candles[column] = 100.0
    features = indicators.compute_features(candles).iloc[config.WARMUP_ROWS:]
    assert np.isfinite(features[config.FEATURE_COLUMNS].to_numpy()).all()


def test_features_are_scale_free():
    """The central M2 claim: multiplying every price by 1000 changes nothing.
    If this fails, the HMM could learn the price era instead of the regime."""
    candles = make_candles()
    scaled = candles.copy()
    scaled[["open", "high", "low", "close"]] *= 1000.0
    a = indicators.compute_features(candles)[config.FEATURE_COLUMNS]
    b = indicators.compute_features(scaled)[config.FEATURE_COLUMNS]
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-9, atol=1e-9)


def test_no_look_ahead():
    """Row t must depend on rows 0..t only: removing the future cannot change the past."""
    candles = make_candles()
    full = indicators.compute_features(candles)[config.FEATURE_COLUMNS].iloc[:400]
    truncated = indicators.compute_features(candles.iloc[:400])[config.FEATURE_COLUMNS]
    np.testing.assert_allclose(full.to_numpy(), truncated.to_numpy(), rtol=1e-12, atol=1e-12)


def test_warmup_is_long_enough():
    """No NaN may survive after WARMUP_ROWS."""
    features = indicators.compute_features(make_candles())
    assert not features[config.FEATURE_COLUMNS].iloc[config.WARMUP_ROWS:].isna().any().any()
