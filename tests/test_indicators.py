"""Tests for M2. Run:  python -m pytest

Each test protects one claim that will be made in the report or the viva.
"""
import numpy as np
import pandas as pd

from src import config
from src.features import indicators

VOL_WINDOW = 50                               # short baseline so tests stay small
WARMUP = config.WARMUP_ROWS + VOL_WINDOW


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


def features_of(candles: pd.DataFrame) -> pd.DataFrame:
    return indicators.compute_features(candles, VOL_WINDOW)[config.FEATURE_COLUMNS]


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


def test_relative_volatility_reads_zero_when_volatility_is_constant():
    """If volatility never changes, 'relative to the past year' must be 0 (= normal)."""
    result = indicators.relative_to_trailing_median(np.full(100, 0.04), 20)
    assert np.isnan(result[:19]).all()
    np.testing.assert_allclose(result[19:], 0.0)
    # a day at twice the usual level reads log(2)
    doubled = np.concatenate((np.full(99, 0.04), [0.08]))
    np.testing.assert_allclose(
        indicators.relative_to_trailing_median(doubled, 20)[-1], np.log(2))


def test_outage_candles_inside_real_data_stay_finite():
    """A block of flat candles (exchange downtime) inside a normal series must
    not produce NaN or infinity."""
    candles = make_candles()
    for column in ("open", "high", "low"):
        candles.loc[300:340, column] = candles.loc[300, "close"]
    candles.loc[300:340, "close"] = candles.loc[300, "close"]
    assert np.isfinite(features_of(candles).iloc[WARMUP:].to_numpy()).all()


def test_features_are_scale_free():
    """The central M2 claim: multiplying every price by 1000 changes nothing.
    If this fails, the HMM could learn the price era instead of the regime."""
    candles = make_candles()
    scaled = candles.copy()
    scaled[["open", "high", "low", "close"]] *= 1000.0
    np.testing.assert_allclose(features_of(candles).to_numpy(),
                               features_of(scaled).to_numpy(), rtol=1e-9, atol=1e-9)


def test_no_look_ahead():
    """Row t must depend on rows 0..t only: removing the future cannot change the past."""
    candles = make_candles()
    full = features_of(candles).iloc[:400]
    truncated = features_of(candles.iloc[:400])
    np.testing.assert_allclose(full.to_numpy(), truncated.to_numpy(), rtol=1e-12, atol=1e-12)


def test_warmup_is_long_enough():
    """No NaN may survive after the warm-up (indicator rows + volatility baseline)."""
    assert not features_of(make_candles()).iloc[WARMUP:].isna().any().any()
