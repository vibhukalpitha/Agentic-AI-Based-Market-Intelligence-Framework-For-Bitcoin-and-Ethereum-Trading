"""M2 - Feature engineering: seven technical indicators, written out explicitly.

Run:  python -m src.features.indicators

WHY no indicator library: every formula here must be explainable line by line.

WHY everything is normalised: over this dataset BTC goes from about $4,000 to
about $126,000. An indicator measured in dollars (EMA, SMA, MACD, ATR, Bollinger)
would mostly tell the HMM *which year it is*. Each one is therefore converted to
a ratio of price, so the model can only learn market behaviour, not price level.

WHY there is no look-ahead: every value in row t uses rows 0..t only. This is
enforced by tests/test_indicators.py (truncating the future must not change
the past).
"""
import numpy as np
import pandas as pd

from src import config


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
def _recursive_smooth(values: np.ndarray, period: int, alpha: float) -> np.ndarray:
    """Recursive average: new = previous + alpha * (value - previous).

    The first output is the plain mean of the first `period` valid values (the
    "seed"), exactly as in the original definitions of EMA and Wilder smoothing.
    Rows before the seed are NaN. Written as a loop on purpose: it is the
    definition itself, with nothing hidden.
    """
    out = np.full(len(values), np.nan)
    valid = np.flatnonzero(~np.isnan(values))
    if len(valid) < period:
        return out
    start = valid[0]
    if np.isnan(values[start:]).any():
        raise ValueError("unexpected NaN inside the series")
    seed = start + period - 1
    out[seed] = values[start:seed + 1].mean()
    for i in range(seed + 1, len(values)):
        out[i] = out[i - 1] + alpha * (values[i] - out[i - 1])
    return out


def ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average, alpha = 2 / (period + 1)."""
    return _recursive_smooth(values, period, 2.0 / (period + 1))


def wilder_smooth(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing, alpha = 1 / period.

    Used by RSI, ATR and ADX. It is slower than an EMA of the same period:
    each new value only moves the average by 1/period of the difference.
    """
    return _recursive_smooth(values, period, 1.0 / period)


def sma(values: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average of the last `period` values."""
    return pd.Series(values).rolling(period).mean().to_numpy()


def _safe_divide(num: np.ndarray, den: np.ndarray, fill: float) -> np.ndarray:
    """num / den, with `fill` where den is exactly zero.

    Zero denominators are real in this data: during exchange downtime Binance
    reports flat candles (open = high = low = close, zero volume). NaN inputs
    stay NaN so warm-up rows are still recognisable.
    """
    out = np.full(len(num), fill, dtype=float)
    ok = den != 0
    out[ok] = num[ok] / den[ok]
    out[np.isnan(num) | np.isnan(den)] = np.nan
    return out


# --------------------------------------------------------------------------
# The indicators
# --------------------------------------------------------------------------
def rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Relative Strength Index (0-100). Bounded already, so used directly.

    RSI = 100 * avg_gain / (avg_gain + avg_loss), which is algebraically the
    same as the textbook 100 - 100 / (1 + avg_gain / avg_loss) but has no
    division by zero when there are no losses. A perfectly flat market
    (no gains, no losses) is neutral: 50.
    """
    change = np.diff(close, prepend=np.nan)
    gain = np.where(change > 0, change, 0.0)
    loss = np.where(change < 0, -change, 0.0)
    gain[0] = loss[0] = np.nan          # no previous close for the first row
    avg_gain = wilder_smooth(gain, period)
    avg_loss = wilder_smooth(loss, period)
    return _safe_divide(100.0 * avg_gain, avg_gain + avg_loss, fill=50.0)


def macd_histogram(close: np.ndarray, fast: int, slow: int, signal: int) -> np.ndarray:
    """MACD histogram in price units (normalised by the caller).

    macd line = EMA(fast) - EMA(slow); signal = EMA of the macd line;
    histogram = macd line - signal. Positive = upward momentum is increasing.
    """
    macd_line = ema(close, fast) - ema(close, slow)
    return macd_line - ema(macd_line, signal)


def bollinger(close: np.ndarray, period: int, num_std: float) -> tuple[np.ndarray, np.ndarray]:
    """Bollinger Bands as two scale-free numbers: (band width, %B).

    width = (upper - lower) / middle   -> how wide the bands are, relative to price
    %B    = (close - lower) / (upper - lower) -> where price sits inside the bands
            (0 = lower band, 1 = upper band, can go outside 0..1)
    Population standard deviation (ddof=0), as in Bollinger's definition.
    """
    mid = sma(close, period)
    std = pd.Series(close).rolling(period).std(ddof=0).to_numpy()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = _safe_divide(upper - lower, mid, fill=0.0)
    pctb = _safe_divide(close - lower, upper - lower, fill=0.5)  # flat market = middle
    return width, pctb


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """True range: the candle's range, extended to include any gap from the
    previous close. First row is NaN (no previous close)."""
    prev_close = np.concatenate(([np.nan], close[:-1]))
    return np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average True Range in price units (normalised by the caller)."""
    return wilder_smooth(true_range(high, low, close), period)


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average Directional Index (0-100): trend STRENGTH, not direction.

    1. +DM / -DM: how far today's high went above yesterday's high, or today's
       low below yesterday's low. Only the larger of the two counts.
    2. +DI / -DI: smoothed DM as a percentage of ATR.
    3. DX = 100 * |+DI - -DI| / (+DI + -DI): how one-sided the movement is.
    4. ADX = Wilder-smoothed DX.
    Already a ratio, so bounded 0-100 and used directly.
    """
    up = np.diff(high, prepend=np.nan)
    down = -np.diff(low, prepend=np.nan)
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm[0] = minus_dm[0] = np.nan

    atr_values = atr(high, low, close, period)
    plus_di = _safe_divide(100.0 * wilder_smooth(plus_dm, period), atr_values, fill=0.0)
    minus_di = _safe_divide(100.0 * wilder_smooth(minus_dm, period), atr_values, fill=0.0)
    dx = _safe_divide(100.0 * np.abs(plus_di - minus_di), plus_di + minus_di, fill=0.0)
    return wilder_smooth(dx, period)


# --------------------------------------------------------------------------
# Feature table
# --------------------------------------------------------------------------
def compute_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Return META_COLUMNS + the nine FEATURE_COLUMNS, same length as the input.

    Warm-up rows are still present here (as NaN); build_features removes them.
    """
    high = candles["high"].to_numpy(dtype=float)
    low = candles["low"].to_numpy(dtype=float)
    close = candles["close"].to_numpy(dtype=float)

    ema_values = ema(close, config.EMA_PERIOD)
    sma_values = sma(close, config.SMA_PERIOD)
    bb_width, bb_pctb = bollinger(close, config.BB_PERIOD, config.BB_NUM_STD)

    out = candles[config.META_COLUMNS].copy()
    out["log_return"] = np.log(close / np.concatenate(([np.nan], close[:-1])))
    out["rsi"] = rsi(close, config.RSI_PERIOD)
    out["macd_hist_norm"] = macd_histogram(
        close, config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL) / close
    out["ema_dist"] = (close - ema_values) / ema_values
    out["sma_dist"] = (close - sma_values) / sma_values
    out["bb_width"] = bb_width
    out["bb_pctb"] = bb_pctb
    out["atr_norm"] = atr(high, low, close, config.ATR_PERIOD) / close
    out["adx"] = adx(high, low, close, config.ADX_PERIOD)
    return out


def build_features(symbol: str, interval: str) -> pd.DataFrame:
    """Load one raw file, compute features, drop warm-up, check, save, report."""
    candles = pd.read_parquet(config.PARQUET_DIR / f"{symbol}_{interval}.parquet")
    features = compute_features(candles).iloc[config.WARMUP_ROWS:].reset_index(drop=True)

    values = features[config.FEATURE_COLUMNS]
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{symbol}_{interval}: NaN or infinite feature after warm-up")
    for column in ("ema_dist", "sma_dist"):
        worst = values[column].abs().max()
        if worst > config.DIST_HARD_LIMIT:
            raise ValueError(
                f"{symbol}_{interval}: |{column}| reaches {worst:.2f} - "
                "normalisation failed, raw price units are leaking through")

    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(config.FEATURES_DIR / f"{symbol}_{interval}_features.parquet",
                        index=False)
    _print_report(symbol, interval, len(candles), features)
    return features


def _print_report(symbol: str, interval: str, raw_rows: int, features: pd.DataFrame) -> None:
    """Sanity report. The last column is the price-era check: the correlation of
    each feature with the price level. A raw dollar indicator would be near
    +1.0 here; a correctly normalised one should be far from it."""
    values = features[config.FEATURE_COLUMNS]
    report = values.agg(["min", "max", "mean", "std"]).T
    report["p1"] = values.quantile(0.01)
    report["p99"] = values.quantile(0.99)
    report["corr_price"] = values.corrwith(features["close"])
    print(f"\n=== {symbol} {interval} ===")
    print(f"raw rows {raw_rows:,} | warm-up dropped {raw_rows - len(features)} | "
          f"feature rows {len(features):,} | "
          f"{features['open_time'].iloc[0]:%Y-%m-%d} -> {features['open_time'].iloc[-1]:%Y-%m-%d}")
    print(report[["min", "p1", "mean", "p99", "max", "std", "corr_price"]]
          .to_string(float_format=lambda x: f"{x:10.4f}"))


def main() -> None:
    # One file at a time: never hold all 2.1 million rows in memory together.
    for symbol in config.SYMBOLS:
        for interval in config.INTERVALS:
            build_features(symbol, interval)
    print("\nAll feature files written to", config.FEATURES_DIR)


if __name__ == "__main__":
    main()
