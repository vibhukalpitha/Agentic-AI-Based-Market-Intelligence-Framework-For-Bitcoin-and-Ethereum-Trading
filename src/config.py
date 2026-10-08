"""Central configuration: every path and constant used by the pipeline.

WHY one file: an examiner must be able to see every number that influences the
results in a single place. No module may hard-code a period, path or threshold.
"""
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"            # git-ignored
PARQUET_DIR = DATA_DIR / "parquet"      # raw OHLCV candles (M1 output)
FEATURES_DIR = DATA_DIR / "features"    # engineered features (M2 output)
MODELS_DIR = ROOT_DIR / "models"        # git-ignored, trained HMMs
RESULTS_DIR = ROOT_DIR / "results"

# --------------------------------------------------------------------------
# Assets and timeframes
# --------------------------------------------------------------------------
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# Ordered slowest -> fastest. The order matters later for propagation analysis.
INTERVALS = ["1d", "4h", "1h", "15m", "5m"]

INTERVAL_STEP = {
    "1d": pd.Timedelta(days=1),
    "4h": pd.Timedelta(hours=4),
    "1h": pd.Timedelta(hours=1),
    "15m": pd.Timedelta(minutes=15),
    "5m": pd.Timedelta(minutes=5),
}

# --------------------------------------------------------------------------
# M1 - data acquisition (Binance Vision monthly kline archives)
# --------------------------------------------------------------------------
BINANCE_VISION_URL = (
    "https://data.binance.vision/data/spot/monthly/klines/"
    "{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"
)

# 15m and 5m start in 2020: early Binance intraday data is thin and less
# representative. This asymmetry is deliberate and is stated in the report.
DOWNLOAD_START = {
    "1d": "2017-08", "4h": "2017-08", "1h": "2017-08",
    "15m": "2020-01", "5m": "2020-01",
}
DOWNLOAD_END = "2026-09"  # last complete month in the frozen dataset

# Binance kline CSV columns, in file order. The 12th ("ignore") is dropped.
BINANCE_CSV_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "num_trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
RAW_COLUMNS = BINANCE_CSV_COLUMNS[:-1]

# Binance SPOT timestamps are milliseconds before 2025-01-01 and microseconds
# from 2025-01-01. Any epoch value above this threshold is microseconds
# (1e14 ms would be the year 5138, so the two ranges cannot overlap).
MICROSECOND_THRESHOLD = 1e14

# Row counts of the verified dataset (2017-08-17 / 2020-01-01 -> 2026-09-30).
# validate.py fails loudly if a file differs from this record.
EXPECTED_ROWS = {"1d": 3332, "4h": 19974, "1h": 79837, "15m": 236488, "5m": 709458}

# --------------------------------------------------------------------------
# M2 - indicator periods
# --------------------------------------------------------------------------
# The published textbook defaults, used identically at every timeframe.
# WHY not tuned: tuning periods per timeframe would add free parameters that
# could be fitted to the result. Standard values keep the features defensible.
RSI_PERIOD = 14         # Wilder (1978)
ATR_PERIOD = 14         # Wilder (1978)
ADX_PERIOD = 14         # Wilder (1978)
MACD_FAST = 12          # Appel's standard MACD
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20          # Bollinger's standard bands
BB_NUM_STD = 2.0
EMA_PERIOD = 20         # short-term trend reference
SMA_PERIOD = 50         # medium-term trend reference

# Rows removed from the start of every feature file.
# WHY 100: the longest hard requirement is SMA(50) = 49 rows, but EMA and
# Wilder averages are recursive and still carry their starting seed for a while.
# After 100 rows the seed's remaining weight is below 0.5% for every indicator
# (Wilder 14: (13/14)^86 = 0.2%; EMA 26: (25/27)^74 = 0.3%).
WARMUP_ROWS = 100

# Volatility baseline: the two volatility features are measured against their
# own median over the previous 365 days (past candles only).
# WHY: crypto volatility has fallen as the market matured. "ATR is 5% of price"
# was calm in 2018 and extreme in 2025, so a model trained on the absolute
# level stops recognising volatile periods in later years. Relative to the
# trailing year, "volatile" means the same thing in every era.
# WHY 365 days: long enough to be a stable reference through a whole market
# phase, short enough to follow the slow decline in volatility.
VOL_BASELINE_DAYS = 365

# The nine columns that are fed to the HMM. Nothing else may be used as input.
FEATURE_COLUMNS = [
    "log_return", "rsi", "macd_hist_norm", "ema_dist", "sma_dist",
    "bb_width_rel", "bb_pctb", "atr_rel", "adx",
]

# Kept in the feature files for bookkeeping only - NEVER model inputs.
# close_time is needed later so that a candle is only used after it has closed.
# atr_norm and bb_width are the plain scale-free values that the two relative
# volatility features are built from; kept so reports can show real units.
META_COLUMNS = ["open_time", "close_time", "close", "atr_norm", "bb_width"]

# Hard failure limit for |ema_dist| and |sma_dist|. A correctly normalised
# distance is a fraction of price; values beyond this mean raw price units
# leaked through (those would be in the thousands).
DIST_HARD_LIMIT = 2.0

# --------------------------------------------------------------------------
# M3 - regime detection (one GaussianHMM per asset and timeframe)
# --------------------------------------------------------------------------
REGIMES_DIR = DATA_DIR / "regimes"      # M3 output: regime label per candle

# The model is fitted on candles up to and including this date and then
# applied, unchanged, to everything after it (train on past, test on future).
# PROPOSED: 2022-12-31 gives training data that contains a full cycle
# (2017 peak, 2018 bear, 2020 crash, 2021 peak, 2022 bear) and leaves
# 2023-01 -> 2026-09 as untouched out-of-sample data.
TRAIN_END = "2022-12-31"

REGIMES = ["Bullish", "Bearish", "Sideways", "Volatile"]
HMM_N_STATES = 4                # PROPOSED: one state per regime
HMM_COVARIANCE_TYPE = "full"    # the nine features are strongly correlated,
                                # which a diagonal covariance cannot represent
HMM_MAX_ITER = 300              # Baum-Welch iterations (stops earlier on convergence)
HMM_TOLERANCE = 1e-4            # log-likelihood gain below which training stops

# Baum-Welch only finds a local optimum, so each model is fitted from several
# random starts and the highest-likelihood fit is kept. Seeds are
# RANDOM_SEED, RANDOM_SEED + 1, ... so every run is repeatable.
HMM_N_RESTARTS = 10

# State counts compared by BIC to check that 4 is a reasonable choice.
HMM_STATE_CANDIDATES = [2, 3, 4, 5, 6]

# OPEN decision: how many consecutive candles a new regime must hold before
# the change counts as a real transition. The count is reported for each of
# these so the decision can be made with the numbers in view.
PERSISTENCE_CANDIDATES = [1, 2, 3, 5, 7]

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED = 42
