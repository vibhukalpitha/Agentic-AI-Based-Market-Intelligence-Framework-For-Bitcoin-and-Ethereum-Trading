import os
import numpy as np
import pandas as pd

# Define paths
base_dir = r"e:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha"
ext_path = os.path.join(base_dir, "External Cryptodata.csv")
int_path = os.path.join(base_dir, "Internal Cryptodata.csv")
out_path = os.path.join(base_dir, "Master_Cleaned_Feature_Store.csv")

print("=========================================================")
print("STEP 1: DATA ALIGNMENT & CLEANING")
print("=========================================================")

# 1. Load both CSV files
print("Loading CSV files...")
ext_df = pd.read_csv(ext_path)
int_df = pd.read_csv(int_path)

# Convert datetime columns to timezone-aware UTC datetime64[ns, UTC]
print("Standardizing datetime columns to UTC...")
ext_df['datetime'] = pd.to_datetime(ext_df['datetime'], utc=True)
int_df['datetime'] = pd.to_datetime(int_df['datetime'], utc=True)

# Sort chronologically
ext_df = ext_df.sort_values('datetime').reset_index(drop=True)
int_df = int_df.sort_values('datetime').reset_index(drop=True)

# For External: Forward-fill macro indicators
macro_cols = [
    'GLD_Open', 'GLD_High', 'GLD_Low', 'GLD_Close', 'GLD_Volume',
    'DXY_open', 'DXY_high', 'DXY_low', 'DXY_close'
]
print("Forward-filling macro indicators in External dataset...")
ext_df[macro_cols] = ext_df[macro_cols].ffill()

# For Internal: Set datetime as temporary index to interpolate method='time'
int_df = int_df.set_index('datetime')

# Interpolate the isolated missing hours in BTC OHLCV columns (limit=3)
btc_ohlcv = ['BTC_open', 'BTC_high', 'BTC_low', 'BTC_close', 'BTC_volume']
print("Interpolating isolated BTC OHLCV missing hours in Internal dataset (time-based, limit=3)...")
int_df[btc_ohlcv] = int_df[btc_ohlcv].interpolate(method='time', limit=3)

# Forward-fill derivatives sentiment ratios with strict limit=4, then fill remaining NaNs with 0
sentiment_cols = [
    'sum_toptrader_long_short_ratio',
    'count_toptrader_long_short_ratio',
    'count_long_short_ratio',
    'sum_taker_long_short_vol_ratio'
]
print("Forward-filling derivatives sentiment ratios (limit=4) and filling remaining NaNs with 0...")
int_df[sentiment_cols] = int_df[sentiment_cols].ffill(limit=4).fillna(0)

# Reset index on internal dataset
int_df = int_df.reset_index()

# Merge cleaned Macro indicators from External into Internal dataset using inner join on datetime
print("Merging cleaned Macro indicators into Internal dataset...")
macro_subset = ext_df[['datetime'] + macro_cols]
merged_df = pd.merge(int_df, macro_subset, on='datetime', how='inner')

# Drop all rows after 2025-07-12 23:00:00 where ETH data is missing
cutoff_dt = pd.to_datetime("2025-07-12 23:00:00", utc=True)
print(f"Filtering dataset to retain valid dual-asset period up to {cutoff_dt}...")
df_aligned = merged_df[merged_df['datetime'] <= cutoff_dt].copy()
df_aligned = df_aligned.sort_values('datetime').reset_index(drop=True)

# Also ensure ETH OHLC is complete in this retained range
eth_ohlcv = ['ETH_open', 'ETH_high', 'ETH_low', 'ETH_close', 'ETH_volume']
# Set datetime index temporarily for time-based interpolation of any isolated 1-3h gap in ETH
df_aligned = df_aligned.set_index('datetime')
df_aligned[eth_ohlcv] = df_aligned[eth_ohlcv].interpolate(method='time', limit=3)
df_aligned = df_aligned.reset_index()

print(f"Aligned dataset shape after cleaning and cutoff: {df_aligned.shape}")

print("\n=========================================================")
print("STEP 2: FEATURE ENGINEERING")
print("=========================================================")

# 1. Logarithmic Returns: r_t = ln(P_t / P_{t-1})
print("Calculating hourly Logarithmic Returns for BTC, ETH, GLD, and DXY...")
df_aligned['BTC_log_return'] = np.log(df_aligned['BTC_close'] / df_aligned['BTC_close'].shift(1))
df_aligned['ETH_log_return'] = np.log(df_aligned['ETH_close'] / df_aligned['ETH_close'].shift(1))
df_aligned['GLD_log_return'] = np.log(df_aligned['GLD_Close'] / df_aligned['GLD_Close'].shift(1))
df_aligned['DXY_log_return'] = np.log(df_aligned['DXY_close'] / df_aligned['DXY_close'].shift(1))

# 2. Parkinson Volatility (24h Rolling):
# Formula: sqrt( 1 / (4 * ln(2) * N) * sum( (ln(High / Low))^2 ) )
print("Calculating Parkinson Volatility (24h rolling) for BTC and ETH...")
def calc_parkinson_vol(df, prefix, window=24):
    h = df[f'{prefix}_high']
    l = df[f'{prefix}_low']
    # Avoid zero division or log(1) edge cases
    ratio = np.where((h > 0) & (l > 0), h / l, 1.0)
    log_hl_sq = (np.log(ratio)) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    rolling_mean_sq = pd.Series(log_hl_sq, index=df.index).rolling(window=window, min_periods=window).mean()
    return np.sqrt(factor * rolling_mean_sq)

df_aligned['BTC_parkinson_vol_24h'] = calc_parkinson_vol(df_aligned, 'BTC', window=24)
df_aligned['ETH_parkinson_vol_24h'] = calc_parkinson_vol(df_aligned, 'ETH', window=24)

# 3. Garman-Klass Volatility (24h Rolling):
# Formula: sqrt( (1 / N) * sum( 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2 ) )
print("Calculating Garman-Klass Volatility (24h rolling) for BTC and ETH...")
def calc_garman_klass_vol(df, prefix, window=24):
    h = df[f'{prefix}_high']
    l = df[f'{prefix}_low']
    c = df[f'{prefix}_close']
    o = df[f'{prefix}_open']
    
    ratio_hl = np.where((h > 0) & (l > 0), h / l, 1.0)
    ratio_co = np.where((c > 0) & (o > 0), c / o, 1.0)
    
    term1 = 0.5 * (np.log(ratio_hl) ** 2)
    term2 = (2.0 * np.log(2.0) - 1.0) * (np.log(ratio_co) ** 2)
    gk_component = term1 - term2
    
    # Clip small negative numerical artifacts if any to zero
    gk_component = np.maximum(gk_component, 0.0)
    rolling_gk = pd.Series(gk_component, index=df.index).rolling(window=window, min_periods=window).mean()
    return np.sqrt(rolling_gk)

df_aligned['BTC_garman_klass_vol_24h'] = calc_garman_klass_vol(df_aligned, 'BTC', window=24)
df_aligned['ETH_garman_klass_vol_24h'] = calc_garman_klass_vol(df_aligned, 'ETH', window=24)

# 4. Taker Order Flow Imbalance (OFI):
# Formula: (TakerBuyQuote - (QuoteVolume - TakerBuyQuote)) / QuoteVolume
# Simplified: (2 * TakerBuyQuote - QuoteVolume) / QuoteVolume
print("Calculating Taker Order Flow Imbalance (OFI) for BTC and ETH...")
def calc_ofi(df, prefix):
    tbq = df[f'{prefix}_taker_buy_quote']
    qv = df[f'{prefix}_quote_volume']
    # Safe division avoiding divide by zero
    ofi = np.where(qv > 0, (2.0 * tbq - qv) / qv, 0.0)
    return ofi

df_aligned['BTC_taker_ofi'] = calc_ofi(df_aligned, 'BTC')
df_aligned['ETH_taker_ofi'] = calc_ofi(df_aligned, 'ETH')

# 5. Normalized Open Interest Delta (24h):
# Formula: Z-score of sum_open_interest using a 24-hour rolling mean and rolling standard deviation
print("Calculating Normalized Open Interest Delta (24h Z-score)...")
oi_rolling_mean = df_aligned['sum_open_interest'].rolling(window=24, min_periods=24).mean()
oi_rolling_std = df_aligned['sum_open_interest'].rolling(window=24, min_periods=24).std()
df_aligned['open_interest_zscore_24h'] = np.where(
    oi_rolling_std > 0,
    (df_aligned['sum_open_interest'] - oi_rolling_mean) / oi_rolling_std,
    0.0
)
# Retain NaN where rolling window has not completed 24 periods
df_aligned.loc[df_aligned.index < 23, 'open_interest_zscore_24h'] = np.nan

# 6. Cyclical Calendar Features:
# Continuous diurnal transformations: hour_sin = sin(2*pi*hour/24), hour_cos = cos(2*pi*hour/24)
print("Calculating continuous cyclical calendar features (hour_sin, hour_cos)...")
hours = df_aligned['datetime'].dt.hour
df_aligned['hour_sin'] = np.sin(2.0 * np.pi * hours / 24.0)
df_aligned['hour_cos'] = np.cos(2.0 * np.pi * hours / 24.0)

print("\n=========================================================")
print("STEP 3: OUTPUT GENERATION & VALIDATION")
print("=========================================================")

print(f"Saving final dataset to {out_path}...")
df_aligned.to_csv(out_path, index=False)
print("Saved successfully!")

print("\n--- Final Dataset Summary ---")
print(f"Final Shape: {df_aligned.shape}")
print(f"Date Range: {df_aligned['datetime'].min()} to {df_aligned['datetime'].max()}")
print(f"Total Columns ({len(df_aligned.columns)}):")
for idx, col in enumerate(df_aligned.columns, start=1):
    print(f"  {idx:2d}. {col}")

print("\n--- Missing Values Check ---")
missing_check = df_aligned.isnull().sum()
cols_with_na = missing_check[missing_check > 0]
if len(cols_with_na) == 0:
    print("Zero missing values across the entire dataset!")
else:
    print("Remaining NaN values (expected rolling initialization windows only):")
    for col, count in cols_with_na.items():
        print(f"  {col}: {count} NaNs ({count/len(df_aligned)*100:.3f}%)")

print("\n--- Head 3 Rows Sample of Generated Features ---")
feature_cols = [
    'datetime', 'BTC_close', 'ETH_close', 'BTC_log_return', 'ETH_log_return',
    'BTC_parkinson_vol_24h', 'ETH_garman_klass_vol_24h', 'BTC_taker_ofi',
    'open_interest_zscore_24h', 'hour_sin', 'hour_cos'
]
print(df_aligned[feature_cols].tail(3).to_string())
