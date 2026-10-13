import pandas as pd
import numpy as np
import os

# File paths
processed_path = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\data\02_processed\macro_cleaned_1h.csv"
features_dir = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\data\03_features"
out_path = os.path.join(features_dir, "macro_feature_store_4h.csv")

os.makedirs(features_dir, exist_ok=True)
print("Loading processed dataset...")
df = pd.read_csv(processed_path)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# 1. Logarithmic Returns (Stationary Data for Models)
print("Calculating Log Returns...")
assets = ['BTC', 'ETH', 'GLD', 'DXY']
for asset in assets:
    # Handle DXY and GLD columns which have different capitalization in your dataset
    col_name = 'GLD_Close' if asset == 'GLD' else 'DXY_close' if asset == 'DXY' else f'{asset}_close'
    df[f'{asset}_log_return'] = np.log(df[col_name] / df[col_name].shift(1))

# 2. Volatility Features (24h Rolling Parkinson & Garman-Klass Volatility)
print("Calculating Volatility Features...")
for asset in ['BTC', 'ETH']:
    high = df[f'{asset}_high']
    low = df[f'{asset}_low']
    open_p = df[f'{asset}_open']
    close_p = df[f'{asset}_close']
    
    # Parkinson Volatility
    rs = np.log(high / low)**2
    df[f'{asset}_parkinson_vol_24h'] = np.sqrt(rs.rolling(window=24).mean() / (4 * np.log(2)))
    
    # Garman-Klass Volatility
    term1 = 0.5 * np.log(high / low)**2
    term2 = (2 * np.log(2) - 1) * np.log(close_p / open_p)**2
    df[f'{asset}_garman_klass_vol_24h'] = np.sqrt((term1 - term2).rolling(window=24).mean())

# 3. Cyclical Time Features (Model-friendly hours)
print("Extracting Cyclical Time Features...")
hours = df.index.hour
df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
df['hour_cos'] = np.cos(2 * np.pi * hours / 24)

# 4. Target Variable Generation (Predicting 4 Hours Ahead)
# 1 = Bullish (Price went up), 0 = Bearish (Price went down)
print("Generating 4h Target Variables...")
for asset in ['BTC', 'ETH']:
    future_price = df[f'{asset}_close'].shift(-4)
    current_price = df[f'{asset}_close']
    df[f'{asset}_target_4h'] = (future_price > current_price).astype(int)

# Drop initial NaN rows created by rolling windows and the final 4 rows created by shift(-4)
df_features = df.dropna()

df_features.to_csv(out_path)
print("\n--- Feature Engineering Complete ---")
print(f"Features created successfully!")
print(f"Final Shape: {df_features.shape}")
print(f"Saved to: {out_path}")