import pandas as pd
import numpy as np
import os

# File paths
processed_path = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\data\02_processed\sp500_cleaned_1h.csv"
features_dir = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\data\03_features"
out_path = os.path.join(features_dir, "sp500_feature_store_4h.csv")

os.makedirs(features_dir, exist_ok=True)
print("Loading processed SP500 dataset...")
df = pd.read_csv(processed_path)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

print("Calculating SP500 Features...")
# 1. SP500 Log Returns
df['SP500_log_return'] = np.log(df['SP500_Close'] / df['SP500_Close'].shift(1))

# 2. SP500 Volatility (24h Rolling Std Dev)
df['SP500_volatility_24h'] = df['SP500_log_return'].rolling(window=24).std()

# 3. Target Variable Generation (Predicting 4 Hours Ahead for Crypto)
print("Generating 4h Target Variables...")
for asset in ['BTC', 'ETH']:
    if f'{asset}_close' in df.columns:
        future_price = df[f'{asset}_close'].shift(-4)
        current_price = df[f'{asset}_close']
        df[f'{asset}_target_4h'] = (future_price > current_price).astype(int)

# Drop NaNs created by rolling windows and shifts
df_features = df.dropna()

df_features.to_csv(out_path)
print("\n--- SP500 Feature Engineering Complete ---")
print(f"Final Shape: {df_features.shape}")
print(f"Saved to: {out_path}")