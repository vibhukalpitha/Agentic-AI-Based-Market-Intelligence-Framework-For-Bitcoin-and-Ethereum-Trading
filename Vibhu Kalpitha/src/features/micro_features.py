import pandas as pd
import numpy as np
import os

# File paths
processed_path = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\data\02_processed\internal_cleaned_1h.csv"
features_dir = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\data\03_features"
out_path = os.path.join(features_dir, "internal_feature_store_4h.csv")

os.makedirs(features_dir, exist_ok=True)
print("Loading processed internal dataset...")
df = pd.read_csv(processed_path)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

print("Calculating Microstructure Features...")

for asset in ['BTC', 'ETH']:
    # 1. Taker Order Flow Imbalance (OFI)
    # Market aggressively buy කරනවද sell කරනවද කියන එක
    if f'{asset}_taker_buy_quote' in df.columns and f'{asset}_quote_volume' in df.columns:
        taker_buy = df[f'{asset}_taker_buy_quote']
        total_vol = df[f'{asset}_quote_volume']
        # OFI = (Buy Volume - Sell Volume) / Total Volume
        # Sell Volume = Total Volume - Buy Volume
        df[f'{asset}_taker_ofi'] = (2 * taker_buy - total_vol) / total_vol.replace(0, np.nan)
        df[f'{asset}_taker_ofi'] = df[f'{asset}_taker_ofi'].fillna(0)

# 2. Normalized Open Interest Delta (24h Z-score)
# Leverage එක සාමාන්‍ය මට්ටමට වඩා කොච්චර වැඩිවෙලාද කියන එක
if 'sum_open_interest' in df.columns:
    rolling_mean = df['sum_open_interest'].rolling(window=24).mean()
    rolling_std = df['sum_open_interest'].rolling(window=24).std()
    df['open_interest_zscore_24h'] = (df['sum_open_interest'] - rolling_mean) / rolling_std.replace(0, np.nan)

# 3. Target Variable Generation (Predicting 4 Hours Ahead)
print("Generating 4h Target Variables...")
for asset in ['BTC', 'ETH']:
    if f'{asset}_close' in df.columns:
        future_price = df[f'{asset}_close'].shift(-4)
        current_price = df[f'{asset}_close']
        df[f'{asset}_target_4h'] = (future_price > current_price).astype(int)

# Drop initial NaN rows created by rolling windows and the final 4 rows created by shift(-4)
df_features = df.dropna()

df_features.to_csv(out_path)
print("\n--- Internal Feature Engineering Complete ---")
print(f"Final Shape: {df_features.shape}")
print(f"Saved to: {out_path}")