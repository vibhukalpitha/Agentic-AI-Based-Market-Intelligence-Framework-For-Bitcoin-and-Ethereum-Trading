import pandas as pd
import os

# File paths setup
raw_path = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\data\01_raw\BTC_ETH_GLD_DXY_1h.csv"
processed_dir = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\data\02_processed"
out_path = os.path.join(processed_dir, "macro_cleaned_1h.csv")

# Create processed directory if it doesn't exist
os.makedirs(processed_dir, exist_ok=True)

print("Loading raw dataset...")
df = pd.read_csv(raw_path)

# 1. Standardize Datetime
print("Standardizing datetime...")
df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
df = df.sort_values('datetime').reset_index(drop=True)

# 2. Handle Macro Indicators (GLD, DXY) 
# Forward Fill (ffill) to carry Friday closing prices over the weekend
print("Cleaning Macro indicators (GLD & DXY)...")
macro_cols = ['GLD_Open', 'GLD_High', 'GLD_Low', 'GLD_Close', 'GLD_Volume',
              'DXY_open', 'DXY_high', 'DXY_low', 'DXY_close']

for col in macro_cols:
    if col in df.columns:
        df[col] = df[col].ffill()

# 3. Handle Crypto Micro-Gaps 
# Interpolate small missing gaps (max 3 hours) for BTC and ETH
print("Interpolating small crypto gaps...")
crypto_cols = [c for c in df.columns if 'BTC_' in c or 'ETH_' in c]
for col in crypto_cols:
    df[col] = df[col].interpolate(method='linear', limit=3)

# 4. Handle Truncated ETH Data
# Drop rows where ETH data is completely missing (e.g., after 2025-07-12)
print("Removing unrecoverable ETH missing periods...")
df_cleaned = df.dropna(subset=['ETH_close'])

# 5. Save Processed Dataset
df_cleaned.to_csv(out_path, index=False)

print("\n--- Preprocessing Complete ---")
print(f"Original Shape: {df.shape}")
print(f"Cleaned Shape: {df_cleaned.shape}")
print(f"Saved to: {out_path}")