import pandas as pd
import os

# File paths setup
raw_path = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\data\01_raw\Internal_Crypto_Data.csv"
processed_dir = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\data\02_processed"
out_path = os.path.join(processed_dir, "internal_cleaned_1h.csv")

os.makedirs(processed_dir, exist_ok=True)

print("Loading raw internal dataset...")
df = pd.read_csv(raw_path)

# 1. Standardize Datetime
print("Standardizing datetime...")
df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
df = df.sort_values('datetime').reset_index(drop=True)

# 2. Handle Crypto Micro-Gaps (Price data only, ignoring 'time' columns)
print("Interpolating small crypto gaps...")
# Filter only columns that have BTC_ or ETH_ but do NOT have 'time' in their name
crypto_cols = [c for c in df.columns if ('BTC_' in c or 'ETH_' in c) and ('time' not in c)]

for col in crypto_cols:
    if col in df.columns:
        # Convert to numeric first to remove any string/text artifacts
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].interpolate(method='linear', limit=3)

# 3. Handle Derivatives & Whale Metrics
print("Cleaning derivatives and sentiment metrics...")
whale_cols = ['sum_toptrader_long_short_ratio', 'count_toptrader_long_short_ratio',
              'count_long_short_ratio', 'sum_taker_long_short_vol_ratio']

for col in whale_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].ffill(limit=4).fillna(0)
        
# Handle open interest gaps
oi_cols = ['sum_open_interest', 'sum_open_interest_value']
for col in oi_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].ffill(limit=4).fillna(0)

# 4. Handle Truncated ETH Data
print("Removing unrecoverable ETH missing periods...")
if 'ETH_close' in df.columns:
    df_cleaned = df.dropna(subset=['ETH_close'])
else:
    df_cleaned = df

# 5. Save Processed Dataset
df_cleaned.to_csv(out_path, index=False)

print("\n--- Internal Preprocessing Complete ---")
print(f"Original Shape: {df.shape}")
print(f"Cleaned Shape: {df_cleaned.shape}")
print(f"Saved to: {out_path}")