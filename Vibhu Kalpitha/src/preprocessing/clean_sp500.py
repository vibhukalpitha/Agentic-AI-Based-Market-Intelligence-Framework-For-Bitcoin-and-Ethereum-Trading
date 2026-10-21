import pandas as pd
import os

# File paths setup
# Make sure the file name matches where you saved the merged SP500 file
raw_path = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\data\01_raw\SP500_with_BTC_ETH_2023_2025.csv"
processed_dir = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\data\02_processed"
out_path = os.path.join(processed_dir, "sp500_cleaned_1h.csv")

os.makedirs(processed_dir, exist_ok=True)
print("Loading raw SP500 dataset...")
df = pd.read_csv(raw_path)

# 1. Standardize Datetime
print("Standardizing datetime...")
df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
df = df.sort_values('Datetime').reset_index(drop=True)

# Standardize time column name for consistency with other datasets
df.rename(columns={'Datetime': 'datetime'}, inplace=True)

# 2. Fill Missing Crypto Values
# Just in case there are small gaps in the merged crypto columns
print("Handling any missing values...")
crypto_cols = [c for c in df.columns if 'BTC_' in c or 'ETH_' in c]
for col in crypto_cols:
    df[col] = df[col].interpolate(method='linear', limit=3)

# Drop any unrecoverable nulls to ensure a clean model input
df_cleaned = df.dropna()

# 3. Save Processed Dataset
df_cleaned.to_csv(out_path, index=False)
print("\n--- SP500 Preprocessing Complete ---")
print(f"Original Shape: {df.shape}")
print(f"Cleaned Shape: {df_cleaned.shape}")
print(f"Saved to: {out_path}")