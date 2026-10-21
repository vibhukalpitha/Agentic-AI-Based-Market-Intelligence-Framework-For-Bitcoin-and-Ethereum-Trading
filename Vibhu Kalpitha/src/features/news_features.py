import pandas as pd
import numpy as np
import os

# File paths
processed_path = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\data\02_processed\news_cleaned_3h.csv"
features_dir = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\data\03_features"
out_path = os.path.join(features_dir, "news_feature_store_4h.csv") # Naming it 4h for consistency, even though it's 3h data

os.makedirs(features_dir, exist_ok=True)
print("Loading processed News dataset...")
df = pd.read_csv(processed_path)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

print("Calculating News Features...")
# 1. Rolling Sentiment Momentum (24h = 8 periods of 3h)
if 'sentiment_score' in df.columns:
    df['sentiment_rolling_24h'] = df['sentiment_score'].rolling(window=8).mean()
    df['sentiment_momentum'] = df['sentiment_score'] - df['sentiment_rolling_24h']

# 2. Confidence weighted sentiment
if 'sentiment_confidence' in df.columns and 'sentiment_score' in df.columns:
    df['weighted_sentiment'] = df['sentiment_score'] * df['sentiment_confidence']

# 3. Target Variable Generation (Predicting 1 step ahead - roughly 3-4 hours)
print("Generating Target Variables...")
for asset in ['BTC', 'ETH']:
    if f'{asset}_close' in df.columns:
        future_price = df[f'{asset}_close'].shift(-1) # Next 3h period
        current_price = df[f'{asset}_close']
        df[f'{asset}_target_4h'] = (future_price > current_price).astype(int)

# Drop NaNs created by rolling windows and shifts
df_features = df.dropna()

df_features.to_csv(out_path)
print("\n--- News Feature Engineering Complete ---")
print(f"Final Shape: {df_features.shape}")
print(f"Saved to: {out_path}")