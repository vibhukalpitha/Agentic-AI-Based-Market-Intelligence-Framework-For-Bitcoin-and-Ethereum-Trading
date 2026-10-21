import pandas as pd
import numpy as np
import os

# File paths
finbert_news_path = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\data\02_processed\news_finbert_sentiment.csv"
features_dir = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\data\03_features"
out_path = os.path.join(features_dir, "news_feature_store_4h.csv") # Overwriting the old one

# Price path
price_path_1 = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\data\01_raw\BTC_ETH_GLD_DXY_1h.csv"
price_path_2 = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\BTC_ETH_GLD_DXY_1h.csv"
price_raw_path = price_path_1 if os.path.exists(price_path_1) else price_path_2

print("Loading FinBERT News dataset...")
df_news = pd.read_csv(finbert_news_path)

# Map FinBERT string labels to numbers
sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
df_news['sentiment_score'] = df_news['finbert_sentiment'].map(sentiment_map).fillna(0)

# The sentiment confidence is the 'finbert_score' now
df_news['sentiment_confidence'] = df_news['finbert_score']

time_col = 'time_interval' if 'time_interval' in df_news.columns else df_news.columns[0]
df_news['datetime'] = pd.to_datetime(df_news[time_col], utc=True).dt.round('h')
df_news = df_news.drop(columns=[time_col, 'finbert_sentiment', 'finbert_score'], errors='ignore')

print("Merging with Crypto Prices...")
df_price = pd.read_csv(price_raw_path)
price_time_col = 'datetime' if 'datetime' in df_price.columns else df_price.columns[0]
df_price['datetime'] = pd.to_datetime(df_price[price_time_col], utc=True).dt.round('h')

price_cols = ['datetime'] + [c for c in df_price.columns if 'BTC_' in c or 'ETH_' in c]
df_price = df_price[price_cols].drop_duplicates(subset=['datetime'])

df_merged = pd.merge(df_news, df_price, on='datetime', how='inner').sort_values('datetime').reset_index(drop=True)
df_merged.set_index('datetime', inplace=True)

print("Calculating FinBERT News Features...")
# Rolling Features
df_merged['sentiment_rolling_24h'] = df_merged['sentiment_score'].rolling(window=8).mean()
df_merged['sentiment_momentum'] = df_merged['sentiment_score'] - df_merged['sentiment_rolling_24h']
df_merged['weighted_sentiment'] = df_merged['sentiment_score'] * df_merged['sentiment_confidence']

print("Generating Target Variables...")
for asset in ['BTC', 'ETH']:
    if f'{asset}_close' in df_merged.columns:
        future_price = df_merged[f'{asset}_close'].shift(-1)
        current_price = df_merged[f'{asset}_close']
        df_merged[f'{asset}_target_4h'] = (future_price > current_price).astype(int)

df_final = df_merged.dropna()
df_final.to_csv(out_path)

print("\n--- FinBERT Feature Engineering Complete ---")
print(f"Final Shape: {df_final.shape}")
print(f"Saved to: {out_path}")