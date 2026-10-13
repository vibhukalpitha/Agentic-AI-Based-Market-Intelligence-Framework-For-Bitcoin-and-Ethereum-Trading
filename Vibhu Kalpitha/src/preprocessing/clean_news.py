import pandas as pd
import os

# File paths
news_raw_path = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\data\01_raw\bitcoin_news_3hour_intervals_with_sentiment.csv"
processed_dir = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\data\02_processed"
out_path = os.path.join(processed_dir, "news_cleaned_3h.csv")

# Look for the price file in likely locations
price_path_1 = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\data\01_raw\BTC_ETH_GLD_DXY_1h.csv"
price_path_2 = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\BTC_ETH_GLD_DXY_1h.csv"
price_raw_path = price_path_1 if os.path.exists(price_path_1) else price_path_2

os.makedirs(processed_dir, exist_ok=True)
print("Loading News dataset...")
df_news = pd.read_csv(news_raw_path, encoding='utf-8-sig')
df_news.columns = df_news.columns.str.strip()

print("Standardizing datetime...")
time_col = 'time_interval' if 'time_interval' in df_news.columns else df_news.columns[0]

# Force round to nearest hour to ensure perfect match
df_news['datetime'] = pd.to_datetime(df_news[time_col], utc=True).dt.round('h')
df_news.drop(columns=[time_col], inplace=True)
df_news = df_news.sort_values('datetime').reset_index(drop=True)

print("Dropping text columns...")
cols_to_drop = ['combined_article_text', 'summary']
df_news.drop(columns=[c for c in cols_to_drop if c in df_news.columns], inplace=True)

print("Encoding sentiment to numerical values...")
sentiment_map = {'Bullish': 1, 'Neutral': 0, 'Bearish': -1, 'bullish': 1, 'neutral': 0, 'bearish': -1}
if 'sentiment' in df_news.columns:
    df_news['sentiment_score'] = df_news['sentiment'].map(sentiment_map).fillna(0)
    df_news.drop(columns=['sentiment'], inplace=True)

print("Merging with Crypto Prices...")
try:
    df_price = pd.read_csv(price_raw_path, encoding='utf-8-sig')
    df_price.columns = df_price.columns.str.strip()
    price_time_col = 'datetime' if 'datetime' in df_price.columns else df_price.columns[0]
    
    # Force round to nearest hour here too
    df_price['datetime'] = pd.to_datetime(df_price[price_time_col], utc=True).dt.round('h')
    
    price_cols = ['datetime'] + [c for c in df_price.columns if 'BTC_' in c or 'ETH_' in c]
    df_price = df_price[price_cols]
    
    # Drop any duplicated hours in price data
    df_price = df_price.drop_duplicates(subset=['datetime'])
    
    # Inner join to safely keep only matching rows
    df_merged = pd.merge(df_news, df_price, on='datetime', how='inner')
    
except Exception as e:
    print(f"Error loading prices: {e}. Saving without prices.")
    df_merged = df_news

df_merged = df_merged.dropna()

df_merged.to_csv(out_path, index=False)
print("\n--- News Preprocessing Complete ---")
print(f"Original News Shape: {df_news.shape}")
print(f"Merged & Cleaned Shape: {df_merged.shape}")
print(f"Saved to: {out_path}")