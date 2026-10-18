import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Paths Setup
# ==========================================
BASE_PATH = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha"
PROCESSED_DIR = os.path.join(BASE_PATH, r"data\02_processed")
FEATURES_1H_DIR = os.path.join(BASE_PATH, r"data\03_features_1h")

os.makedirs(FEATURES_1H_DIR, exist_ok=True)

def find_time_col(df):
    for col in df.columns:
        if 'time' in col.lower() or 'date' in col.lower():
            return col
    return None

# ==========================================
# 2. Process News Sentiment (Convert to 1H)
# ==========================================
print("Processing News Sentiment to 1-Hour...")
news_path = os.path.join(PROCESSED_DIR, "news_finbert_sentiment.csv")
df_news = pd.read_csv(news_path, engine='python', on_bad_lines='skip', encoding_errors='ignore')

# 🚀 FIX: Force 'finbert_score' to be a Number
if 'finbert_score' in df_news.columns:
    df_news['finbert_score'] = pd.to_numeric(df_news['finbert_score'], errors='coerce')

t_col = find_time_col(df_news)
if t_col:
    df_news['datetime'] = pd.to_datetime(df_news[t_col], errors='coerce', utc=True).dt.floor('1h').dt.tz_localize(None)
    if t_col != 'datetime':
        df_news = df_news.drop(columns=[t_col])

# Group by 1-hour and calculate mean
num_cols = df_news.select_dtypes(include=[np.number]).columns.tolist()
if 'datetime' in df_news.columns:
    df_news_1h = df_news.groupby('datetime')[num_cols].mean().reset_index()
    df_news_1h.to_csv(os.path.join(FEATURES_1H_DIR, "news_feature_store_1h.csv"), index=False)
    print(f"✅ news_feature_store_1h.csv created! Features: {df_news_1h.shape[1]}")

# ==========================================
# 3. Process Macro 1H & Add Targets
# ==========================================
print("\nProcessing Macro 1H...")
macro_path = os.path.join(PROCESSED_DIR, "macro_cleaned_1h.csv")
df_macro = pd.read_csv(macro_path, engine='python', on_bad_lines='skip', encoding_errors='ignore')

t_col = find_time_col(df_macro)
if t_col:
    df_macro['datetime'] = pd.to_datetime(df_macro[t_col], errors='coerce', utc=True).dt.floor('1h').dt.tz_localize(None)
    if t_col != 'datetime':
        df_macro = df_macro.drop(columns=[t_col])

df_macro = df_macro.groupby('datetime').last().reset_index()

if 'ETH_close' in df_macro.columns:
    df_macro['ETH_target_1h'] = (df_macro['ETH_close'].shift(-1) > df_macro['ETH_close']).astype(int)
if 'BTC_close' in df_macro.columns:
    df_macro['BTC_target_1h'] = (df_macro['BTC_close'].shift(-1) > df_macro['BTC_close']).astype(int)

df_macro.to_csv(os.path.join(FEATURES_1H_DIR, "macro_feature_store_1h.csv"), index=False)
print("✅ macro_feature_store_1h.csv created!")

# ==========================================
# 4. Process Internal 1H
# ==========================================
print("\nProcessing Internal 1H...")
internal_path = os.path.join(PROCESSED_DIR, "internal_cleaned_1h.csv")
df_internal = pd.read_csv(internal_path, engine='python', on_bad_lines='skip', encoding_errors='ignore')

t_col = find_time_col(df_internal)
if t_col:
    df_internal['datetime'] = pd.to_datetime(df_internal[t_col], errors='coerce', utc=True).dt.floor('1h').dt.tz_localize(None)
    if t_col != 'datetime':
        df_internal = df_internal.drop(columns=[t_col])

df_internal = df_internal.groupby('datetime').last().reset_index()
df_internal.to_csv(os.path.join(FEATURES_1H_DIR, "internal_feature_store_1h.csv"), index=False)
print("✅ internal_feature_store_1h.csv created!")

# ==========================================
# 5. Process SP500 1H
# ==========================================
print("\nProcessing SP500 1H...")
sp500_path = os.path.join(PROCESSED_DIR, "sp500_cleaned_1h.csv")
df_sp500 = pd.read_csv(sp500_path, engine='python', on_bad_lines='skip', encoding_errors='ignore')

t_col = find_time_col(df_sp500)
if t_col:
    df_sp500['datetime'] = pd.to_datetime(df_sp500[t_col], errors='coerce', utc=True).dt.floor('1h').dt.tz_localize(None)
    if t_col != 'datetime':
        df_sp500 = df_sp500.drop(columns=[t_col])

df_sp500 = df_sp500.groupby('datetime').last().reset_index()
df_sp500.to_csv(os.path.join(FEATURES_1H_DIR, "sp500_feature_store_1h.csv"), index=False)
print("✅ sp500_feature_store_1h.csv created!")