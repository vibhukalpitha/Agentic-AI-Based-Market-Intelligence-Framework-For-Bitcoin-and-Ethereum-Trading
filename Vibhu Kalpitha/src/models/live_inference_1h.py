import os
import time
import joblib
import pandas as pd
import numpy as np
import ccxt
import yfinance as yf
import feedparser
from textblob import TextBlob
import schedule
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Local Paths Setup (1H Only)
# ==========================================
BASE_PATH = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha"
MODELS_1H_ETH_DIR = os.path.join(BASE_PATH, r"src\models\saved models\1_hr_ETH")
MODELS_1H_BTC_DIR = os.path.join(BASE_PATH, r"src\models\saved models\1_hr_BTC")

print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing 1-Hour AI Trading Framework...")

# ==========================================
# 2. Model Loading (1H)
# ==========================================
models = {'ETH': {}, 'BTC': {}}

try:
    # Load ETH 1H Models
    models['ETH']['macro'] = joblib.load(os.path.join(MODELS_1H_ETH_DIR, "eth_macro_best_model.joblib"))
    models['ETH']['internal'] = joblib.load(os.path.join(MODELS_1H_ETH_DIR, "eth_internal_best_model.joblib"))
    models['ETH']['sp500'] = joblib.load(os.path.join(MODELS_1H_ETH_DIR, "eth_sp500_best_model.joblib"))
    models['ETH']['news'] = joblib.load(os.path.join(MODELS_1H_ETH_DIR, "eth_news_best_model.joblib"))
    models['ETH']['meta'] = joblib.load(os.path.join(MODELS_1H_ETH_DIR, "eth_meta_learner_1h.joblib"))

    # Load BTC 1H Models
    models['BTC']['macro'] = joblib.load(os.path.join(MODELS_1H_BTC_DIR, "btc_macro_best_model.joblib"))
    models['BTC']['internal'] = joblib.load(os.path.join(MODELS_1H_BTC_DIR, "btc_internal_best_model.joblib"))
    models['BTC']['sp500'] = joblib.load(os.path.join(MODELS_1H_BTC_DIR, "btc_sp500_best_model.joblib"))
    models['BTC']['news'] = joblib.load(os.path.join(MODELS_1H_BTC_DIR, "btc_news_best_model.joblib"))
    models['BTC']['meta'] = joblib.load(os.path.join(MODELS_1H_BTC_DIR, "btc_meta_learner_1h.joblib"))
    
    print("✅ All 1-Hour AI Models Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading 1H models: {e}")
    exit()

# ==========================================
# 3. Free Data Fetchers (With Anti-Ban Sleep)
# ==========================================
def fetch_crypto_data(symbol, timeframe='1h', limit=5):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df.iloc[-1]

def fetch_sp500_live():
    try:
        sp500 = yf.download('^GSPC', period='1d', interval='1h', progress=False)
        return float(sp500['Close'].iloc[-1])
    except:
        return 0.0

def fetch_crypto_news_sentiment():
    try:
        url = "https://cointelegraph.com/rss"
        feed = feedparser.parse(url)
        sentiments = [TextBlob(entry.title).sentiment.polarity for entry in feed.entries[:5]]
        return float(np.mean(sentiments)) if sentiments else 0.0
    except:
        return 0.0

# ==========================================
# 4. Feature Builder
# ==========================================
def create_safe_features(model, real_time_value_dict):
    features = getattr(model, 'feature_names_in_', getattr(model, 'feature_names_', None))
    df = pd.DataFrame(columns=features)
    df.loc[0] = 0.0 
    for col in features:
        for key, val in real_time_value_dict.items():
            if key.lower() in col.lower():
                df.at[0, col] = val
    return df

# ==========================================
# 5. Inference Logic
# ==========================================
def run_1h_inference():
    print("\n" + "="*70)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Executing 1-Hour Market Analysis...")
    
    sp500_price = fetch_sp500_live()
    time.sleep(2)  # Prevent rate limit
    news_sentiment = fetch_crypto_news_sentiment()
    
    for coin in ['ETH', 'BTC']:
        print(f"\n--- {coin}/USDT (1-Hour) ---")
        time.sleep(2)  # Delay between requests
        try:
            data = fetch_crypto_data(f'{coin}/USDT', '1h')
            rt_dict = {'close': data['close'], 'volume': data['volume'], 'sp500': sp500_price, 'finbert': news_sentiment}
            
            m_mods = models[coin]
            m_probs = m_mods['macro'].predict_proba(create_safe_features(m_mods['macro'], rt_dict))[0][1]
            i_probs = m_mods['internal'].predict_proba(create_safe_features(m_mods['internal'], rt_dict))[0][1]
            s_probs = m_mods['sp500'].predict_proba(create_safe_features(m_mods['sp500'], rt_dict))[0][1]
            n_probs = m_mods['news'].predict_proba(create_safe_features(m_mods['news'], rt_dict))[0][1]
            
            X_meta = pd.DataFrame({'Macro_Expert': [m_probs], 'Whale_Expert': [i_probs], 'SP500_Expert': [s_probs], 'News_Expert': [n_probs]})
            meta_conf = m_mods['meta'].predict_proba(X_meta)[0][1] * 100
            
            signal = "BUY 🟢" if meta_conf > 50 else "SELL 🔴"
            print(f"[1H SIGNAL] {signal} (Conf: {meta_conf:.2f}%) | Experts -> M: {m_probs:.2f}, W: {i_probs:.2f}, S: {s_probs:.2f}, N: {n_probs:.2f}")
        except Exception as e:
            print(f"⚠️ Error for {coin}: {e}")

    print("="*70)

if __name__ == "__main__":
    run_1h_inference()
    schedule.every().hour.at(":01").do(run_1h_inference)
    print("\n⏳ 1H Scheduler Active: Waiting for the next hourly close...")
    while True:
        schedule.run_pending()
        time.sleep(1)