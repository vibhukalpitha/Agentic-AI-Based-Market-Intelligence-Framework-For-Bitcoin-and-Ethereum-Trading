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
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df.iloc[-1]
    except Exception as e:
        print(f"⚠️ Error fetching {symbol}: {e}")
        return None

def fetch_macro_market_data():
    """Live S&P 500, Gold (GLD), සහ DXY දත්ත yfinance හරහා ලබා ගැනීම"""
    macro_data = {'sp500': 0.0, 'gold': 0.0, 'dxy': 0.0}
    
    # 1. S&P 500 (^GSPC)
    try:
        sp500 = yf.download('^GSPC', period='5d', interval='1h', progress=False)
        if not sp500.empty:
            close_val = sp500['Close'].dropna().iloc[-1]
            macro_data['sp500'] = float(close_val.values[0] if isinstance(close_val, pd.Series) else close_val)
    except Exception:
        pass

    # 2. Gold / GLD (GLD or GC=F)
    try:
        gold = yf.download('GLD', period='5d', interval='1h', progress=False)
        if not gold.empty:
            close_val = gold['Close'].dropna().iloc[-1]
            macro_data['gold'] = float(close_val.values[0] if isinstance(close_val, pd.Series) else close_val)
    except Exception:
        pass

    # 3. US Dollar Index / DXY (DX-Y.NYB or UUP)
    try:
        dxy = yf.download('DX-Y.NYB', period='5d', interval='1h', progress=False)
        if not dxy.empty:
            close_val = dxy['Close'].dropna().iloc[-1]
            macro_data['dxy'] = float(close_val.values[0] if isinstance(close_val, pd.Series) else close_val)
    except Exception:
        pass

    return macro_data

def fetch_crypto_news_sentiment():
    try:
        url = "https://cointelegraph.com/rss"
        feed = feedparser.parse(url)
        sentiments = [TextBlob(entry.title).sentiment.polarity for entry in feed.entries[:5]]
        return float(np.mean(sentiments)) if sentiments else 0.0
    except:
        return 0.0

# ==========================================
# 4. Smart Feature Builder
# ==========================================
def create_safe_features(model, real_time_dict):
    features = getattr(model, 'feature_names_in_', getattr(model, 'feature_names_', None))
    if features is None:
        return pd.DataFrame()
        
    df = pd.DataFrame(0.0, index=[0], columns=features)
    
    for col in features:
        c_low = col.lower()
        
        # Crypto OHLCV mappings
        if 'btc_close' in c_low and 'btc_close' in real_time_dict:
            df.at[0, col] = real_time_dict['btc_close']
        elif 'btc_open' in c_low and 'btc_open' in real_time_dict:
            df.at[0, col] = real_time_dict['btc_open']
        elif 'btc_high' in c_low and 'btc_high' in real_time_dict:
            df.at[0, col] = real_time_dict['btc_high']
        elif 'btc_low' in c_low and 'btc_low' in real_time_dict:
            df.at[0, col] = real_time_dict['btc_low']
        elif 'btc_vol' in c_low and 'btc_volume' in real_time_dict:
            df.at[0, col] = real_time_dict['btc_volume']

        elif 'eth_close' in c_low and 'eth_close' in real_time_dict:
            df.at[0, col] = real_time_dict['eth_close']
        elif 'eth_open' in c_low and 'eth_open' in real_time_dict:
            df.at[0, col] = real_time_dict['eth_open']
        elif 'eth_high' in c_low and 'eth_high' in real_time_dict:
            df.at[0, col] = real_time_dict['eth_high']
        elif 'eth_low' in c_low and 'eth_low' in real_time_dict:
            df.at[0, col] = real_time_dict['eth_low']
        elif 'eth_vol' in c_low and 'eth_volume' in real_time_dict:
            df.at[0, col] = real_time_dict['eth_volume']

        # Macro: Gold mappings
        elif any(k in c_low for k in ['gld', 'gold']) and 'gold' in real_time_dict:
            df.at[0, col] = real_time_dict['gold']

        # Macro: DXY mappings
        elif 'dxy' in c_low and 'dxy' in real_time_dict:
            df.at[0, col] = real_time_dict['dxy']

        # SP500 mappings
        elif any(k in c_low for k in ['sp500', 'sp_500', 'gspc', 'spy']) and 'sp500' in real_time_dict:
            df.at[0, col] = real_time_dict['sp500']

        # Sentiment / News mappings
        elif any(k in c_low for k in ['finbert', 'sentiment', 'score', 'news', 'polarity']) and 'sentiment' in real_time_dict:
            df.at[0, col] = real_time_dict['sentiment']
            
    return df

# ==========================================
# 5. Inference Logic
# ==========================================
def run_1h_inference():
    print("\n" + "="*70)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Executing 1-Hour Market Analysis...")
    
    # 1. Macro, SP500 & News Data Fetching
    macro_info = fetch_macro_market_data()
    print(f"📊 Macro Fetched -> SP500: {macro_info['sp500']:.2f} | Gold: {macro_info['gold']:.2f} | DXY: {macro_info['dxy']:.2f}")
    
    time.sleep(1)
    news_sentiment = fetch_crypto_news_sentiment()
    print(f"📰 News Sentiment -> Polarity: {news_sentiment:.4f}")
    
    # 2. Crypto Data Fetching (BTC & ETH)
    btc_candle = fetch_crypto_data('BTC/USDT', '1h')
    time.sleep(1)
    eth_candle = fetch_crypto_data('ETH/USDT', '1h')

    if btc_candle is None or eth_candle is None:
        print("⚠️ Could not fetch exchange candles. Skipping cycle.")
        print("="*70)
        return

    # Master Real-time feature dictionary
    rt_master = {
        'btc_close': float(btc_candle['close']),
        'btc_open': float(btc_candle['open']),
        'btc_high': float(btc_candle['high']),
        'btc_low': float(btc_candle['low']),
        'btc_volume': float(btc_candle['volume']),
        
        'eth_close': float(eth_candle['close']),
        'eth_open': float(eth_candle['open']),
        'eth_high': float(eth_candle['high']),
        'eth_low': float(eth_candle['low']),
        'eth_volume': float(eth_candle['volume']),
        
        'sp500': macro_info['sp500'],
        'gold': macro_info['gold'],
        'dxy': macro_info['dxy'],
        'sentiment': news_sentiment
    }
    
    for coin in ['ETH', 'BTC']:
        print(f"\n--- {coin}/USDT (1-Hour) ---")
        try:
            m_mods = models[coin]
            
            # Predict Probabilities
            m_probs = m_mods['macro'].predict_proba(create_safe_features(m_mods['macro'], rt_master))[0][1]
            i_probs = m_mods['internal'].predict_proba(create_safe_features(m_mods['internal'], rt_master))[0][1]
            s_probs = m_mods['sp500'].predict_proba(create_safe_features(m_mods['sp500'], rt_master))[0][1]
            n_probs = m_mods['news'].predict_proba(create_safe_features(m_mods['news'], rt_master))[0][1]
            
            # Meta Learner Late Fusion
            X_meta = pd.DataFrame({
                'Macro_Expert': [m_probs],
                'Whale_Expert': [i_probs],
                'SP500_Expert': [s_probs],
                'News_Expert': [n_probs]
            })
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