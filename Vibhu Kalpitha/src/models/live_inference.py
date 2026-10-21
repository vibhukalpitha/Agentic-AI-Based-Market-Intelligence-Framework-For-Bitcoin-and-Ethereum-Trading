import os
import time
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import schedule
import ccxt
import yfinance as yf
import feedparser

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    from transformers import pipeline
    sentiment_analyzer = pipeline("text-classification", model="ProsusAI/finbert")
except Exception:
    sentiment_analyzer = None

# ==========================================
# 1. Configuration & Paths
# ==========================================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ETH_MODELS_DIR = os.path.join(BASE_PATH, r"src\models\saved models\ETH")
BTC_MODELS_DIR = os.path.join(BASE_PATH, r"src\models\saved models")

# ==========================================
# 2. Load All Models (ETH & BTC)
# ==========================================
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading BTC and ETH AI Models into Memory...")
try:
    # Load ETH Models
    eth_macro = joblib.load(os.path.join(ETH_MODELS_DIR, "eth_macro_best_model.joblib"))
    eth_internal = joblib.load(os.path.join(ETH_MODELS_DIR, "eth_internal_best_model.joblib"))
    eth_sp500 = joblib.load(os.path.join(ETH_MODELS_DIR, "eth_sp500_best_model.joblib"))
    eth_news = joblib.load(os.path.join(ETH_MODELS_DIR, "eth_news_best_model.joblib"))
    eth_meta = joblib.load(os.path.join(ETH_MODELS_DIR, "eth_meta_learner.joblib"))
    
    # Load BTC Models (Exact matching names)
    btc_macro = joblib.load(os.path.join(BTC_MODELS_DIR, "macro_best_model.joblib"))
    btc_internal = joblib.load(os.path.join(BTC_MODELS_DIR, "internal_best_model.joblib"))
    btc_sp500 = joblib.load(os.path.join(BTC_MODELS_DIR, "sp500_best_model.joblib"))
    btc_news = joblib.load(os.path.join(BTC_MODELS_DIR, "news_best_model.joblib"))
    btc_meta = joblib.load(os.path.join(BTC_MODELS_DIR, "meta_learner_best_model.joblib"))
    
    print(f"✅ All 10 Models Loaded Successfully! Multi-Coin System Ready.\n")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

# ==========================================
# 3. Helper Functions
# ==========================================
def get_features(model):
    if hasattr(model, 'feature_names_in_'): return list(model.feature_names_in_)
    elif hasattr(model, 'booster_'): return model.booster_.feature_name()
    elif hasattr(model, 'feature_name_'): return model.feature_name_
    return []

def create_safe_dataframe(model, proxy_value=0.0):
    expected = get_features(model)
    df = pd.DataFrame(0.0, index=[0], columns=expected)
    for col in df.columns:
        if any(k in col.lower() for k in ['vol', 'flow', 'sentiment', 'score', 'close', 'price']):
            df[col] = proxy_value
    return df

def calculate_parkinson(high, low):
    return np.sqrt((1.0 / (4.0 * np.log(2.0))) * (np.log(high / low) ** 2))

def calculate_garman_klass(open_p, high, low, close):
    term1 = 0.5 * (np.log(high / low) ** 2)
    term2 = (2.0 * np.log(2.0) - 1.0) * (np.log(close / open_p) ** 2)
    return np.sqrt(term1 - term2)

# ==========================================
# 4. Global Market Data Fetching
# ==========================================
def fetch_global_macro_data():
    exchange = ccxt.binance()
    
    eth_ohlcv = exchange.fetch_ohlcv('ETH/USDT', timeframe='4h', limit=5)
    btc_ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='4h', limit=5)
    
    try:
        gld_data = yf.download("GLD", period="5d", interval="1h", progress=False)
        dxy_data = yf.download("DX-Y.NYB", period="5d", interval="1h", progress=False)
        
        gld_open = float(gld_data['Open'].values[-1])
        gld_high = float(gld_data['High'].values[-1])
        gld_low = float(gld_data['Low'].values[-1])
        gld_close = float(gld_data['Close'].values[-1])
        gld_vol = float(gld_data['Volume'].values[-1])
        gld_log_re = np.log(gld_close / float(gld_data['Close'].values[-2]))
        
        dxy_open = float(dxy_data['Open'].values[-1])
        dxy_high = float(dxy_data['High'].values[-1])
        dxy_low = float(dxy_data['Low'].values[-1])
        dxy_close = float(dxy_data['Close'].values[-1])
        dxy_log_re = np.log(dxy_close / float(dxy_data['Close'].values[-2]))
    except Exception:
        gld_open = gld_high = gld_low = gld_close = 240.0
        gld_vol = 1000.0; gld_log_re = 0.0
        dxy_open = dxy_high = dxy_low = dxy_close = 103.0
        dxy_log_re = 0.0

    eth_last = eth_ohlcv[-1]; eth_prev = eth_ohlcv[-2]
    btc_last = btc_ohlcv[-1]; btc_prev = btc_ohlcv[-2]
    
    master_dict = {
        'ETH_open': eth_last[1], 'ETH_high': eth_last[2], 'ETH_low': eth_last[3], 'ETH_close': eth_last[4], 'ETH_volum': eth_last[5],
        'BTC_open': btc_last[1], 'BTC_high': btc_last[2], 'BTC_low': btc_last[3], 'BTC_close': btc_last[4], 'BTC_volum': btc_last[5],
        'GLD_Open': gld_open, 'GLD_High': gld_high, 'GLD_Low': gld_low, 'GLD_Close': gld_close, 'GLD_Volum': gld_vol,
        'DXY_open': dxy_open, 'DXY_high': dxy_high, 'DXY_low': dxy_low, 'DXY_close': dxy_close,
        
        'ETH_log_re': np.log(eth_last[4] / eth_prev[4]),
        'BTC_log_re': np.log(btc_last[4] / btc_prev[4]),
        'GLD_log_re': gld_log_re,
        'DXY_log_re': dxy_log_re,
        
        'ETH_parkin': calculate_parkinson(eth_last[2], eth_last[3]),
        'ETH_garma': calculate_garman_klass(eth_last[1], eth_last[2], eth_last[3], eth_last[4]),
        'BTC_parkin': calculate_parkinson(btc_last[2], btc_last[3]),
        'BTC_garma': calculate_garman_klass(btc_last[1], btc_last[2], btc_last[3], btc_last[4]),
        
        'hour_sin': np.sin(2 * np.pi * datetime.utcnow().hour / 24.0),
        'hour_cos': np.cos(2 * np.pi * datetime.utcnow().hour / 24.0)
    }
    return master_dict

def fetch_whale_data(symbol, threshold):
    try:
        exchange = ccxt.binance()
        trades = exchange.fetch_trades(symbol, limit=1000)
        buy_vol = sum([t['amount'] for t in trades if t['amount'] > threshold and t['side'] == 'buy'])
        sell_vol = sum([t['amount'] for t in trades if t['amount'] > threshold and t['side'] == 'sell'])
        return buy_vol - sell_vol
    except Exception:
        return 0.0

def fetch_market_sentiments():
    try:
        sp500_close = float(yf.Ticker("^GSPC").history(period="1d", interval="1h")['Close'].values[-1])
    except Exception:
        sp500_close = 0.0
    
    try:
        feed = feedparser.parse('https://cointelegraph.com/rss')
        combined_text = ". ".join([entry.title for entry in feed.entries[:5]])
        news_score = 0.0
        if sentiment_analyzer and combined_text:
            res = sentiment_analyzer(combined_text[:512])[0]
            news_score = float(res['score']) if res['label'] == 'positive' else -float(res['score']) if res['label'] == 'negative' else 0.0
    except Exception:
        news_score = 0.0
    
    return sp500_close, news_score

# ==========================================
# 5. Core Prediction Engine
# ==========================================
def safe_predict(model, df):
    if type(model).__name__ == 'LGBMClassifier':
        return model.predict_proba(df.values)[0][1]
    return model.predict_proba(df)[0][1]

def run_coin_inference(coin_name, macro_data, whale_pressure, sp500_val, news_val, models):
    base_macro, base_internal, base_sp500, base_news, meta_learner = models
    
    expected_macro = get_features(base_macro)
    df_macro = pd.DataFrame([macro_data])
    for col in expected_macro:
        if col not in df_macro.columns: df_macro[col] = 0.0
    df_macro = df_macro[expected_macro]
    
    df_internal = create_safe_dataframe(base_internal, whale_pressure)
    df_sp500 = create_safe_dataframe(base_sp500, sp500_val)
    df_news = create_safe_dataframe(base_news, news_val)
    
    prob_macro = safe_predict(base_macro, df_macro)
    prob_internal = safe_predict(base_internal, df_internal)
    prob_sp500 = safe_predict(base_sp500, df_sp500)
    prob_news = safe_predict(base_news, df_news)
    
    meta_features = pd.DataFrame([[prob_macro, prob_internal, prob_sp500, prob_news]], 
                                 columns=['Macro_Expert', 'Whale_Expert', 'SP500_Expert', 'News_Expert'])
    
    decision = meta_learner.predict(meta_features)[0]
    confidence = meta_learner.predict_proba(meta_features)[0][1]
    
    print(f"[{coin_name}] Macro: {prob_macro:.2f} | Whale: {prob_internal:.2f} | SP500: {prob_sp500:.2f} | News: {prob_news:.2f}")
    if decision == 1 and confidence > 0.60:
        print(f"🚀 {coin_name} SIGNAL: BUY (Confidence: {confidence:.2%})")
    elif decision == 0 and confidence < 0.40:
        print(f"🔻 {coin_name} SIGNAL: SELL/SHORT (Confidence: {(1 - confidence):.2%})")
    else:
        print(f"⚖️ {coin_name} SIGNAL: HOLD (Market Uncertain - Confidence: {confidence:.2%})")

def generate_multi_coin_signal():
    print("=" * 65)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Executing Multi-Coin Live Inference...")
    
    master_macro = fetch_global_macro_data()
    eth_whales = fetch_whale_data('ETH/USDT', threshold=10.0)
    btc_whales = fetch_whale_data('BTC/USDT', threshold=1.0)
    sp500_val, news_val = fetch_market_sentiments()
    
    print("\n--- ETHEREUM (ETH) ---")
    run_coin_inference("ETH", master_macro, eth_whales, sp500_val, news_val, 
                       (eth_macro, eth_internal, eth_sp500, eth_news, eth_meta))
    
    print("\n--- BITCOIN (BTC) ---")
    run_coin_inference("BTC", master_macro, btc_whales, sp500_val, news_val, 
                       (btc_macro, btc_internal, btc_sp500, btc_news, btc_meta))
    print("=" * 65)

if __name__ == "__main__":
    generate_multi_coin_signal()