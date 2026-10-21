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
    print("Loading FinBERT Model for Live Sentiment Analysis...")
    sentiment_analyzer = pipeline("text-classification", model="ProsusAI/finbert")
except Exception:
    sentiment_analyzer = None
    print("⚠️ FinBERT pipeline not initialized. Using neutral sentiment proxy.")

# ==========================================
# 1. Configuration & Paths
# ==========================================
BASE_PATH = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha"
MODELS_DIR = os.path.join(BASE_PATH, r"src\models\saved models\ETH")

# ==========================================
# 2. Load Models
# ==========================================
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading AI Models into Memory...")
try:
    base_macro = joblib.load(os.path.join(MODELS_DIR, "eth_macro_best_model.joblib"))
    base_internal = joblib.load(os.path.join(MODELS_DIR, "eth_internal_best_model.joblib"))
    base_sp500 = joblib.load(os.path.join(MODELS_DIR, "eth_sp500_best_model.joblib"))
    base_news = joblib.load(os.path.join(MODELS_DIR, "eth_news_best_model.joblib"))
    meta_learner = joblib.load(os.path.join(MODELS_DIR, "eth_meta_learner.joblib"))
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] All 5 Models Loaded Successfully! System Ready.\n")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

# ==========================================
# 3. Helper Functions
# ==========================================
def get_features(model):
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    elif hasattr(model, 'booster_'):
        return model.booster_.feature_name()
    elif hasattr(model, 'feature_name_'):
        return model.feature_name_
    return []

def create_safe_dataframe(model, proxy_value=0.0):
    expected_features = get_features(model)
    df = pd.DataFrame(0.0, index=[0], columns=expected_features)
    for col in df.columns:
        if any(k in col.lower() for k in ['vol', 'flow', 'sentiment', 'score', 'close', 'price']):
            df[col] = proxy_value
    return df

# ==========================================
# 4. Feature Engineering Functions
# ==========================================
def calculate_parkinson(high, low):
    return np.sqrt((1.0 / (4.0 * np.log(2.0))) * (np.log(high / low) ** 2))

def calculate_garman_klass(open_p, high, low, close):
    term1 = 0.5 * (np.log(high / low) ** 2)
    term2 = (2.0 * np.log(2.0) - 1.0) * (np.log(close / open_p) ** 2)
    return np.sqrt(term1 - term2)

# ==========================================
# 5. Live Data Fetching (Macro with DXY & Gold)
# ==========================================
def fetch_live_macro_data():
    expected_features = get_features(base_macro)
    
    # 1. Fetch ETH/USDT 4H candles from Binance
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv('ETH/USDT', timeframe='4h', limit=5)
    eth_df = pd.DataFrame(ohlcv, columns=['timestamp', 'ETH_open', 'ETH_high', 'ETH_low', 'ETH_close', 'ETH_volum'])
    
    # 2. Fetch Gold (GLD) and DXY from Yahoo Finance
    try:
        gld_data = yf.download("GLD", period="5d", interval="1h", progress=False)
        dxy_data = yf.download("DX-Y.NYB", period="5d", interval="1h", progress=False)
        
        gld_last = gld_data.iloc[-1]
        gld_prev = gld_data.iloc[-2]
        
        dxy_last = dxy_data.iloc[-1]
        dxy_prev = dxy_data.iloc[-2]
        
        gld_open = float(gld_last['Open'])
        gld_high = float(gld_last['High'])
        gld_low = float(gld_last['Low'])
        gld_close = float(gld_last['Close'])
        gld_vol = float(gld_last['Volume'])
        gld_log_re = np.log(gld_close / float(gld_prev['Close']))
        
        dxy_open = float(dxy_last['Open'])
        dxy_high = float(dxy_last['High'])
        dxy_low = float(dxy_last['Low'])
        dxy_close = float(dxy_last['Close'])
        dxy_log_re = np.log(dxy_close / float(dxy_prev['Close']))
    except Exception as e:
        print(f"⚠️ Yahoo Finance fetch fallback: {e}")
        gld_open = gld_high = gld_low = gld_close = 240.0
        gld_vol = 1000.0
        gld_log_re = 0.0
        dxy_open = dxy_high = dxy_low = dxy_close = 103.0
        dxy_log_re = 0.0

    # 3. Last ETH candle & metrics
    last_eth = eth_df.iloc[-1]
    prev_eth = eth_df.iloc[-2]
    
    eth_open = float(last_eth['ETH_open'])
    eth_high = float(last_eth['ETH_high'])
    eth_low = float(last_eth['ETH_low'])
    eth_close = float(last_eth['ETH_close'])
    eth_volum = float(last_eth['ETH_volum'])
    eth_log_re = np.log(eth_close / float(prev_eth['ETH_close']))
    
    eth_parkin = calculate_parkinson(eth_high, eth_low)
    eth_garma = calculate_garman_klass(eth_open, eth_high, eth_low, eth_close)
    
    now = datetime.utcnow()
    hour = now.hour
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    
    # 4. Construct feature dictionary matching the exact dataset schema
    data_dict = {
        'ETH_open': eth_open,
        'ETH_high': eth_high,
        'ETH_low': eth_low,
        'ETH_close': eth_close,
        'ETH_volum': eth_volum,
        'GLD_Open': gld_open,
        'GLD_High': gld_high,
        'GLD_Low': gld_low,
        'GLD_Close': gld_close,
        'GLD_Volum': gld_vol,
        'DXY_open': dxy_open,
        'DXY_high': dxy_high,
        'DXY_low': dxy_low,
        'DXY_close': dxy_close,
        'ETH_log_re': eth_log_re,
        'GLD_log_re': gld_log_re,
        'DXY_log_re': dxy_log_re,
        'ETH_parkin': eth_parkin,
        'ETH_garma': eth_garma,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos
    }
    
    live_df = pd.DataFrame([data_dict])
    
    for col in expected_features:
        if col not in live_df.columns:
            live_df[col] = 0.0
            
    return live_df[expected_features]

def fetch_live_internal_data():
    try:
        exchange = ccxt.binance()
        trades = exchange.fetch_trades('ETH/USDT', limit=1000)
        whale_buy_vol = sum([t['amount'] for t in trades if t['amount'] > 10 and t['side'] == 'buy'])
        whale_sell_vol = sum([t['amount'] for t in trades if t['amount'] > 10 and t['side'] == 'sell'])
        whale_pressure = whale_buy_vol - whale_sell_vol
    except Exception:
        whale_pressure = 0.0
    return create_safe_dataframe(base_internal, proxy_value=whale_pressure)

def fetch_live_sp500_data():
    try:
        ticker = yf.Ticker("^GSPC")
        hist = ticker.history(period="1d", interval="1h")
        latest_close = float(hist['Close'].iloc[-1])
    except Exception:
        latest_close = 0.0
    return create_safe_dataframe(base_sp500, proxy_value=latest_close)

def fetch_live_news_data():
    sentiment_score = 0.0
    try:
        feed = feedparser.parse('https://cointelegraph.com/rss')
        titles = [entry.title for entry in feed.entries[:5]]
        combined_text = ". ".join(titles)
        if sentiment_analyzer and combined_text:
            result = sentiment_analyzer(combined_text[:512])[0]
            if result['label'] == 'positive':
                sentiment_score = float(result['score'])
            elif result['label'] == 'negative':
                sentiment_score = -float(result['score'])
            else:
                sentiment_score = 0.0
    except Exception:
        sentiment_score = 0.0
    return create_safe_dataframe(base_news, proxy_value=sentiment_score)

# ==========================================
# 6. Core Prediction Engine
# ==========================================
def safe_predict(model, df):
    if type(model).__name__ == 'LGBMClassifier':
        return model.predict_proba(df.values)[0][1]
    return model.predict_proba(df)[0][1]

def generate_trade_signal():
    print("-" * 65)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Executing Real-Time Multi-Asset Inference Pipeline...")
    
    try:
        live_macro = fetch_live_macro_data()
        live_internal = fetch_live_internal_data()
        live_sp500 = fetch_live_sp500_data()
        live_news = fetch_live_news_data()
        
        prob_macro = safe_predict(base_macro, live_macro)
        prob_internal = safe_predict(base_internal, live_internal)
        prob_sp500 = safe_predict(base_sp500, live_sp500)
        prob_news = safe_predict(base_news, live_news)
        
        print(f"  Macro Expert (ETH + DXY + Gold) : {prob_macro:.4f}")
        print(f"  Whale Expert (Binance Trades)   : {prob_internal:.4f}")
        print(f"  SP500 Expert (Yahoo Finance)    : {prob_sp500:.4f}")
        print(f"  News Expert  (CoinTelegraph)    : {prob_news:.4f}")
        
        meta_features = pd.DataFrame(
            [[prob_macro, prob_internal, prob_sp500, prob_news]],
            columns=['Macro_Expert', 'Whale_Expert', 'SP500_Expert', 'News_Expert']
        )
        
        final_decision = meta_learner.predict(meta_features)[0]
        final_confidence = meta_learner.predict_proba(meta_features)[0][1]
        
        print("\n=== FINAL AI TRADING SIGNAL ===")
        if final_decision == 1 and final_confidence > 0.60:
            print(f"🚀 SIGNAL: BUY ETH (Confidence: {final_confidence:.2%})")
        elif final_decision == 0 and final_confidence < 0.40:
            print(f"🔻 SIGNAL: SELL/SHORT ETH (Confidence: {(1 - final_confidence):.2%})")
        else:
            print(f"⚖️ SIGNAL: HOLD (Market Uncertain - Confidence: {final_confidence:.2%})")
            
    except Exception as e:
        print(f"❌ Error during inference: {e}")

# ==========================================
# 7. Runner / Scheduler
# ==========================================
if __name__ == "__main__":
    generate_trade_signal()