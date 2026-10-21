import os
import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Local Paths Setup
# ==========================================
BASE_PATH = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha"
DATA_DIR = os.path.join(BASE_PATH, r"data\03_features_1h")
MODELS_1H_DIR = os.path.join(BASE_PATH, r"src\models\saved models\1_hr_ETH")

os.makedirs(MODELS_1H_DIR, exist_ok=True)

# ==========================================
# 2. Data Loading & Cleaning
# ==========================================
def load_and_prep_1h(filename):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing File: {filepath}")
    
    df = pd.read_csv(filepath, engine='python', on_bad_lines='skip', encoding_errors='ignore')
    
    text_cols = ['combined_article_text', 'summary', 'title', 'text', 'finbert_sentiment', 'symbol', 'pair', 'BTC_open_time', 'BTC_close_time', 'ETH_open_time', 'ETH_close_time']
    df = df.drop(columns=[c for c in text_cols if c in df.columns], errors='ignore')
    
    for col in df.columns:
        if col != 'datetime':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df = df.fillna(0)
    return df

def merge_target(df, target_df):
    if 'ETH_target_1h' in df.columns:
        return df
    return pd.merge(df, target_df, on='datetime', how='inner')

# ==========================================
# 3. Model Training & Selection Pipeline
# ==========================================
def train_and_select_best(df, target_col, dataset_name):
    print(f"\n[{dataset_name} Expert] Evaluating Algorithms...")
    
    drop_cols = ['datetime'] + [c for c in df.columns if 'target' in c]
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df[target_col].astype(int)
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    models_to_test = {
        'LightGBM': LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1),
        'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1),
        'LogReg': LogisticRegression(class_weight='balanced', max_iter=1000)
    }
    
    best_model_name = ""
    best_auc = -1.0
    best_model_obj = None
    
    for model_name, model in models_to_test.items():
        auc_scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            try:
                model.fit(X_train, y_train)
                probs = model.predict_proba(X_test)[:, 1]
                auc_scores.append(roc_auc_score(y_test, probs))
            except Exception:
                auc_scores.append(0.5)
                
        mean_auc = np.mean(auc_scores)
        print(f"  -> {model_name} ROC-AUC: {mean_auc:.4f}")
        
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_model_name = model_name
            best_model_obj = model
            
    print(f"🏆 Best Model for {dataset_name}: {best_model_name} (AUC: {best_auc:.4f})")
    
    best_model_obj.fit(X, y)
    save_path = os.path.join(MODELS_1H_DIR, f"eth_{dataset_name.lower()}_best_model.joblib")
    joblib.dump(best_model_obj, save_path)
    print(f"✅ Saved to: {save_path}")

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    print(f"Loading Feature Stores from: {DATA_DIR}")
    try:
        df_macro = load_and_prep_1h("macro_feature_store_1h.csv")
        df_internal = load_and_prep_1h("internal_feature_store_1h.csv")
        df_sp500 = load_and_prep_1h("sp500_feature_store_1h.csv")
        df_news = load_and_prep_1h("news_feature_store_1h.csv")
        
        target_df = df_macro[['datetime', 'ETH_target_1h']].copy()
        
        df_internal = merge_target(df_internal, target_df)
        df_sp500 = merge_target(df_sp500, target_df)
        df_news = merge_target(df_news, target_df)
        
        train_and_select_best(df_macro, target_col='ETH_target_1h', dataset_name='Macro')
        train_and_select_best(df_internal, target_col='ETH_target_1h', dataset_name='Internal')
        train_and_select_best(df_sp500, target_col='ETH_target_1h', dataset_name='SP500')
        train_and_select_best(df_news, target_col='ETH_target_1h', dataset_name='News')
        
        print("\n🎉 All 4 Base Models Trained & Saved Successfully!")
        
    except Exception as e:
        print(f"\n❌ Execution Failed: {e}")