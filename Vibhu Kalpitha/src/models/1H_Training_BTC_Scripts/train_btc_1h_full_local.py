import os
import warnings
import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings('ignore')

# ==========================================
# 1. VS Code Local Paths Setup (BTC)
# ==========================================
DATA_DIR = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\data\03_features_1h"
MODELS_BTC_DIR = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\src\models\saved models\1_hr_BTC"

os.makedirs(MODELS_BTC_DIR, exist_ok=True)

# ==========================================
# 2. Data Loading & Cleaning
# ==========================================
def load_data(filename):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing File: {filepath}")
        
    df = pd.read_csv(filepath, engine='python')
    
    # Datetime timezone fix
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)
        
    text_cols = [
        'combined_article_text', 'summary', 'title', 'text', 
        'finbert_sentiment', 'symbol', 'pair', 
        'BTC_open_time', 'BTC_close_time', 'ETH_open_time', 'ETH_close_time'
    ]
    df = df.drop(columns=[c for c in text_cols if c in df.columns], errors='ignore')
    for col in df.columns:
        if col != 'datetime':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.fillna(0)

print("Loading 1H Datasets for BTC...")
df_macro = load_data("final_1h_dataset_with_macro.csv")
df_internal = load_data("internal_feature_store_1h.csv")
df_sp500 = load_data("sp500_feature_store_1h.csv")
df_news = load_data("news_feature_store_1h.csv")

# Target Dataframe (BTC)
target_df = df_macro[['datetime', 'BTC_target_1h']].copy()

def attach_target(df, target_df):
    if 'BTC_target_1h' in df.columns:
        return df
    return pd.merge(df, target_df, on='datetime', how='inner')

df_macro_train = df_macro.drop(columns=['ETH_target_1h'], errors='ignore')
df_internal_train = attach_target(df_internal, target_df)
df_sp500_train = attach_target(df_sp500, target_df)
df_news_train = attach_target(df_news, target_df)

# ==========================================
# 3. Base Model Training (Independent Experts)
# ==========================================
def train_and_select_best_base(df, target_col, dataset_name):
    print(f"\n[{dataset_name} Expert] Evaluating Algorithms on {df.shape[1] - 2} features...")
    
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
    
    best_auc = -1.0
    best_model_obj = None
    best_model_name = ""
    
    for model_name, model in models_to_test.items():
        auc_scores = []
        for train_idx, test_idx in tscv.split(X):
            try:
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                probs = model.predict_proba(X.iloc[test_idx])[:, 1]
                auc_scores.append(roc_auc_score(y.iloc[test_idx], probs))
            except Exception:
                auc_scores.append(0.5)
        
        mean_auc = np.mean(auc_scores)
        print(f"  -> {model_name} ROC-AUC: {mean_auc:.4f}")
        
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_model_obj = model
            best_model_name = model_name
            
    print(f"🏆 Best Model for BTC {dataset_name}: {best_model_name} (AUC: {best_auc:.4f})")
    best_model_obj.fit(X, y)
    save_path = os.path.join(MODELS_BTC_DIR, f"btc_{dataset_name.lower()}_best_model.joblib")
    joblib.dump(best_model_obj, save_path)
    return best_model_obj

base_macro = train_and_select_best_base(df_macro_train, 'BTC_target_1h', 'Macro')
base_internal = train_and_select_best_base(df_internal_train, 'BTC_target_1h', 'Internal')
base_sp500 = train_and_select_best_base(df_sp500_train, 'BTC_target_1h', 'SP500')
base_news = train_and_select_best_base(df_news_train, 'BTC_target_1h', 'News')

# ==========================================
# 4. Meta-Learner Training (Late Fusion)
# ==========================================
print("\n--- Aligning for BTC Meta-Learner ---")
df_merged = df_macro.drop(columns=['BTC_target_1h', 'ETH_target_1h'], errors='ignore').set_index('datetime') \
    .join(df_internal.set_index('datetime'), how='left', rsuffix='_dup1') \
    .join(df_sp500.set_index('datetime'), how='left', rsuffix='_dup2') \
    .join(df_news.set_index('datetime'), how='left', rsuffix='_dup3') \
    .join(target_df.set_index('datetime'), how='left')

df_merged = df_merged.ffill().fillna(0).reset_index()
dup_cols = [c for c in df_merged.columns if '_dup' in c]
df_merged = df_merged.drop(columns=dup_cols)

def get_probs(model, df):
    features = getattr(model, 'feature_names_in_', getattr(model, 'feature_names_', None))
    return model.predict_proba(df[features])[:, 1]

X_meta = pd.DataFrame({
    'Macro_Expert': get_probs(base_macro, df_merged),
    'Whale_Expert': get_probs(base_internal, df_merged),
    'SP500_Expert': get_probs(base_sp500, df_merged),
    'News_Expert': get_probs(base_news, df_merged)
})
y_meta = df_merged['BTC_target_1h'].astype(int)

tscv = TimeSeriesSplit(n_splits=5)
meta_model = LogisticRegression(class_weight='balanced')

acc_list, f1_list, auc_list = [], [], []
for train_idx, test_idx in tscv.split(X_meta):
    X_train, X_test = X_meta.iloc[train_idx], X_meta.iloc[test_idx]
    y_train, y_test = y_meta.iloc[train_idx], y_meta.iloc[test_idx]
    
    meta_model.fit(X_train, y_train)
    preds = meta_model.predict(X_test)
    probs = meta_model.predict_proba(X_test)[:, 1]
    
    acc_list.append(accuracy_score(y_test, preds))
    f1_list.append(f1_score(y_test, preds, zero_division=0))
    current_auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5
    auc_list.append(current_auc)

print(f"BTC Meta-Learner Accuracy: {np.mean(acc_list):.4f}")
print(f"BTC Meta-Learner F1-Score: {np.mean(f1_list):.4f}")
print(f"BTC Meta-Learner ROC-AUC: {np.mean(auc_list):.4f}")

meta_model.fit(X_meta, y_meta)

importance = pd.DataFrame({
    'Expert': X_meta.columns,
    'Weight (Trust Level)': meta_model.coef_[0]
}).sort_values(by='Weight (Trust Level)', ascending=False)

print("\n=== BTC AI Trust Levels (Weights) ===")
print(importance.to_markdown(index=False))

meta_path = os.path.join(MODELS_BTC_DIR, "btc_meta_learner_1h.joblib")
joblib.dump(meta_model, meta_path)
print(f"\n Models successfully saved to: {MODELS_BTC_DIR}")