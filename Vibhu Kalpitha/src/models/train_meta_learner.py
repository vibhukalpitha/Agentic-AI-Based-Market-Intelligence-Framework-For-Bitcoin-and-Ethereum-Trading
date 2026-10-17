import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Paths Setup
# ==========================================
base_path = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha"
features_dir = os.path.join(base_path, r"data\03_features")
models_dir = os.path.join(base_path, r"src\models\saved models")
reports_dir = os.path.join(base_path, r"reports\figures")

os.makedirs(reports_dir, exist_ok=True)

# ==========================================
# 2. Data Loading & Time Alignment (FIXED)
# ==========================================
def load_and_prep(filename):
    filepath = os.path.join(features_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath, engine='python', on_bad_lines='skip', encoding_errors='ignore')
    
    text_cols = ['combined_article_text', 'summary', 'title', 'text', 'finbert_sentiment', 'symbol', 'pair', 'BTC_open_time', 'BTC_close_time', 'ETH_open_time', 'ETH_close_time']
    df = df.drop(columns=[c for c in text_cols if c in df.columns], errors='ignore')
    
    time_cols = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
    if time_cols:
        df = df.rename(columns={time_cols[0]: 'datetime'})
        # Convert to datetime and Floor to nearest 4H to ensure matching
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce').dt.floor('4h')
    
    for col in df.columns:
        if col != 'datetime':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Group by exact 4H timeframe to remove duplicates from same window
    df = df.groupby('datetime').last().reset_index()
    return df

print("Loading and aligning datasets using 4H exact matching...")
df_macro = load_and_prep("macro_feature_store_4h.csv")
df_internal = load_and_prep("internal_feature_store_4h.csv")
df_sp500 = load_and_prep("sp500_feature_store_4h.csv")
df_news = load_and_prep("news_feature_store_4h.csv")

# FIX: Distinct suffixes for each join to avoid MergeError
df_merged = df_macro.set_index('datetime') \
    .join(df_internal.set_index('datetime'), how='outer', rsuffix='_dup1') \
    .join(df_sp500.set_index('datetime'), how='outer', rsuffix='_dup2') \
    .join(df_news.set_index('datetime'), how='outer', rsuffix='_dup3')

# Forward fill missing values and drop remaining NaNs
df_merged = df_merged.ffill().dropna().reset_index()

# Drop duplicate columns
dup_cols = [c for c in df_merged.columns if '_dup' in c]
df_merged = df_merged.drop(columns=dup_cols).sort_values('datetime').reset_index(drop=True)

print(f"Data aligned successfully! Overlapping timestamps for Late Fusion: {df_merged.shape[0]}")

# ==========================================
# 3. Load Base Models
# ==========================================
print("\nLoading Base Models...")
try:
    model_macro = joblib.load(os.path.join(models_dir, "macro_best_model.joblib"))
    model_internal = joblib.load(os.path.join(models_dir, "internal_best_model.joblib"))
    model_sp500 = joblib.load(os.path.join(models_dir, "sp500_best_model.joblib"))
    model_news = joblib.load(os.path.join(models_dir, "news_best_model.joblib"))
except Exception as e:
    print(f"\n❌ Error loading models! Check if the filenames are exactly correct in '{models_dir}'.\nDetails: {e}")
    exit()

# ==========================================
# 4. Generate Meta-Features (Probabilities)
# ==========================================
print("Generating Expert Predictions (Probabilities)...")
def get_probs(model, df):
    features = None
    try:
        # Try standard scikit-learn attribute
        features = model.feature_names_in_
    except AttributeError:
        # FIX: Safe fallback for LightGBM models which use a different attribute structure
        if hasattr(model, 'booster_'):
            features = model.booster_.feature_name()
        elif hasattr(model, 'feature_name_'):
            features = model.feature_name_
            
    if features is None:
        raise ValueError(f"Could not determine feature names for {type(model).__name__}")
        
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features required by the model: {missing}")
        
    return model.predict_proba(df[features])[:, 1]

X_meta = pd.DataFrame({
    'Macro_Expert': get_probs(model_macro, df_merged),
    'Whale_Expert': get_probs(model_internal, df_merged),
    'SP500_Expert': get_probs(model_sp500, df_merged),
    'News_Expert': get_probs(model_news, df_merged)
})

y_meta = df_merged['BTC_target_4h'].astype(int)

# ==========================================
# 5. Train Late Fusion Meta-Learner
# ==========================================
print("\n--- Training Meta-Learner (Logistic Regression) ---")
tscv = TimeSeriesSplit(n_splits=5)
meta_model = LogisticRegression(class_weight='balanced')

acc_list, f1_list, auc_list = [], [], []

for train_idx, test_idx in tscv.split(X_meta):
    X_train, X_meta_test = X_meta.iloc[train_idx], X_meta.iloc[test_idx]
    y_train, y_meta_test = y_meta.iloc[train_idx], y_meta.iloc[test_idx]
    
    meta_model.fit(X_train, y_train)
    preds = meta_model.predict(X_meta_test)
    probs = meta_model.predict_proba(X_meta_test)[:, 1]
    
    acc_list.append(accuracy_score(y_meta_test, preds))
    f1_list.append(f1_score(y_meta_test, preds, zero_division=0))
    auc_list.append(roc_auc_score(y_meta_test, probs))

print(f"Meta-Learner Accuracy: {np.mean(acc_list):.4f}")
print(f"Meta-Learner F1-Score: {np.mean(f1_list):.4f}")
print(f"Meta-Learner ROC-AUC: {np.mean(auc_list):.4f}")

# ==========================================
# 6. Analyze Model Importance & Save
# ==========================================
meta_model.fit(X_meta, y_meta) 

importance = pd.DataFrame({
    'Expert': X_meta.columns,
    'Weight (Trust Level)': meta_model.coef_[0]
}).sort_values(by='Weight (Trust Level)', ascending=False)

print("\n=== Which Expert Does the AI Trust Most? ===")
print(importance.to_markdown(index=False))

meta_model_path = os.path.join(models_dir, "meta_learner_best_model.joblib")
joblib.dump(meta_model, meta_model_path)
print(f"\n✅ Meta-Learner successfully saved to: {meta_model_path}")