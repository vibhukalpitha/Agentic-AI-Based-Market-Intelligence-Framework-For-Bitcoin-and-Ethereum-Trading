import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Local Paths Setup
# ==========================================
BASE_PATH = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha"
DATA_DIR = os.path.join(BASE_PATH, r"data\03_features_1h")
MODELS_1H_DIR = os.path.join(BASE_PATH, r"src\models\saved models\1_hr_ETH")

print("Loading 1H Datasets and Aligning Timestamps (Fixing Overfitting)...")

def load_data(filename):
    filepath = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(filepath, engine='python')
    text_cols = ['combined_article_text', 'summary', 'title', 'text', 'finbert_sentiment', 'symbol', 'pair']
    df = df.drop(columns=[c for c in text_cols if c in df.columns], errors='ignore')
    return df

df_macro = load_data("macro_feature_store_1h.csv")
df_internal = load_data("internal_feature_store_1h.csv")
df_sp500 = load_data("sp500_feature_store_1h.csv")
df_news = load_data("news_feature_store_1h.csv")

target_df = df_macro[['datetime', 'ETH_target_1h']].copy()

# LEFT JOIN on Macro Data (Prevents row drop and overfitting)
df_merged = df_macro.drop(columns=['ETH_target_1h']).set_index('datetime') \
    .join(df_internal.set_index('datetime'), how='left', rsuffix='_dup1') \
    .join(df_sp500.set_index('datetime'), how='left', rsuffix='_dup2') \
    .join(df_news.set_index('datetime'), how='left', rsuffix='_dup3') \
    .join(target_df.set_index('datetime'), how='left')

# Forward fill missing weekend/news data, fill initial gaps with 0
df_merged = df_merged.ffill().fillna(0).reset_index()

dup_cols = [c for c in df_merged.columns if '_dup' in c]
df_merged = df_merged.drop(columns=dup_cols)

print(f"✅ Data Aligned! Total Data Points: {df_merged.shape[0]}\n")

# ==========================================
# 2. Load Saved Base Models
# ==========================================
print("Loading Best Base Models from Local Folder...")
base_macro = joblib.load(os.path.join(MODELS_1H_DIR, "eth_macro_best_model.joblib"))
base_internal = joblib.load(os.path.join(MODELS_1H_DIR, "eth_internal_best_model.joblib"))
base_sp500 = joblib.load(os.path.join(MODELS_1H_DIR, "eth_sp500_best_model.joblib"))
base_news = joblib.load(os.path.join(MODELS_1H_DIR, "eth_news_best_model.joblib"))

# ==========================================
# 3. Generate Expert Predictions
# ==========================================
def get_probs(model, df):
    features = getattr(model, 'feature_names_in_', None)
    if features is None and hasattr(model, 'feature_names_'):
        features = model.feature_names_
    return model.predict_proba(df[features])[:, 1]

X_meta = pd.DataFrame({
    'Macro_Expert': get_probs(base_macro, df_merged),
    'Whale_Expert': get_probs(base_internal, df_merged),
    'SP500_Expert': get_probs(base_sp500, df_merged),
    'News_Expert': get_probs(base_news, df_merged)
})
y_meta = df_merged['ETH_target_1h'].astype(int)

# ==========================================
# 4. Train Meta-Learner
# ==========================================
print("--- Training 1H ETH Meta-Learner ---")
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
    auc_list.append(roc_auc_score(y_test, probs))

print(f"Meta-Learner Accuracy: {np.mean(acc_list):.4f}")
print(f"Meta-Learner F1-Score: {np.mean(f1_list):.4f}")
print(f"Meta-Learner ROC-AUC: {np.mean(auc_list):.4f}")

# Train on full data to learn weights
meta_model.fit(X_meta, y_meta)

importance = pd.DataFrame({
    'Expert': X_meta.columns,
    'Weight (Trust Level)': meta_model.coef_[0]
}).sort_values(by='Weight (Trust Level)', ascending=False)

print("\n=== AI Trust Levels (Weights) ===")
print(importance.to_string(index=False))

# Save directly to the target directory
meta_path = os.path.join(MODELS_1H_DIR, "eth_meta_learner_1h.joblib")
joblib.dump(meta_model, meta_path)
print(f"\n✅ Meta-Learner Model Saved to: {meta_path}")