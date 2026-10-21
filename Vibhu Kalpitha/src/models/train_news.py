import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import shap
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Paths Setup (For Local VS Code)
# ==========================================
feature_path = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\data\03_features\news_feature_store_4h.csv"
reports_dir = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\reports\figures"
models_dir = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\src\models\saved models"

os.makedirs(reports_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# ==========================================
# 2. Load and Prepare Data
# ==========================================
print("Loading News Sentiment Feature Store...")
if not os.path.exists(feature_path):
    raise FileNotFoundError(f"කරුණාකර '{feature_path}' file එක තියෙනවාදැයි පරීක්ෂා කරන්න!")

df = pd.read_csv(
    feature_path,
    engine='python',
    on_bad_lines='skip',
    encoding_errors='ignore'
)

text_drop = ['combined_article_text', 'summary', 'title', 'text', 'finbert_sentiment']
df = df.drop(columns=[c for c in text_drop if c in df.columns], errors='ignore')

time_cols = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
if time_cols:
    sort_col = time_cols[0]
    df[sort_col] = pd.to_datetime(df[sort_col], errors='coerce')
    df = df.sort_values(sort_col).reset_index(drop=True)
    df = df.drop(columns=[sort_col])

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

target_col = 'BTC_target_4h'
drop_cols = [c for c in df.columns if 'target' in c]

X = df.drop(columns=drop_cols)
y = df[target_col].astype(int)

print(f"Dataset Loaded. Features count: {X.shape[1]}, Total samples: {X.shape[0]}")

# ==========================================
# FIX: Handle Class Imbalance
# ==========================================
neg_count = (y == 0).sum()
pos_count = (y == 1).sum()
scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

# ==========================================
# 3. Time-Series Cross Validation & Comparison
# ==========================================
tscv = TimeSeriesSplit(n_splits=5)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, scale_pos_weight=scale_weight, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42, verbose=-1)
}

results = []

print("\n--- Training News Models with TimeSeriesSplit ---")
for model_name, model in models.items():
    acc_list, f1_list, auc_list = [], [], []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
        
        acc_list.append(accuracy_score(y_test, preds))
        f1_list.append(f1_score(y_test, preds, zero_division=0))
        auc_list.append(roc_auc_score(y_test, probs))
        
    results.append({
        "Model": model_name,
        "Accuracy": np.mean(acc_list),
        "F1-Score": np.mean(f1_list),
        "ROC-AUC": np.mean(auc_list)
    })

df_results = pd.DataFrame(results)
print("\n=== Model Comparison Results ===")
print(df_results.to_markdown(index=False))

# ==========================================
# 4. Explainable AI (SHAP) Generation
# ==========================================
print("\nGenerating SHAP Explanations for News Metrics (using XGBoost)...")
best_xgb = XGBClassifier(n_estimators=100, scale_pos_weight=scale_weight, eval_metric='logloss', random_state=42)
best_xgb.fit(X, y)

explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False)
plt.title("SHAP: FinBERT News Sentiment Impact on BTC", fontsize=14)
plt.tight_layout()

shap_save_path = os.path.join(reports_dir, "news_shap_importance.png")
plt.savefig(shap_save_path)
print(f"SHAP plot saved to {shap_save_path}")

# ==========================================
# 5. Dynamically Save the Best Model
# ==========================================
best_model_name = df_results.sort_values(by="F1-Score", ascending=False).iloc[0]["Model"]
print(f"\n--- Best Model based on F1-Score is {best_model_name} ---")

best_model = models[best_model_name]
best_model.fit(X, y) 

model_filename = os.path.join(models_dir, "news_best_model.joblib")
joblib.dump(best_model, model_filename)
print(f"✅ Best News Model ({best_model_name}) successfully saved to: {model_filename}")