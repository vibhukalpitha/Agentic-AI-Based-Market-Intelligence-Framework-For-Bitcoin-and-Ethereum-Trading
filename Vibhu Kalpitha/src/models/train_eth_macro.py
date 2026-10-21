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
# 1. Local File Paths Setup (GitHub Ready)
# ==========================================
base_path = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha"
feature_path = os.path.join(base_path, r"data\03_features\macro_feature_store_4h.csv")

# Specific paths for ETH outputs
models_dir = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\src\models\saved models\ETH"
reports_dir = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\reports\figures\ETH"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

model_filename = os.path.join(models_dir, "eth_macro_best_model.joblib")
plot_filename = os.path.join(reports_dir, "eth_macro_shap_summary.jpg")

# ==========================================
# 2. Data Loading & Preprocessing
# ==========================================
print("Loading Macro Feature Store for ETH...")
if not os.path.exists(feature_path):
    raise FileNotFoundError(f"Dataset not found at: {feature_path}")

df = pd.read_csv(feature_path, engine='python', on_bad_lines='skip', encoding_errors='ignore')

# Drop text and redundant time columns
text_cols = ['combined_article_text', 'summary', 'title', 'text', 'finbert_sentiment', 'symbol', 'pair', 'BTC_open_time', 'BTC_close_time', 'ETH_open_time', 'ETH_close_time']
df = df.drop(columns=[c for c in text_cols if c in df.columns], errors='ignore')

time_cols = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
if time_cols:
    sort_col = time_cols[0]
    df[sort_col] = pd.to_datetime(df[sort_col], errors='coerce')
    df = df.sort_values(sort_col).reset_index(drop=True)
    df = df.drop(columns=[sort_col])

# Convert all features to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

# Extract ETH Target
target_col = 'ETH_target_4h'
if target_col not in df.columns:
    raise KeyError(f"Target column '{target_col}' missing!")

drop_cols = [c for c in df.columns if 'target' in c]
X = df.drop(columns=drop_cols)
y = df[target_col].astype(int)

print(f"Data ready. Features: {X.shape[1]}, Total Samples: {X.shape[0]}")

# Calculate Class Weights
neg_count = (y == 0).sum()
pos_count = (y == 1).sum()
scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

# ==========================================
# 3. Time-Series Cross Validation Training
# ==========================================
tscv = TimeSeriesSplit(n_splits=5)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, scale_pos_weight=scale_weight, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42, verbose=-1)
}

results = []

print("\n--- Training ETH Macro Baselines ---")
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
print("\n=== Model Evaluation ===")
print(df_results.to_markdown(index=False))

# ==========================================
# 4. Save Best Model & Generate SHAP
# ==========================================
best_model_name = df_results.sort_values(by="F1-Score", ascending=False).iloc[0]["Model"]
print(f"\n🏆 Selected Best Model: {best_model_name}")

best_model = models[best_model_name]
best_model.fit(X, y)

joblib.dump(best_model, model_filename)
print(f"✅ Model saved to: {model_filename}")

print("\n📊 Generating SHAP Analysis...")
plt.figure(figsize=(8, 10))

# Subsample background data to fix SHAP warning for large datasets
X_background = shap.sample(X, 100)

if best_model_name in ["Random Forest", "XGBoost", "LightGBM"]:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X)
    
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values_to_plot = shap_values[1]
    elif len(shap_values.shape) == 3 and shap_values.shape[2] == 2:
        shap_values_to_plot = shap_values[:, :, 1]
    else:
        shap_values_to_plot = shap_values
else:
    # Logistic Regression
    explainer = shap.LinearExplainer(best_model, X_background)
    shap_values_to_plot = explainer.shap_values(X)

shap.summary_plot(shap_values_to_plot, X, show=False)
plt.title(f"SHAP: Macro Metrics Driving ETH ({best_model_name})", fontsize=12, pad=15)
plt.tight_layout()
plt.savefig(plot_filename, format='jpg', dpi=300, bbox_inches='tight')
print(f"✅ SHAP JPG saved to: {plot_filename}")