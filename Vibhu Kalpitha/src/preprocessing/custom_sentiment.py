import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# File paths
raw_news_path = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\data\01_raw\bitcoin_news_3hour_intervals_with_sentiment.csv"
processed_dir = r"E:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha\data\02_processed"
out_path = os.path.join(processed_dir, "news_finbert_sentiment.csv")

os.makedirs(processed_dir, exist_ok=True)

print("Loading raw news data...")
df = pd.read_csv(raw_news_path, encoding='utf-8-sig')
df.columns = df.columns.str.strip()

text_col = 'combined_article_text' if 'combined_article_text' in df.columns else df.columns[1]

# Drop old sentiment columns
cols_to_drop = ['sentiment', 'sentiment_confidence', 'prob_bearish', 'prob_neutral', 'prob_bullish']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
df = df.dropna(subset=[text_col])

print("Downloading and Loading FinBERT model... (This might take a minute)")
# Using ProsusAI/finbert (Industry standard for Financial text)
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Use GPU if available, else CPU
device = 0 if torch.cuda.is_available() else -1
nlp = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

print(f"Running Sentiment Analysis on {len(df)} rows... (Please wait, this will take some time)")

labels = []
scores = []

for i, text in enumerate(df[text_col]):
    try:
        # FinBERT token limit is 512, so we truncate long articles
        result = nlp(str(text)[:1500], truncation=True, max_length=512)[0]
        labels.append(result['label'])
        scores.append(result['score'])
    except Exception as e:
        labels.append("neutral")
        scores.append(0.0)
    
    # Print progress every 500 rows so you know it's not stuck
    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1} / {len(df)} rows...")

df['finbert_sentiment'] = labels
df['finbert_score'] = scores

df.to_csv(out_path, index=False)
print(f"\n--- Custom FinBERT Sentiment Analysis Complete ---")
print(f"Saved to: {out_path}")