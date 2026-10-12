import os
import json
import numpy as np
import pandas as pd

base = r"e:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha"
ext_path = os.path.join(base, "External Cryptodata.csv")
int_path = os.path.join(base, "Internal Cryptodata.csv")

def inspect_dataset(path, name):
    print(f"==================================================")
    print(f"AUDITING: {name}")
    print(f"==================================================")
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print("\nColumns and dtypes:")
    for c, dt in zip(df.columns, df.dtypes):
        print(f"  - {c}: {dt}")
    
    print("\nHead (2 rows):")
    print(df.head(2).to_dict(orient='records'))
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing_Count': missing, 'Missing_Pct': missing_pct})
    print("\nMissing Values:")
    print(missing_df[missing_df['Missing_Count'] > 0])
    
    # Duplicate rows
    dup_rows = df.duplicated().sum()
    print(f"\nExact duplicate rows: {dup_rows}")
    
    # Date column check
    date_cols = [c for c in df.columns if any(k in c.lower() for k in ['date', 'time', 'timestamp'])]
    for date_col in date_cols:
        print(f"\nDate column check: '{date_col}'")
        dup_timestamps = df[date_col].duplicated().sum()
        print(f"Duplicate timestamps: {dup_timestamps}")
        
        parsed = pd.to_datetime(df[date_col], errors='coerce')
        nat_count = parsed.isna().sum()
        print(f"NaT count: {nat_count}")
        valid = parsed.dropna().sort_values()
        if len(valid) > 0:
            print(f"Range: {valid.iloc[0]} to {valid.iloc[-1]}")
            diffs = valid.diff()
            print("Top 5 frequency diffs:")
            print(diffs.value_counts().head(5))
            
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nNumeric columns count: {len(num_cols)}")
    for c in num_cols:
        neg_count = (df[c] < 0).sum()
        zero_count = (df[c] == 0).sum()
        consec = (df[c] == df[c].shift(1)).sum()
        print(f"Col '{c}': min={df[c].min()}, max={df[c].max()}, neg={neg_count}, zero={zero_count}, consec_repeats={consec}")
        
    return df

ext_df = inspect_dataset(ext_path, "External Cryptodata")
int_df = inspect_dataset(int_path, "Internal Cryptodata")
