import os
import json
import numpy as np
import pandas as pd

base = r"e:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha"
ext = pd.read_csv(os.path.join(base, "External Cryptodata.csv"))
inter = pd.read_csv(os.path.join(base, "Internal Cryptodata.csv"))

print("=== 1. BASIC SHAPES & RANGES ===")
print("External shape:", ext.shape)
print("Internal shape:", inter.shape)

ext['dt'] = pd.to_datetime(ext['datetime'], errors='coerce')
inter['dt'] = pd.to_datetime(inter['datetime'], errors='coerce')

print(f"External date range: {ext['dt'].min()} to {ext['dt'].max()}")
print(f"Internal date range: {inter['dt'].min()} to {inter['dt'].max()}")

# Missing values tables
print("\n=== 2. MISSING VALUES SUMMARY ===")
print("External Missing:")
ext_miss = pd.DataFrame({'count': ext.isnull().sum(), 'pct': (ext.isnull().sum()/len(ext))*100})
print(ext_miss[ext_miss['count'] > 0])

print("\nInternal Missing:")
int_miss = pd.DataFrame({'count': inter.isnull().sum(), 'pct': (inter.isnull().sum()/len(inter))*100})
print(int_miss[int_miss['count'] > 0])

# OHLC logic check
print("\n=== 3. OHLC INTEGRITY CHECKS ===")
def check_ohlc(df, prefix):
    o, h, l, c = f"{prefix}_open", f"{prefix}_high", f"{prefix}_low", f"{prefix}_close"
    # check if columns exist with exact or case variation
    cols = {col.lower(): col for col in df.columns}
    o_col, h_col, l_col, c_col = cols.get(o.lower()), cols.get(h.lower()), cols.get(l.lower()), cols.get(c.lower())
    if not (o_col and h_col and l_col and c_col):
        print(f"OHLC cols for {prefix} not found!")
        return
    sub = df[[o_col, h_col, l_col, c_col]].dropna()
    h_lt_l = (sub[h_col] < sub[l_col]).sum()
    h_lt_o = (sub[h_col] < sub[o_col]).sum()
    h_lt_c = (sub[h_col] < sub[c_col]).sum()
    l_gt_o = (sub[l_col] > sub[o_col]).sum()
    l_gt_c = (sub[l_col] > sub[c_col]).sum()
    print(f"[{prefix}] High < Low: {h_lt_l}, High < Open: {h_lt_o}, High < Close: {h_lt_c}, Low > Open: {l_gt_o}, Low > Close: {l_gt_c}")

print("External checks:")
check_ohlc(ext, "BTC")
check_ohlc(ext, "ETH")
check_ohlc(ext, "GLD")
check_ohlc(ext, "DXY")

print("Internal checks:")
check_ohlc(inter, "BTC")
check_ohlc(inter, "ETH")

# GLD & DXY Missing Structure Analysis
print("\n=== 4. GLD & DXY MISSING STRUCTURE ===")
ext['dow'] = ext['dt'].dt.dayofweek # 0=Mon, 4=Fri, 5=Sat, 6=Sun
ext['hour'] = ext['dt'].dt.hour
weekend_mask = ext['dow'].isin([5, 6])
print("External Weekend rows:", weekend_mask.sum())
print("GLD missing on weekends:", ext.loc[weekend_mask, 'GLD_Close'].isnull().sum(), "out of", weekend_mask.sum())
print("GLD missing on weekdays:", ext.loc[~weekend_mask, 'GLD_Close'].isnull().sum(), "out of", (~weekend_mask).sum())
print("DXY missing on weekends:", ext.loc[weekend_mask, 'DXY_close'].isnull().sum(), "out of", weekend_mask.sum())
print("DXY missing on weekdays:", ext.loc[~weekend_mask, 'DXY_close'].isnull().sum(), "out of", (~weekend_mask).sum())

# ETH missing range in both
print("\n=== 5. ETH MISSING PERIODS ===")
eth_miss_ext = ext[ext['ETH_close'].isnull()]
print(f"External ETH missing count: {len(eth_miss_ext)}, date span: {eth_miss_ext['dt'].min()} to {eth_miss_ext['dt'].max()}")
eth_miss_int = inter[inter['ETH_close'].isnull()]
print(f"Internal ETH missing count: {len(eth_miss_int)}, date span: {eth_miss_int['dt'].min()} to {eth_miss_int['dt'].max()}")

# Overlap & Agreement Analysis
print("\n=== 6. OVERLAP & VALUE AGREEMENT ===")
merged = pd.merge(ext, inter, on='datetime', suffixes=('_ext', '_int'))
print(f"Overlapping timestamps: {len(merged)}")
print(f"Overlap date span: {merged['dt_ext'].min()} to {merged['dt_ext'].max()}")

for asset in ['BTC', 'ETH']:
    for metric in ['open', 'high', 'low', 'close', 'volume']:
        col_ext = f"{asset}_{metric}_ext"
        col_int = f"{asset}_{metric}_int"
        diff = (merged[col_ext] - merged[col_int]).abs()
        max_diff = diff.max()
        mean_diff = diff.mean()
        mismatch_cnt = (diff > 1e-4).sum()
        print(f"Comparison {asset}_{metric}: max_abs_diff={max_diff:.6f}, mean_abs_diff={mean_diff:.6f}, mismatches(>1e-4)={mismatch_cnt}")

# Correlation and features
print("\n=== 7. CORRELATION OF RETURNS ===")
# Compute log returns on overlapping cleaned slices
merged_clean = merged.dropna(subset=['BTC_close_ext', 'BTC_close_int', 'ETH_close_ext', 'ETH_close_int']).copy()
merged_clean = merged_clean.sort_values('dt_ext').reset_index(drop=True)
merged_clean['BTC_ret'] = np.log(merged_clean['BTC_close_int'] / merged_clean['BTC_close_int'].shift(1))
merged_clean['ETH_ret'] = np.log(merged_clean['ETH_close_int'] / merged_clean['ETH_close_int'].shift(1))
merged_clean['GLD_ret'] = np.log(merged_clean['GLD_Close'] / merged_clean['GLD_Close'].shift(1))
merged_clean['DXY_ret'] = np.log(merged_clean['DXY_close'] / merged_clean['DXY_close'].shift(1))

ret_corr = merged_clean[['BTC_ret', 'ETH_ret', 'GLD_ret', 'DXY_ret']].corr()
print("Return Correlation Matrix:")
print(ret_corr)

price_corr = merged_clean[['BTC_close_int', 'ETH_close_int', 'GLD_Close', 'DXY_close']].corr()
print("\nPrice Level Correlation Matrix:")
print(price_corr)

# Internal market sentiment indicators correlation
print("\nInternal Derivatives Metrics Correlation with BTC_ret:")
int_clean = inter.sort_values('dt').reset_index(drop=True)
int_clean['BTC_ret'] = np.log(int_clean['BTC_close'] / int_clean['BTC_close'].shift(1))
int_clean['ETH_ret'] = np.log(int_clean['ETH_close'] / int_clean['ETH_close'].shift(1))
deriv_cols = ['sum_open_interest', 'sum_open_interest_value', 'count_toptrader_long_short_ratio', 
              'sum_toptrader_long_short_ratio', 'count_long_short_ratio', 'sum_taker_long_short_vol_ratio',
              'BTC_ret', 'ETH_ret']
print(int_clean[deriv_cols].corr())

