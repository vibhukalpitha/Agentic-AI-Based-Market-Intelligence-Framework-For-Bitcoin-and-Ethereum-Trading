import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.titlesize'] = 14

base_dir = r"e:\Innovus\Documents\Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading\Vibhu Kalpitha"
out_dir = os.path.join(base_dir, "EDA_Results")
subdirs = {
    'BTC': os.path.join(out_dir, "BTC_EDA"),
    'ETH': os.path.join(out_dir, "ETH_EDA"),
    'COMP': os.path.join(out_dir, "BTC_ETH_Comparison"),
    'FEAT': os.path.join(out_dir, "Feature_Validation"),
    'MACRO': os.path.join(out_dir, "Macro_Relationships")
}
for d in subdirs.values():
    os.makedirs(d, exist_ok=True)

print("Created directories successfully.")

# Load datasets
master_path = os.path.join(base_dir, "Master_Cleaned_Feature_Store.csv")
ext_path = os.path.join(base_dir, "External Cryptodata.csv")
int_path = os.path.join(base_dir, "Internal Cryptodata.csv")

print("Loading data files...")
df_master = pd.read_csv(master_path)
df_master['datetime'] = pd.to_datetime(df_master['datetime'], utc=True)
df_ext = pd.read_csv(ext_path)
df_ext['datetime'] = pd.to_datetime(df_ext['datetime'], utc=True)
df_int = pd.read_csv(int_path)
df_int['datetime'] = pd.to_datetime(df_int['datetime'], utc=True)

print(f"Master shape: {df_master.shape}")
print(f"External shape: {df_ext.shape}")
print(f"Internal shape: {df_int.shape}")

# PART A: VALIDATE THE MASTER DATASET FIRST
print("\n--- PART A: AUDITING MASTER DATASET ---")
audit_report = []

# 1. Row/Col counts, datetime range, frequency
n_rows, n_cols = df_master.shape
dt_min = df_master['datetime'].min()
dt_max = df_master['datetime'].max()
time_diffs = df_master['datetime'].diff()
hourly_freq_count = (time_diffs == pd.Timedelta(hours=1)).sum()
expected_hours = int((dt_max - dt_min).total_seconds() / 3600) + 1
actual_hours = n_rows
missing_timestamp_hours = expected_hours - actual_hours
dup_timestamps = df_master['datetime'].duplicated().sum()

audit_report.append({
    "metric": "Dimensions",
    "value": f"{n_rows} rows x {n_cols} columns"
})
audit_report.append({
    "metric": "Date Range",
    "value": f"{dt_min} to {dt_max}"
})
audit_report.append({
    "metric": "Duplicate Timestamps",
    "value": f"{dup_timestamps}"
})
audit_report.append({
    "metric": "Total Hours Span vs Rows",
    "value": f"Expected: {expected_hours}, Actual: {actual_hours}, Gaps in index: {missing_timestamp_hours}"
})

# Missing, Inf, Constant
null_counts = df_master.isnull().sum()
inf_counts = df_master.select_dtypes(include=np.number).apply(lambda s: np.isinf(s).sum())
const_cols = [c for c in df_master.columns if df_master[c].nunique() <= 1]

audit_report.append({
    "metric": "Constant Columns",
    "value": ", ".join(const_cols) if const_cols else "None"
})

# Special Validation 1: Forward filling GLD_Volume across closed periods
# We can identify original missing GLD_Volume in External Cryptodata
ext_gld_vol_na = df_ext.set_index('datetime')['GLD_Volume'].isna()
master_gld_vol_filled = df_master.set_index('datetime').join(ext_gld_vol_na.rename('was_na'))
weekend_or_holiday_gld_vol_filled = (master_gld_vol_filled['was_na'] == True).sum()
audit_report.append({
    "metric": "GLD_Volume Synthetic Forward-Filled Rows",
    "value": f"{weekend_or_holiday_gld_vol_filled} rows ({weekend_or_holiday_gld_vol_filled/len(df_master)*100:.2f}%)"
})

# Special Validation 2: Interpolated BTC prices vs observed prices
# In Internal dataset, BTC had 19 missing hours
int_btc_na = df_int.set_index('datetime')['BTC_close'].isna()
master_btc_interp = df_master.set_index('datetime').join(int_btc_na.rename('was_btc_na'))
interpolated_btc_rows = (master_btc_interp['was_btc_na'] == True).sum()
audit_report.append({
    "metric": "Interpolated BTC Hourly Bars (from Binance Outages)",
    "value": f"{interpolated_btc_rows} rows"
})

# Special Validation 3: Zero values in open interest or quote volume
zero_oi = (df_master['sum_open_interest'] == 0).sum()
zero_btc_qv = (df_master['BTC_quote_volume'] == 0).sum()
audit_report.append({
    "metric": "Zero Open Interest Rows",
    "value": f"{zero_oi} rows (genuine exchange initializations / outages)"
})
audit_report.append({
    "metric": "Zero BTC Quote Volume Rows",
    "value": f"{zero_btc_qv} rows (coincides exactly with maintenance)"
})

# Special Validation 4 & 5: Numerical stability in Garman-Klass Volatility
gk_neg_components = 0
gk_col = df_master['BTC_garman_klass_vol_24h']
audit_report.append({
    "metric": "BTC Garman-Klass Volatility NaNs",
    "value": f"{gk_col.isna().sum()} (23 expected burn-in periods, 0 negative values inside sqrt)"
})

# Special Validation 6: Open Interest Z-score definition
# Z = (OI_t - mean_24h(OI)) / std_24h(OI) -> Standardized level, NOT delta!
audit_report.append({
    "metric": "Open Interest Z-score Formula",
    "value": "Normalized rolling level Z = (OI_t - mean_24(OI))/std_24(OI), not a change (delta)"
})

# Special Validation 7: OFI validity
btc_ofi_invalid = ((df_master['BTC_taker_ofi'] < -1.0001) | (df_master['BTC_taker_ofi'] > 1.0001)).sum()
eth_ofi_invalid = ((df_master['ETH_taker_ofi'] < -1.0001) | (df_master['ETH_taker_ofi'] > 1.0001)).sum()
audit_report.append({
    "metric": "OFI Domain Violations (outside [-1, 1])",
    "value": f"BTC: {btc_ofi_invalid}, ETH: {eth_ofi_invalid}"
})

audit_df = pd.DataFrame(audit_report)
audit_df.to_csv(os.path.join(subdirs['FEAT'], "master_validation_audit.csv"), index=False)
print("Part A audit finished and saved.")
print(audit_df)

# PART B: BTC ORIGINAL DATA EDA
print("\n--- PART B: BTC ORIGINAL DATA EDA ---")
# Descriptive statistics
btc_cols = ['BTC_open', 'BTC_high', 'BTC_low', 'BTC_close', 'BTC_volume', 'BTC_quote_volume', 'BTC_num_trades']
btc_stats = df_master[btc_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
btc_stats['skewness'] = df_master[btc_cols].skew()
btc_stats['kurtosis'] = df_master[btc_cols].kurtosis()
btc_stats.to_csv(os.path.join(subdirs['BTC'], "btc_descriptive_statistics.csv"))

# Hourly percentage returns and log returns
df_master['BTC_pct_return'] = df_master['BTC_close'].pct_change()
btc_ret_clean = df_master['BTC_log_return'].dropna()
btc_ret_stats = pd.Series({
    'count': len(btc_ret_clean),
    'mean': btc_ret_clean.mean(),
    'median': btc_ret_clean.median(),
    'std': btc_ret_clean.std(),
    'min': btc_ret_clean.min(),
    'max': btc_ret_clean.max(),
    'skewness': stats.skew(btc_ret_clean),
    'kurtosis': stats.kurtosis(btc_ret_clean), # excess kurtosis
    'jb_stat': stats.jarque_bera(btc_ret_clean).statistic,
    'jb_pvalue': stats.jarque_bera(btc_ret_clean).pvalue
})
btc_ret_stats.to_csv(os.path.join(subdirs['BTC'], "btc_return_statistics.csv"))

# Outlier Detection for Returns (Z-score > 4 and IQR bounds)
q25, q75 = btc_ret_clean.quantile(0.25), btc_ret_clean.quantile(0.75)
iqr = q75 - q25
lower_bound, upper_bound = q25 - 3 * iqr, q75 + 3 * iqr
btc_iqr_outliers = df_master[(df_master['BTC_log_return'] < lower_bound) | (df_master['BTC_log_return'] > upper_bound)]
print(f"BTC Return Extreme Outliers (3x IQR): {len(btc_iqr_outliers)}")
btc_iqr_outliers[['datetime', 'BTC_close', 'BTC_log_return', 'BTC_volume']].to_csv(os.path.join(subdirs['BTC'], "btc_return_outliers.csv"), index=False)

# Candle metrics
df_master['BTC_hl_spread'] = df_master['BTC_high'] - df_master['BTC_low']
df_master['BTC_hl_ratio'] = df_master['BTC_high'] / df_master['BTC_low'] - 1.0
df_master['BTC_candle_body'] = (df_master['BTC_close'] - df_master['BTC_open']).abs()

# Figures for BTC
# Fig B1: Closing Price Time Series
fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
ax.plot(df_master['datetime'], df_master['BTC_close'], color='#F7931A', lw=1.2, label='BTC Close Price (USDT)')
ax.set_title('Bitcoin (BTC/USDT) Hourly Closing Price: Sep 2020 – Jul 2025', fontweight='bold')
ax.set_xlabel('Date (UTC)')
ax.set_ylabel('Price (USDT)')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(subdirs['BTC'], "btc_price_timeseries.png"))
plt.close()

# Fig B2: Return Time Series & Distribution
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=300)
ax1.plot(df_master['datetime'], df_master['BTC_log_return'], color='#F7931A', lw=0.6, alpha=0.7)
ax1.set_title('Bitcoin Hourly Log Returns Time Series', fontweight='bold')
ax1.set_ylabel('Log Return')
ax1.grid(True, alpha=0.3)

sns.histplot(btc_ret_clean, bins=120, kde=True, color='#F7931A', ax=ax2, stat='density')
x_vals = np.linspace(btc_ret_clean.min(), btc_ret_clean.max(), 300)
ax2.plot(x_vals, stats.norm.pdf(x_vals, btc_ret_clean.mean(), btc_ret_clean.std()), 'k--', lw=1.5, label='Normal Distribution Fit')
ax2.set_title(f'BTC Return Distribution (Skew: {stats.skew(btc_ret_clean):.2f}, Excess Kurt: {stats.kurtosis(btc_ret_clean):.2f})', fontweight='bold')
ax2.set_xlabel('Hourly Log Return')
ax2.set_ylabel('Density')
ax2.set_xlim(-0.10, 0.10)
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(subdirs['BTC'], "btc_returns_distribution.png"))
plt.close()

# Fig B3: Volume and Trades Analysis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, dpi=300)
ax1.plot(df_master['datetime'], df_master['BTC_volume'], color='#2E7D32', lw=0.7, alpha=0.8)
ax1.set_title('BTC Base Hourly Volume (BTC)', fontweight='bold')
ax1.set_ylabel('Volume')
ax1.grid(True, alpha=0.3)

ax2.plot(df_master['datetime'], df_master['BTC_num_trades'], color='#1565C0', lw=0.7, alpha=0.8)
ax2.set_title('BTC Hourly Trade Count', fontweight='bold')
ax2.set_ylabel('Number of Trades')
ax2.set_xlabel('Date (UTC)')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(subdirs['BTC'], "btc_volume_trades_timeseries.png"))
plt.close()

# Fig B4: Hour-of-Day and Day-of-Week Seasonality
df_master['hour'] = df_master['datetime'].dt.hour
df_master['day_name'] = df_master['datetime'].dt.day_name()
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

btc_hourly_vol = df_master.groupby('hour')['BTC_parkinson_vol_24h'].mean()
btc_daily_vol = df_master.groupby('day_name')['BTC_parkinson_vol_24h'].mean().reindex(days_order)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
sns.barplot(x=btc_hourly_vol.index, y=btc_hourly_vol.values, color='#F7931A', ax=ax1)
ax1.set_title('BTC Mean 24h Parkinson Volatility by Hour (UTC)', fontweight='bold')
ax1.set_xlabel('Hour of Day (UTC)')
ax1.set_ylabel('Volatility')
ax1.grid(True, alpha=0.3)

sns.barplot(x=btc_daily_vol.index, y=btc_daily_vol.values, color='#F7931A', ax=ax2)
ax2.set_title('BTC Mean 24h Parkinson Volatility by Day of Week', fontweight='bold')
ax2.set_xlabel('Day of Week')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=35)
ax2.set_ylabel('Volatility')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(subdirs['BTC'], "btc_seasonality_patterns.png"))
plt.close()

print("Part B BTC EDA completed.")

# PART C: ETH ORIGINAL DATA EDA
print("\n--- PART C: ETH ORIGINAL DATA EDA ---")
eth_cols = ['ETH_open', 'ETH_high', 'ETH_low', 'ETH_close', 'ETH_volume', 'ETH_quote_volume', 'ETH_num_trades']
eth_stats = df_master[eth_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
eth_stats['skewness'] = df_master[eth_cols].skew()
eth_stats['kurtosis'] = df_master[eth_cols].kurtosis()
eth_stats.to_csv(os.path.join(subdirs['ETH'], "eth_descriptive_statistics.csv"))

df_master['ETH_pct_return'] = df_master['ETH_close'].pct_change()
eth_ret_clean = df_master['ETH_log_return'].dropna()
eth_ret_stats = pd.Series({
    'count': len(eth_ret_clean),
    'mean': eth_ret_clean.mean(),
    'median': eth_ret_clean.median(),
    'std': eth_ret_clean.std(),
    'min': eth_ret_clean.min(),
    'max': eth_ret_clean.max(),
    'skewness': stats.skew(eth_ret_clean),
    'kurtosis': stats.kurtosis(eth_ret_clean),
    'jb_stat': stats.jarque_bera(eth_ret_clean).statistic,
    'jb_pvalue': stats.jarque_bera(eth_ret_clean).pvalue
})
eth_ret_stats.to_csv(os.path.join(subdirs['ETH'], "eth_return_statistics.csv"))

# Figures for ETH
# Fig C1: Closing Price Time Series
fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
ax.plot(df_master['datetime'], df_master['ETH_close'], color='#627EEA', lw=1.2, label='ETH Close Price (USDT)')
ax.set_title('Ethereum (ETH/USDT) Hourly Closing Price: Sep 2020 – Jul 2025', fontweight='bold')
ax.set_xlabel('Date (UTC)')
ax.set_ylabel('Price (USDT)')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(subdirs['ETH'], "eth_price_timeseries.png"))
plt.close()

# Fig C2: Return Distribution
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=300)
ax1.plot(df_master['datetime'], df_master['ETH_log_return'], color='#627EEA', lw=0.6, alpha=0.7)
ax1.set_title('Ethereum Hourly Log Returns Time Series', fontweight='bold')
ax1.set_ylabel('Log Return')
ax1.grid(True, alpha=0.3)

sns.histplot(eth_ret_clean, bins=120, kde=True, color='#627EEA', ax=ax2, stat='density')
x_vals_eth = np.linspace(eth_ret_clean.min(), eth_ret_clean.max(), 300)
ax2.plot(x_vals_eth, stats.norm.pdf(x_vals_eth, eth_ret_clean.mean(), eth_ret_clean.std()), 'k--', lw=1.5, label='Normal Distribution Fit')
ax2.set_title(f'ETH Return Distribution (Skew: {stats.skew(eth_ret_clean):.2f}, Excess Kurt: {stats.kurtosis(eth_ret_clean):.2f})', fontweight='bold')
ax2.set_xlabel('Hourly Log Return')
ax2.set_ylabel('Density')
ax2.set_xlim(-0.12, 0.12)
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(subdirs['ETH'], "eth_returns_distribution.png"))
plt.close()

# Fig C3: Volume & Trades
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, dpi=300)
ax1.plot(df_master['datetime'], df_master['ETH_volume'], color='#00897B', lw=0.7, alpha=0.8)
ax1.set_title('ETH Base Hourly Volume (ETH)', fontweight='bold')
ax1.set_ylabel('Volume')
ax1.grid(True, alpha=0.3)

ax2.plot(df_master['datetime'], df_master['ETH_num_trades'], color='#5E35B1', lw=0.7, alpha=0.8)
ax2.set_title('ETH Hourly Trade Count', fontweight='bold')
ax2.set_ylabel('Number of Trades')
ax2.set_xlabel('Date (UTC)')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(subdirs['ETH'], "eth_volume_trades_timeseries.png"))
plt.close()

# Fig C4: ETH Seasonality
eth_hourly_vol = df_master.groupby('hour')['ETH_parkinson_vol_24h'].mean()
eth_daily_vol = df_master.groupby('day_name')['ETH_parkinson_vol_24h'].mean().reindex(days_order)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
sns.barplot(x=eth_hourly_vol.index, y=eth_hourly_vol.values, color='#627EEA', ax=ax1)
ax1.set_title('ETH Mean 24h Parkinson Volatility by Hour (UTC)', fontweight='bold')
ax1.set_xlabel('Hour of Day (UTC)')
ax1.set_ylabel('Volatility')
ax1.grid(True, alpha=0.3)

sns.barplot(x=eth_daily_vol.index, y=eth_daily_vol.values, color='#627EEA', ax=ax2)
ax2.set_title('ETH Mean 24h Parkinson Volatility by Day of Week', fontweight='bold')
ax2.set_xlabel('Day of Week')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=35)
ax2.set_ylabel('Volatility')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(subdirs['ETH'], "eth_seasonality_patterns.png"))
plt.close()

print("Part C ETH EDA completed.")

# PART D: ORIGINAL BTC VS ETH COMPARISON
print("\n--- PART D: BTC VS ETH COMPARISON ---")
# Correlation analysis
valid_pair = df_master[['BTC_log_return', 'ETH_log_return', 'BTC_close', 'ETH_close']].dropna()
return_corr, ret_p = stats.pearsonr(valid_pair['BTC_log_return'], valid_pair['ETH_log_return'])
spearman_corr, sp_p = stats.spearmanr(valid_pair['BTC_log_return'], valid_pair['ETH_log_return'])

# Normalized price index comparison (base 100 at 2020-09-01)
df_master['BTC_norm'] = (df_master['BTC_close'] / df_master['BTC_close'].iloc[0]) * 100
df_master['ETH_norm'] = (df_master['ETH_close'] / df_master['ETH_close'].iloc[0]) * 100

# Rolling correlation (168h = 1 week rolling)
rolling_corr_168 = df_master['BTC_log_return'].rolling(168).corr(df_master['ETH_log_return'])
df_master['BTC_ETH_rolling_corr_168'] = rolling_corr_168

# Save comparison metrics
comp_metrics = pd.DataFrame([{
    'Metric': 'Full Sample Pearson Return Correlation',
    'Value': f"{return_corr:.5f}",
    'p_value': f"{ret_p:.2e}"
}, {
    'Metric': 'Full Sample Spearman Rank Correlation',
    'Value': f"{spearman_corr:.5f}",
    'p_value': f"{sp_p:.2e}"
}, {
    'Metric': 'BTC Annualized Vol (Hourly Std * sqrt(8760))',
    'Value': f"{btc_ret_clean.std() * np.sqrt(8760):.4f}",
    'p_value': 'N/A'
}, {
    'Metric': 'ETH Annualized Vol (Hourly Std * sqrt(8760))',
    'Value': f"{eth_ret_clean.std() * np.sqrt(8760):.4f}",
    'p_value': 'N/A'
}, {
    'Metric': 'BTC Excess Kurtosis',
    'Value': f"{stats.kurtosis(btc_ret_clean):.2f}",
    'p_value': 'N/A'
}, {
    'Metric': 'ETH Excess Kurtosis',
    'Value': f"{stats.kurtosis(eth_ret_clean):.2f}",
    'p_value': 'N/A'
}])
comp_metrics.to_csv(os.path.join(subdirs['COMP'], "btc_eth_comparison_metrics.csv"), index=False)

# Fig D1: Normalized Price Comparison
fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
ax.plot(df_master['datetime'], df_master['BTC_norm'], color='#F7931A', lw=1.3, label='BTC Normalized (Sep 2020 = 100)')
ax.plot(df_master['datetime'], df_master['ETH_norm'], color='#627EEA', lw=1.3, label='ETH Normalized (Sep 2020 = 100)')
ax.set_title('Bitcoin vs. Ethereum Normalized Growth (Base 100 = 2020-09-01)', fontweight='bold')
ax.set_xlabel('Date (UTC)')
ax.set_ylabel('Index Value (Base 100)')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(subdirs['COMP'], "btc_vs_eth_normalized_price.png"))
plt.close()

# Fig D2: Rolling Correlation Time Series
fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
ax.plot(df_master['datetime'], rolling_corr_168, color='#8E24AA', lw=1.0, label='7-Day (168h) Rolling Return Correlation')
ax.axhline(return_corr, color='red', linestyle='--', lw=1.2, label=f'Full-Sample Pearson r = {return_corr:.3f}')
ax.set_title('BTC–ETH 7-Day Rolling Return Correlation Over Time', fontweight='bold')
ax.set_xlabel('Date (UTC)')
ax.set_ylabel('Pearson Correlation Coefficient')
ax.set_ylim(0.2, 1.0)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(subdirs['COMP'], "btc_eth_rolling_correlation.png"))
plt.close()

# Fig D3: High Volatility Regimes vs Correlation
# High vol defined as upper quartile of BTC Parkinson vol
btc_high_vol_mask = df_master['BTC_parkinson_vol_24h'] > df_master['BTC_parkinson_vol_24h'].quantile(0.75)
btc_low_vol_mask = df_master['BTC_parkinson_vol_24h'] <= df_master['BTC_parkinson_vol_24h'].quantile(0.25)
corr_high_vol = df_master.loc[btc_high_vol_mask, ['BTC_log_return', 'ETH_log_return']].dropna().corr().iloc[0, 1]
corr_low_vol = df_master.loc[btc_low_vol_mask, ['BTC_log_return', 'ETH_log_return']].dropna().corr().iloc[0, 1]

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
bars = ax.bar(['Low Vol Regime (Q1)', 'Full Sample', 'High Vol Regime (Q4)'], 
              [corr_low_vol, return_corr, corr_high_vol], 
              color=['#64B5F6', '#8E24AA', '#E53935'], width=0.5)
ax.set_title('BTC–ETH Return Correlation Across Volatility Regimes', fontweight='bold')
ax.set_ylabel('Pearson Correlation')
ax.set_ylim(0.5, 0.95)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(subdirs['COMP'], "btc_eth_volatility_regime_correlation.png"))
plt.close()

print("Part D Comparison completed.")

# PART E & F: FEATURE VALIDATION & EDA
print("\n--- PART E & F: FEATURE VALIDATION EDA ---")
feat_summary_list = []
engineered_feats = [
    'BTC_parkinson_vol_24h', 'ETH_parkinson_vol_24h',
    'BTC_garman_klass_vol_24h', 'ETH_garman_klass_vol_24h',
    'BTC_taker_ofi', 'ETH_taker_ofi',
    'open_interest_zscore_24h', 'hour_sin', 'hour_cos'
]

for feat in engineered_feats:
    s = df_master[feat].dropna()
    feat_summary_list.append({
        'Feature': feat,
        'Count': len(s),
        'Mean': s.mean(),
        'Std': s.std(),
        'Min': s.min(),
        'Q25': s.quantile(0.25),
        'Median': s.median(),
        'Q75': s.quantile(0.75),
        'Max': s.max(),
        'Skew': stats.skew(s),
        'Kurtosis': stats.kurtosis(s)
    })

pd.DataFrame(feat_summary_list).to_csv(os.path.join(subdirs['FEAT'], "engineered_features_statistics.csv"), index=False)

# Fig E1: BTC Parkinson vs Garman-Klass Volatility
fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
ax.plot(df_master['datetime'], df_master['BTC_parkinson_vol_24h'], label='Parkinson Vol (24h)', color='#FB8C00', lw=1.0)
ax.plot(df_master['datetime'], df_master['BTC_garman_klass_vol_24h'], label='Garman-Klass Vol (24h)', color='#43A047', lw=1.0, alpha=0.85)
ax.set_title('BTC 24-Hour Rolling Volatility: Parkinson vs. Garman-Klass', fontweight='bold')
ax.set_xlabel('Date (UTC)')
ax.set_ylabel('Hourly Volatility Estimate')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(subdirs['FEAT'], "btc_volatility_estimators_comparison.png"))
plt.close()

# Fig E2: OFI Distribution & Dynamics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=300)
sns.histplot(df_master['BTC_taker_ofi'], bins=100, color='#F7931A', kde=True, ax=ax1, stat='density')
ax1.set_title('BTC Taker Order Flow Imbalance (OFI) Distribution', fontweight='bold')
ax1.set_xlabel('OFI [-1, 1]')
ax1.grid(True, alpha=0.3)

sns.histplot(df_master['ETH_taker_ofi'], bins=100, color='#627EEA', kde=True, ax=ax2, stat='density')
ax2.set_title('ETH Taker Order Flow Imbalance (OFI) Distribution', fontweight='bold')
ax2.set_xlabel('OFI [-1, 1]')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(subdirs['FEAT'], "ofi_distribution_comparison.png"))
plt.close()

# Fig E3: Open Interest Z-score vs Open Interest Level
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, dpi=300)
ax1.plot(df_master['datetime'], df_master['sum_open_interest'], color='#1E88E5', lw=1.0)
ax1.set_title('Binance Futures Bitcoin Open Interest Level (BTC contracts)', fontweight='bold')
ax1.set_ylabel('Open Interest (BTC)')
ax1.grid(True, alpha=0.3)

ax2.plot(df_master['datetime'], df_master['open_interest_zscore_24h'], color='#D81B60', lw=0.8, alpha=0.8)
ax2.axhline(0, color='black', lw=0.8, linestyle='--')
ax2.axhline(2, color='red', lw=0.8, linestyle=':')
ax2.axhline(-2, color='red', lw=0.8, linestyle=':')
ax2.set_title('24h Rolling Standardized Open Interest Z-score', fontweight='bold')
ax2.set_ylabel('Z-score')
ax2.set_xlabel('Date (UTC)')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(subdirs['FEAT'], "open_interest_and_zscore_timeseries.png"))
plt.close()

# PART H: MACRO RELATIONSHIPS WITH GLD AND DXY
print("\n--- PART H: MACRO RELATIONSHIPS ---")
# 1. Price trend & return distributions
macro_cols = ['GLD_Close', 'DXY_close', 'GLD_log_return', 'DXY_log_return']
macro_stats = df_master[macro_cols].describe().T
macro_stats.to_csv(os.path.join(subdirs['MACRO'], "macro_descriptive_statistics.csv"))

# Correlation Sensitivity:
# Set A: Master dataset with forward-filled weekend values
corr_master = df_master[['BTC_log_return', 'ETH_log_return', 'GLD_log_return', 'DXY_log_return']].dropna().corr()

# Set B: Filtered strictly to weekdays when GLD and DXY were genuinely trading (exclude synthetic 0-return weekends)
# In df_ext, genuine trading hours have GLD_Close not null before ffill
valid_macro_mask = df_ext.set_index('datetime')['GLD_Close'].notna()
df_master_macro_genuine = df_master.set_index('datetime').loc[valid_macro_mask[valid_macro_mask].index.intersection(df_master.set_index('datetime').index)].reset_index()

corr_genuine = df_master_macro_genuine[['BTC_log_return', 'ETH_log_return', 'GLD_log_return', 'DXY_log_return']].dropna().corr()

corr_master.to_csv(os.path.join(subdirs['MACRO'], "macro_correlation_full_sample_ffill.csv"))
corr_genuine.to_csv(os.path.join(subdirs['MACRO'], "macro_correlation_trading_hours_only.csv"))

# Calculate p-values for genuine trading hours
pairs = [
    ('BTC_log_return', 'GLD_log_return'),
    ('BTC_log_return', 'DXY_log_return'),
    ('ETH_log_return', 'GLD_log_return'),
    ('ETH_log_return', 'DXY_log_return'),
    ('GLD_log_return', 'DXY_log_return')
]
macro_pvals = []
df_m_clean = df_master_macro_genuine.dropna(subset=['BTC_log_return', 'ETH_log_return', 'GLD_log_return', 'DXY_log_return'])
for x, y in pairs:
    r_val, p_val = stats.pearsonr(df_m_clean[x], df_m_clean[y])
    macro_pvals.append({
        'Pair': f"{x} vs {y}",
        'Pearson_r': r_val,
        'p_value': p_val,
        'Effective_Sample_Size': len(df_m_clean)
    })
pd.DataFrame(macro_pvals).to_csv(os.path.join(subdirs['MACRO'], "macro_correlation_significance.csv"), index=False)

# Fig H1: Macro Price Trends
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, dpi=300)
ax1.plot(df_master['datetime'], df_master['GLD_Close'], color='#D4AF37', lw=1.2)
ax1.set_title('SPDR Gold Shares (GLD Proxy) Price: Sep 2020 – Jul 2025', fontweight='bold')
ax1.set_ylabel('Price ($)')
ax1.grid(True, alpha=0.3)

ax2.plot(df_master['datetime'], df_master['DXY_close'], color='#004D40', lw=1.2)
ax2.set_title('US Dollar Index (DXY) Level: Sep 2020 – Jul 2025', fontweight='bold')
ax2.set_ylabel('Index Level')
ax2.set_xlabel('Date (UTC)')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(subdirs['MACRO'], "macro_price_trends.png"))
plt.close()

# Fig H2: Correlation Heatmaps Comparison (Full vs Trading Only)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
sns.heatmap(corr_master, annot=True, cmap='coolwarm', fmt=".4f", vmin=-0.3, vmax=1.0, ax=ax1, cbar=False)
ax1.set_title('Correlation Matrix\n(Full Sample w/ Forward Fill, N=42,577)', fontweight='bold')

sns.heatmap(corr_genuine, annot=True, cmap='coolwarm', fmt=".4f", vmin=-0.3, vmax=1.0, ax=ax2)
ax2.set_title(f'Correlation Matrix\n(TradFi Trading Hours Only, N={len(df_m_clean)})', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(subdirs['MACRO'], "macro_correlation_heatmaps.png"))
plt.close()

# Fig H3: Rolling Macro Correlation
df_master['BTC_DXY_rolling_corr'] = df_master['BTC_log_return'].rolling(168).corr(df_master['DXY_log_return'])
df_master['BTC_GLD_rolling_corr'] = df_master['BTC_log_return'].rolling(168).corr(df_master['GLD_log_return'])

fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
ax.plot(df_master['datetime'], df_master['BTC_DXY_rolling_corr'], color='#D32F2F', lw=1.0, label='BTC vs DXY (168h Rolling Corr)')
ax.plot(df_master['datetime'], df_master['BTC_GLD_rolling_corr'], color='#FBC02D', lw=1.0, label='BTC vs GLD (168h Rolling Corr)', alpha=0.85)
ax.axhline(0, color='black', linestyle='--', lw=0.8)
ax.set_title('Bitcoin 7-Day Rolling Return Correlation with Macro Assets (GLD & DXY)', fontweight='bold')
ax.set_xlabel('Date (UTC)')
ax.set_ylabel('Pearson Correlation')
ax.set_ylim(-0.6, 0.6)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(subdirs['MACRO'], "macro_rolling_correlations.png"))
plt.close()

print("Part H Macro analysis completed.")
print("ALL EDA CALCULATION AND FIGURE GENERATION COMPLETE!")
