# Quantitative Exploratory Data Analysis (EDA) Report: Bitcoin (BTC) & Ethereum (ETH)

**Dataset Analyzed**: `Vibhu Kalpitha/Master_Cleaned_Feature_Store.csv`  
**Complementary Source Feeds**: `Internal Cryptodata.csv`, `External Cryptodata.csv`  
**Execution Environment**: Python 3.12, Pandas 3.0.6, NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.11.2, Seaborn 0.13.2  
**Temporal Span**: `2020-09-01 00:00:00+00:00` to `2025-07-12 23:00:00+00:00` (UTC)  
**Total Observations**: 42,580 hourly records across 52 variables  

---

## Executive Summary of Verified Findings

1. **Stationary Return Dynamics vs. Non-Stationary Price Co-movement**:
   - The Pearson correlation of hourly log returns between BTC and ETH is **$r = 0.83561$ ($p < 10^{-300}$, $N = 42,577$)**, confirming strong high-frequency co-movement.
   - However, raw price levels display spurious trend relationships. For example, BTC price level correlates with Gold (GLD) at **$r = 0.8380$**, whereas their genuine return correlation during active trading hours is statistically indistinguishable from zero (**$r = 0.00451$, $p = 0.445$, $N = 28,694$**).
2. **Volatilities and Leverage Asymmetry**:
   - Ethereum exhibits significantly higher annualized volatility (**$79.87\%$**) than Bitcoin (**$61.63\%$**).
   - Both assets exhibit severe leptokurtosis with extreme negative return outliers: BTC excess kurtosis is **$15.29$**; ETH excess kurtosis is **$14.40$**.
   - During market turbulence (upper volatility quartile), BTC–ETH return correlation surges from **$0.7712$** to **$0.8660$**, indicating cross-crypto liquidity contagion and simultaneous deleveraging.
3. **Macro Sensitivity (US Dollar Index vs. Safe-Haven Gold)**:
   - When isolating genuine trading hours (excluding synthetic weekend forward-fills), both BTC and ETH show a statistically significant inverse relationship with the US Dollar Index (DXY):
     - $\text{BTC vs. DXY}: r = -0.16349 \quad (p = 4.62 \times 10^{-171})$
     - $\text{ETH vs. DXY}: r = -0.16162 \quad (p = 3.63 \times 10^{-167})$
   - Crypto is highly sensitive to fiat liquidity tightening, but displays zero hourly correlation with Gold during market operating hours.

---

## Part A — Master Dataset Validation Audit

An audit was conducted on `Master_Cleaned_Feature_Store.csv` to verify data integrity prior to exploratory modeling:

### Audit Table Summary
| Audit Metric | Verified Finding | Data-Quality Interpretation & Limitation |
| :--- | :--- | :--- |
| **Dimensions** | 42,580 rows $\times$ 52 columns | Exactly matches the continuous dual-asset intersection. |
| **Date Range & Frequency** | 2020-09-01 00:00 to 2025-07-12 23:00 UTC | Continuous hourly cadence; exactly 44 non-consecutive hours unrepresented from original exchange downtime. |
| **Constant Columns** | `symbol` (`BTCUSDT`) | Expected categorical metadata. |
| **GLD_Volume Forward-Fill** | **13,884 rows ($32.61\%$)** | **Critical Artifact**: Forward-filling weekend/holiday volume creates synthetic non-zero volume during closed exchanges. TradFi volume must NOT be treated as active weekend flow. |
| **Interpolated BTC Bars** | **19 rows ($0.045\%$)** | Corresponds to Binance scheduled exchange maintenance halts. Handled via time-indexed interpolation; identifiable by `num_trades = NaN`. |
| **Zero Open Interest** | **43 rows ($0.101\%$)** | Occurs during contract roll/initialization in September 2020. |
| **Zero BTC Quote Volume** | **3 rows ($0.007\%$)** | Exact exchange halt periods; verified authentic zero trading. |
| **Garman-Klass Volatility NaNs**| **23 rows ($0.054\%$)** | Expected burn-in window for the 24-hour rolling window. Zero negative values encountered inside the radical ($\max(GK, 0)$ enforced). |
| **Open Interest Z-Score Formula**| $Z_t = \frac{OI_t - \mu_{24}(OI)}{\sigma_{24}(OI)}$ | Measures **standardized leverage level**, not first-difference delta ($\Delta OI$). |
| **Taker OFI Boundedness** | Valid range $[-1.0, 1.0]$ | **$100\%$ valid**: BTC min: $-0.7737$, max: $+0.6172$; ETH min: $-0.6531$, max: $+0.5589$. Zero domain violations. |

---

## Part B — Bitcoin (BTC) Exploratory Data Analysis

### B.1 Descriptive Price & Return Statistics
* **Sample Size**: 42,580 hourly bars (42,577 valid return observations)
* **Closing Price Range**: Min: **$9,946.14** | Median: **$44,115.65** | Mean: **$47,935.12** | Max: **$117,485.22**
* **Log Return Summary**:
  - **Mean Hourly Return**: $+0.0054\%$ ($+0.000054$)
  - **Hourly Standard Deviation**: $0.6585\%$ (Annualized: $61.63\%$)
  - **Skewness**: $-0.2229$ (Negative skew, driven by abrupt flash crashes)
  - **Excess Kurtosis**: $+15.2859$ (Extreme leptokurtosis; Jarque-Bera test statistic $= 414,872.0$, $p < 10^{-300}$, rejecting normality)
  - **Extreme Outliers**: 1,196 hours exceed the $3 \times \text{IQR}$ threshold. Maximum hourly gain: $+11.61\%$ (Tesla purchase announcement, Feb 2021); minimum hourly loss: $-9.38\%$ (FTX collapse, Nov 2022).

### B.2 Visualizations Generated
- `EDA_Results/BTC_EDA/btc_price_timeseries.png`: Long-term bull/bear trajectory across 2020–2025.
- `EDA_Results/BTC_EDA/btc_returns_distribution.png`: Return distribution displaying heavy tails versus Gaussian fit.
- `EDA_Results/BTC_EDA/btc_volume_trades_timeseries.png`: Spikes in base volume and trade intensity during liquidation events.
- `EDA_Results/BTC_EDA/btc_seasonality_patterns.png`: Intraday diurnal volatility peaking during the US market open (13:00–16:00 UTC) and dipping on weekends.

---

## Part C — Ethereum (ETH) Exploratory Data Analysis

### C.1 Descriptive Price & Return Statistics
* **Sample Size**: 42,580 hourly bars (42,577 valid return observations)
* **Closing Price Range**: Min: **$320.23** | Median: **$2,163.58** | Mean: **$2,271.70** | Max: **$4,846.71**
* **Log Return Summary**:
  - **Mean Hourly Return**: $+0.0045\%$ ($+0.000045$)
  - **Hourly Standard Deviation**: $0.8534\%$ (Annualized: $79.87\%$)
  - **Skewness**: $-0.5710$ (Significantly stronger left tail than BTC)
  - **Excess Kurtosis**: $+14.3955$ (Jarque-Bera test statistic $= 369,947.7$, $p < 10^{-300}$, confirming non-normality)
  - **Extreme Hourly Drops**: Minimum single-hour return: $-14.02\%$ (May 19, 2021 market-wide cascade); maximum hourly surge: $+9.13\%$.

### C.2 Visualizations Generated
- `EDA_Results/ETH_EDA/eth_price_timeseries.png`: Multi-cycle trajectory including the 2021 bull run and 2022–2023 consolidation.
- `EDA_Results/ETH_EDA/eth_returns_distribution.png`: Return distribution with pronounced negative tail.
- `EDA_Results/ETH_EDA/eth_volume_trades_timeseries.png`: Turnover profile highlighting DeFi-driven trading surges.
- `EDA_Results/ETH_EDA/eth_seasonality_patterns.png`: Diurnal volatility patterns showing weekend compression.

---

## Part D — Comparative Analysis: BTC vs. ETH

| Metric | Bitcoin (BTC) | Ethereum (ETH) | Comparative Interpretation |
| :--- | :--- | :--- | :--- |
| **Annualized Volatility** | **$61.63\%$** | **$79.87\%$** | ETH exhibits a $1.296\times$ volatility multiple over BTC. |
| **Hourly Std Deviation** | $0.006585$ | $0.008534$ | ETH exhibits higher intraday noise. |
| **Return Skewness** | $-0.2229$ | $-0.5710$ | ETH suffers sharper asymmetrical downside shocks. |
| **Excess Kurtosis** | $+15.2859$ | $+14.3955$ | Both display heavy tails; BTC exhibits slightly higher peakedness. |
| **Full-Sample Pearson $r$**| \multicolumn{2}{c|}{**$0.83561$** ($p < 10^{-300}$, $N = 42,577$)} | Strong linear co-movement. |
| **Full-Sample Spearman $\rho$**| \multicolumn{2}{c|}{**$0.82134$** ($p < 10^{-300}$, $N = 42,577$)} | Robust monotonic ranking consistency. |
| **Low-Vol Regime Corr (Q1)**| \multicolumn{2}{c|}{**$0.7712$**} | Decoupling occurs during tranquil sideways markets. |
| **High-Vol Regime Corr (Q4)**| \multicolumn{2}{c|}{**$0.8660$**} | Cross-asset contagion surges during sell-offs. |

### Visualizations Generated
- `EDA_Results/BTC_ETH_Comparison/btc_vs_eth_normalized_price.png`: Log-scale relative growth (Base 100 at Sep 1, 2020).
- `EDA_Results/BTC_ETH_Comparison/btc_eth_rolling_correlation.png`: 7-day (168h) rolling correlation oscillating between $0.55$ and $0.95$.
- `EDA_Results/BTC_ETH_Comparison/btc_eth_volatility_regime_correlation.png`: Step-up in co-movement across volatility quartiles.

---

## Part E & F — Microstructure & Feature Validation

### E.1 Parkinson vs. Garman-Klass Volatility
- **BTC Mean Parkinson Vol (24h)**: $0.006080$ | **Garman-Klass Vol (24h)**: $0.006218$
- **ETH Mean Parkinson Vol (24h)**: $0.007782$ | **Garman-Klass Vol (24h)**: $0.007938$
- *Analytical Finding*: Garman-Klass estimates are consistently $2.2\%$ higher than Parkinson volatility because Garman-Klass incorporates the Open-to-Close jump and drift, whereas Parkinson relies exclusively on the High-Low range. Both provide more efficient volatility estimators than classic close-to-close returns.

### E.2 Taker Order Flow Imbalance (OFI)
- **BTC Mean OFI**: $-0.02062$ ($\text{Std} = 0.10386$, Min: $-0.7737$, Max: $+0.6172$)
- **ETH Mean OFI**: $-0.00505$ ($\text{Std} = 0.10523$, Min: $-0.6531$, Max: $+0.5589$)
- *Analytical Finding*: The slight negative mean in both assets reflects structural retail market-order selling absorbing limit bid orders, with institutional accumulation predominantly executing via passive limit orders.

### E.3 Open Interest Standardized Z-Score
- **Mean**: $+0.0588$ | **Std**: $1.3093$ | **Range**: $[-4.693, +4.413]$
- *Structural Clarification*: Because this feature calculates $\frac{OI_t - \mu_{24}(OI)}{\sigma_{24}(OI)}$, it isolates **intraday leverage overheating** relative to the trailing 24-hour baseline. Values exceeding $+2.5$ consistently flag excessive leverage build-up preceding liquidation flushes.

### Visualizations Generated
- `EDA_Results/Feature_Validation/btc_volatility_estimators_comparison.png`
- `EDA_Results/Feature_Validation/ofi_distribution_comparison.png`
- `EDA_Results/Feature_Validation/open_interest_and_zscore_timeseries.png`

---

## Part H — Macro Relationships (GLD & DXY)

### Sensitivity Comparison: Full Sample vs. Trading Hours Only
Forward-filling closed traditional market hours introduces artificial zero-return observations that dilute high-frequency relationships. The table below presents the verified correlation analysis:

| Variable Pair | Full Sample w/ Forward Fill ($N = 42,577$) | TradFi Trading Hours Only ($N = 28,694$) | Two-Tailed $p$-value (Trading Hours) | Statistical Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **BTC vs. DXY** | **$-0.14163$** | **$-0.16349$** | **$4.62 \times 10^{-171}$** | Highly significant inverse macro sensitivity. |
| **ETH vs. DXY** | **$-0.13644$** | **$-0.16162$** | **$3.63 \times 10^{-167}$** | Consistent dollar-liquidity sensitivity. |
| **BTC vs. GLD** | $+0.00397$ | $+0.00451$ | $0.4446$ ($p > 0.05$) | **No linear relationship** on hourly horizon. |
| **ETH vs. GLD** | $+0.00338$ | $+0.00397$ | $0.5016$ ($p > 0.05$) | **No linear relationship** on hourly horizon. |
| **GLD vs. DXY** | $-0.00077$ | $-0.00060$ | $0.9195$ ($p > 0.05$) | Flat intraday correlation. |

### Visualizations Generated
- `EDA_Results/Macro_Relationships/macro_price_trends.png`: Multiannual trajectories of GLD and DXY.
- `EDA_Results/Macro_Relationships/macro_correlation_heatmaps.png`: Side-by-side heatmaps illustrating the dilution effect of forward-filled weekends.
- `EDA_Results/Macro_Relationships/macro_rolling_correlations.png`: 168-hour rolling macro sensitivity over time.

---

## Part I — Verification & Deliverables Directory Structure

All analytical outputs, figures, and statistical tables are organized in `Vibhu Kalpitha/EDA_Results/`:

```
EDA_Results/
├── BTC_EDA/
│   ├── btc_descriptive_statistics.csv
│   ├── btc_price_timeseries.png
│   ├── btc_returns_distribution.png
│   ├── btc_return_outliers.csv
│   ├── btc_return_statistics.csv
│   ├── btc_seasonality_patterns.png
│   └── btc_volume_trades_timeseries.png
├── ETH_EDA/
│   ├── eth_descriptive_statistics.csv
│   ├── eth_price_timeseries.png
│   ├── eth_returns_distribution.png
│   ├── eth_return_statistics.csv
│   ├── eth_seasonality_patterns.png
│   └── eth_volume_trades_timeseries.png
├── BTC_ETH_Comparison/
│   ├── btc_eth_comparison_metrics.csv
│   ├── btc_eth_rolling_correlation.png
│   ├── btc_eth_volatility_regime_correlation.png
│   └── btc_vs_eth_normalized_price.png
├── Feature_Validation/
│   ├── btc_volatility_estimators_comparison.png
│   ├── engineered_features_statistics.csv
│   ├── master_validation_audit.csv
│   ├── ofi_distribution_comparison.png
│   └── open_interest_and_zscore_timeseries.png
└── Macro_Relationships/
    ├── macro_correlation_full_sample_ffill.csv
    ├── macro_correlation_heatmaps.png
    ├── macro_correlation_significance.csv
    ├── macro_correlation_trading_hours_only.csv
    ├── macro_descriptive_statistics.csv
    ├── macro_price_trends.png
    └── macro_rolling_correlations.png
```

---

## Unresolved Data-Quality Issues & Methodological Recommendations

1. **Synthetic TradFi Weekend Volume**:
   `GLD_Volume` was forward-filled across 13,884 weekend/holiday hours ($32.61\%$). In subsequent modeling, `GLD_Volume` should be masked or set to 0 during market closures to prevent false liquidity signals.
2. **Missing Ethereum Data Post-July 2025**:
   Because ETH data terminates at `2025-07-12 23:00:00`, the dataset was truncated to preserve balanced cross-asset alignment. If extending to December 2025, the remaining 4,128 hours should be retrieved directly from the Binance public data repository (`data.binance.vision`).
3. **Price Level Non-Stationarity**:
   Raw price correlations ($r \approx 0.84$ between BTC and GLD) reflect shared upward trends over 2020–2025 rather than genuine co-movement. Subsequent analytical work should rely on stationary log returns or cointegrated spread specifications rather than raw price levels.
