# Component 01 — Adaptive Multi-Timeframe Market Regime Intelligence Engine

Part of the group project *Agentic AI-Based Market Intelligence Framework for Bitcoin and
Ethereum Trading*. Everything belonging to this component lives in this folder,
`component_01/`.

Branch: `Multi_Timeframe_Market_Regime_Intelligence_Engine`

This component detects the market regime (Bullish, Bearish, Sideways, Volatile) of
BTCUSDT and ETHUSDT independently at five timeframes (Daily, 4-hour, 1-hour, 15-minute,
5-minute), and then measures how far those five answers agree. It describes the current
market state. It does not predict price.

### Status

| Module | Purpose | State |
|---|---|---|
| M1 | Data acquisition and validation | Built |
| M2 | Feature engineering (9 scale-free features) | Built |
| M3 | Regime detection (one Gaussian HMM per asset and timeframe) | Built |
| M4 | Regime fusion (dominant regime, Alignment Score, Confidence Score) | Built |
| M5 | Propagation between timeframes (direction and support depth of each regime flip) | Built |
| M9 | Hypothesis tests H1 and H2 | Built and run |
| Output | One JSON record per asset for the Fusion Engine | Built (proposal, see `docs/OUTPUT_FORMAT.md`) |
| M6 | Regime Stability Score (probability the Daily regime holds 3 more days) | Built and evaluated |
| M8 | Conflict archetypes with base rates, and the plain-language explanation | Built and tested |
| Novelty 3 | A method for, and test of, context-adaptive timeframe weights | Built and tested: not supported, so the fixed weights are retained |
| M7 | Historical similarity (the past stretches most like the last 10 days) | Built |

### Setup

Python 3.13 on Windows (PowerShell). **Run every command from inside the
`component_01` folder.** The Python environment is created one level up, at the
repository root, because the repository path is long and Windows limits file paths to
260 characters:

```
cd component_01
python -m venv ..\.venv
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` pins the exact library versions used for the committed results.

### Running the pipeline

Run the steps in this order. Each one prints a report of what it produced.

| Step | Command | Time | Output |
|---|---|---|---|
| 1. Download data | `..\.venv\Scripts\python.exe -m src.data.download` | long, about 2.1 million candles | `data/parquet/` |
| 2. Check data | `..\.venv\Scripts\python.exe -m src.data.validate` | seconds | report only |
| 3. Build features | `..\.venv\Scripts\python.exe -m src.features.indicators` | about 1 minute | `data/features/` |
| 3b. Explore the data (EDA) | `..\.venv\Scripts\python.exe -W ignore -m src.eda.explore` | about 1 minute | `results/eda/` |
| 4. Train regime models | `..\.venv\Scripts\python.exe -W ignore -m src.regime.train --interval 1d` | minutes for `1d`; over an hour for `5m` | `models/`, `data/regimes/`, `results/` |
| 4b. State-count check (BIC) for 1h, 15m, 5m | `..\.venv\Scripts\python.exe -W ignore -m src.regime.bic_check --symbol BTCUSDT --interval 1h` (then `--summary`) | about 40 min for `1h`, 1 hour for `15m`, 4 hours for `5m` | `results/bic/` |
| 4c. Convergence check, BTC 5m | `..\.venv\Scripts\python.exe -W ignore -m src.regime.convergence_check` | about 6 minutes | `results/convergence_check_BTCUSDT_5m.json` |
| 5. Fuse the timeframes | `..\.venv\Scripts\python.exe -W ignore -m src.fusion.alignment` | about 10 seconds | `data/fusion/` |
| 6. Propagation | `..\.venv\Scripts\python.exe -W ignore -m src.fusion.propagation` | seconds | `data/propagation/` |
| 7. Test H1 | `..\.venv\Scripts\python.exe -W ignore -m src.experiments.h1_precursor` | about 4 minutes | `results/h1_*` |
| 8. Test H2 | `..\.venv\Scripts\python.exe -W ignore -m src.experiments.h2_propagation --period test` | about 2 minutes | `results/h2_*` |
| 8b. H2 placebo control | `..\.venv\Scripts\python.exe -W ignore -m src.experiments.h2_placebo` | about 2 minutes | `results/h2_placebo_test.json` |
| 8c. Test H2b | `..\.venv\Scripts\python.exe -W ignore -m src.experiments.h2b_failed_propagation --period test` | about 3 minutes | `results/h2b_*` |
| 9. Stability score | `..\.venv\Scripts\python.exe -W ignore -m src.stability.stability` | about 20 seconds | `data/stability/`, `results/stability_model_*` |
| 9b. Evaluate it | `..\.venv\Scripts\python.exe -W ignore -m src.experiments.m6_evaluation --period test` | seconds | `results/m6_*` |
| 9c. Model-quality charts | `..\.venv\Scripts\python.exe -W ignore -m src.figures.model_quality_charts` | seconds | `results/model_quality/` |
| 10. Output record | `..\.venv\Scripts\python.exe -W ignore -m src.output.snapshot --symbol BTCUSDT` | seconds | printed JSON |

Step 4 is run once per timeframe: `1d`, `4h`, `1h`, `15m`, `5m`. Two options help:

- `--skip-bic` skips the comparison of state counts, which is slow on `1h`, `15m` and `5m`.
- `--relabel` reuses a model that is already trained and only reprints its report and
  rewrites its outputs. Use it to view results without retraining.

### Where the results are

- `results/` holds, for every asset and timeframe, a chart of the detected regimes
  (`*_regimes.png`) and a full record of the model (`*_hmm_report.json`): settings,
  random seed, state profiles, regime shares, transition matrix and transition counts.
- `results/h1_*` and `results/h2_*` hold the hypothesis test results, with the design
  settings recorded inside each JSON file.
- `results/sample_output_*.json` are example records for the Fusion Engine.
- `results/eda/` holds the exploratory data analysis: charts, tables and a summary
  (`EDA_SUMMARY.md`) of data quality, price history, volatility over the years, the
  shape of returns, feature correlations and activity by hour and weekday.
- `results/model_quality/` holds four charts of model quality, drawn from the result
  files above (fit on unseen data, state count, meaning of the regimes on unseen data,
  and the Stability Score against its baselines), with `MODEL_QUALITY_SUMMARY.md`. The
  regime models are unsupervised, so there is no accuracy; these charts show how they
  are judged instead.
- `results/bic/` and `results/convergence_check_*.json` hold the two training checks
  (steps 4b and 4c). Neither changes a saved model or label. In all ten models BIC
  keeps improving up to six states, so four states is a design choice (one per named
  regime), not a number the data selects. Training the BTC 5m model, which stopped at
  the iteration limit, on to convergence changed 0.11% of its labels.
- `data/` and `models/` (inside `component_01/`) are not in Git because of their size.
  Steps 1 to 6 rebuild them.

### Findings so far

Models were trained on data up to 2022-12-31 and tested, unchanged, on 2023–2026.
An effect of 0.5 means no difference.

| Hypothesis | Result |
|---|---|
| H1: cross-timeframe alignment declines before a Daily regime transition | **Not supported.** BTC 0.398 (95% interval 0.337–0.460), ETH 0.440 (0.377–0.503). For BTC alignment was slightly higher before transitions. Alignment added nothing beyond volatility. |
| H2: regime changes already present at faster timeframes are more durable | **Not established at the Daily level** (the effect is no larger than a placebo). **A small effect at the 1h and 15m levels** that exceeds the placebo by 0.025–0.086. |
| H2b: a regime change that the next slower timeframe does not take up is noise | **Not established for 4h changes taken up by Daily** (too few cases). **A small effect at the 5m level and for ETH at 15m** that exceeds the placebo by 0.040–0.071. |

**Regime Stability Score (M6), test period.** The delivered score (the Daily model's own
signals) beats both baselines for both assets:

| | BTC | ETH |
|---|---|---|
| Brier score: constant baseline / HMM-only baseline / delivered score (lower is better) | 0.182 / 0.185 / 0.151 | 0.223 / 0.223 / 0.199 |
| AUC of the delivered score (0.5 = no skill) | 0.734 | 0.665 |
| Adding the cross-timeframe signals | a small further gain | no clear gain |

It is well calibrated for BTC except at the top, and too confident for ETH at high values.
The model was chosen on training data before this evaluation.

**Conflict archetypes (Novelty 4), test period.** The four patterns carry information:
the Daily regime changed within 3 days on 31% (BTC) and 35% (ETH) of "Opposed" days
against 14% and 20% of "Aligned" days. Using the training base rates as a forecast beat
a constant rate for BTC but not clearly for ETH, and every rate was higher in the test
period than in training, so base rates are relative, not exact.

**Context-adaptive weights (Novelty 3), test period. Not supported.** Weights learned
per timeframe, and weights learned per timeframe and volatility context, were no better
than the fixed weights at judging whether the Daily regime would hold (Brier, adaptive
minus fixed: BTC +0.0028, ETH +0.0013; both intervals include zero). The fixed weights
are kept. The 4-hour timeframe carries almost all of the learned weight.

The H1, H2 and H2b designs were fixed before their results were computed. The H2 placebo
control was added after the H2 result was known and is reported as an additional check.

### Tests

```
..\.venv\Scripts\python.exe -W ignore -m pytest
```

The tests check, among other things, that the features do not depend on the price
level, and that no feature, regime label or alignment value uses data from the future.

### Method notes

- Data source: Binance Vision public spot klines, <https://data.binance.vision>.
- All settings (indicator periods, training cut-off, model options, fusion weights) are
  in `src/config.py`, each with its reason.
- What the four regime names mean in each model is stated in
  `docs/REGIME_DEFINITIONS.md`. In particular, "Volatile" is the more active of the two
  non-directional states and is not always the most volatile state.
- Models are fitted on data up to 2022-12-31 and applied unchanged to later data.
- Regime labels use past data only (filtered probabilities), and the fusion step uses
  only candles that have already closed.
