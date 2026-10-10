# Agentic-AI-Based-Market-Intelligence-Framework-For-Bitcoin-and-Ethereum-Trading

## Component 02 — Adaptive Multi-Timeframe Market Regime Intelligence Engine

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
| M6–M8 | Stability score, historical similarity, explanation | Not built yet |

### Setup

Python 3.13 on Windows (PowerShell), from the repository folder:

```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` pins the exact library versions used for the committed results.

### Running the pipeline

Run the steps in this order. Each one prints a report of what it produced.

| Step | Command | Time | Output |
|---|---|---|---|
| 1. Download data | `.\.venv\Scripts\python.exe -m src.data.download` | long, about 2.1 million candles | `data/parquet/` |
| 2. Check data | `.\.venv\Scripts\python.exe -m src.data.validate` | seconds | report only |
| 3. Build features | `.\.venv\Scripts\python.exe -m src.features.indicators` | about 1 minute | `data/features/` |
| 4. Train regime models | `.\.venv\Scripts\python.exe -W ignore -m src.regime.train --interval 1d` | minutes for `1d`; over an hour for `5m` | `models/`, `data/regimes/`, `results/` |
| 5. Fuse the timeframes | `.\.venv\Scripts\python.exe -W ignore -m src.fusion.alignment` | about 10 seconds | `data/fusion/` |
| 6. Propagation | `.\.venv\Scripts\python.exe -W ignore -m src.fusion.propagation` | seconds | `data/propagation/` |
| 7. Test H1 | `.\.venv\Scripts\python.exe -W ignore -m src.experiments.h1_precursor` | about 1 minute | `results/h1_*` |
| 8. Test H2 | `.\.venv\Scripts\python.exe -W ignore -m src.experiments.h2_propagation --period test` | about 1 minute | `results/h2_*` |
| 9. Output record | `.\.venv\Scripts\python.exe -W ignore -m src.output.snapshot --symbol BTCUSDT` | seconds | printed JSON |

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
- `data/` and `models/` are not in Git because of their size. Steps 1 to 6 rebuild them.

### Tests

```
.\.venv\Scripts\python.exe -W ignore -m pytest
```

The tests check, among other things, that the features do not depend on the price
level, and that no feature, regime label or alignment value uses data from the future.

### Method notes

- Data source: Binance Vision public spot klines, <https://data.binance.vision>.
- All settings (indicator periods, training cut-off, model options, fusion weights) are
  in `src/config.py`, each with its reason.
- Models are fitted on data up to 2022-12-31 and applied unchanged to later data.
- Regime labels use past data only (filtered probabilities), and the fusion step uses
  only candles that have already closed.
