# CLAUDE.md — Project Context

> **Place this file at the root of the repository.** Claude Code reads it automatically
> and will then have full context on every request without needing re-explanation.

---

## 1. Who and What

| | |
|---|---|
| **Student** | K.A.U. Gimhani — IT23269866 |
| **Institution** | Sri Lanka Institute of Information Technology (SLIIT) |
| **Project ID** | J26-DS-312 |
| **Module** | IT4010 Research Project |
| **Group project** | Agentic AI-Based Market Intelligence Framework for Bitcoin and Ethereum Trading |
| **My component** | **Adaptive Multi-Timeframe Market Regime Intelligence Engine** (Component 02) |
| **Git branch** | `Multi_Timeframe_Market_Regime_Intelligence_Engine` |
| **Supervisor** | Ms. Malithi Nawarathne |
| **Co-supervisor** | Ms. Fathima Fanoon |

This is an **undergraduate research project**, not a commercial product. Correctness,
reproducibility and defensibility matter more than features or polish. Every design
choice must be explainable to an examination panel.

---

## 2. What My Component Does

It takes Bitcoin and Ethereum price data and independently determines the market
**regime** — Bullish, Bearish, Sideways, or Volatile — at five different timeframes.
It then studies the **relationship between those five answers**: whether they agree,
where any disagreement started, and in which direction it is spreading.

**It does NOT predict price.** It describes the current market state, assesses how
stable that state is, and explains its reasoning in plain language.

One-sentence version:

> My component independently detects the market regime at five timeframes, then studies
> the disagreement between them — because my claim is that this disagreement is an early
> warning that the regime is about to change.

### Where it sits in the group project

Four independent engines run in parallel; a Fusion Intelligence Engine combines them.

| Engine | Owner | Question |
|---|---|---|
| **Multi-Timeframe Market Regime (this one)** | Gimhani | What is the market doing now, and how stable is it? |
| Retrieval-Augmented Cross-Market & Capital Flow | Kalpitha W.K.V | Why is the market moving? |
| AI-Based Market Integrity | R P S Jayawardana | Can current market behaviour be trusted? |
| Evidence-Guided Market Scenario | H A A Dilshan | What plausible futures could follow? |

This component has **no upstream dependency** on the other three. The only hard
dependency is downstream: the Fusion Engine consumes this component's output.

---

## 3. The Central Research Hypothesis

Everything in this component exists to test one claim:

> **Cross-timeframe regime disagreement is not noise to be averaged away.
> It is structured, directional, and predictive of regime change.**

This is what makes the work research rather than engineering: **it can be false.**
If alignment does not fall before regime transitions, the hypothesis is refuted — and
that is still a valid, reportable finding, because nobody has tested it.

Two formal hypotheses:

- **H1** — Cross-timeframe alignment declines measurably in the period preceding a
  regime transition, compared with matched stable periods.
- **H2** — Regime changes propagate directionally between timeframes, and cascade depth
  distinguishes durable change from noise.

---

## 4. The Four Novelties

### Novelty 1 — Conflict as a regime-transition precursor
Alignment is computed as a **continuous time series**, not a snapshot. Its level, its
slope, and the HMM transition probabilities combine into a **Regime Stability Score** —
the estimated probability that the current regime persists.

*Why novel:* existing regime-transition early-warning research uses signals from
**outside** the asset (cross-asset spillover networks, topological structure). Using an
asset's own internal timeframe disagreement as the precursor appears unexplored.

### Novelty 2 — Regime propagation direction and cascade depth
Measures **where** a regime change started and **how far** it travelled up the hierarchy.

Three diagnosable states:
- **Bottom-up cascade** — starts short, climbs upward → genuine new trend
- **Top-down decay** — long timeframe weakens first → structural exhaustion
- **Failed propagation** — flip never reaches the next level → it was noise

That third case is the most practically valuable output: a principled, learned filter
for false breakouts.

*Why novel:* lead-lag analysis in finance is almost entirely **cross-asset**. Applying
it **between timeframes of the same asset** appears unexplored.

### Novelty 3 — Context-adaptive timeframe weighting
Weights are **not fixed**. In calm trending markets the short timeframes are mostly
noise; in high-volatility markets the long timeframes are slow and stale. The system
learns which timeframe is informative under which conditions.

*Also:* this is what makes "Adaptive" in the component name mean something specific
beyond periodic retraining.

### Novelty 4 — Conflict archetypes with outcome-calibrated explanation
Disagreement patterns are classified into named archetypes, each paired with a
**historical base rate** — e.g. "this pattern resolved into full regime change 38% of
the time." The explanation becomes a diagnosis with evidence, not template prose.

*Why novel:* financial explainability is usually generic (SHAP/LIME on a prediction).
A domain-specific conflict taxonomy with measured base rates is a different kind of
explanation.

---

## 5. The Five Timeframes

**Daily → 4-Hour → 1-Hour → 15-Minute → 5-Minute**

Changed in October 2026 on the lecturer's instruction (previously Monthly → 1-Hour).

### Justification (use this, not the old one)
> These five cover the full intraday decision range — from the daily structure that
> frames the day, down to the five-minute level where entries actually happen. The noisy
> short timeframes are included **deliberately**, because noise is part of what is being
> studied: Novelty 2 tests whether a change starting at 5-minute propagates upward or
> dies out. That can only be tested if the short timeframes are present.

### ⚠️ Do NOT use this old justification
> ~~"Below 1-Hour the data becomes too noisy for regime detection"~~

The set now includes 15m and 5m. That statement contradicts the design and will be
caught immediately.

---

## 6. The Data — Already Acquired and Verified

**Phase 1 is complete.** Do not re-download unless explicitly asked.

| | |
|---|---|
| **Source** | Binance Vision — `https://data.binance.vision` |
| **Docs** | `https://github.com/binance/binance-public-data` |
| **Assets** | BTCUSDT, ETHUSDT (Binance Spot) |
| **Format** | Parquet |
| **Location** | `data/parquet/` |
| **Total** | ~2.1 million candles across 10 files |
| **Cost** | Free, no API key |

### Verified file inventory

| File | Rows | Period | Gaps |
|---|---|---|---|
| `BTCUSDT_1d.parquet` | 3,332 | 2017-08-17 → 2026-09-30 | 0 |
| `BTCUSDT_4h.parquet` | 19,974 | 2017-08-17 → 2026-09-30 | 9 |
| `BTCUSDT_1h.parquet` | 79,837 | 2017-08-17 → 2026-09-30 | 29 |
| `BTCUSDT_15m.parquet` | 236,488 | 2020-01-01 → 2026-09-30 | 15 |
| `BTCUSDT_5m.parquet` | 709,458 | 2020-01-01 → 2026-09-30 | 15 |
| `ETHUSDT_1d.parquet` | 3,332 | 2017-08-17 → 2026-09-30 | 0 |
| `ETHUSDT_4h.parquet` | 19,974 | 2017-08-17 → 2026-09-30 | 9 |
| `ETHUSDT_1h.parquet` | 79,837 | 2017-08-17 → 2026-09-30 | 29 |
| `ETHUSDT_15m.parquet` | 236,488 | 2020-01-01 → 2026-09-30 | 15 |
| `ETHUSDT_5m.parquet` | 709,458 | 2020-01-01 → 2026-09-30 | 15 |

Verified: row counts match date-range arithmetic, zero duplicate timestamps, price
ranges match known market history (BTC low $3,189 = Dec 2018 trough), all timestamps
parse to valid dates.

**Note:** 15m and 5m start in 2020, not 2017 — early Binance intraday data is thinner
and less representative. This asymmetry is deliberate and must be stated in the report.

### Schema (11 columns)

```
open_time        datetime64   candle open timestamp
open             float64
high             float64
low              float64
close            float64
volume           float64      base asset volume
close_time       datetime64
quote_volume     float64
num_trades       int64        useful activity measure — keep it
taker_buy_base   float64
taker_buy_quote  float64
```

Binance's 12th column ("ignore") was already dropped.

---

## 7. ⚠️ CRITICAL CONSTRAINTS — Read Before Writing Any Code

These are failures that would **silently** corrupt results. Each has cost real projects weeks.

### 7.1 Indicators MUST be normalised to scale-free form

Over this dataset BTC moves from **$4,261 → $126,011**. Raw price-unit indicators
would cause the HMM to segment the data by **price era** rather than by market regime.
The "regimes" would effectively become a calendar and H1 would be meaningless.

| Indicator | Raw problem | Use instead |
|---|---|---|
| RSI | none — bounded 0–100 | **use directly** |
| ADX | none — bounded 0–100 | **use directly** |
| MACD | price units | `macd_hist / close` |
| EMA | price units | `(close - ema) / ema` |
| SMA | price units | `(close - sma) / sma` |
| Bollinger | price units | `(upper-lower)/mid` and `%B` |
| ATR | price units | `atr / close` |

All seven TAF-required indicators are still used — just expressed as **relative**
quantities. Defensible as: *"indicators are normalised to scale-free form so the model
learns regime structure rather than price level."*

### 7.2 Circularity in the H1 test

Transitions are defined using the **Daily** HMM's output. The Alignment Score also
includes Daily's contribution. So Daily would be **partly predicting its own change**.

➡️ **When testing H1, compute the Alignment Score EXCLUDING Daily.** A sharp examiner
will look for this.

### 7.3 Alignment will be noisy at short timeframes

5m and 15m regimes flip constantly. Raw alignment may fluctuate so much it drowns the
H1 signal. Three mitigations, all required:
1. **Smooth** the alignment series (rolling mean), don't use instantaneous values
2. **Weight fast timeframes low**
3. Test on **slope** as well as level — "declining for N periods" is more robust

### 7.4 Time-series validation only

**Never** use random train/test splits or `shuffle=True`. Always walk-forward: train on
past, test on future. Random shuffling leaks future information and inflates results.

### 7.5 Data must not be committed to Git

`BTCUSDT_5m.parquet` is 48 MB; the folder is ~140 MB. Commit the **download script**,
not the data. `.gitignore` must include `data/`, `*.parquet`, `models/`.

### 7.6 Timestamp units (already handled, don't break it)

Binance SPOT timestamps are **milliseconds before 2025-01-01** and **microseconds from
2025-01-01**. The existing data is correctly parsed. If writing any new loader, detect
the unit per row by magnitude (threshold `> 1e14` → microseconds).

---

## 8. Architecture — Nine Modules

| # | Module | Input | Output | Trained? |
|---|---|---|---|---|
| **M1** | Data acquisition | Binance Vision | 10 Parquet files | No |
| **M2** | Feature engineering | Raw OHLCV | 9 normalised features per timeframe | No |
| **M3** | Regime detection | Features | **5 trained HMMs** → regime + confidence | **Yes** |
| **M4** | Regime fusion | 5 labels + confidences | Dominant regime, Alignment Score, Confidence Score | No |
| **M5** | Propagation analysis | Regime label time series | Flip events, direction, cascade depth | No |
| **M6** | Stability scoring | Alignment level/slope + transition probs | Regime Stability Score | No |
| **M7** | Historical similarity | Feature vectors | Closest past match (supporting evidence only) | No |
| **M8** | Explanation | All of the above | Plain-language text + base rate | No |
| **M9** | **Experiment harness** | Historical outputs | H1 and H2 test results | No |

**M9 is not optional.** It is the module that makes this research rather than a
dashboard, and the one most likely to be skipped under time pressure. **Build it early.**

### Model design — critical

- **Five separate HMMs, one per timeframe.** Same algorithm, five independently trained
  instances. NOT one model across all timeframes.
- **Reason (statistical):** each timeframe has different ranges, volatility and noise.
  One model would try to learn one pattern fitting five different behaviours.
- **Reason (structural):** the novelty claim of *independent* detection only holds if
  the models are genuinely independent.
- **Algorithm:** `hmmlearn.hmm.GaussianHMM`, 4 states, unsupervised (Baum-Welch).
- **State mapping:** HMM outputs State 0–3, not names. Inspect each state's mean return,
  volatility and trend strength, then map to regime names **once** per model. Document
  the mapping.

### Fusion — decision-level (late), NOT early

```
Alignment Score = 100 × Σ(weight × confidence | agreeing with dominant)
                        ÷ Σ(weight × confidence | all five)

Confidence Score = Alignment Score × mean(confidence | agreeing)
```

**Why late fusion matters:** early fusion concatenates all timeframes' features into one
model before any decision. That permanently destroys the information about *which*
timeframe disagreed — making the entire hypothesis untestable. Late fusion preserves it.

---

## 9. Build Phases — Current Position

| Phase | Modules | Status | Checkpoint |
|---|---|---|---|
| **1 — Foundation** | M1 | ✅ **COMPLETE** | Data verified |
| **2a — Features** | M2 | ✅ **COMPLETE** (7 Oct 2026) | Feature ranges sane, no price-level drift |
| **2b — Core detection** | M3 (start Daily) | ⬅️ **NEXT** | Regimes match known history; **count transitions** |
| **3 — The research** | M9 + M4 | pending | Run H1 test **early** |
| **4 — Extension** | M5, M6 | pending | H2 testable |
| **5 — Delivery** | M7, M8, API | pending | Fusion Engine integration |

### The most important upcoming number

Phase 2b must **count how many Daily regime transitions exist** in the history. H1's
statistical power depends entirely on it:
- **150+ transitions** → good, proceed as planned
- **~30 transitions** → experimental design needs rethinking

This is unknown until M3 runs. Do not design the full H1 experiment before knowing it.

---

## 10. Repository Structure

```
.
├── CLAUDE.md                  # this file
├── README.md
├── requirements.txt
├── .gitignore
├── data/                      # GITIGNORED
│   ├── parquet/               # raw OHLCV (10 files)
│   └── features/              # engineered features
├── models/                    # GITIGNORED — trained HMMs
├── results/                   # experiment outputs, figures
├── notebooks/                 # exploration only, not pipeline
├── src/
│   ├── config.py              # paths, constants, indicator periods
│   ├── data/
│   │   ├── download.py        # M1 — Binance Vision downloader
│   │   └── validate.py        # M1 — integrity checks
│   ├── features/
│   │   └── indicators.py      # M2 — normalised indicators
│   ├── regime/
│   │   ├── hmm_model.py       # M3 — model wrapper
│   │   ├── train.py           # M3 — training per timeframe
│   │   └── state_mapping.py   # M3 — state → regime name
│   ├── fusion/
│   │   ├── alignment.py       # M4 — fusion + scores
│   │   └── propagation.py     # M5 — cascade analysis
│   ├── stability/
│   │   └── stability.py       # M6
│   ├── similarity/
│   │   └── dtw_match.py       # M7
│   ├── explain/
│   │   └── explainer.py       # M8
│   └── experiments/
│       ├── h1_precursor.py    # M9 — H1 test
│       └── h2_propagation.py  # M9 — H2 test
└── tests/
```

### Dependencies

```
pandas
pyarrow
numpy
hmmlearn
scikit-learn
scipy
matplotlib
tslearn          # DTW for M7
```

---

## 11. Coding Conventions for This Project

- **Python 3.10+**, type hints on function signatures.
- **No hidden magic numbers** — all constants in `src/config.py` with a comment on why.
- **Every indicator implemented explicitly, not via a library.** Reason: I must be able
  to explain Wilder's smoothing or ADX line by line in a viva. "I used a library" is a
  weaker answer. (`pandas-ta` also breaks against newer numpy/pandas in Colab.)
- **Reproducibility:** set and record random seeds; save fitted models; log parameters.
- **One timeframe at a time** when processing — do not load all 2.1M rows at once.
- **Docstrings explain WHY, not just what.** This code will be read by an examiner.
- **Print sanity reports** after every pipeline stage. Never assume a step worked.
- Prefer **clear and verifiable** over clever or compact.

---

## 12. Status of Every Design Decision

### ✅ DECIDED
- Five timeframes: Daily, 4H, 1H, 15m, 5m
- Assets: BTCUSDT, ETHUSDT
- Seven TAF indicators: RSI, MACD, EMA, SMA, Bollinger Bands, ATR, ADX
- Four regimes: Bullish, Bearish, Sideways, Volatile
- Five independent HMMs, one per timeframe
- Decision-level (late) fusion
- Indicators normalised to scale-free form
- Daily is the anchor timeframe for defining transitions
- Historical similarity is **supporting evidence only**
- TAF amendment for expanded novelty — **supervisor approved (Oct 2026)**
- **Per-asset models** (decided 7 Oct 2026): BTC and ETH each get their own five HMMs,
  10 models in total. Keeps detection independent, and ETH acts as a second test of
  the same hypothesis.

### 🟡 PROPOSED (working assumptions — may change)
- Starting weights: Daily 30%, 4H 25%, 1H 20%, 15m 15%, 5m 10%
  *(Novelty 3 replaces these with learned context-dependent weights)*
- 4 HMM states mapping 1:1 onto the 4 regimes
- Rolling-window retraining for adaptivity
- DTW for historical similarity matching
- Random Forest / XGBoost as secondary comparison models

### 🔴 OPEN (needs a decision)
- **5m training time** — not benchmarked. 709,458 rows is much larger than the others.
  If infeasible, train on a recent window (defensible on microstructure grounds).
- **Transition persistence threshold** — how many periods must a Daily regime hold to
  count as a real transition rather than flicker?
- **Cascade window length** — how long to wait before declaring propagation failed?
- **Context definition for adaptive weights** — volatility buckets, ADX bands, or both?
- **Binance data licensing terms** — must be verified and cited directly.

---

## 13. Scope Boundary with Teammate (Important)

Novelties 1 and 4 touch "what happens next," which brushes against H A A Dilshan's
Evidence-Guided Market Scenario Engine. The agreed distinction:

| | This component | Dilshan's component |
|---|---|---|
| Direction in time | **Present** | **Future** |
| Output | State label + stability probability | Ranked price scenarios |
| Historical use | Base rate **for own archetypes** | Episode retrieval for scenarios |
| Question | "Is what's happening now about to stop?" | "What might happen next?" |

**Rule:** this component must never generate ranked future price scenarios or retrieve
historical episodes as scenario evidence. It computes a statistic about its own
diagnostic categories.

---

## 14. Things to NEVER Do

1. ❌ Feed raw price-unit indicators (EMA, SMA, MACD, ATR, Bollinger) into the HMM
2. ❌ Train one model across all five timeframes
3. ❌ Use early/feature-level fusion
4. ❌ Include Daily in the Alignment Score when testing H1
5. ❌ Use random train/test splits or `shuffle=True` on time-series data
6. ❌ Commit `data/`, `*.parquet`, or `models/` to Git
7. ❌ Claim "below 1-Hour is too noisy" as the timeframe justification
8. ❌ Generate future price scenarios (that is Dilshan's scope)
9. ❌ Predict price — this component describes state, not direction
10. ❌ Skip M9, the experiment harness
11. ❌ Present a 🟡 PROPOSED or 🔴 OPEN item as a settled decision
12. ❌ Invent citations, results, or numbers — this is a research project under academic
    integrity rules. If a value is unknown, say so.

---

## 15. How I Want Claude Code to Work With Me

- **Explain before coding.** I need to understand every line well enough to defend it
  in a viva. Don't just produce working code.
- **Flag assumptions.** If something is ambiguous, ask rather than guess.
- **Be honest about uncertainty.** If a technique might not work, say so upfront.
- **Build incrementally** with a sanity check after each stage.
- **Prefer the defensible choice** over the clever one.
- **Tell me when I'm wrong.** If a request would damage the research validity
  (e.g. introduce leakage, break the hypothesis test), say so instead of complying.
- **One phase at a time.** Do not jump ahead to later modules.

---

## 16. Current Immediate Task

**Phase 2b — Core Detection (M3), Daily timeframe first**

Phase 2a is done: `python -m src.features.indicators` writes the 10 feature files to
`data/features/`, and `python -m pytest` passes (scale-free and no-look-ahead tests).

### Phase 2a facts to carry forward
- Indicator periods (textbook defaults, same at every timeframe): RSI/ATR/ADX 14,
  MACD 12/26/9, Bollinger 20 / 2 std, EMA 20, SMA 50. All in `src/config.py`.
- Warm-up: first 100 rows of every file dropped. Daily features start 2017-11-25.
- `sma_dist` on Daily reaches +1.10 (BTC, Dec 2017) and 1st-99th percentile is
  -0.33 to +0.49. That is genuine market behaviour, not a normalisation failure.
- **Residual era effect:** `atr_norm` and `bb_width` correlate -0.2 to -0.3 with price
  level on Daily (crypto volatility fell as the market matured). Watch in Phase 2b that
  the HMM does not simply split the history into "early high-vol" and "late low-vol".
- Features are NOT yet standardised (rsi/adx are 0-100, log_return is ~0.03). Scaling
  must be fitted on the training window only, inside the walk-forward loop.

### Phase 2b must
1. Train per asset (decided): BTC Daily first, then ETH Daily as the second test.
2. Train the Daily `GaussianHMM`, compare 3 / 4 / 5 states by BIC, fixed seed.
3. Map states to regime names from each state's mean return, volatility and ADX.
4. Use **forward-only (filtered) probabilities** for labels, never `predict()` over the
   whole history: Viterbi/smoothed labels at time t use future data and would
   invalidate H1.
5. **Count the Daily regime transitions** and report the number.
