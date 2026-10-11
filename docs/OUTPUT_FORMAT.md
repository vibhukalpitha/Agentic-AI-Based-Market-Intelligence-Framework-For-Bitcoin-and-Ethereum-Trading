# Output of the Market Regime Engine (Component 02)

**Status: proposal, schema version 0.3.** The group has not yet agreed a common format.
This is what Component 02 can deliver today; field names can be changed to fit the
Fusion Intelligence Engine.

## What you get

One JSON record per asset per moment. It answers: *what is the market regime at each
timeframe right now, how far do the timeframes agree, and where did the latest changes
start?* It describes the present state. It never contains a price forecast.

## How to get it

From the command line:

```
.\.venv\Scripts\python.exe -W ignore -m src.output.snapshot --symbol BTCUSDT
.\.venv\Scripts\python.exe -W ignore -m src.output.snapshot --symbol ETHUSDT --as-of 2025-03-01T12:00
```

From Python:

```python
from src.output.snapshot import build_snapshot

record = build_snapshot("BTCUSDT")                        # latest available moment
record = build_snapshot("ETHUSDT", "2025-03-01T12:00")    # any past moment (UTC)
```

`symbol` is `BTCUSDT` or `ETHUSDT`. A record for a past moment uses only candles that
had closed by that moment, so it is exactly what the engine would have said then.

Full examples: `results/sample_output_BTCUSDT.json` and `results/sample_output_ETHUSDT.json`.

## Top-level fields

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | text | Version of this format. Changes when fields change. |
| `component` | text | Always `market_regime_engine`. |
| `symbol` | text | `BTCUSDT` or `ETHUSDT`. |
| `as_of` | time | The moment the record describes (UTC, ISO 8601). |
| `timeframes` | object | State of each timeframe. See below. |
| `fusion` | object | The five timeframes combined. See below. |
| `propagation` | object | Where the latest regime change at each timeframe started. See below. |
| `stability` | object or null | Regime Stability Score. See below. `null` where no score exists (before 2021). |
| `explanation` | null | Plain-language explanation. **Not built yet**; currently always `null`. |
| `caveats` | list of text | Limits a consumer must respect. See the last section. |

## `timeframes`

One entry for each of `1d`, `4h`, `1h`, `15m`, `5m`:

| Field | Type | Meaning |
|---|---|---|
| `regime` | text | `Bullish`, `Bearish`, `Sideways` or `Volatile`. |
| `confidence` | number 0–1 | The model's probability for that regime. |
| `probabilities` | object | Probability of each of the four regimes; they sum to 1. |
| `candles_in_regime` | whole number | How many candles in a row this regime has held. |
| `last_closed_candle` | time | Close time of the candle the state comes from. |
| `regime_profile` | object | What this regime name means in this particular model. See below. |

`regime_profile` (new in version 0.2) has four fields, all measured on the training period:

| Field | Type | Meaning |
|---|---|---|
| `typical_rsi` | number 0–100 | Average RSI in this regime. Above 50: gains outweigh losses. |
| `volatility_vs_norm` | number | Typical price range as a multiple of its own past-year norm. 1.00 = normal, 2.00 = twice. |
| `share_of_training_time` | number 0–1 | Share of the training period spent in this regime. |
| `is_most_volatile_state` | true / false | Whether this is the most volatile of the model's four states. |

The four names and their rules are defined in `docs/REGIME_DEFINITIONS.md`. In short:
`Bullish` and `Bearish` are the states with the highest and lowest average RSI; of the two
left, `Volatile` is the more active and `Sideways` the quieter. **`Volatile` is not always
the most volatile state**: in 5 of the 10 models that is `Bearish`. Check
`volatility_vs_norm` and `is_most_volatile_state` before treating `Volatile` as turbulence.

## `fusion`

| Field | Type | Meaning |
|---|---|---|
| `dominant_regime` | text | The regime with the largest weighted vote. |
| `alignment_score` | number 0–100 | Share of the weighted vote that backs the dominant regime. 100 = all five agree. |
| `confidence_score` | number 0–100 | `alignment_score` × mean confidence of the agreeing timeframes. |
| `timeframes_agreeing` | list | Timeframes whose regime equals the dominant regime. |
| `timeframes_disagreeing` | list | The others. |
| `weights` | object | Weight of each timeframe in the vote (1d 0.30, 4h 0.25, 1h 0.20, 15m 0.15, 5m 0.10). |

## `stability`

How settled the current Daily regime is (new in version 0.3):

| Field | Type | Meaning |
|---|---|---|
| `probability_regime_persists` | number 0–1 | Estimated probability that the Daily regime stays unchanged on each of the next `horizon_days` days. |
| `horizon_days` | whole number | Always 3. |
| `regime` | text | The Daily regime the score refers to. |
| `daily_candle` | time | Open time of the Daily candle the score was computed from (the last one closed at `as_of`). |
| `in_training_period` | true / false | `true` if that day lies in the period the model was fitted on (up to 2022). Such scores are not out-of-sample. |
| `baseline_always` | number 0–1 | How often the regime persisted in the training period, for comparison. A score near this value says nothing special. |

The score comes from the Daily model's own signals: its confidence, how long the regime
has lasted, its transition probability and volatility. It describes the present state
only; it does not say which regime would come next and it is not a price forecast.

## `propagation`

One entry per timeframe, describing the flip that began its current regime
(`null` if that regime began before the data starts):

| Field | Type | Meaning |
|---|---|---|
| `from_regime`, `to_regime` | text | The regime before and after the flip. |
| `flip_candle_open` | time | Open time of the first candle of the new regime. |
| `support_depth` | whole number | How many faster timeframes, in an unbroken row, were already in the new regime before that candle opened. |
| `faster_timeframes` | whole number | How many faster timeframes exist below this one (the maximum possible depth). |
| `direction` | text | `bottom-up` (depth ≥ 1: the change was already present below), `top-down` (depth 0: this timeframe moved first), or `not_applicable` for `5m`. |
| `adopted_by_slower` | text | Whether the next slower timeframe took up the change: `already_there`, `adopted`, `pending`, `failed`, or `not_applicable` for `1d`. A change counts as adopted if the slower timeframe shows it within 3 of its own candles. |

## Caveats — please read before using the numbers

These come from this component's own tests (training data to 2022, tested on 2023–2026):

1. **`confidence` is not a calibrated accuracy.** It is above 0.99 about 75% of the time.
2. **A regime name can differ in meaning between timeframes**, because each timeframe
   has its own independent model. `regime_profile` states the meaning for each one.
3. **`stability.probability_regime_persists` ranks days usefully but is too confident
   at the top.** In testing, a day whose regime persisted received the higher score 73% of
   the time for BTC and 67% for ETH. But when the score was above 90%, the regime
   persisted about 86–93% of the time for BTC and only about 80% for ETH. Treat high
   values as "more settled than usual", not as an exact probability.
3b. **`alignment_score` is a description, not a warning.** It did not anticipate Daily
   regime changes in testing.
4. **`bottom-up` changes lasted somewhat longer than `top-down` ones at the 1h and 15m
   timeframes.** At the Daily timeframe the difference could not be separated from a
   background effect (a placebo check gave a similar result), so do not rely on
   `direction` for Daily flips.
5. **Data ends on 2026-09-30.** The record is built from saved files, not from a live feed.
