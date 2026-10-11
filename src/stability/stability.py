"""M6 - Regime Stability Score.

Run:  python -m src.stability.stability      (fits on the training period only)

The score answers one question at the close of every day:

    "What is the probability that the Daily regime will still be the same
     for each of the next STABILITY_HORIZON_DAYS days?"

It describes how settled the PRESENT state is. It says nothing about price
and nothing about which regime would come next.

Four scores are produced so the delivered one can be judged against simpler ones:

  B0  base rate    the share of training days whose regime did persist (a constant)
  B1  HMM only     the Daily model's own chance of staying in its state for
                   the horizon: (self-transition probability) ^ horizon
  S1  own signals  logistic regression on the Daily model's own signals
  S2  S1 + cross-timeframe signals (agreement with Daily, alignment, support depth)

WHY the choice between S1 and S2 is made on training data. H1 found that
alignment does not anticipate Daily transitions, but that was seen on the test
period. Picking the model from test-period findings and then scoring it on the
same period would flatter it. So the delivered model is chosen by time-ordered
cross-validation inside the training period, by a rule written down in advance
(select_model). Both S1 and S2 are still evaluated on the test period, which
also answers honestly whether the cross-timeframe signals add anything.

No look-ahead: every input for day t is known at the close of day t; the
outcome uses days t+1 .. t+horizon; the model is fitted on training days only.
"""
import json

import numpy as np
import pandas as pd

from src import config
from src.experiments import h1_precursor
from src.experiments.stats import logistic_coefficients
from src.fusion import propagation

OWN_SIGNALS = ["hmm_persist_logit", "confidence_logit", "log_age", "volatility"]
CROSS_SIGNALS = ["agree_share", "alignment_ex1d", "support_depth"]
MODELS = {"S1": OWN_SIGNALS, "S2": OWN_SIGNALS + CROSS_SIGNALS}
ONE_DAY = pd.Timedelta(days=1)


def _logit(p: np.ndarray) -> np.ndarray:
    """log(p / (1 - p)), with p kept away from exactly 0 and 1."""
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def persistence_target(regime: np.ndarray, horizon: int) -> np.ndarray:
    """1.0 if the regime on each of the next `horizon` days equals today's,
    0.0 if it changes on any of them, NaN for the last `horizon` days
    (their future is not in the data)."""
    target = np.full(len(regime), np.nan)
    for t in range(len(regime) - horizon):
        target[t] = float((regime[t + 1:t + horizon + 1] == regime[t]).all())
    return target


def candles_in_regime(regime: np.ndarray) -> np.ndarray:
    """How many days in a row, up to and including today, the regime has held."""
    age = np.ones(len(regime), dtype=int)
    for t in range(1, len(regime)):
        if regime[t] == regime[t - 1]:
            age[t] = age[t - 1] + 1
    return age


def build_inputs(symbol: str) -> pd.DataFrame:
    """One row per day: the regime, the seven signals, the outcome and the period."""
    horizon = config.STABILITY_HORIZON_DAYS
    daily = propagation.load_labels(symbol, "1d")
    regime = daily["regime"].to_numpy()

    with open(config.RESULTS_DIR / f"{symbol}_1d_hmm_report.json") as file:
        transition = json.load(file)["transition_matrix"]
    stay = np.array([transition[name][name] for name in regime])

    volatility = pd.read_parquet(config.FEATURES_DIR / f"{symbol}_1d_features.parquet",
                                 columns=["open_time", "atr_rel"]).set_index("open_time")["atr_rel"]

    frame = pd.DataFrame({
        "regime": regime,
        "hmm_persist": stay ** horizon,
        "hmm_persist_logit": _logit(stay ** horizon),
        "confidence_logit": _logit(daily["confidence"].to_numpy(dtype=float)),
        "log_age": np.log(candles_in_regime(regime)),
    }, index=pd.DatetimeIndex(daily["open_time"], name="date"))
    frame["volatility"] = volatility.reindex(frame.index).to_numpy()

    # Cross-timeframe signals, all as they stood at the close of the day.
    close = daily["known_time"].to_numpy()
    faster = config.INTERVALS[1:]
    weights = np.array([config.TIMEFRAME_WEIGHTS[i] for i in faster])
    agrees = np.column_stack([
        propagation.labels_as_of(propagation.load_labels(symbol, i), close) == regime
        for i in faster])
    frame["agree_share"] = (agrees * weights).sum(axis=1) / weights.sum()

    alignment = h1_precursor.daily_alignment(symbol)
    covered = alignment["coverage"] >= config.H1_MIN_DAY_COVERAGE
    frame["alignment_ex1d"] = alignment["standard"].where(covered).reindex(frame.index).to_numpy()

    # Support depth of the flip that began the current regime (M5).
    flips = pd.read_parquet(config.PROPAGATION_DIR / f"{symbol}_flips.parquet")
    flips = flips[flips["interval"] == "1d"].set_index("open_time")["depth"]
    run_start = frame.index[np.arange(len(frame)) - candles_in_regime(regime) + 1]
    frame["support_depth"] = flips.reindex(run_start).to_numpy()

    frame["target"] = persistence_target(regime, horizon)
    train_end = pd.Timestamp(config.TRAIN_END)
    # A training day's outcome must also lie inside the training period.
    frame["period"] = np.where(frame.index + horizon * ONE_DAY <= train_end, "train",
                               np.where(frame.index > train_end, "test", "gap"))
    return frame.dropna(subset=MODELS["S2"] + ["target"])


def fit(train: pd.DataFrame, signals: list[str]) -> dict:
    """Standardise with training statistics, then fit the logistic regression."""
    x = train[signals].to_numpy(dtype=float)
    mean, spread = x.mean(axis=0), x.std(axis=0)
    spread[spread == 0] = 1.0
    coefficients = logistic_coefficients((x - mean) / spread, train["target"].to_numpy())
    return {"signals": signals, "mean": mean.tolist(), "std": spread.tolist(),
            "intercept": float(coefficients[0]), "coefficients": coefficients[1:].tolist()}


def predict(model: dict, frame: pd.DataFrame) -> np.ndarray:
    """Probability that the regime persists, for each row."""
    x = (frame[model["signals"]].to_numpy(dtype=float) - model["mean"]) / model["std"]
    return 1.0 / (1.0 + np.exp(-(model["intercept"] + x @ np.array(model["coefficients"]))))


def log_loss(target: np.ndarray, probability: np.ndarray) -> float:
    """Average surprise of the forecasts; lower is better. Penalises
    confident wrong answers heavily."""
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    return float(-np.mean(target * np.log(p) + (1 - target) * np.log(1 - p)))


def cross_validate(train: pd.DataFrame) -> pd.DataFrame:
    """Log loss of S1 and S2 on each later block, fitted on the blocks before it.

    The last `horizon` rows before each validation block are left out of the
    fit, because their outcomes reach into that block.
    """
    edges = np.linspace(0, len(train), config.STABILITY_CV_BLOCKS + 1).astype(int)
    rows = []
    for k in range(1, config.STABILITY_CV_BLOCKS):
        past = train.iloc[:edges[k] - config.STABILITY_HORIZON_DAYS]
        block = train.iloc[edges[k]:edges[k + 1]]
        row = {"fold": k, "fit_days": len(past), "validation_days": len(block)}
        for name, signals in MODELS.items():
            row[name] = log_loss(block["target"].to_numpy(), predict(fit(past, signals), block))
        rows.append(row)
    return pd.DataFrame(rows)


def select_model(cv: pd.DataFrame) -> str:
    """The rule, fixed in advance: deliver the simpler model S1 unless S2 has a
    lower average cross-validated log loss AND is better in most folds."""
    s2_better_on_average = cv["S2"].mean() < cv["S1"].mean()
    s2_better_in_most_folds = (cv["S2"] < cv["S1"]).sum() > len(cv) / 2
    return "S2" if s2_better_on_average and s2_better_in_most_folds else "S1"


def run(symbol: str) -> dict:
    inputs = build_inputs(symbol)
    train = inputs[inputs["period"] == "train"]

    cv = cross_validate(train)
    chosen = select_model(cv)
    models = {name: fit(train, signals) for name, signals in MODELS.items()}
    base_rate = float(train["target"].mean())

    scores = inputs.copy()
    scores["B0"] = base_rate
    scores["B1"] = scores["hmm_persist"]
    for name, model in models.items():
        scores[name] = predict(model, scores)
    scores["delivered"] = scores[chosen]
    config.STABILITY_DIR.mkdir(parents=True, exist_ok=True)
    scores.reset_index().to_parquet(config.STABILITY_DIR / f"{symbol}_stability.parquet",
                                    index=False)

    record = {"symbol": symbol, "horizon_days": config.STABILITY_HORIZON_DAYS,
              "train_days": int(len(train)), "train_from": str(train.index[0].date()),
              "train_to": str(train.index[-1].date()), "base_rate": base_rate,
              "cross_validation": cv.to_dict(orient="records"),
              "delivered_model": chosen, "models": models}
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / f"stability_model_{symbol}.json", "w") as file:
        json.dump(record, file, indent=2)

    # Report on the TRAINING period only. Test-period performance is the job
    # of src/experiments/m6_evaluation.py and is not looked at here.
    print(f"\n=== {symbol}: Regime Stability Score (fitted on training data only) ===")
    print(f"training days {len(train)} ({record['train_from']} -> {record['train_to']}) | "
          f"regime persisted {config.STABILITY_HORIZON_DAYS} days on {base_rate:.1%} of them")
    print("\ntime-ordered cross-validation inside the training period (log loss, lower is better):")
    print(cv.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"average   S1 {cv['S1'].mean():.4f} | S2 {cv['S2'].mean():.4f}   "
          f"-> delivered model: {chosen}")
    print("\nstandardised coefficients (positive = raises the chance the regime persists):")
    for name, model in models.items():
        print(f"  {name}: " + ", ".join(
            f"{signal} {coefficient:+.3f}"
            for signal, coefficient in zip(model["signals"], model["coefficients"])))
    return record


def main() -> None:
    for symbol in config.SYMBOLS:
        run(symbol)
    print("\nSaved: data/stability/<symbol>_stability.parquet, "
          "results/stability_model_<symbol>.json")


if __name__ == "__main__":
    main()
