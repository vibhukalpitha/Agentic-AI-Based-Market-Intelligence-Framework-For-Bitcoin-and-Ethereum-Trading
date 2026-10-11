"""Novelty 3 - A method for, and test of, context-adaptive timeframe weighting.

Run:  python -m src.fusion.adaptive_weights      (fits on the training period only)

RESULT (test period 2023-2026, src/experiments/n3_adaptive_weights.py): NOT
SUPPORTED. Context-adaptive weights were no better than fixed weights (Brier
difference +0.0028 for BTC and +0.0013 for ETH, both intervals including
zero), and weights learned without context were no better either. The fixed
weights in src/config.py are therefore retained. The 4-hour timeframe carries
almost all of the learned weight. This module is kept because the method and
the test are the contribution: it shows the question was asked properly.

The claim. The weight a timeframe deserves is not fixed: in a calm market the
fast timeframes are mostly noise, and in an active market the slow ones are
stale. So the weights should depend on the market's condition.

Made testable. "Informative" needs an object. Here a timeframe is informative
if its agreement with the Daily regime helps to judge whether that regime will
hold for the next STABILITY_HORIZON_DAYS days - the same outcome as the Regime
Stability Score (M6). Three ways of using the four faster timeframes are
compared, each added to the Daily model's own signals:

  F  fixed     one number: the share of faster timeframes backing Daily,
               combined with the proposed fixed weights (0.25/0.20/0.15/0.10)
  L  learned   one weight per timeframe, learned from the training period
  A  adaptive  one weight per timeframe PER CONTEXT, learned from training

Context. Two conditions of the Daily market, split at a threshold that needs
no tuning: volatility relative to its own past-year norm.
    calm    atr_rel <= 0   (at or below the yearly norm)
    active  atr_rel  > 0   (above it)

Novelty 3 is supported only if A beats both F and L on the test period.

As with M6, the model to deliver is chosen by time-ordered cross-validation
inside the training period, by a rule fixed in advance (select_model).
"""
import json

import numpy as np
import pandas as pd

from src import config
from src.stability import stability

FASTER = config.INTERVALS[1:]
AGREE = [f"agree_{i}" for i in FASTER]
CALM = [f"{name}_calm" for name in AGREE]
ACTIVE = [f"{name}_active" for name in AGREE]
MODELS = {
    "F": stability.OWN_SIGNALS + ["agree_share"],
    "L": stability.OWN_SIGNALS + AGREE,
    "A": stability.OWN_SIGNALS + ["active"] + CALM + ACTIVE,
}


def add_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the context flag and each timeframe's agreement split by context."""
    out = frame.copy()
    out["active"] = (out["volatility"] > 0).astype(float)
    for name in AGREE:
        out[f"{name}_calm"] = out[name] * (1.0 - out["active"])
        out[f"{name}_active"] = out[name] * out["active"]
    return out


def _beats(cv: pd.DataFrame, richer: str, simpler: str) -> bool:
    """Better on average AND in most folds."""
    return bool(cv[richer].mean() < cv[simpler].mean()
                and (cv[richer] < cv[simpler]).sum() > len(cv) / 2)


def select_model(cv: pd.DataFrame) -> str:
    """The rule, fixed in advance: keep the simplest model (F) unless a richer
    one clearly wins. L must beat F; A must then beat whichever is in place."""
    chosen = "L" if _beats(cv, "L", "F") else "F"
    return "A" if _beats(cv, "A", chosen) else chosen


def learned_weights(models: dict) -> pd.DataFrame:
    """The learned weight of each timeframe: how much the log-odds of the
    Daily regime persisting rise when that timeframe is in the Daily regime.

    Coefficients are converted back from standardised units, so the weights
    of different timeframes and contexts can be compared directly.
    """
    def raw(model: dict, signal: str) -> float:
        k = model["signals"].index(signal)
        return model["coefficients"][k] / model["std"][k]

    return pd.DataFrame({
        "learned": [raw(models["L"], name) for name in AGREE],
        "adaptive_calm": [raw(models["A"], name) for name in CALM],
        "adaptive_active": [raw(models["A"], name) for name in ACTIVE],
        "fixed_proposed": [config.TIMEFRAME_WEIGHTS[i] for i in FASTER],
    }, index=FASTER)


def run(symbol: str) -> dict:
    inputs = add_context(stability.build_inputs(symbol))
    train = inputs[inputs["period"] == "train"]

    cv = stability.cross_validate(train, MODELS)
    chosen = select_model(cv)
    models = {name: stability.fit(train, signals) for name, signals in MODELS.items()}
    weights = learned_weights(models)

    scores = inputs[["regime", "target", "period", "active"]].copy()
    for name, model in models.items():
        scores[name] = stability.predict(model, inputs)
    config.FUSION_DIR.mkdir(parents=True, exist_ok=True)
    scores.reset_index().to_parquet(config.FUSION_DIR / f"{symbol}_adaptive_weights.parquet",
                                    index=False)

    record = {"symbol": symbol, "horizon_days": config.STABILITY_HORIZON_DAYS,
              "context": "calm: Daily atr_rel <= 0; active: Daily atr_rel > 0",
              "train_days": int(len(train)), "train_active_share": float(train["active"].mean()),
              "cross_validation": cv.to_dict(orient="records"), "delivered_model": chosen,
              "learned_weights": weights.to_dict(orient="index"), "models": models}
    with open(config.RESULTS_DIR / f"adaptive_weights_{symbol}.json", "w") as file:
        json.dump(record, file, indent=2)

    print(f"\n=== {symbol}: context-adaptive weights (fitted on training data only) ===")
    print(f"training days {len(train)} | active (volatility above its yearly norm) on "
          f"{train['active'].mean():.1%} of them")
    print("\ntime-ordered cross-validation inside the training period (log loss, lower is better):")
    print(cv.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("average   " + " | ".join(f"{name} {cv[name].mean():.4f}" for name in MODELS)
          + f"   -> delivered model: {chosen}")
    print("\nlearned weight of each timeframe (rise in the log-odds that the Daily regime "
          "persists when the timeframe backs it):")
    print(weights.to_string(float_format=lambda v: f"{v:+.3f}"))
    return record


def main() -> None:
    for symbol in config.SYMBOLS:
        run(symbol)
    print("\nSaved: data/fusion/<symbol>_adaptive_weights.parquet, "
          "results/adaptive_weights_<symbol>.json")


if __name__ == "__main__":
    main()
