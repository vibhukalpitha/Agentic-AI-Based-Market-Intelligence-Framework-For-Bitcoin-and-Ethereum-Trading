"""M3 - Turning anonymous HMM states (0, 1, 2, 3) into regime names.

The HMM is unsupervised: it finds four groups but does not know what they
mean, and the state numbers change from one fit to the next. The names are
assigned ONCE per model, on training data only, by a fixed written rule - not
by looking at a chart and choosing. The rule is the same for every asset and
timeframe.

Rule history (for the report): the first version ranked direction by average
log_return. On the Daily training data that mislabelled a state, because the
average daily return inside a state (about 0.3%) is tiny next to its daily
spread (3-5%). The rule was revised once, using training data only and before
any hypothesis test was run, to rank direction by trend position (sma_dist).
"""
import numpy as np
import pandas as pd

from src import config

# Columns used to describe a state, in original (unscaled) units.
PROFILE_COLUMNS = ["sma_dist", "atr_rel", "log_return", "atr_norm", "adx", "rsi"]


def state_profiles(features: pd.DataFrame, states: np.ndarray) -> pd.DataFrame:
    """Average behaviour of the market while it was in each state."""
    grouped = features.assign(state=states).groupby("state")
    profile = grouped[PROFILE_COLUMNS].mean()
    profile.insert(0, "days", grouped.size())
    profile.insert(1, "share", profile["days"] / len(features))
    return profile


def map_states(profile: pd.DataFrame) -> dict[int, str]:
    """The naming rule (requires exactly four states):

      1. Direction first, from average sma_dist (how far price sits above or
         below its 50-period average):
            highest -> Bullish, lowest -> Bearish.
      2. The two states left have no strong direction. They are separated by
         average atr_rel (range relative to its own past-year norm):
            higher -> Volatile, lower -> Sideways.

    WHY trend position and not average return: a regime's direction is whether
    price is above or below its trend. sma_dist measures exactly that and is
    stable; the average of daily returns is dominated by noise.
    """
    if len(profile) != len(config.REGIMES):
        raise ValueError("the naming rule is defined for four states only")
    by_trend = profile["sma_dist"].sort_values()
    bearish, bullish = by_trend.index[0], by_trend.index[-1]
    by_volatility = profile.drop(index=[bearish, bullish])["atr_rel"].sort_values()
    sideways, volatile = by_volatility.index
    return {int(bullish): "Bullish", int(bearish): "Bearish",
            int(sideways): "Sideways", int(volatile): "Volatile"}


def mapping_warnings(profile: pd.DataFrame, mapping: dict[int, str]) -> list[str]:
    """Honest checks on whether the names really fit what the model found."""
    state_of = {name: state for state, name in mapping.items()}
    by_name = {name: profile.loc[state] for name, state in state_of.items()}
    warnings = []
    if by_name["Bullish"]["sma_dist"] <= 0 or by_name["Bullish"]["log_return"] <= 0:
        warnings.append("'Bullish' state is not above trend with a positive average return")
    if by_name["Bearish"]["sma_dist"] >= 0 or by_name["Bearish"]["log_return"] >= 0:
        warnings.append("'Bearish' state is not below trend with a negative average return")
    if by_name["Volatile"]["atr_rel"] <= 0:
        warnings.append("'Volatile' state is not above its past-year volatility norm")
    if by_name["Sideways"]["atr_rel"] >= 0:
        warnings.append("'Sideways' state is not below its past-year volatility norm")
    most_volatile = mapping[int(profile["atr_rel"].idxmax())]
    if most_volatile != "Volatile":
        warnings.append(f"the most volatile state is '{most_volatile}', not 'Volatile'")
    for name, row in by_name.items():
        if row["share"] < 0.05:
            warnings.append(f"'{name}' covers only {row['share']:.1%} of training candles")
    return warnings
