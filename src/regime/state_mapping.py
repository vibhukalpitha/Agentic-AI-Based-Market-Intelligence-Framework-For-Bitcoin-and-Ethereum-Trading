"""M3 - Turning anonymous HMM states (0, 1, 2, 3) into regime names.

The HMM is unsupervised: it finds four groups but does not know what they
mean, and the state numbers change from one fit to the next. The names are
assigned ONCE per model, on training data only, by a fixed written rule - not
by looking at a chart and choosing. The rule is the same for every asset and
timeframe.
"""
import numpy as np
import pandas as pd

from src import config

# Columns used to describe a state, in original (unscaled) units.
PROFILE_COLUMNS = ["log_return", "atr_norm", "adx", "sma_dist", "rsi"]


def state_profiles(features: pd.DataFrame, states: np.ndarray) -> pd.DataFrame:
    """Average behaviour of the market while it was in each state."""
    grouped = features.assign(state=states).groupby("state")
    profile = grouped[PROFILE_COLUMNS].mean()
    profile.insert(0, "days", grouped.size())
    profile.insert(1, "share", profile["days"] / len(features))
    return profile


def map_states(profile: pd.DataFrame) -> dict[int, str]:
    """The naming rule (requires exactly four states):

      1. Volatile = the state with the highest average atr_norm
                    (largest price range relative to price).
      2. Of the other three, ranked by average log_return:
            highest -> Bullish, lowest -> Bearish, middle -> Sideways.

    WHY volatility is assigned first: a volatile market is defined by the size
    of its moves, not their direction, so it must be separated before the
    remaining states are ordered by direction.
    """
    if len(profile) != len(config.REGIMES):
        raise ValueError("the naming rule is defined for four states only")
    volatile = profile["atr_norm"].idxmax()
    by_return = profile.drop(index=volatile)["log_return"].sort_values()
    bearish, sideways, bullish = by_return.index
    return {int(bullish): "Bullish", int(bearish): "Bearish",
            int(sideways): "Sideways", int(volatile): "Volatile"}


def mapping_warnings(profile: pd.DataFrame, mapping: dict[int, str]) -> list[str]:
    """Honest checks on whether the names really fit what the model found."""
    by_name = {name: profile.loc[state] for state, name in mapping.items()}
    warnings = []
    if by_name["Bullish"]["log_return"] <= 0:
        warnings.append("'Bullish' state has a non-positive average return")
    if by_name["Bearish"]["log_return"] >= 0:
        warnings.append("'Bearish' state has a non-negative average return")
    for name, row in by_name.items():
        if row["share"] < 0.05:
            warnings.append(f"'{name}' covers only {row['share']:.1%} of training days")
    return warnings
