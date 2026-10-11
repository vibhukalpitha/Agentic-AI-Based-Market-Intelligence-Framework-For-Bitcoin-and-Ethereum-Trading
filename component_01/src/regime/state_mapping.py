"""M3 - Turning anonymous HMM states (0, 1, 2, 3) into regime names.

The HMM is unsupervised: it finds four groups but does not know what they
mean, and the state numbers change from one fit to the next. The names are
assigned ONCE per model, on training data only, by a fixed written rule - not
by looking at a chart and choosing. The rule is the same for every asset and
timeframe.

Rule history (must be disclosed in the report). Every revision used
training-period state profiles only, before any hypothesis test was run:
  v1  Volatile = highest atr_norm; the rest ranked by average log_return.
      Failed: average return inside a state (about 0.3% a day) is tiny next
      to its daily spread (3-5%), so the ranking was mostly noise.
  v2  Direction ranked by average sma_dist.
      Failed on ETH 1h: two states had the same average sma_dist. The size of
      sma_dist grows with volatility, so it mixes direction with volatility.
  v3  (current) Direction ranked by average RSI. RSI divides gains by total
      movement, so it measures direction independently of the size of moves.
      v3 gives the same names as v2 on every Daily and 4-hour model.
"""
import numpy as np
import pandas as pd

from src import config

# Columns used to describe a state, in original (unscaled) units.
PROFILE_COLUMNS = ["rsi", "atr_rel", "sma_dist", "log_return", "atr_norm", "adx"]

RSI_NEUTRAL = 50.0   # RSI above 50 = gains outweigh losses, below 50 = the reverse


def state_profiles(features: pd.DataFrame, states: np.ndarray) -> pd.DataFrame:
    """Average behaviour of the market while it was in each state."""
    grouped = features.assign(state=states).groupby("state")
    profile = grouped[PROFILE_COLUMNS].mean()
    profile.insert(0, "candles", grouped.size())
    profile.insert(1, "share", profile["candles"] / len(features))
    return profile


def map_states(profile: pd.DataFrame) -> dict[int, str]:
    """The naming rule (requires exactly four states):

      1. Direction first, from average RSI:
            highest -> Bullish, lowest -> Bearish.
      2. The two states left are separated by average atr_rel (price range
         relative to its own past-year norm):
            higher -> Volatile, lower -> Sideways.
    """
    if len(profile) != len(config.REGIMES):
        raise ValueError("the naming rule is defined for four states only")
    by_direction = profile["rsi"].sort_values()
    bearish, bullish = by_direction.index[0], by_direction.index[-1]
    by_volatility = profile.drop(index=[bearish, bullish])["atr_rel"].sort_values()
    sideways, volatile = by_volatility.index
    return {int(bullish): "Bullish", int(bearish): "Bearish",
            int(sideways): "Sideways", int(volatile): "Volatile"}


def mapping_warnings(profile: pd.DataFrame, mapping: dict[int, str]) -> list[str]:
    """Honest checks on whether the names really fit what the model found."""
    state_of = {name: state for state, name in mapping.items()}
    by_name = {name: profile.loc[state] for name, state in state_of.items()}
    warnings = []
    if by_name["Bullish"]["rsi"] <= RSI_NEUTRAL or by_name["Bullish"]["log_return"] <= 0:
        warnings.append("'Bullish' state does not have RSI above 50 with a positive average return")
    if by_name["Bearish"]["rsi"] >= RSI_NEUTRAL or by_name["Bearish"]["log_return"] >= 0:
        warnings.append("'Bearish' state does not have RSI below 50 with a negative average return")
    if by_name["Volatile"]["atr_rel"] <= 0:
        warnings.append("'Volatile' state is not above its past-year volatility norm")
    if by_name["Sideways"]["atr_rel"] >= 0:
        warnings.append("'Sideways' state is not below its past-year volatility norm")
    most_volatile = mapping[int(profile["atr_rel"].idxmax())]
    if most_volatile != "Volatile":
        warnings.append(f"the most volatile state is '{most_volatile}', not 'Volatile'")
    ranked = profile["rsi"].sort_values()
    if ranked.iloc[1] - ranked.iloc[0] < 3.0 or ranked.iloc[-1] - ranked.iloc[-2] < 3.0:
        warnings.append("two states have nearly the same average RSI; the direction ranking is close")
    for name, row in by_name.items():
        if row["share"] < 0.05:
            warnings.append(f"'{name}' covers only {row['share']:.1%} of training candles")
    return warnings
