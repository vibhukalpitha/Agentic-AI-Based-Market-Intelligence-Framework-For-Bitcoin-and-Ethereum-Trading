"""M8 - Plain-language explanation of the engine's output.

The explanation is assembled from facts the other modules have already
produced. Every number in it is either the current state or a statistic
measured on the training period:

  - the Daily regime and how long it has held
  - how the four faster timeframes stood AT THE LAST DAILY CLOSE, the
    conflict archetype that makes, and that archetype's base rate
    ("this pattern was followed by a Daily regime change N% of the time")
  - whether the faster timeframes have moved since that close
  - the Regime Stability Score
  - how the current Daily regime began (bottom-up or top-down)

WHY the archetype is tied to the last Daily close: its base rate was measured
at Daily closes, so quoting it for a pattern seen mid-day would compare unlike
things.

Scope: it describes the present state and how settled it is. It never says
what the next regime will be and never mentions price direction; that
belongs to another component of the group project.
"""
import json

import pandas as pd

from src import config

REGIME_ORDER = {name: k for k, name in enumerate(config.REGIMES)}
FASTER = config.INTERVALS[1:]
TIMEFRAME_NAMES = {"1d": "Daily", "4h": "4-hour", "1h": "1-hour", "15m": "15-minute",
                   "5m": "5-minute"}


def load_archetypes(symbol: str) -> tuple[pd.DataFrame | None, dict | None]:
    """Saved archetype of every day, and the training-period base rates."""
    table_path = config.EXPLAIN_DIR / f"{symbol}_archetypes.parquet"
    rates_path = config.RESULTS_DIR / f"archetype_base_rates_{symbol}.json"
    if not table_path.exists() or not rates_path.exists():
        return None, None
    with open(rates_path) as file:
        return pd.read_parquet(table_path).set_index("date"), json.load(file)


def _join(names: list[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _was(names: list[str], past: bool) -> str:
    if past:
        return "was" if len(names) == 1 else "were"
    return "is" if len(names) == 1 else "are"


def describe_faster(faster: dict[str, str], daily_regime: str, past: bool) -> str:
    """How the four faster timeframes stand relative to the Daily regime."""
    same = [TIMEFRAME_NAMES[i] for i in FASTER if faster[i] == daily_regime]
    if len(same) == len(FASTER):
        return f"all four faster timeframes {_was(FASTER, past)} also {daily_regime}"
    groups: dict[str, list[str]] = {}
    for interval in FASTER:
        if faster[interval] != daily_regime:
            groups.setdefault(faster[interval], []).append(TIMEFRAME_NAMES[interval])
    others = "; ".join(f"the {_join(names)} {_was(names, past)} {regime}"
                       for regime, names in sorted(groups.items(),
                                                   key=lambda item: REGIME_ORDER[item[0]]))
    if same:
        return f"the {_join(same)} {_was(same, past)} also {daily_regime}; {others}"
    return f"none of the four faster timeframes {'was' if past else 'is'} in that regime: {others}"


def build(symbol: str, timeframes: dict, stability: dict | None, propagation: dict,
          archetype_row: pd.Series | None, rates: dict | None,
          faster_at_close: dict[str, str] | None, close_date: str | None) -> dict | None:
    """The `explanation` block of the output record, or None if no archetype
    is available for the last Daily close."""
    if archetype_row is None or rates is None or faster_at_close is None:
        return None
    daily = timeframes["1d"]
    horizon = rates["horizon_days"]
    name = archetype_row["archetype"]
    info = rates["archetypes"][name]

    days = daily["candles_in_regime"]
    sentences = [
        f"{symbol}: the Daily regime is {daily['regime']} and has held for "
        f"{days} day{'s' if days != 1 else ''}.",
        f"At the last Daily close ({close_date}), "
        f"{describe_faster(faster_at_close, daily['regime'], past=True)}.",
        f"This is the \"{name}\" pattern: {rates['definitions'][name]}.",
        f"In the training period ({rates['train_from'][:4]}-{rates['train_to'][:4]}) the Daily "
        f"regime changed within {horizon} days on {info['change_rate']:.0%} of the "
        f"{info['days']} days with this pattern, against {rates['overall_change_rate']:.0%} "
        f"of all days.",
    ]
    faster_now = {i: timeframes[i]["regime"] for i in FASTER}
    if faster_now != faster_at_close:
        sentences.append("Since that close the faster timeframes have moved: now "
                         f"{describe_faster(faster_now, daily['regime'], past=False)}.")
    if stability is not None:
        sentences.append(
            f"The stability score puts the chance that {stability['regime']} holds for "
            f"{stability['horizon_days']} more days at "
            f"{stability['probability_regime_persists']:.0%}.")
    flip = propagation.get("1d")
    if flip is not None:
        origin = ("at least one faster timeframe had already turned "
                  f"{flip['to_regime']} before it" if flip["direction"] == "bottom-up"
                  else f"no faster timeframe had turned {flip['to_regime']} before it")
        sentences.append(
            f"The current Daily regime began on {flip['flip_candle_open'][:10]} as a "
            f"{flip['direction']} change: {origin}.")
    profile = daily.get("regime_profile")
    if profile is not None and daily["regime"] == "Volatile" and not profile["is_most_volatile_state"]:
        sentences.append(
            "Note: in this Daily model \"Volatile\" is the more active of the two "
            f"non-directional states (about {profile['volatility_vs_norm']:.2f} times its "
            "yearly norm), not the most volatile state.")

    return {
        "archetype": name,
        "archetype_definition": rates["definitions"][name],
        "daily_close": close_date,
        "faster_timeframes_at_close": faster_at_close,
        "faster_timeframes_agreeing": int(archetype_row["n_agree"]),
        "opposing_regime": (None if pd.isna(archetype_row["opposing_regime"])
                            else archetype_row["opposing_regime"]),
        "base_rate_regime_change": round(info["change_rate"], 4),
        "base_rate_days": info["days"],
        "overall_rate_regime_change": round(rates["overall_change_rate"], 4),
        "horizon_days": horizon,
        "text": " ".join(sentences),
    }
