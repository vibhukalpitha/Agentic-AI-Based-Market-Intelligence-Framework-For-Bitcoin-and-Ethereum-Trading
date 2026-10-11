"""Tests for Novelty 4 (archetypes) and M8 (explanation). Run:  python -m pytest"""
import numpy as np
import pandas as pd

from src import config
from src.experiments import n4_archetypes
from src.explain import archetypes, explainer


def test_each_archetype_rule():
    daily = np.array(["Bullish"] * 6)
    #                  4h         1h          15m         5m
    faster = np.array([
        ["Bullish", "Bullish", "Bullish", "Bullish"],     # 4 agree            -> Aligned
        ["Bullish", "Bullish", "Bullish", "Sideways"],    # 3 agree            -> Aligned
        ["Bullish", "Bullish", "Sideways", "Bearish"],    # 2 agree            -> Partly backed
        ["Bullish", "Sideways", "Sideways", "Sideways"],  # 3 share another    -> Opposed
        ["Sideways", "Sideways", "Sideways", "Sideways"], # 4 share another    -> Opposed
        ["Bullish", "Sideways", "Bearish", "Volatile"],   # 1 agrees, no group -> Fragmented
    ])
    name, n_agree, opposing = archetypes.classify(daily, faster)
    assert list(name) == ["Aligned", "Aligned", "Partly backed", "Opposed", "Opposed", "Fragmented"]
    assert list(n_agree) == [4, 3, 2, 1, 0, 1]
    assert list(opposing) == [None, None, None, "Sideways", "Sideways", None]


def test_every_day_gets_exactly_one_archetype():
    rng = np.random.default_rng(config.RANDOM_SEED)
    daily = rng.choice(config.REGIMES, 2000)
    faster = rng.choice(config.REGIMES, (2000, 4))
    name, _, _ = archetypes.classify(daily, faster)
    assert set(name) == set(archetypes.ARCHETYPES)
    assert set(archetypes.DEFINITIONS) == set(archetypes.ARCHETYPES)


def test_base_rates_are_the_share_of_days_with_a_change():
    train = pd.DataFrame({
        "archetype": ["Aligned"] * 4 + ["Opposed"] * 4 + ["Fragmented"] * 2 + ["Partly backed"] * 2,
        "changed": [0, 0, 0, 1,  1, 1, 0, 0,  1, 0,  0, 0],
    })
    rates = archetypes.base_rates(train)
    assert rates["archetypes"]["Aligned"] == {"days": 4, "change_rate": 0.25}
    assert rates["archetypes"]["Opposed"]["change_rate"] == 0.5
    assert rates["overall_change_rate"] == 4 / 12 and rates["training_days"] == 12


def test_contrast_is_the_difference_in_change_rates():
    rng = np.random.default_rng(config.RANDOM_SEED)
    frame = pd.DataFrame({
        "archetype": np.tile(["Opposed", "Aligned"], 210),
        "changed": np.tile([1.0, 0.0, 1.0, 0.0, 0.0, 0.0], 70),   # Opposed 2/3, Aligned 0
    })
    result = n4_archetypes.contrast(frame, "Opposed", "Aligned", rng)
    assert abs(result["difference"] - 2 / 3) < 1e-9
    assert result["ci_low"] > 0


# ---------------------------------------------------------------- explanation
RATES = {
    "horizon_days": 3, "train_from": "2021-01-01", "train_to": "2022-12-28",
    "overall_change_rate": 0.191, "definitions": archetypes.DEFINITIONS,
    "archetypes": {name: {"days": 100 + k, "change_rate": 0.08 + 0.05 * k}
                   for k, name in enumerate(archetypes.ARCHETYPES)},
}
CLOSE_DATE = "2026-09-30"


def make_timeframes(daily: str, faster: list[str], days: int = 13) -> dict:
    regimes = dict(zip(config.INTERVALS, [daily] + faster))
    return {i: {"regime": regime, "candles_in_regime": days, "regime_profile": None}
            for i, regime in regimes.items()}


def explain(daily: str, faster_at_close: list[str], faster_now: list[str] | None = None,
            stability: dict | None = None, flip: dict | None = None) -> dict:
    """Build an explanation the way the output record does."""
    name, n_agree, opposing = archetypes.classify(np.array([daily]), np.array([faster_at_close]))
    row = pd.Series({"archetype": name[0], "n_agree": n_agree[0], "opposing_regime": opposing[0]})
    timeframes = make_timeframes(daily, faster_now or faster_at_close)
    return explainer.build("BTCUSDT", timeframes, stability, {"1d": flip}, row, RATES,
                           dict(zip(config.INTERVALS[1:], faster_at_close)), CLOSE_DATE)


def test_explanation_quotes_the_archetype_and_its_base_rate():
    block = explain(
        "Bullish", ["Sideways"] * 4,
        stability={"regime": "Bullish", "horizon_days": 3, "probability_regime_persists": 0.8888},
        flip={"to_regime": "Bullish", "direction": "top-down",
              "flip_candle_open": "2026-09-18T00:00:00.000Z"})

    assert block["archetype"] == "Opposed" and block["opposing_regime"] == "Sideways"
    assert block["base_rate_regime_change"] == 0.23 and block["base_rate_days"] == 103
    assert block["daily_close"] == CLOSE_DATE
    text = block["text"]
    assert "the Daily regime is Bullish and has held for 13 days" in text
    assert ("At the last Daily close (2026-09-30), none of the four faster timeframes was in "
            "that regime: the 4-hour, 1-hour, 15-minute and 5-minute were Sideways.") in text
    assert 'This is the "Opposed" pattern' in text
    assert "on 23% of the 103 days with this pattern, against 19% of all days" in text
    assert "holds for 3 more days at 89%" in text
    assert "began on 2026-09-18 as a top-down change" in text
    assert "Since that close" not in text            # nothing has moved


def test_archetype_always_matches_the_timeframes_described():
    """The pattern named in the text must be the pattern of the timeframes the
    text describes, i.e. those at the last Daily close - even when the faster
    timeframes have moved since."""
    block = explain("Bearish", ["Bullish", "Sideways", "Volatile", "Bearish"],
                    faster_now=["Bearish"] * 4)
    assert block["archetype"] == "Fragmented"
    assert block["faster_timeframes_agreeing"] == 1
    text = block["text"]
    assert ("At the last Daily close (2026-09-30), the 5-minute was also Bearish; the 4-hour "
            "was Bullish; the 1-hour was Sideways; the 15-minute was Volatile.") in text
    assert ("Since that close the faster timeframes have moved: now all four faster "
            "timeframes are also Bearish.") in text


def test_explanation_never_talks_about_price_or_the_next_regime():
    """Scope rule: describe the present state; no price direction, no forecast
    of which regime comes next."""
    for daily, faster in (("Bullish", ["Bullish"] * 4),
                          ("Bearish", ["Bearish", "Bearish", "Sideways", "Volatile"]),
                          ("Volatile", ["Bullish", "Sideways", "Bearish", "Volatile"])):
        text = explain(daily, faster, faster_now=["Sideways"] * 4)["text"].lower()
        for forbidden in ("price", "will rise", "will fall", "buy", "sell", "target",
                          "next regime", "expect"):
            assert forbidden not in text


def test_all_agreeing_and_bottom_up_wording():
    block = explain("Bearish", ["Bearish"] * 4,
                    flip={"to_regime": "Bearish", "direction": "bottom-up",
                          "flip_candle_open": "2025-02-24T00:00:00.000Z"})
    assert block["archetype"] == "Aligned"
    assert "all four faster timeframes were also Bearish" in block["text"]
    assert ("as a bottom-up change: at least one faster timeframe had already turned Bearish "
            "before it") in block["text"]


def test_no_explanation_without_an_archetype_or_the_close_state():
    timeframes = make_timeframes("Bullish", ["Bullish"] * 4)
    faster = dict(zip(config.INTERVALS[1:], ["Bullish"] * 4))
    row = pd.Series({"archetype": "Aligned", "n_agree": 4, "opposing_regime": None})
    assert explainer.build("BTCUSDT", timeframes, None, {"1d": None}, None, RATES,
                           faster, CLOSE_DATE) is None
    assert explainer.build("BTCUSDT", timeframes, None, {"1d": None}, row, RATES,
                           None, CLOSE_DATE) is None
