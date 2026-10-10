"""Tests for M9 (H1 experiment). Run:  python -m pytest

These protect the definitions the H1 result depends on.
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.experiments import h1_precursor as h1


def test_transition_start_is_the_first_day_of_the_new_regime():
    #                 0    1    2    3    4    5    6    7    8
    regimes = np.array(["A", "A", "A", "B", "A", "B", "B", "B", "B"])
    is_start, settled = h1.transition_starts(regimes, persistence=3)
    # The single B on day 3 is flicker. The run B B B starts on day 5 and is
    # confirmed on day 7; the transition day must be day 5, not day 7.
    assert list(np.flatnonzero(is_start)) == [5]
    assert list(settled) == ["A"] * 7 + ["B", "B"]


def test_persistence_one_counts_every_change():
    regimes = np.array(["A", "B", "A", "A"])
    is_start, _ = h1.transition_starts(regimes, persistence=1)
    assert list(np.flatnonzero(is_start)) == [1, 2]


def test_matched_auc_hand_calculation():
    # Group x: transition 1 vs stable 2, 3  -> lower in 2 of 2 pairs -> 1.0
    # Group y: transition 5 vs stable 4, 5  -> lower 0, tie 1 of 2   -> 0.25
    values = np.array([1.0, 2.0, 3.0, 5.0, 4.0, 5.0])
    is_event = np.array([True, False, False, True, False, False])
    strata = np.array(["x", "x", "x", "y", "y", "y"])
    auc, n_events, n_controls = h1.matched_auc(values, is_event, strata)
    assert (n_events, n_controls) == (2, 4)
    assert auc == pytest.approx((1.0 + 0.25) / 2)


def test_groups_without_a_partner_are_skipped():
    values = np.array([1.0, 2.0, 9.0])
    is_event = np.array([True, False, True])
    strata = np.array(["x", "x", "lonely"])
    auc, n_events, n_controls = h1.matched_auc(values, is_event, strata)
    assert (auc, n_events, n_controls) == (1.0, 1, 1)


def test_permutation_test_finds_a_real_difference_and_ignores_noise():
    rng = np.random.default_rng(config.RANDOM_SEED)
    strata = np.repeat(["a", "b", "c", "d"], 30)
    is_event = np.tile(np.r_[np.ones(10, bool), np.zeros(20, bool)], 4)

    noise = rng.normal(60, 10, len(strata))
    assert h1.permutation_p_value(noise, is_event, strata, rng) > 0.05

    lower_before = noise - 12 * is_event          # transitions clearly lower
    assert h1.permutation_p_value(lower_before, is_event, strata, rng) < 0.01


def test_matching_removes_a_difference_that_is_only_between_groups():
    """If group 'a' is simply lower than group 'b' and holds most transitions,
    an unmatched test would see an effect. The matched one must not."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    strata = np.repeat(["a", "b"], 60)
    is_event = np.r_[np.ones(40, bool), np.zeros(20, bool), np.ones(5, bool), np.zeros(55, bool)]
    values = np.where(strata == "a", 40.0, 70.0) + rng.normal(0, 5, 120)
    auc, _, _ = h1.matched_auc(values, is_event, strata)
    assert abs(auc - 0.5) < 0.1
    assert h1.permutation_p_value(values, is_event, strata, rng) > 0.05


def test_logistic_regression_recovers_known_coefficients():
    rng = np.random.default_rng(config.RANDOM_SEED)
    x = rng.normal(size=(4000, 2))
    probability = 1 / (1 + np.exp(-(0.5 + 1.5 * x[:, 0] - 1.0 * x[:, 1])))
    y = (rng.random(4000) < probability).astype(float)
    np.testing.assert_allclose(h1.logistic_coefficients(x, y), [0.5, 1.5, -1.0], atol=0.15)


def test_window_never_includes_the_transition_day(monkeypatch):
    """Put an extreme alignment value on every transition day. If any window
    included its transition day, a level would be pulled far away from 50."""
    days = pd.date_range("2023-06-01", periods=120, freq="D")
    regimes = np.array((["Bullish"] * 20 + ["Bearish"] * 20) * 3)
    is_start, _ = h1.transition_starts(regimes, config.H1_PERSISTENCE)

    table = pd.DataFrame({"regime": regimes, "confidence": 0.9, "atr_rel": 0.0}, index=days)
    align = pd.DataFrame({v: 50.0 for v in h1.ALIGNMENT_VARIANTS}, index=days)
    align.loc[days[is_start], h1.ALIGNMENT_VARIANTS] = -1000.0
    align["coverage"] = 1.0
    monkeypatch.setattr(h1, "daily_table", lambda symbol: table)
    monkeypatch.setattr(h1, "daily_alignment", lambda symbol: align)
    monkeypatch.setattr(h1, "four_hour_changes", lambda symbol: pd.Series(1, index=days))

    windows = h1.build_windows("TEST", config.H1_PERSISTENCE)
    transitions = windows[windows["kind"] == "transition"]
    assert len(transitions) == 5
    assert (transitions["level_standard"] == 50.0).all()
    assert (transitions["regime"] != regimes[is_start][:len(transitions)]).all()  # the OLD regime
    # Stable days are at least a week from every transition.
    starts = days[is_start]
    for day in windows[windows["kind"] == "stable"]["date"]:
        assert min(abs((day - starts).days)) > config.H1_STABLE_MARGIN_DAYS
