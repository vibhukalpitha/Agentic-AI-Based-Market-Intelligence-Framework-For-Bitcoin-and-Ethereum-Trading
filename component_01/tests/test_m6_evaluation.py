"""Tests for the evaluation of the Regime Stability Score. Run:  python -m pytest"""
import numpy as np
import pandas as pd

from src import config
from src.experiments import m6_evaluation as m6


def test_brier_score_values():
    target = np.array([1.0, 0.0, 1.0, 0.0])
    assert m6.brier(target, target) == 0.0                       # perfect
    assert m6.brier(target, np.full(4, 0.5)) == 0.25             # coin flip
    assert m6.brier(target, 1 - target) == 1.0                   # always wrong


def test_auc_values():
    target = np.array([1.0, 1.0, 0.0, 0.0])
    assert m6.auc(target, np.array([0.9, 0.8, 0.3, 0.1])) == 1.0  # perfect ranking
    assert m6.auc(target, np.array([0.1, 0.2, 0.8, 0.9])) == 0.0  # reversed
    assert m6.auc(target, np.full(4, 0.7)) == 0.5                 # a constant has no skill
    assert np.isnan(m6.auc(np.ones(4), np.full(4, 0.7)))          # only one class present


def test_calibration_table_groups_days_by_forecast():
    probability = np.repeat([0.5, 0.6, 0.7, 0.8, 0.9], 20)
    rng = np.random.default_rng(config.RANDOM_SEED)
    target = (rng.random(100) < probability).astype(float)
    table = m6.calibration_table(target, probability)
    assert len(table) == m6.CALIBRATION_BINS and (table["days"] == 20).all()
    np.testing.assert_allclose(table["mean_forecast"], [0.5, 0.6, 0.7, 0.8, 0.9])
    assert table["mean_forecast"].is_monotonic_increasing


def test_block_differences_have_the_right_sign_and_bracket_the_estimate():
    rng = np.random.default_rng(config.RANDOM_SEED)
    days = 420
    target = (rng.random(days) < 0.8).astype(float)
    frame = pd.DataFrame({
        "target": target,
        "good": np.where(target == 1, 0.9, 0.3),     # informative
        "flat": np.full(days, 0.8),                  # a constant
    })
    result = m6.weekly_block_differences(frame, [("good", "flat")], rng)["good_minus_flat"]
    assert result["brier_difference"] < 0            # negative = "good" is better
    assert result["ci_low"] <= result["brier_difference"] <= result["ci_high"]
    assert result["ci_high"] < 0
