"""Tests for M6 (Regime Stability Score). Run:  python -m pytest"""
import numpy as np
import pandas as pd

from src import config
from src.stability import stability


def test_target_looks_only_at_the_next_days():
    regime = np.array(["A", "A", "A", "A", "B", "B", "B", "A"])
    target = stability.persistence_target(regime, horizon=3)
    # Day 0: days 1-3 are A A A -> persists. Day 1: days 2-4 are A A B -> changes.
    # Day 4: days 5-7 are B B A -> changes. The last 3 days have no known future.
    np.testing.assert_array_equal(target[:5], [1.0, 0.0, 0.0, 0.0, 0.0])
    assert np.isnan(target[5:]).all()


def test_target_of_a_day_does_not_depend_on_days_beyond_the_horizon():
    regime = np.array(["A"] * 6 + ["B"] * 4)
    changed_later = regime.copy()
    changed_later[9] = "A"                      # alter a day far in the future
    np.testing.assert_array_equal(stability.persistence_target(regime, 3)[:5],
                                  stability.persistence_target(changed_later, 3)[:5])


def test_candles_in_regime_restarts_at_every_change():
    regime = np.array(["A", "A", "B", "B", "B", "A"])
    assert list(stability.candles_in_regime(regime)) == [1, 2, 1, 2, 3, 1]


def make_frame(days: int = 600) -> pd.DataFrame:
    """Synthetic days where a higher 'confidence_logit' truly raises persistence
    and the cross-timeframe signals are pure noise."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    frame = pd.DataFrame(rng.normal(size=(days, 7)), columns=stability.MODELS["S2"],
                         index=pd.date_range("2021-01-01", periods=days, freq="D"))
    probability = 1 / (1 + np.exp(-(1.0 + 1.5 * frame["confidence_logit"])))
    frame["target"] = (rng.random(days) < probability).astype(float)
    return frame


def test_fit_recovers_the_signal_and_predictions_are_probabilities():
    frame = make_frame()
    model = stability.fit(frame, stability.MODELS["S1"])
    coefficients = dict(zip(model["signals"], model["coefficients"]))
    assert coefficients["confidence_logit"] > 1.0
    assert abs(coefficients["volatility"]) < 0.4
    probability = stability.predict(model, frame)
    assert ((probability > 0) & (probability < 1)).all()


def test_scaling_uses_the_fitting_rows_only():
    """Predictions for early rows must not change when later rows are added
    to the frame being scored (the model keeps its own mean and std)."""
    frame = make_frame()
    model = stability.fit(frame.iloc[:300], stability.MODELS["S2"])
    np.testing.assert_allclose(stability.predict(model, frame.iloc[:300]),
                               stability.predict(model, frame)[:300])


def test_cross_validation_never_fits_on_the_block_it_predicts():
    frame = make_frame()
    cv = stability.cross_validate(frame)
    assert len(cv) == config.STABILITY_CV_BLOCKS - 1
    # Each fold is fitted on strictly fewer days than precede its validation block.
    block = len(frame) // config.STABILITY_CV_BLOCKS
    for fold in cv.itertuples():
        assert fold.fit_days == fold.fold * block - config.STABILITY_HORIZON_DAYS
    assert (cv["fit_days"].diff().dropna() > 0).all()


def test_simpler_model_is_kept_unless_the_richer_one_clearly_wins():
    worse = pd.DataFrame({"S1": [0.50, 0.50, 0.50, 0.50], "S2": [0.52, 0.49, 0.53, 0.51]})
    mixed = pd.DataFrame({"S1": [0.50, 0.50, 0.50, 0.50], "S2": [0.20, 0.51, 0.51, 0.51]})
    better = pd.DataFrame({"S1": [0.50, 0.50, 0.50, 0.50], "S2": [0.45, 0.48, 0.51, 0.47]})
    assert stability.select_model(worse) == "S1"
    assert stability.select_model(mixed) == "S1"      # better on average, but in 1 fold only
    assert stability.select_model(better) == "S2"


def test_log_loss_punishes_confident_wrong_forecasts():
    target = np.array([1.0, 0.0])
    assert stability.log_loss(target, np.array([0.9, 0.1])) < stability.log_loss(
        target, np.array([0.6, 0.4])) < stability.log_loss(target, np.array([0.1, 0.9]))
