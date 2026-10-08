"""Tests for M3. Run:  python -m pytest

The key claim protected here: regime probabilities never use future data.
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.regime import hmm_model, state_mapping


@pytest.fixture(scope="module")
def fitted():
    """A small HMM fitted on synthetic data with two clearly different halves."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    x = np.vstack([rng.normal(-2, 1, (150, 3)), rng.normal(2, 1, (150, 3)),
                   rng.normal(-2, 1, (150, 3))])
    model, _ = hmm_model.fit_hmm(x, n_states=2, covariance_type="full")
    return model, x


def test_filtered_probabilities_use_no_future_data(fitted):
    """Cutting off the future must not change any earlier probability."""
    model, x = fitted
    full = hmm_model.filtered_probabilities(model, x)
    truncated = hmm_model.filtered_probabilities(model, x[:200])
    np.testing.assert_allclose(full[:200], truncated, rtol=1e-12, atol=1e-12)


def test_filtered_matches_hmmlearn_on_last_candle(fitted):
    """On the final candle there is no future, so the filtered answer must equal
    hmmlearn's own (smoothed) answer. This checks the forward algorithm is right."""
    model, x = fitted
    ours = hmm_model.filtered_probabilities(model, x)[-1]
    theirs = model.predict_proba(x)[-1]
    np.testing.assert_allclose(ours, theirs, atol=1e-8)


def test_filtered_probabilities_sum_to_one(fitted):
    model, x = fitted
    np.testing.assert_allclose(hmm_model.filtered_probabilities(model, x).sum(axis=1), 1.0)


def test_confirmed_states_removes_flicker():
    states = np.array([0, 0, 1, 0, 0, 1, 1, 1, 0])
    # persistence 3: the single 1 is ignored; the run of three 1s is accepted
    # on its third candle; the final lone 0 is not yet confirmed.
    expected = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1])
    np.testing.assert_array_equal(hmm_model.confirmed_states(states, 3), expected)
    np.testing.assert_array_equal(hmm_model.confirmed_states(states, 1), states)
    assert hmm_model.count_transitions(expected) == 1


def test_state_naming_rule():
    profile = pd.DataFrame({
        "share": [0.25, 0.25, 0.25, 0.25],
        "sma_dist": [0.25, -0.18, -0.02, 0.03],
        "atr_rel": [0.0, 0.4, -0.3, 0.1],
    })
    assert state_mapping.map_states(profile) == {
        0: "Bullish", 1: "Bearish", 2: "Sideways", 3: "Volatile"}
