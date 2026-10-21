"""Tests for Novelty 3 (context-adaptive weights). Run:  python -m pytest"""
import numpy as np
import pandas as pd

from src import config
from src.fusion import adaptive_weights as aw
from src.stability import stability


def make_inputs(days: int = 1200, context_matters: bool = True) -> pd.DataFrame:
    """Synthetic days. If context_matters, the 4h timeframe's agreement raises
    persistence only in the ACTIVE context, and the 5m's only in the calm one."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    frame = pd.DataFrame(rng.normal(size=(days, 4)), columns=stability.OWN_SIGNALS,
                         index=pd.date_range("2021-01-01", periods=days, freq="D"))
    for name in aw.AGREE:
        frame[name] = (rng.random(days) < 0.5).astype(float)
    weights = np.array([config.TIMEFRAME_WEIGHTS[i] for i in aw.FASTER])
    frame["agree_share"] = frame[aw.AGREE].to_numpy() @ weights / weights.sum()
    active = frame["volatility"] > 0
    if context_matters:
        z = np.where(active, 2.5 * frame["agree_4h"], 2.5 * frame["agree_5m"]) - 0.5
    else:
        z = 1.5 * frame["agree_4h"] - 0.5
    frame["target"] = (rng.random(days) < 1 / (1 + np.exp(-z))).astype(float)
    return aw.add_context(frame)


def test_context_split_is_at_the_yearly_norm_and_partitions_agreement():
    frame = make_inputs()
    assert ((frame["volatility"] > 0) == (frame["active"] == 1.0)).all()
    for name in aw.AGREE:
        # every agreement is counted in exactly one context
        np.testing.assert_array_equal(frame[f"{name}_calm"] + frame[f"{name}_active"], frame[name])
        assert (frame.loc[frame["active"] == 1.0, f"{name}_calm"] == 0).all()


def test_adaptive_model_recovers_weights_that_depend_on_context():
    frame = make_inputs(context_matters=True)
    models = {name: stability.fit(frame, signals) for name, signals in aw.MODELS.items()}
    weights = aw.learned_weights(models)
    assert weights.loc["4h", "adaptive_active"] > 1.5 > 0.6 > abs(weights.loc["4h", "adaptive_calm"])
    assert weights.loc["5m", "adaptive_calm"] > 1.5 > 0.6 > abs(weights.loc["5m", "adaptive_active"])
    assert list(weights["fixed_proposed"]) == [0.25, 0.20, 0.15, 0.10]


def test_selection_keeps_fixed_weights_unless_a_richer_model_clearly_wins():
    same = pd.DataFrame({"F": [0.50] * 4, "L": [0.51, 0.49, 0.52, 0.50], "A": [0.60] * 4})
    learned = pd.DataFrame({"F": [0.50] * 4, "L": [0.45, 0.46, 0.51, 0.44], "A": [0.47] * 4})
    adaptive = pd.DataFrame({"F": [0.50] * 4, "L": [0.45, 0.46, 0.51, 0.44], "A": [0.40] * 4})
    lucky = pd.DataFrame({"F": [0.50] * 4, "L": [0.10, 0.51, 0.51, 0.51], "A": [0.60] * 4})
    assert aw.select_model(same) == "F"
    assert aw.select_model(learned) == "L"
    assert aw.select_model(adaptive) == "A"
    assert aw.select_model(lucky) == "F"       # better on average, but in one fold only


def test_cross_validation_prefers_adaptive_only_when_context_really_matters():
    with_context = stability.cross_validate(make_inputs(context_matters=True), aw.MODELS)
    without = stability.cross_validate(make_inputs(context_matters=False), aw.MODELS)
    assert aw.select_model(with_context) == "A"
    assert aw.select_model(without) != "A"
