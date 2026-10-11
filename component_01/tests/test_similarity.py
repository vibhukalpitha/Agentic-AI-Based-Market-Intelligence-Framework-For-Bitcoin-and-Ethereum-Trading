"""Tests for M7 (historical similarity). Run:  python -m pytest"""
import numpy as np
import pandas as pd

from src import config
from src.similarity import dtw_match


def test_dtw_of_a_stretch_with_itself_is_zero():
    rng = np.random.default_rng(config.RANDOM_SEED)
    stretch = rng.normal(size=(10, 9))
    assert dtw_match.dtw_distances(stretch, stretch[None])[0] == 0.0


def test_dtw_hand_calculation():
    # One feature. Query 0, 1, 2; candidate 0, 2, 2.
    # cost      total (cheapest path so far)
    # 0 2 2     0 2 4
    # 1 1 1     1 1 2
    # 2 0 0     3 1 1      -> bottom-right 1, divided by 3 days
    query = np.array([[0.0], [1.0], [2.0]])
    candidate = np.array([[[0.0], [2.0], [2.0]]])
    np.testing.assert_allclose(dtw_match.dtw_distances(query, candidate), [1.0 / 3.0])


def test_dtw_forgives_a_shape_that_is_one_day_out_of_step():
    """The same bump one day later is far in a day-by-day comparison but close in DTW."""
    bump = np.array([0, 0, 0, 5, 5, 0, 0, 0, 0, 0], dtype=float)[:, None]
    shifted = np.roll(bump, 1, axis=0)
    day_by_day = np.abs(bump - shifted).sum() / len(bump)
    assert dtw_match.dtw_distances(bump, shifted[None])[0] == 0.0 < day_by_day


def test_matches_come_only_from_before_the_stretch_and_do_not_overlap():
    rng = np.random.default_rng(config.RANDOM_SEED)
    x = rng.normal(size=(400, 4))
    window, end = 10, 300
    x[100:110] = x[end - window + 1:end + 1]          # plant an exact copy in the past
    x[350:360] = x[end - window + 1:end + 1]          # and one in the FUTURE
    matches = dtw_match.find_matches(x, end, window, 3)

    assert matches[0] == (109, 0.0)                   # the past copy is found first
    ends = [match_end for match_end, _ in matches]
    assert max(ends) < end - window + 1               # nothing from the stretch itself or later
    assert all(abs(a - b) >= window for k, a in enumerate(ends) for b in ends[k + 1:])


def test_matches_do_not_change_when_the_future_is_deleted():
    rng = np.random.default_rng(config.RANDOM_SEED)
    x = rng.normal(size=(300, 4))
    assert dtw_match.find_matches(x, 200, 10, 3) == dtw_match.find_matches(x[:201], 200, 10, 3)
    assert dtw_match.find_matches(x, 12, 10, 3) == []   # not enough history before it


def test_similar_history_reports_dates_and_regimes_but_no_outcomes():
    rng = np.random.default_rng(config.RANDOM_SEED)
    days = pd.date_range("2022-01-01", periods=200, freq="D")
    regimes = pd.DataFrame({"open_time": days, "regime": rng.choice(config.REGIMES, 200)})
    x = rng.normal(size=(200, 9))
    block = dtw_match.similar_history(regimes, x, days[150])

    assert block["window_days"] == config.SIMILARITY_WINDOW_DAYS
    assert block["regime_now"] == regimes["regime"].iloc[150]
    assert len(block["matches"]) == config.SIMILARITY_MATCHES
    assert set(block["matches"][0]) == {"window_start", "window_end", "distance", "regime_then"}
    assert all(pd.Timestamp(m["window_end"]) < days[150 - config.SIMILARITY_WINDOW_DAYS + 1]
               for m in block["matches"])
    assert block["matches_in_same_regime"] == sum(
        m["regime_then"] == block["regime_now"] for m in block["matches"])
    assert dtw_match.similar_history(regimes, x, pd.Timestamp("2030-01-01")) is None
