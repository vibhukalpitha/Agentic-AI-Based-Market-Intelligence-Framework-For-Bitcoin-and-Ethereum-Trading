"""Tests for the similarity charts' calculations. Run:  python -m pytest"""
import numpy as np

from src.figures.similarity_charts import dtw_path, percentile
from src.similarity.dtw_match import dtw_distances


def test_traced_path_reproduces_the_dtw_distance():
    rng = np.random.default_rng(42)
    for _ in range(20):
        a, b = rng.normal(size=(10, 9)), rng.normal(size=(10, 9))
        _, traced = dtw_path(a, b)
        assert abs(traced - dtw_distances(a, b[None])[0]) < 1e-12


def test_path_runs_from_first_to_last_day_in_small_steps():
    rng = np.random.default_rng(0)
    pairs, _ = dtw_path(rng.normal(size=(10, 3)), rng.normal(size=(10, 3)))
    assert pairs[0] == (0, 0) and pairs[-1] == (9, 9)
    for (i, j), (k, m) in zip(pairs, pairs[1:]):
        assert (k - i, m - j) in {(1, 1), (1, 0), (0, 1)}


def test_identical_stretches_are_paired_day_by_day():
    a = np.arange(30.0).reshape(10, 3)
    pairs, distance = dtw_path(a, a)
    assert pairs == [(k, k) for k in range(10)] and distance == 0.0


def test_percentile_counts_stretches_that_are_further_away():
    distances = np.array([1.0, 2.0, 3.0, 4.0])
    assert percentile(1.0, distances) == 75.0          # 3 of 4 are further away
    assert percentile(4.0, distances) == 0.0
