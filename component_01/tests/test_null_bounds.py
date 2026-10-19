"""Tests for the null-result bounds. Run:  python -m pytest"""
from src.experiments.null_bounds import classify


def test_each_status():
    assert classify(0.01, 0.20, 0.06) == "supported"            # whole interval above 0
    assert classify(-0.16, -0.04, 0.06) == "opposite direction"  # whole interval below 0
    assert classify(-0.12, 0.003, 0.06) == "ruled out"          # small effect excluded
    assert classify(-0.13, 0.11, 0.06) == "inconclusive"         # small effect still possible


def test_boundary_counts_as_inconclusive():
    """An interval that just reaches the small effect cannot rule it out."""
    assert classify(-0.1, 0.06, 0.06) == "inconclusive"
