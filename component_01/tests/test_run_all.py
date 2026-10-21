"""Tests for the pipeline runner. Run:  python -m pytest"""
import importlib.util

from src import run_all

ALL_STEPS = run_all.BUILD_STEPS + run_all.REPRODUCE_STEPS


def position(module: str, *args: str) -> int:
    """Index of the first step running this module (with these arguments, if given)."""
    for k, step in enumerate(ALL_STEPS):
        if step.module == module and all(a in step.args for a in args):
            return k
    raise AssertionError(f"no step runs {module} {args}")


def test_every_step_points_to_an_existing_module():
    for step in ALL_STEPS:
        assert importlib.util.find_spec(step.module) is not None, step.module


def test_no_step_trains_or_downloads():
    for step in ALL_STEPS:
        assert step.module != "src.data.download"
        if step.module == "src.regime.train":
            assert "--relabel" in step.args          # labels from the saved models only
        assert step.module not in ("src.regime.convergence_check",)
        if step.module == "src.regime.bic_check":
            assert step.args == ("--summary",)       # never the multi-hour comparison


def test_order_follows_the_data_dependencies():
    features = position("src.features.indicators")
    labels = [position("src.regime.train", i) for i in ("1d", "4h", "1h", "15m", "5m")]
    fusion = position("src.fusion.alignment")
    flips = position("src.fusion.propagation")
    stability = position("src.stability.stability")
    archetypes = position("src.explain.archetypes")
    assert features < min(labels)
    assert max(labels) < fusion and max(labels) < flips
    assert fusion < stability and flips < stability          # stability reads both
    assert stability < archetypes < position("src.output.snapshot")
    assert position("src.fusion.adaptive_weights") < position("src.experiments.n3_adaptive_weights")
    assert position("src.regime.model_quality") < position("src.figures.model_quality_charts")
    for needed in ("src.experiments.h1_precursor", "src.experiments.h2_placebo",
                   "src.experiments.h2b_failed_propagation", "src.experiments.n3_adaptive_weights"):
        assert position(needed) < position("src.experiments.null_bounds")
    assert position("pytest") == len(ALL_STEPS) - 1          # tests run last


def test_only_the_frozen_inventory_check_is_skippable_in_the_live_workspace():
    skippable = [step.module for step in ALL_STEPS if step.frozen_data_only]
    assert skippable == ["src.data.validate"]


def test_fingerprint_comparison(tmp_path):
    before = {"a": "1", "b": "2", "c": "3"}
    after = {"a": "1", "b": "9", "d": "4"}
    assert run_all.compare(before, after) == {"changed": ["b"], "added": ["d"], "removed": ["c"]}
