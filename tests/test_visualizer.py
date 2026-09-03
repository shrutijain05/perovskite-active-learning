"""
tests/test_visualizer.py
Confirms the plotting functions actually run against a real completed
loop and produce a saved file — not just that the code has no syntax
errors, but that it works against the real IterationRecord/surrogate
objects the loop produces.
"""

import matplotlib

matplotlib.use("Agg")  # headless backend for CI / no-display environments

import numpy as np
import pytest

from dft_runner.benchmark_engine import BenchmarkEvaluationBackend
from perov_core.space import generate_composition_space
from pipeline.loop import ActiveLearningLoop, run_random_search_baseline
from pipeline.visualizer import plot_bo_vs_random, plot_optimization_trajectory, plot_parity


@pytest.fixture
def completed_loop(tmp_path):
    space = generate_composition_space(["Cs", "MA", "FA"], ["Pb", "Sn"], ["Cl", "Br", "I"], 0.1)
    evaluator = BenchmarkEvaluationBackend(cache_path=tmp_path / "cache.json")
    loop = ActiveLearningLoop(1.9, space, evaluator, n_initial=5)
    loop.run(n_iterations=10)
    return loop


def test_trajectory_plot_saves_a_file(completed_loop, tmp_path):
    out = tmp_path / "trajectory.png"
    fig = plot_optimization_trajectory(completed_loop.history, target_bandgap_eV=1.9, save_path=out)
    assert out.exists()
    assert out.stat().st_size > 0
    assert fig is not None


def test_parity_plot_saves_a_file(completed_loop, tmp_path):
    X_test = np.array(completed_loop.X_observed)
    y_test = np.array(completed_loop.y_observed)
    out = tmp_path / "parity.png"
    fig = plot_parity(completed_loop.surrogate, X_test, y_test, save_path=out)
    assert out.exists()
    assert out.stat().st_size > 0
    assert fig is not None


def test_bo_vs_random_plot_saves_a_file(completed_loop, tmp_path):
    space = generate_composition_space(["Cs", "MA", "FA"], ["Pb", "Sn"], ["Cl", "Br", "I"], 0.1)
    ev_random = BenchmarkEvaluationBackend(cache_path=tmp_path / "random_cache.json")
    random_result = run_random_search_baseline(1.9, space, ev_random, n_evaluations=15)

    out = tmp_path / "bo_vs_random.png"
    fig = plot_bo_vs_random(
        completed_loop.history, random_result["history"], target_bandgap_eV=1.9, save_path=out
    )
    assert out.exists()
    assert fig is not None
