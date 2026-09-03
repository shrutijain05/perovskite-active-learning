"""
tests/test_loop.py
Validates the closed-loop orchestrator: history bookkeeping, monotonic
best-error tracking, reproducibility given a fixed seed, and — the claim
that actually matters — that it converges toward the target and does
measurably better than picking candidates at random on the same budget.
"""

import numpy as np
import pytest

from dft_runner.benchmark_engine import BenchmarkEvaluationBackend
from perov_core.space import generate_composition_space
from pipeline.loop import ActiveLearningLoop, _composition_key, run_random_search_baseline


@pytest.fixture
def candidate_space():
    # 3 A x 2 B x 3 X at 10% resolution -> a few thousand candidates,
    # small enough to run fast in a test, large enough to be a real search.
    return generate_composition_space(
        a_site_ions=["Cs", "MA", "FA"],
        b_site_ions=["Pb", "Sn"],
        x_site_ions=["Cl", "Br", "I"],
        resolution=0.1,
    )


@pytest.fixture
def evaluator(tmp_path):
    return BenchmarkEvaluationBackend(cache_path=tmp_path / "cache.json")


def test_history_length_matches_seed_plus_iterations(candidate_space, evaluator):
    loop = ActiveLearningLoop(
        target_bandgap_eV=1.9, candidate_space=candidate_space, evaluator=evaluator, n_initial=5
    )
    loop.run(n_iterations=10)
    assert len(loop.history) == 15  # 5 seed + 10 loop iterations


def test_best_error_so_far_is_monotonically_non_increasing(candidate_space, evaluator):
    loop = ActiveLearningLoop(
        target_bandgap_eV=1.9, candidate_space=candidate_space, evaluator=evaluator, n_initial=5
    )
    loop.run(n_iterations=15)
    best_so_far = [h.best_error_so_far_eV for h in loop.history]
    assert all(b1 >= b2 for b1, b2 in zip(best_so_far, best_so_far[1:]))


def test_no_composition_evaluated_twice(candidate_space, evaluator):
    loop = ActiveLearningLoop(
        target_bandgap_eV=1.9, candidate_space=candidate_space, evaluator=evaluator, n_initial=5
    )
    loop.run(n_iterations=15)
    keys = [_composition_key(h.composition) for h in loop.history]
    assert len(keys) == len(set(keys))


def test_same_seed_reproduces_identical_run(candidate_space, evaluator, tmp_path):
    ev1 = BenchmarkEvaluationBackend(cache_path=tmp_path / "c1.json")
    ev2 = BenchmarkEvaluationBackend(cache_path=tmp_path / "c2.json")
    loop1 = ActiveLearningLoop(1.9, candidate_space, ev1, n_initial=5, random_state=7)
    loop2 = ActiveLearningLoop(1.9, candidate_space, ev2, n_initial=5, random_state=7)
    result1 = loop1.run(n_iterations=10)
    result2 = loop2.run(n_iterations=10)
    assert result1["best_bandgap_eV"] == result2["best_bandgap_eV"]
    assert [h.bandgap_eV for h in loop1.history] == [h.bandgap_eV for h in loop2.history]


def test_converges_within_tolerance_given_enough_budget(candidate_space, evaluator):
    """The real claim: given a reasonable budget, the loop should land
    within a tight tolerance of an achievable target. 1.9 eV sits well
    inside the mixed I/Br range this search space can reach."""
    loop = ActiveLearningLoop(
        target_bandgap_eV=1.9, candidate_space=candidate_space, evaluator=evaluator, n_initial=5
    )
    result = loop.run(n_iterations=30, convergence_tol=0.02)
    assert result["best_error_eV"] <= 0.02
    assert result["converged_at_iteration"] is not None


def test_raises_on_too_few_initial_points(candidate_space, evaluator):
    with pytest.raises(ValueError):
        ActiveLearningLoop(1.9, candidate_space, evaluator, n_initial=1)


def test_raises_when_candidate_space_smaller_than_n_initial(evaluator):
    tiny_space = generate_composition_space(["Cs"], ["Pb"], ["I"], resolution=1.0)
    with pytest.raises(ValueError):
        ActiveLearningLoop(1.9, tiny_space, evaluator, n_initial=5)


def test_active_learning_beats_random_search_on_same_budget(candidate_space, evaluator, tmp_path):
    """The whole point of the project, checked directly: for the same
    number of evaluations, the acquisition-driven loop should land at
    least as close to target as blind random search — usually closer.
    Uses epsilon=0.0 (fully greedy) so this is a clean head-to-head."""
    ev_bo = BenchmarkEvaluationBackend(cache_path=tmp_path / "bo.json")
    ev_random = BenchmarkEvaluationBackend(cache_path=tmp_path / "random.json")

    loop = ActiveLearningLoop(
        target_bandgap_eV=1.9,
        candidate_space=candidate_space,
        evaluator=ev_bo,
        n_initial=5,
        epsilon=0.0,
        random_state=123,
    )
    bo_result = loop.run(n_iterations=20)

    random_result = run_random_search_baseline(
        target_bandgap_eV=1.9,
        candidate_space=candidate_space,
        evaluator=ev_random,
        n_evaluations=25,  # same total budget as 5 seed + 20 loop iterations
        random_state=123,
    )

    assert bo_result["best_error_eV"] <= random_result["best_error_eV"]
