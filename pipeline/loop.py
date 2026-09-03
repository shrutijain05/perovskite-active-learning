"""
pipeline/loop.py
The closed-loop orchestrator — this is the piece that turns Phases 1-3 from
three separate, independently-tested modules into an actual active-learning
search: seed with random points, fit the GP, score every untested candidate,
evaluate the best-scoring one, fold it back into the training set, repeat.

Also ships a plain random-search baseline (`run_random_search_baseline`) —
not the full BO-vs-random benchmark study (that's Phase 6, over many seeds
with real statistics), just enough to sanity-check during development that
the acquisition-driven loop is actually smarter than picking blindly.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from dft_runner.base import EvaluationBackend
from perov_core.descriptors import PerovskiteComposition
from surrogate.acquisition import score_candidates
from surrogate.gp_model import PerovskiteGPSurrogate


def _composition_key(comp: PerovskiteComposition) -> str:
    """Same canonicalization scheme as the evaluation backends' cache key —
    kept independent (not imported) since this is about deduplicating
    candidates within a search, not about caching evaluator results."""

    def _fmt(site: Dict[str, float]) -> str:
        return ",".join(f"{ion}:{frac:.6f}" for ion, frac in sorted(site.items()))

    return f"A[{_fmt(comp.a_site)}]B[{_fmt(comp.b_site)}]X[{_fmt(comp.x_site)}]"


@dataclass
class IterationRecord:
    """One row of the loop's history — everything needed to reconstruct a
    trajectory plot or debug a run after the fact."""

    iteration: int
    composition: PerovskiteComposition
    bandgap_eV: float
    error_eV: float
    best_error_so_far_eV: float
    was_random_exploration: bool = False


class ActiveLearningLoop:
    """Closed-loop active learning search for a target perovskite bandgap.

    Parameters
    ----------
    target_bandgap_eV : the bandgap you're trying to hit.
    candidate_space : the pool of PerovskiteComposition objects to search
        over (typically from perov_core.space.generate_composition_space).
    evaluator : anything implementing EvaluationBackend.evaluate() — the
        offline BenchmarkEvaluationBackend, the MaterialsProjectEvaluationBackend,
        or (Phase 7 stretch) a real DFT runner. The loop doesn't care which.
    delta : half-width of the "close enough to target" window used by the
        Target-EI acquisition function.
    n_initial : number of random seed points evaluated before the GP starts
        making decisions. Needs at least 2 for a GP fit to mean anything.
    epsilon : probability of picking a uniformly random untested candidate
        instead of the top acquisition score, each iteration. Exists
        because Target-EI can collapse into pure exploitation once sigma
        shrinks to ~0 everywhere near the target — a small random-restart
        chance keeps the search from getting stuck in a local pocket of
        the composition space. Set to 0.0 to disable and go fully greedy.
    min_std_floor : passed through to the acquisition function so a
        near-zero predictive std doesn't zero out every candidate's score
        at once (same collapse risk, addressed a second way).
    random_state : seeds both the initial-point selection and the epsilon-
        greedy draws, so two runs with the same seed reproduce exactly.
    """

    def __init__(
        self,
        target_bandgap_eV: float,
        candidate_space: List[PerovskiteComposition],
        evaluator: EvaluationBackend,
        delta: float = 0.05,
        n_initial: int = 5,
        epsilon: float = 0.1,
        min_std_floor: float = 1e-3,
        random_state: int = 42,
    ):
        if n_initial < 2:
            raise ValueError("n_initial must be >= 2 — a GP needs at least 2 points to fit")
        if len(candidate_space) < n_initial:
            raise ValueError(
                f"candidate_space has {len(candidate_space)} compositions, "
                f"fewer than n_initial={n_initial}"
            )

        self.target = target_bandgap_eV
        self.candidate_space = candidate_space
        self.evaluator = evaluator
        self.delta = delta
        self.n_initial = n_initial
        self.epsilon = epsilon
        self.min_std_floor = min_std_floor
        self.rng = np.random.default_rng(random_state)

        self.surrogate = PerovskiteGPSurrogate(random_state=random_state)
        self.X_observed: List[np.ndarray] = []
        self.y_observed: List[float] = []
        self.history: List[IterationRecord] = []
        self._evaluated_keys: set = set()

    def _evaluate_and_record(
        self, comp: PerovskiteComposition, iteration: int, was_random: bool = False
    ) -> float:
        bandgap = self.evaluator.evaluate(comp)
        self.X_observed.append(comp.compute_features())
        self.y_observed.append(bandgap)
        self._evaluated_keys.add(_composition_key(comp))

        error = abs(bandgap - self.target)
        prev_best = self.history[-1].best_error_so_far_eV if self.history else float("inf")
        self.history.append(
            IterationRecord(
                iteration=iteration,
                composition=comp,
                bandgap_eV=bandgap,
                error_eV=error,
                best_error_so_far_eV=min(error, prev_best),
                was_random_exploration=was_random,
            )
        )
        return bandgap

    def _seed_initial_points(self) -> None:
        indices = self.rng.choice(len(self.candidate_space), size=self.n_initial, replace=False)
        for idx in indices:
            self._evaluate_and_record(self.candidate_space[int(idx)], iteration=0)

    def _remaining_candidates(self) -> List[PerovskiteComposition]:
        return [c for c in self.candidate_space if _composition_key(c) not in self._evaluated_keys]

    def run(
        self, n_iterations: int = 20, convergence_tol: Optional[float] = None
    ) -> Dict[str, Any]:
        """Runs the closed loop for up to n_iterations. Stops early if
        convergence_tol is set and the best error-to-target drops at or
        below it, or if the candidate space is exhausted.
        """
        if not self.history:
            self._seed_initial_points()
            if convergence_tol is not None and self.best_error() <= convergence_tol:
                return self._summary(converged_at_iteration=0)

        for i in range(1, n_iterations + 1):
            remaining = self._remaining_candidates()
            if not remaining:
                break

            self.surrogate.fit(np.array(self.X_observed), np.array(self.y_observed))

            use_random = self.rng.random() < self.epsilon
            if use_random:
                next_comp = remaining[int(self.rng.integers(len(remaining)))]
            else:
                X_remaining = np.array([c.compute_features() for c in remaining])
                scores = score_candidates(
                    self.surrogate,
                    X_remaining,
                    strategy="target_ei",
                    target=self.target,
                    delta=self.delta,
                    min_std=self.min_std_floor,
                )
                next_comp = remaining[int(np.argmax(scores))]

            self._evaluate_and_record(next_comp, iteration=i, was_random=use_random)

            if convergence_tol is not None and self.best_error() <= convergence_tol:
                return self._summary(converged_at_iteration=i)

        return self._summary(converged_at_iteration=None)

    def best_index(self) -> int:
        return int(np.argmin([h.error_eV for h in self.history]))

    def best_composition(self) -> PerovskiteComposition:
        return self.history[self.best_index()].composition

    def best_bandgap(self) -> float:
        return self.history[self.best_index()].bandgap_eV

    def best_error(self) -> float:
        return self.history[self.best_index()].error_eV

    def _summary(self, converged_at_iteration: Optional[int]) -> Dict[str, Any]:
        return {
            "best_composition": self.best_composition(),
            "best_bandgap_eV": self.best_bandgap(),
            "best_error_eV": self.best_error(),
            "n_evaluations": len(self.y_observed),
            "converged_at_iteration": converged_at_iteration,
            "history": self.history,
        }


def run_random_search_baseline(
    target_bandgap_eV: float,
    candidate_space: List[PerovskiteComposition],
    evaluator: EvaluationBackend,
    n_evaluations: int = 25,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Evaluates n_evaluations uniformly-random candidates — no GP, no
    acquisition function. This is the baseline the real Phase 6 benchmark
    compares ActiveLearningLoop against; here it's mainly for a quick
    sanity check that the smart loop actually beats blind guessing on the
    same search space and evaluator.
    """
    rng = np.random.default_rng(random_state)
    if len(candidate_space) < n_evaluations:
        raise ValueError(
            f"candidate_space has {len(candidate_space)} compositions, "
            f"fewer than n_evaluations={n_evaluations}"
        )
    indices = rng.choice(len(candidate_space), size=n_evaluations, replace=False)

    history: List[IterationRecord] = []
    best_so_far = float("inf")
    for i, idx in enumerate(indices):
        comp = candidate_space[int(idx)]
        bandgap = evaluator.evaluate(comp)
        error = abs(bandgap - target_bandgap_eV)
        best_so_far = min(error, best_so_far)
        history.append(
            IterationRecord(
                iteration=i,
                composition=comp,
                bandgap_eV=bandgap,
                error_eV=error,
                best_error_so_far_eV=best_so_far,
                was_random_exploration=True,
            )
        )

    best = min(history, key=lambda h: h.error_eV)
    return {
        "best_composition": best.composition,
        "best_bandgap_eV": best.bandgap_eV,
        "best_error_eV": best.error_eV,
        "n_evaluations": len(history),
        "history": history,
    }
