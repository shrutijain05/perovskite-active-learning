"""
surrogate/acquisition.py
Acquisition functions: turn a surrogate's (mean, std) predictions into a
single score for "how worth testing is this candidate right now."

Target-oriented Expected Improvement is the default (used by Phase 4's
closed loop) — it's built for "land near a specific bandgap," not "find
the maximum." UCB/LCB serve a different use case: mapping how far a
composition space can push the bandgap in either direction, rather than
hitting one target value.
"""

from typing import Callable, Dict

import numpy as np
from scipy.stats import norm


def target_expected_improvement(
    mean: np.ndarray, std: np.ndarray, target: float, delta: float = 0.05, min_std: float = 1e-6
) -> np.ndarray:
    """Scores each candidate by how likely it is to fall within
    [target - delta, target + delta], weighted by remaining uncertainty.

    A confident prediction sitting dead-center on target can still score
    lower than an uncertain one that merely overlaps the window — because
    testing the confident one teaches you almost nothing you didn't
    already believe, while the uncertain one might resolve into exactly
    the answer you want.
    """
    mean = np.asarray(mean, dtype=np.float64)
    std = np.maximum(np.asarray(std, dtype=np.float64), min_std)
    z_upper = (target + delta - mean) / std
    z_lower = (target - delta - mean) / std
    prob_in_range = norm.cdf(z_upper) - norm.cdf(z_lower)
    return prob_in_range * std


def upper_confidence_bound(mean: np.ndarray, std: np.ndarray, kappa: float = 2.0) -> np.ndarray:
    """mean + kappa * std — use for searching out the MAXIMUM achievable
    bandgap. Larger kappa biases more toward unexplored/uncertain regions."""
    return np.asarray(mean, dtype=np.float64) + kappa * np.asarray(std, dtype=np.float64)


def lower_confidence_bound(mean: np.ndarray, std: np.ndarray, kappa: float = 2.0) -> np.ndarray:
    """mean - kappa * std — the mirror image of UCB, for searching out the
    MINIMUM achievable bandgap."""
    return np.asarray(mean, dtype=np.float64) - kappa * np.asarray(std, dtype=np.float64)


ACQUISITION_FUNCTIONS: Dict[str, Callable] = {
    "target_ei": target_expected_improvement,
    "ucb": upper_confidence_bound,
    "lcb": lower_confidence_bound,
}


def score_candidates(
    surrogate, X_candidates: np.ndarray, strategy: str = "target_ei", **kwargs
) -> np.ndarray:
    """Convenience entry point for the Phase 4 closed loop: predicts with
    the surrogate, then scores every candidate with the named strategy.

    Example: score_candidates(surrogate, X, strategy="target_ei", target=1.34)
    """
    if strategy not in ACQUISITION_FUNCTIONS:
        raise ValueError(
            f"Unknown strategy '{strategy}' — choose from {list(ACQUISITION_FUNCTIONS)}"
        )
    mean, std = surrogate.predict(X_candidates)
    return ACQUISITION_FUNCTIONS[strategy](mean, std, **kwargs)
