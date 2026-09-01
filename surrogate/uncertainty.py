"""
surrogate/uncertainty.py
Checks whether the GP's stated confidence intervals are actually honest —
if it says "68% confidence," does the true value fall inside that interval
roughly 68% of the time on held-out data? A model that's consistently
overconfident (intervals too narrow) makes the acquisition function
under-explore; consistently underconfident wastes evaluations re-checking
things it was already sure about.
"""

from typing import Dict

import numpy as np
from scipy.stats import norm


def empirical_coverage(
    y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, confidence: float = 0.68
) -> float:
    """Fraction of points where y_true actually falls inside the
    surrogate's stated confidence interval at the given confidence level.

    A well-calibrated model returns a value close to `confidence` itself —
    e.g. ~0.68 for the default 68% interval (one standard deviation on
    each side of a normal distribution).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.maximum(np.asarray(std, dtype=np.float64), 1e-12)

    z = norm.ppf(0.5 + confidence / 2)
    lower = mean - z * std
    upper = mean + z * std
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def calibration_report(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> Dict[str, float]:
    """Empirical coverage at a few standard confidence levels, for a quick
    at-a-glance calibration check (e.g. printed in a notebook)."""
    return {
        f"coverage_{int(c * 100)}pct": empirical_coverage(y_true, mean, std, confidence=c)
        for c in (0.5, 0.68, 0.95)
    }
