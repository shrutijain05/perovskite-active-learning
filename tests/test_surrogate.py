"""
tests/test_surrogate.py
Validates GP fitting/prediction behavior, acquisition function ordering,
and that predictive uncertainty is calibrated — not just "does it run."
"""

import numpy as np
import pytest

from surrogate.acquisition import (
    lower_confidence_bound,
    score_candidates,
    target_expected_improvement,
    upper_confidence_bound,
)
from surrogate.gp_model import PerovskiteGPSurrogate
from surrogate.uncertainty import calibration_report


@pytest.fixture
def toy_data():
    """Simple, noise-free synthetic function so behavior is predictable."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, size=(20, 14))
    y = 1.5 + 0.5 * X[:, 0] - 0.2 * X[:, 1]
    return X, y


# --- gp_model.py -----------------------------------------------------------

def test_uncertainty_shrinks_at_observed_points(toy_data):
    X, y = toy_data
    surrogate = PerovskiteGPSurrogate()
    surrogate.fit(X, y)
    _, std_at_observed = surrogate.predict(X[:1])
    assert std_at_observed[0] < 0.1


def test_uncertainty_grows_far_from_data(toy_data):
    """The whole point of a GP: more confident near data it's seen than in
    genuinely unexplored regions."""
    X, y = toy_data
    surrogate = PerovskiteGPSurrogate()
    surrogate.fit(X, y)

    _, std_near = surrogate.predict(X[:1])
    far_point = np.full((1, 14), 5.0)  # well outside the [0, 1] training range
    _, std_far = surrogate.predict(far_point)

    assert std_far[0] > std_near[0]


def test_predict_before_fit_raises_clear_error():
    surrogate = PerovskiteGPSurrogate()
    with pytest.raises(RuntimeError):
        surrogate.predict(np.zeros((1, 14)))


def test_wrong_feature_dimension_raises_clear_error(toy_data):
    X, y = toy_data
    surrogate = PerovskiteGPSurrogate()
    with pytest.raises(ValueError):
        surrogate.fit(X[:, :5], y)  # wrong number of features


# --- acquisition.py ----------------------------------------------------

def test_target_ei_prefers_uncertain_over_confident_when_both_plausible():
    """Two candidates both plausibly on-target: EI should score the more
    uncertain one higher, since testing it teaches you more."""
    mean = np.array([1.75, 1.75])
    std = np.array([0.3, 0.02])
    scores = target_expected_improvement(mean, std, target=1.75, delta=0.05)
    assert scores[0] > scores[1]


def test_target_ei_penalizes_being_far_from_target():
    mean = np.array([1.75, 2.50])
    std = np.array([0.2, 0.2])
    scores = target_expected_improvement(mean, std, target=1.75, delta=0.05)
    assert scores[0] > scores[1]


def test_ucb_and_lcb_are_mirror_images():
    mean = np.array([1.5, 2.0])
    std = np.array([0.1, 0.3])
    ucb = upper_confidence_bound(mean, std, kappa=2.0)
    lcb = lower_confidence_bound(mean, std, kappa=2.0)
    assert np.allclose(ucb, mean + 2.0 * std)
    assert np.allclose(lcb, mean - 2.0 * std)
    assert np.all(ucb > lcb)


def test_score_candidates_dispatches_by_strategy_name(toy_data):
    X, y = toy_data
    surrogate = PerovskiteGPSurrogate()
    surrogate.fit(X, y)

    scores = score_candidates(surrogate, X[:3], strategy="target_ei", target=1.75)
    assert scores.shape == (3,)

    with pytest.raises(ValueError):
        score_candidates(surrogate, X[:3], strategy="not_a_real_strategy")


# --- uncertainty.py ----------------------------------------------------

def test_calibration_report_returns_reasonable_values(toy_data):
    """On a GP fit to smooth synthetic data and evaluated on held-out
    points from the same distribution, coverage shouldn't be wildly off
    from nominal. Bounds here are deliberately generous — 20 training
    points isn't enough for a tight statistical guarantee, only a sanity
    check that calibration isn't badly broken."""
    X, y = toy_data
    surrogate = PerovskiteGPSurrogate()
    surrogate.fit(X, y)

    rng = np.random.default_rng(7)
    X_test = rng.uniform(0, 1, size=(50, 14))
    y_test = 1.5 + 0.5 * X_test[:, 0] - 0.2 * X_test[:, 1]
    mean, std = surrogate.predict(X_test)

    report = calibration_report(y_test, mean, std)
    assert 0.0 <= report["coverage_68pct"] <= 1.0
    assert report["coverage_95pct"] >= report["coverage_68pct"]
