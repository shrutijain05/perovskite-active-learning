"""
surrogate/gp_model.py
Gaussian Process surrogate: predicts a bandgap mean and uncertainty for any
composition's feature vector, given whatever data has been observed so far.
"""

from typing import Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from perov_core.descriptors import N_FEATURES


class PerovskiteGPSurrogate:
    """Wraps scikit-learn's GaussianProcessRegressor with the kernel choice
    appropriate for this project: Matern 5/2 assumes the underlying
    composition -> bandgap function is smooth but not infinitely so, a
    physically reasonable assumption for a real material property surface.
    A small WhiteKernel term accounts for the fact that even DFT or
    database-looked-up bandgaps aren't perfectly noise-free ground truth.
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        n_restarts_optimizer: int = 10,
        random_state: int = 42,
    ):
        kernel = Matern(length_scale=np.ones(N_FEATURES) * length_scale, nu=2.5) + WhiteKernel(
            noise_level=1e-3
        )
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=True,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != N_FEATURES:
            raise ValueError(f"Expected X with shape (n_samples, {N_FEATURES}), got {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]} values")
        if X.shape[0] < 2:
            raise ValueError("Need at least 2 observations to fit a GP usefully")
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict() — the surrogate has no data yet.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != N_FEATURES:
            raise ValueError(f"Expected X with shape (n_samples, {N_FEATURES}), got {X.shape}")
        mean, std = self.model.predict(X, return_std=True)
        return mean, std
