"""
pipeline/visualizer.py
Two plots that turn an ActiveLearningLoop run into evidence rather than an
assertion: a parity plot (is the GP's prediction actually trustworthy?) and
an optimization trajectory (did the search actually converge, and how
fast?). Both take matplotlib Axes optionally, so they can be composed into
a single figure (e.g. side-by-side in a notebook) or used standalone.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from pipeline.loop import IterationRecord
from surrogate.gp_model import PerovskiteGPSurrogate


def plot_optimization_trajectory(
    history: List[IterationRecord],
    target_bandgap_eV: float,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plots |bandgap - target| for every evaluation, plus the running best.
    A converging search shows the orange 'best so far' line dropping and
    flattening out near zero — this is the plot that makes an efficiency
    claim ('reached target in N evaluations') visually verifiable rather
    than just asserted in a README.
    """
    iterations = [h.iteration for h in history]
    errors = [h.error_eV for h in history]
    best_errors = [h.best_error_so_far_eV for h in history]
    random_mask = [h.was_random_exploration for h in history]

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7, 4.5))

    greedy_x = [i for i, r in zip(iterations, random_mask) if not r]
    greedy_y = [e for e, r in zip(errors, random_mask) if not r]
    random_x = [i for i, r in zip(iterations, random_mask) if r]
    random_y = [e for e, r in zip(errors, random_mask) if r]

    ax.plot(greedy_x, greedy_y, "o", alpha=0.5, color="tab:blue", label="acquisition-chosen")
    if random_x:
        ax.plot(random_x, random_y, "^", alpha=0.6, color="tab:green", label="epsilon-random")
    ax.plot(iterations, best_errors, "-", linewidth=2, color="tab:orange", label="best so far")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    ax.set_xlabel("Evaluation index")
    ax.set_ylabel(f"|bandgap - target| (eV), target = {target_bandgap_eV:.3f} eV")
    ax.set_title("Active learning optimization trajectory")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_parity(
    surrogate: PerovskiteGPSurrogate,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Predicted-vs-true bandgap on held-out points, with 95% CI error
    bars. Points hugging the diagonal with honest-sized error bars is what
    a trustworthy surrogate looks like; points far off the diagonal with
    tiny error bars is what an overconfident, wrong one looks like.
    """
    mean, std = surrogate.predict(X_test)
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(5, 5))

    ax.errorbar(
        y_test, mean, yerr=1.96 * std, fmt="o", alpha=0.6, ecolor="lightgray", capsize=2
    )
    lo = min(float(np.min(y_test)), float(np.min(mean))) - 0.1
    hi = max(float(np.max(y_test)), float(np.max(mean))) + 0.1
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="perfect prediction")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True bandgap (eV)")
    ax.set_ylabel("GP predicted bandgap (eV)")
    ax.set_title("Parity plot (95% CI)")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_bo_vs_random(
    bo_history: List[IterationRecord],
    random_history: List[IterationRecord],
    target_bandgap_eV: float,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Overlays one BO run's best-so-far curve against one random-search
    run's, on the same evaluation budget. A single-seed version of the
    comparison Phase 6 formalizes properly across many seeds — useful
    during development to sanity-check the acquisition function is
    actually earning its complexity, not just decoration.
    """
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7, 4.5))
    ax.plot(
        [h.iteration for h in bo_history],
        [h.best_error_so_far_eV for h in bo_history],
        "-",
        linewidth=2,
        color="tab:orange",
        label="active learning (BO)",
    )
    ax.plot(
        [h.iteration for h in random_history],
        [h.best_error_so_far_eV for h in random_history],
        "-",
        linewidth=2,
        color="tab:gray",
        label="random search",
    )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Evaluation index")
    ax.set_ylabel(f"best |bandgap - target| so far (eV), target = {target_bandgap_eV:.3f} eV")
    ax.set_title("Active learning vs. random search")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
