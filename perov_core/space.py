"""
perov_core/space.py
Generates the candidate composition search space: every combination of
A-site, B-site, and X-site mixing ratios at a chosen resolution.

Grid size grows fast — this is exactly why Phase 2 onward searches smartly
instead of evaluating every candidate. With 3 A-site ions, 2 B-site ions,
and 3 X-site ions at resolution=0.1 (10% steps), the grid is already in the
low thousands; resolution=0.05 pushes well past that.
"""

import itertools
from typing import Iterator, List

from perov_core.descriptors import PerovskiteComposition


def _mixing_ratios(n_components: int, resolution: float) -> Iterator[List[float]]:
    """Yields every combination of `n_components` fractions that are
    multiples of `resolution` and sum to 1.0.

    Example: n_components=2, resolution=0.5 -> [1.0, 0.0], [0.5, 0.5], [0.0, 1.0]
    """
    steps = round(1.0 / resolution)
    if n_components == 1:
        yield [1.0]
        return
    for combo in itertools.product(range(steps + 1), repeat=n_components - 1):
        used = sum(combo)
        if used > steps:
            continue
        last = steps - used
        yield [round(c * resolution, 6) for c in combo] + [round(last * resolution, 6)]


def generate_composition_space(
    a_site_ions: List[str],
    b_site_ions: List[str],
    x_site_ions: List[str],
    resolution: float = 0.1,
) -> List[PerovskiteComposition]:
    """Builds every PerovskiteComposition combining mixing ratios across
    the three sites at the given resolution. Zero-fraction ions are
    dropped from each composition's site dict rather than kept at 0.0.
    """
    candidates: List[PerovskiteComposition] = []
    for a_fracs in _mixing_ratios(len(a_site_ions), resolution):
        a_site = {ion: frac for ion, frac in zip(a_site_ions, a_fracs) if frac > 0}
        for b_fracs in _mixing_ratios(len(b_site_ions), resolution):
            b_site = {ion: frac for ion, frac in zip(b_site_ions, b_fracs) if frac > 0}
            for x_fracs in _mixing_ratios(len(x_site_ions), resolution):
                x_site = {ion: frac for ion, frac in zip(x_site_ions, x_fracs) if frac > 0}
                candidates.append(
                    PerovskiteComposition(a_site=a_site, b_site=b_site, x_site=x_site)
                )
    return candidates
