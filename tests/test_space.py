"""
tests/test_space.py
Validates the composition-space generator: correct grid size, valid
fractions, and clean feature extraction across an entire generated grid.
"""

import numpy as np

from perov_core.space import generate_composition_space


def test_grid_size_matches_expected_combinatorics():
    # 2 A-site ions, 1 B-site ion, 2 X-site ions, resolution=0.5 (50% steps)
    # -> 3 A-site splits x 1 B-site split x 3 X-site splits = 9 compositions
    space = generate_composition_space(
        a_site_ions=["Cs", "MA"],
        b_site_ions=["Pb"],
        x_site_ions=["Br", "I"],
        resolution=0.5,
    )
    assert len(space) == 9


def test_all_fractions_sum_to_one():
    space = generate_composition_space(
        a_site_ions=["Cs", "FA"],
        b_site_ions=["Pb", "Sn"],
        x_site_ions=["Br", "I"],
        resolution=0.5,
    )
    for comp in space:
        for site in (comp.a_site, comp.b_site, comp.x_site):
            assert np.isclose(sum(site.values()), 1.0)


def test_full_grid_features_have_no_nans():
    space = generate_composition_space(
        a_site_ions=["Cs", "MA", "FA"],
        b_site_ions=["Pb", "Sn"],
        x_site_ions=["Cl", "Br", "I"],
        resolution=0.25,
    )
    features = np.array([comp.compute_features() for comp in space])
    assert not np.isnan(features).any()
    assert features.shape[1] == 14  # matches the surrogate's expected input dimension
