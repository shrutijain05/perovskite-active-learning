"""
tests/test_descriptors.py
Validates physical descriptor calculations against known, real perovskite
compositions and their accepted stability ranges — not just "does it run."
"""

import numpy as np
import pytest

from perov_core.descriptors import PerovskiteComposition


def test_tolerance_factor_mapbi3():
    """MAPbI3 is a textbook stable 3D perovskite; its tolerance factor
    should land within the commonly cited stable range."""
    comp = PerovskiteComposition(a_site={"MA": 1.0}, b_site={"Pb": 1.0}, x_site={"I": 1.0})
    tolerance_factor = comp.compute_features()[0]
    assert 0.88 <= tolerance_factor <= 0.98


@pytest.mark.parametrize(
    "a_site,b_site,x_site",
    [
        ({"MA": 1.0}, {"Pb": 1.0}, {"I": 1.0}),   # MAPbI3
        ({"Cs": 1.0}, {"Pb": 1.0}, {"Br": 1.0}),  # CsPbBr3
        ({"FA": 1.0}, {"Sn": 1.0}, {"I": 1.0}),   # FASnI3
    ],
)
def test_known_materials_fall_in_stable_range(a_site, b_site, x_site):
    """Broader check: all three sit in the generally accepted
    perovskite-forming range (0.8 <= t <= 1.0, mu > 0.41)."""
    comp = PerovskiteComposition(a_site=a_site, b_site=b_site, x_site=x_site)
    tolerance_factor, octahedral_factor = comp.compute_features()[:2]
    assert 0.8 <= tolerance_factor <= 1.0
    assert octahedral_factor > 0.41


def test_invalid_fractions_raise_error():
    with pytest.raises(ValueError):
        PerovskiteComposition(a_site={"Cs": 0.5, "MA": 0.3}, b_site={"Pb": 1.0}, x_site={"I": 1.0})


def test_unknown_ion_raises_error():
    with pytest.raises(ValueError):
        PerovskiteComposition(a_site={"K": 1.0}, b_site={"Pb": 1.0}, x_site={"I": 1.0})


def test_mixed_composition_features_are_finite():
    """A genuinely mixed composition — the kind the search will actually
    propose — should still produce a clean, finite feature vector."""
    comp = PerovskiteComposition(
        a_site={"Cs": 0.2, "FA": 0.8},
        b_site={"Pb": 1.0},
        x_site={"Br": 0.3, "I": 0.7},
    )
    features = comp.compute_features()
    assert np.all(np.isfinite(features))
    assert features.shape == (14,)
