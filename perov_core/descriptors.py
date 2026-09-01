"""
perov_core/descriptors.py
Computes physical, crystallographic, and chemical descriptors for perovskite
alloys, and turns a composition into the 14-dimensional feature vector the
Phase 2 Gaussian Process surrogate expects.

Ionic radii source: R.D. Shannon, "Revised effective ionic radii and
systematic studies of interatomic distances in halides and chalcogenides,"
Acta Cryst. A32, 751-767 (1976). Values for the organic A-site cations
(MA, FA) are commonly-used effective radii from later perovskite-specific
literature, not true Shannon radii — MA/FA aren't simple ions, so treating
them as if they have one fixed "size" is an approximation worth flagging
rather than treating as exact.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np

IONIC_RADII: Dict[str, float] = {
    "Cs": 1.88, "MA": 2.17, "FA": 2.53,   # A-site cations (Angstroms)
    "Pb": 1.19, "Sn": 1.15,               # B-site metals
    "Cl": 1.81, "Br": 1.96, "I": 2.20,    # X-site halides
}

ELECTRONEGATIVITY: Dict[str, float] = {
    "Cs": 0.79, "MA": 2.50, "FA": 2.55,
    "Pb": 2.33, "Sn": 1.96,
    "Cl": 3.16, "Br": 2.96, "I": 2.66,
}

_KNOWN_A_SITE = {"Cs", "MA", "FA"}
_KNOWN_B_SITE = {"Pb", "Sn"}
_KNOWN_X_SITE = {"Cl", "Br", "I"}

N_FEATURES = 14  # length of the vector returned by compute_features() below


@dataclass
class PerovskiteComposition:
    """A single ABX3 composition, specified as mixing fractions per site.

    Example: {"Cs": 0.2, "FA": 0.8} means 20% Cs / 80% FA on the A-site.
    Fractions within each site must sum to 1.0.
    """

    a_site: Dict[str, float]
    b_site: Dict[str, float]
    x_site: Dict[str, float]

    def __post_init__(self) -> None:
        self._validate_site("a_site", self.a_site, _KNOWN_A_SITE)
        self._validate_site("b_site", self.b_site, _KNOWN_B_SITE)
        self._validate_site("x_site", self.x_site, _KNOWN_X_SITE)

    @staticmethod
    def _validate_site(name: str, site: Dict[str, float], known_ions: set) -> None:
        if not site:
            raise ValueError(f"{name} cannot be empty")
        for ion in site:
            if ion not in known_ions:
                raise ValueError(
                    f"Unknown ion '{ion}' in {name} — expected one of {sorted(known_ions)}"
                )
        total = sum(site.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"{name} fractions must sum to 1.0, got {total:.4f}: {site}")

    def compute_features(self) -> np.ndarray:
        """Extracts the 14-dim physical descriptor vector used downstream
        by the surrogate model:
        [tolerance_factor, octahedral_factor, r_a, r_b, r_x, delta_chi_bx,
         x_Cs, x_MA, x_FA, x_Pb, x_Sn, x_Cl, x_Br, x_I]
        """
        r_a = sum(IONIC_RADII[ion] * frac for ion, frac in self.a_site.items())
        r_b = sum(IONIC_RADII[ion] * frac for ion, frac in self.b_site.items())
        r_x = sum(IONIC_RADII[ion] * frac for ion, frac in self.x_site.items())

        tolerance_factor = (r_a + r_x) / (np.sqrt(2) * (r_b + r_x))
        octahedral_factor = r_b / r_x

        chi_b = sum(ELECTRONEGATIVITY[ion] * frac for ion, frac in self.b_site.items())
        chi_x = sum(ELECTRONEGATIVITY[ion] * frac for ion, frac in self.x_site.items())
        delta_chi_bx = abs(chi_b - chi_x)

        fractions = [
            self.a_site.get("Cs", 0.0), self.a_site.get("MA", 0.0), self.a_site.get("FA", 0.0),
            self.b_site.get("Pb", 0.0), self.b_site.get("Sn", 0.0),
            self.x_site.get("Cl", 0.0), self.x_site.get("Br", 0.0), self.x_site.get("I", 0.0),
        ]

        return np.array(
            [tolerance_factor, octahedral_factor, r_a, r_b, r_x, delta_chi_bx, *fractions],
            dtype=np.float64,
        )
