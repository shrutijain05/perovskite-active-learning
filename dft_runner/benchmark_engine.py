"""
dft_runner/benchmark_engine.py
The "offline reference book": a completely offline evaluator covering all
18 pure perovskites in this project's A x B x X space (3 A-site cations x
2 B-site metals x 3 X-site halides), plus multilinear interpolation with
an optical bowing correction for any mixed composition. No network, no
API key, sub-millisecond, never fails for a composition this dataset can
reach — that's the safety net the live evaluator (materials_project_backend.py)
falls back to.

Bandgap physics used here:
  Eg(x) = (1-x)*E1 + x*E2 - b*x*(1-x)
This is the standard "Vegard's law plus bowing" equation used throughout
the mixed-halide perovskite literature (e.g. Noh et al. 2013, and the
bowing parameters below). The last term captures "bowing": mixed
compositions typically sit BELOW a straight line between the two pure
end-members, not on it. This module generalizes it to all three sites at
once by applying the same pairwise correction independently per site and
summing — exact for single-site mixing, a reasonable but experimentally
unvalidated extension when two sites are mixed simultaneously.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from perov_core.descriptors import PerovskiteComposition

CACHE_PATH = Path(__file__).parent / "reference_data" / "cache.json"

# --------------------------------------------------------------------------
# The 18 pure perovskites: every combination of {Cs, MA, FA} x {Pb, Sn} x
# {Cl, Br, I}. Confidence varies by entry — grouped and flagged below
# rather than presented as uniformly precise.
#
# Well-corroborated by multiple independent sources (searched, not just
# recalled): MA/FA/Cs-Pb-I and -Br (literature-fit endpoints used to
# derive the bowing parameters below, so kept internally consistent with
# them); Cs-Sn-Br (1.75 eV, several independent papers agree); Cs-Sn-I;
# FA-Sn-I (1.41 eV, several sources agree); Cs-Sn-Cl (~2.8 eV, one
# detailed experimental source).
#
# Estimated by analogy/trend, NOT independently confirmed: MA-Sn-Cl,
# FA-Pb-Cl, FA-Sn-Cl, FA-Sn-Br. These fill out the full 18-entry grid so
# every combination in this project's search space has *something* to
# interpolate from, but treat them as placeholders to replace with real
# values (computed or measured) before trusting them for anything beyond
# keeping the pipeline runnable.
# --------------------------------------------------------------------------
REFERENCE_BANDGAPS: Dict[Tuple[str, str, str], float] = {
    ("Cs", "Pb", "Cl"): 3.00,
    ("Cs", "Pb", "Br"): 2.30,
    ("Cs", "Pb", "I"): 1.73,
    ("Cs", "Sn", "Cl"): 2.80,
    ("Cs", "Sn", "Br"): 1.75,
    ("Cs", "Sn", "I"): 1.30,
    ("MA", "Pb", "Cl"): 3.00,
    ("MA", "Pb", "Br"): 2.28,
    ("MA", "Pb", "I"): 1.58,
    ("MA", "Sn", "Cl"): 2.90,  # estimated by analogy — not independently confirmed
    ("MA", "Sn", "Br"): 1.80,
    ("MA", "Sn", "I"): 1.20,
    ("FA", "Pb", "Cl"): 2.90,  # estimated by analogy — not independently confirmed
    ("FA", "Pb", "Br"): 2.23,
    ("FA", "Pb", "I"): 1.48,
    ("FA", "Sn", "Cl"): 2.90,  # estimated by analogy — not independently confirmed
    ("FA", "Sn", "Br"): 1.85,  # estimated by analogy — not independently confirmed
    ("FA", "Sn", "I"): 1.41,
}

# Optical bowing parameters (eV), keyed by sorted ion pair. X-site (halide)
# values are literature-cited for MAPb-based systems specifically (mixed-
# halide phase segregation literature): I/Br = 0.33, Cl/Br = 0.09. Cl/I
# has no direct source found here — 0.45 is an extrapolation (Cl/I are
# the most dissimilar halide pair, so bowing should exceed either
# adjacent pair) and is the least certain of the three. A/B-site bowing
# is far less consistently characterized in the literature than halide
# bowing; Pb/Sn alloys in particular are known to show unusually large,
# sometimes non-monotonic bowing near certain compositions that a single
# scalar term does not fully capture — treat 0.30 as a rough placeholder,
# not a citation.
BOWING_PARAMETERS: Dict[Tuple[str, str], float] = {
    ("Br", "I"): 0.33,
    ("Br", "Cl"): 0.09,
    ("Cl", "I"): 0.45,  # extrapolated, not directly sourced
    ("Cs", "MA"): 0.05,  # placeholder
    ("Cs", "FA"): 0.05,  # placeholder
    ("FA", "MA"): 0.02,  # placeholder
    ("Pb", "Sn"): 0.30,  # placeholder — real Pb/Sn bowing is more complex than this
}


class BenchmarkEvaluationBackend:
    """Looks up (or interpolates, with bowing) a bandgap for a composition
    without running DFT. Caches every result to disk so a whole search
    loop can run offline after the first pass over any given composition.
    Implements the shared EvaluationBackend protocol (see base.py).
    """

    def __init__(self, cache_path: Path = CACHE_PATH):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, float] = self._load_cache()

    def _load_cache(self) -> Dict[str, float]:
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                return json.load(f)
        return {}

    def _save_cache(self) -> None:
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f, indent=2, sort_keys=True)

    @staticmethod
    def _cache_key(composition: PerovskiteComposition) -> str:
        def _fmt(site: Dict[str, float]) -> str:
            return ",".join(f"{ion}:{frac:.6f}" for ion, frac in sorted(site.items()))

        return (
            f"A[{_fmt(composition.a_site)}]"
            f"B[{_fmt(composition.b_site)}]"
            f"X[{_fmt(composition.x_site)}]"
        )

    def evaluate(self, composition: PerovskiteComposition) -> float:
        """Returns a bandgap (eV): from cache, from an exact end-member
        match, or from bowing-corrected interpolation. Always returns a
        value or raises clearly — a search loop can't gracefully handle a
        silently missing result."""
        key = self._cache_key(composition)
        if key in self._cache:
            return self._cache[key]

        bandgap = self._exact_end_member_match(composition)
        if bandgap is None:
            bandgap = self._interpolate_with_bowing(composition)

        self._cache[key] = bandgap
        self._save_cache()
        return bandgap

    @staticmethod
    def _exact_end_member_match(composition: PerovskiteComposition) -> Optional[float]:
        """Only matches pure (single ion per site) compositions present
        directly in REFERENCE_BANDGAPS."""
        if (
            len(composition.a_site) != 1
            or len(composition.b_site) != 1
            or len(composition.x_site) != 1
        ):
            return None
        a_ion = next(iter(composition.a_site))
        b_ion = next(iter(composition.b_site))
        x_ion = next(iter(composition.x_site))
        return REFERENCE_BANDGAPS.get((a_ion, b_ion, x_ion))

    @staticmethod
    def _linear_interpolation(composition: PerovskiteComposition) -> Tuple[float, float]:
        """Multilinear (Vegard's-law) weighted average across whichever
        end-members have reference data. Returns (weighted_sum, weight_total)
        rather than the ratio directly, so the caller can detect missing
        coverage (weight_total == 0)."""
        weight_total = 0.0
        weighted_sum = 0.0
        for a_ion, a_frac in composition.a_site.items():
            for b_ion, b_frac in composition.b_site.items():
                for x_ion, x_frac in composition.x_site.items():
                    end_member = REFERENCE_BANDGAPS.get((a_ion, b_ion, x_ion))
                    if end_member is None:
                        continue
                    w = a_frac * b_frac * x_frac
                    weighted_sum += w * end_member
                    weight_total += w
        return weighted_sum, weight_total

    @staticmethod
    def _bowing_correction(composition: PerovskiteComposition) -> float:
        """Sum of pairwise bowing corrections (b * f_i * f_j) across all
        three sites. Reduces exactly to the standard binary bowing
        equation when only one site is mixed."""
        total = 0.0
        for site in (composition.a_site, composition.b_site, composition.x_site):
            ions = list(site.items())
            for i in range(len(ions)):
                for j in range(i + 1, len(ions)):
                    ion_i, frac_i = ions[i]
                    ion_j, frac_j = ions[j]
                    b = BOWING_PARAMETERS.get(tuple(sorted((ion_i, ion_j))))
                    if b is not None:
                        total += b * frac_i * frac_j
        return total

    @classmethod
    def _interpolate_with_bowing(cls, composition: PerovskiteComposition) -> float:
        weighted_sum, weight_total = cls._linear_interpolation(composition)
        if weight_total == 0.0:
            raise ValueError(
                "No reference end-members found to interpolate this composition "
                "from — every A/B/X ion combination present is missing from "
                "REFERENCE_BANDGAPS. Add a value for at least one relevant "
                "end-member, or expand the reference set."
            )
        linear_estimate = weighted_sum / weight_total
        return linear_estimate - cls._bowing_correction(composition)
