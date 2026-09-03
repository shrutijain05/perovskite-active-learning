"""
dft_runner/materials_project_backend.py
The "live online search": tries a real Materials Project lookup over the
web, and falls back to the offline BenchmarkEvaluationBackend on ANY
failure — no API key, no internet, MP has no entry for this composition,
the request errors out, anything. The fallback exists so a search loop
never has to special-case "did the live call work this time" — it just
calls .evaluate() and always gets a number back.

Scope decision: live queries are only attempted for compositions MP can
be searched for unambiguously — pure (single ion per site) compositions
where the A-site cation is Cs. MA and FA are molecular cations
(methylammonium, formamidinium), not simple elements, and Materials
Project's core inorganic-materials index doesn't represent them the way
it represents elemental formulas — searching for them reliably by formula
isn't something this module attempts. Mixed compositions are skipped for
the same reason: MP indexes specific ordered structures, not arbitrary
fractional alloys. Everything outside that narrow case routes straight to
the offline backend without even trying a live call.

HONESTY NOTE — unlike every other module in this project, the *live*
query path (_query_materials_project) has not been run against a real
Materials Project account: building this required a working MP_API_KEY,
which wasn't available while writing it. What IS fully tested is the
fallback mechanism itself — every way this can fail gracefully. Treat
_query_materials_project as a best-effort starting point: confirm it
works once you have your own key set up (Phase 0 already asked you to
get one), and expect to adjust field/method names if the mp-api client's
interface has moved since this was written.
"""

import os
from typing import Optional

from dft_runner.benchmark_engine import BenchmarkEvaluationBackend
from perov_core.descriptors import PerovskiteComposition

# Only these A-site ions map onto a simple chemical formula MP can be
# searched for directly.
_MP_SEARCHABLE_A_SITE = {"Cs"}


class MaterialsProjectEvaluationBackend:
    """Tries a live Materials Project lookup; falls back to the offline
    BenchmarkEvaluationBackend on any failure. Implements the shared
    EvaluationBackend protocol (see base.py).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        fallback: Optional[BenchmarkEvaluationBackend] = None,
    ):
        self.api_key = api_key or os.environ.get("MP_API_KEY")
        self.fallback = fallback or BenchmarkEvaluationBackend()

    def evaluate(self, composition: PerovskiteComposition) -> float:
        live_value = self._try_live_lookup(composition)
        if live_value is not None:
            return live_value
        return self.fallback.evaluate(composition)

    def _try_live_lookup(self, composition: PerovskiteComposition) -> Optional[float]:
        if not self.api_key or not self._is_mp_searchable(composition):
            return None
        try:
            return self._query_materials_project(composition)
        except Exception:
            # Network error, auth failure, no matching entry, an mp-api
            # interface change — any of it lands here. A search loop
            # needs a bandgap back, not a stack trace, so this is a
            # deliberate broad catch, not a lazy one.
            return None

    @staticmethod
    def _is_mp_searchable(composition: PerovskiteComposition) -> bool:
        if (
            len(composition.a_site) != 1
            or len(composition.b_site) != 1
            or len(composition.x_site) != 1
        ):
            return False  # mixed compositions: skip straight to offline
        a_ion = next(iter(composition.a_site))
        return a_ion in _MP_SEARCHABLE_A_SITE

    def _query_materials_project(self, composition: PerovskiteComposition) -> Optional[float]:
        """UNVERIFIED against a real account — see module docstring."""
        from mp_api.client import MPRester  # lazy import: mp-api is an
        # optional dependency (pip install -e ".[live]"), not required
        # just to import this module or use the offline path.

        a_ion = next(iter(composition.a_site))
        b_ion = next(iter(composition.b_site))
        x_ion = next(iter(composition.x_site))
        formula = f"{a_ion}{b_ion}{x_ion}3"

        with MPRester(api_key=self.api_key) as mpr:
            docs = mpr.materials.summary.search(formula=formula, fields=["band_gap"])
        if not docs:
            return None
        return float(docs[0].band_gap)
