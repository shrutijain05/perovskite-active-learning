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

HONESTY NOTE / CHANGELOG — the live query path (_query_materials_project)
was originally written without access to a real Materials Project account
and shipped with a real bug: it passed `formula=formula` (a bare string)
to mpr.materials.summary.search(), but every documented example and
working sample passes `formula=[formula]` (a list) — the client validates
this and raises, which the broad except-all below then swallowed
silently, making a real bug indistinguishable from an expected fallback.
Fixed now. If it still doesn't work for you, pass debug=True (see below)
to see the actual exception instead of guessing.
"""

import os
from typing import Optional

from dft_runner.benchmark_engine import BenchmarkEvaluationBackend
from perov_core.descriptors import PerovskiteComposition

# Only these A-site ions map onto a simple chemical formula MP can be
# searched for directly.
_MP_SEARCHABLE_A_SITE = {"Cs"}

_dotenv_loaded = False


def _ensure_dotenv_loaded() -> None:
    """Loads variables from a .env file (if present) into os.environ, once
    per process. Without this, MP_API_KEY sitting in .env is invisible to
    os.environ.get() unless the shell exported it first — an easy trap
    that makes "I added the key" not actually take effect. Silently a
    no-op if python-dotenv isn't installed or no .env file is found, so
    this never breaks anything for someone not using the live backend."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    try:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:
        pass
    _dotenv_loaded = True


class MaterialsProjectEvaluationBackend:
    """Tries a live Materials Project lookup first; falls back to the
    offline BenchmarkEvaluationBackend on any failure — no key, no
    internet, no matching MP entry, a request error, anything. This is
    already "live-first, offline-fallback": evaluate() always attempts
    _try_live_lookup() before ever touching self.fallback. Implements the
    shared EvaluationBackend protocol (see base.py).

    Pass debug=True to print the actual exception (and whether a live
    call was even attempted) instead of silently falling back — without
    it, a real bug and an expected fallback look identical from the
    outside, which is exactly what made the original formula-list bug
    invisible.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        fallback: Optional[BenchmarkEvaluationBackend] = None,
        debug: bool = False,
    ):
        if api_key is None:
            _ensure_dotenv_loaded()
        self.api_key = api_key or os.environ.get("MP_API_KEY")
        self.fallback = fallback or BenchmarkEvaluationBackend()
        self.debug = debug
        self.last_source: Optional[str] = None  # "live" or "fallback", set after each evaluate()

    def evaluate(self, composition: PerovskiteComposition) -> float:
        live_value = self._try_live_lookup(composition)
        if live_value is not None:
            self.last_source = "live"
            return live_value
        self.last_source = "fallback"
        return self.fallback.evaluate(composition)

    def _try_live_lookup(self, composition: PerovskiteComposition) -> Optional[float]:
        if not self.api_key:
            if self.debug:
                print("[MP debug] no api_key set — skipping live lookup, going to fallback")
            return None
        if not self._is_mp_searchable(composition):
            if self.debug:
                print(
                    f"[MP debug] composition {composition} is outside live-searchable scope "
                    f"(needs pure Cs on A-site, single ion per site) — going straight to fallback"
                )
            return None
        try:
            return self._query_materials_project(composition)
        except Exception as e:
            # Network error, auth failure, no matching entry, an mp-api
            # interface change — any of it lands here. A search loop
            # needs a bandgap back, not a stack trace, so this stays a
            # deliberate broad catch — but with debug=True you see why.
            if self.debug:
                print(f"[MP debug] live lookup failed ({type(e).__name__}: {e}) — falling back")
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
        from mp_api.client import MPRester  # lazy import: mp-api is an
        # optional dependency (pip install -e ".[live]"), not required
        # just to import this module or use the offline path.

        a_ion = next(iter(composition.a_site))
        b_ion = next(iter(composition.b_site))
        x_ion = next(iter(composition.x_site))
        formula = f"{a_ion}{b_ion}{x_ion}3"

        with MPRester(api_key=self.api_key) as mpr:
            # formula must be a list — mp-api validates the search kwargs
            # and rejects a bare string; this was the original bug here.
            docs = mpr.materials.summary.search(formula=[formula], fields=["band_gap"])
        if not docs:
            if self.debug:
                print(f"[MP debug] query for formula={formula!r} returned no matching entries")
            return None
        band_gap = docs[0].band_gap
        if band_gap is None:
            if self.debug:
                print(f"[MP debug] found an entry for {formula!r} but band_gap field is null")
            return None
        return float(band_gap)
