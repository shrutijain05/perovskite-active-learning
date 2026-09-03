"""
tests/test_benchmark_engine.py
Validates the offline mock evaluation backend: all 18 end-member lookups,
bowing-corrected interpolation (checked against a hand calculation, not
just "it ran"), caching, and the documented failure mode.
"""

import json

import pytest

from dft_runner.benchmark_engine import (
    BOWING_PARAMETERS,
    REFERENCE_BANDGAPS,
    BenchmarkEvaluationBackend,
)
from perov_core.descriptors import PerovskiteComposition


@pytest.fixture
def backend(tmp_path):
    """Points at a throwaway cache file so tests never touch the real
    dft_runner/reference_data/cache.json."""
    return BenchmarkEvaluationBackend(cache_path=tmp_path / "test_cache.json")


def test_all_18_end_members_are_present():
    a_sites, b_sites, x_sites = {"Cs", "MA", "FA"}, {"Pb", "Sn"}, {"Cl", "Br", "I"}
    expected_keys = {(a, b, x) for a in a_sites for b in b_sites for x in x_sites}
    assert set(REFERENCE_BANDGAPS.keys()) == expected_keys
    assert len(REFERENCE_BANDGAPS) == 18


def test_exact_end_member_returns_reference_value(backend):
    comp = PerovskiteComposition(a_site={"MA": 1.0}, b_site={"Pb": 1.0}, x_site={"I": 1.0})
    bandgap = backend.evaluate(comp)
    assert bandgap == REFERENCE_BANDGAPS[("MA", "Pb", "I")]


def test_bowing_correction_matches_hand_calculation(backend):
    """The exact formula from the module docstring, computed independently
    here rather than trusting the implementation to grade itself."""
    comp = PerovskiteComposition(
        a_site={"MA": 1.0}, b_site={"Pb": 1.0}, x_site={"Br": 0.5, "I": 0.5}
    )
    bandgap = backend.evaluate(comp)
    linear_average = (
        0.5 * REFERENCE_BANDGAPS[("MA", "Pb", "I")] + 0.5 * REFERENCE_BANDGAPS[("MA", "Pb", "Br")]
    )
    expected = linear_average - BOWING_PARAMETERS[("Br", "I")] * 0.5 * 0.5
    assert bandgap == pytest.approx(expected)


def test_bowing_pulls_result_below_plain_linear_average(backend):
    """Downward bowing is the well-documented direction for halide mixing:
    the modeled value should sit below what plain interpolation predicts,
    not on top of it."""
    comp = PerovskiteComposition(
        a_site={"MA": 1.0}, b_site={"Pb": 1.0}, x_site={"Br": 0.3, "I": 0.7}
    )
    bandgap = backend.evaluate(comp)
    linear_average = (
        0.7 * REFERENCE_BANDGAPS[("MA", "Pb", "I")] + 0.3 * REFERENCE_BANDGAPS[("MA", "Pb", "Br")]
    )
    assert bandgap < linear_average


def test_pure_composition_has_zero_bowing_correction():
    """A pure end-member never reaches the bowing formula via evaluate()
    (it short-circuits to an exact lookup) — but the correction function
    itself should still be a no-op on it, since there are no ion pairs at
    any site to correct for."""
    comp = PerovskiteComposition(a_site={"Cs": 1.0}, b_site={"Pb": 1.0}, x_site={"Br": 1.0})
    assert BenchmarkEvaluationBackend._bowing_correction(comp) == 0.0


def test_result_is_cached_and_reused_across_instances(backend):
    comp = PerovskiteComposition(a_site={"Cs": 1.0}, b_site={"Pb": 1.0}, x_site={"Br": 1.0})
    first = backend.evaluate(comp)

    reloaded = BenchmarkEvaluationBackend(cache_path=backend.cache_path)
    second = reloaded.evaluate(comp)

    assert first == second
    assert backend._cache_key(comp) in reloaded._cache


def test_unknown_ion_rejected_before_it_ever_reaches_the_backend():
    """Every pure end-member now has a value, so the old 'no reference
    data at all' case can't happen for a valid composition anymore.
    Confirm the failure now happens earlier and more usefully — at
    PerovskiteComposition construction, not silently inside evaluate()."""
    with pytest.raises(ValueError):
        PerovskiteComposition(a_site={"K": 1.0}, b_site={"Pb": 1.0}, x_site={"I": 1.0})


def test_cache_file_is_valid_json_on_disk(backend):
    comp = PerovskiteComposition(a_site={"FA": 1.0}, b_site={"Pb": 1.0}, x_site={"I": 1.0})
    backend.evaluate(comp)
    with open(backend.cache_path) as f:
        data = json.load(f)  # raises if not valid JSON
    assert len(data) == 1
