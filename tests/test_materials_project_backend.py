"""
tests/test_materials_project_backend.py
Tests the graceful-fallback mechanism — the part of this module that's
actually verifiable without a real Materials Project API key. The live
query path itself is exercised only via a mocked stand-in; see the module
docstring in materials_project_backend.py for what "verified" means here.
"""

import pytest

from dft_runner.benchmark_engine import REFERENCE_BANDGAPS, BenchmarkEvaluationBackend
from dft_runner.materials_project_backend import MaterialsProjectEvaluationBackend
from perov_core.descriptors import PerovskiteComposition


@pytest.fixture
def offline_backend(tmp_path):
    return BenchmarkEvaluationBackend(cache_path=tmp_path / "cache.json")


def test_falls_back_when_no_api_key(offline_backend):
    backend = MaterialsProjectEvaluationBackend(api_key=None, fallback=offline_backend)
    comp = PerovskiteComposition(a_site={"Cs": 1.0}, b_site={"Pb": 1.0}, x_site={"I": 1.0})
    assert backend.evaluate(comp) == REFERENCE_BANDGAPS[("Cs", "Pb", "I")]


def test_falls_back_for_mixed_compositions_even_with_a_key(offline_backend):
    """Even with a key present, mixed compositions skip the live lookup
    entirely — MP indexes specific ordered structures, not arbitrary
    fractional alloys."""
    backend = MaterialsProjectEvaluationBackend(
        api_key="fake-key-for-testing", fallback=offline_backend
    )
    comp = PerovskiteComposition(
        a_site={"Cs": 1.0}, b_site={"Pb": 1.0}, x_site={"Br": 0.5, "I": 0.5}
    )
    assert backend.evaluate(comp) == offline_backend.evaluate(comp)


def test_falls_back_for_organic_a_site_even_with_a_key(offline_backend):
    """MA/FA aren't simple elements MP can be searched for by formula —
    these route straight to the offline backend regardless of API key."""
    backend = MaterialsProjectEvaluationBackend(
        api_key="fake-key-for-testing", fallback=offline_backend
    )
    comp = PerovskiteComposition(a_site={"MA": 1.0}, b_site={"Pb": 1.0}, x_site={"I": 1.0})
    assert backend.evaluate(comp) == offline_backend.evaluate(comp)


def test_falls_back_when_live_query_raises(offline_backend, monkeypatch):
    """Simulates a live query failure (bad network, expired key, an
    mp-api interface change) — evaluate() should still return a usable
    number, not propagate the exception."""
    backend = MaterialsProjectEvaluationBackend(
        api_key="fake-key-for-testing", fallback=offline_backend
    )

    def _boom(self, composition):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(MaterialsProjectEvaluationBackend, "_query_materials_project", _boom)

    comp = PerovskiteComposition(a_site={"Cs": 1.0}, b_site={"Pb": 1.0}, x_site={"I": 1.0})
    assert backend.evaluate(comp) == REFERENCE_BANDGAPS[("Cs", "Pb", "I")]


def test_uses_live_value_when_query_succeeds(offline_backend, monkeypatch):
    """When the live path does return something, it should be preferred
    over the offline fallback."""
    backend = MaterialsProjectEvaluationBackend(
        api_key="fake-key-for-testing", fallback=offline_backend
    )

    def _fake_live_value(self, composition):
        return 1.234  # deliberately different from the offline reference value

    monkeypatch.setattr(
        MaterialsProjectEvaluationBackend, "_query_materials_project", _fake_live_value
    )

    comp = PerovskiteComposition(a_site={"Cs": 1.0}, b_site={"Pb": 1.0}, x_site={"I": 1.0})
    assert backend.evaluate(comp) == 1.234
