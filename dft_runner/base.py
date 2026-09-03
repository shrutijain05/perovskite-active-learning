"""
dft_runner/base.py
Shared interface both evaluation backends implement, so the rest of the
pipeline (Phase 4 onward) can be pointed at either one interchangeably —
two evaluators, one contract.
"""

from typing import Protocol

from perov_core.descriptors import PerovskiteComposition


class EvaluationBackend(Protocol):
    def evaluate(self, composition: PerovskiteComposition) -> float:
        """Returns a bandgap (eV) for the given composition."""
        ...
