"""
dft_runner/log_parser.py
Parses Quantum ESPRESSO SCF output logs to extract electronic structure
properties. This belongs to the real-DFT evaluation path, which is
deferred (see qe_generator.py note in ROADMAP.md) — but the parser is
built and tested now, so that path has a working piece ready whenever a
real pw.x run needs reading.
"""

import re
from typing import Dict, Optional


class QELogParser:
    @staticmethod
    def parse_scf_output(log_text: str) -> Dict[str, Optional[float]]:
        """Parses Fermi energy, total energy, HOMO/LUMO levels (used to
        derive bandgap), and convergence status from QE SCF log text."""
        results: Dict[str, Optional[float]] = {
            "converged": False,
            "total_energy_Ry": None,
            "fermi_energy_eV": None,
            "bandgap_eV": None,
        }

        if "convergence has been achieved" in log_text:
            results["converged"] = True

        total_energy_match = re.search(r"!\s+total energy\s+=\s+([-+]?\d+\.\d+)\s+Ry", log_text)
        if total_energy_match:
            results["total_energy_Ry"] = float(total_energy_match.group(1))

        fermi_match = re.search(
            r"the Fermi energy is\s+([-+]?\d+\.\d+)\s+ev", log_text, re.IGNORECASE
        )
        if fermi_match:
            results["fermi_energy_eV"] = float(fermi_match.group(1))

        homo_match = re.search(
            r"^\s*highest occupied level\s+\(ev\):\s+([-+]?\d+\.\d+)", log_text, re.MULTILINE
        )
        lumo_match = re.search(
            r"^\s*lowest unoccupied level\s+\(ev\):\s+([-+]?\d+\.\d+)", log_text, re.MULTILINE
        )
        if homo_match and lumo_match:
            homo = float(homo_match.group(1))
            lumo = float(lumo_match.group(1))
            results["bandgap_eV"] = max(0.0, lumo - homo)

        return results
