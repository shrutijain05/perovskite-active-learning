"""
tests/test_dft_parser.py
Validates QE log parsing against a realistic fixture SCF output block —
the way pw.x actually writes it, not isolated string patterns.
"""

import pytest

from dft_runner.log_parser import QELogParser

CONVERGED_LOG = """
     Program PWSCF v.7.2 starts on  1Sep2026 at  9:15: 2

     This program is part of the open-source Quantum ESPRESSO suite

     bravais-lattice index     =            0

     iteration #  1     ecut=    60.00 Ry     beta! = 0.70
     iteration # 12     ecut=    60.00 Ry     beta! = 0.70
     convergence has been achieved in  12 iterations

!    total energy              =    -456.32918273 Ry

     estimated scf accuracy    <       0.00000012 Ry

     the Fermi energy is     3.8172 ev

     highest occupied, lowest unoccupied level (ev):     3.4521    5.0123

     highest occupied level (ev):     3.4521
     lowest unoccupied level (ev):     5.0123

     PWSCF        :   4m12.33s CPU   4m20.01s WALL
"""

UNCONVERGED_LOG = """
     Program PWSCF v.7.2 starts on  1Sep2026 at  9:15: 2

     iteration # 100     ecut=    60.00 Ry
     convergence NOT achieved after 100 iterations: stopping
"""


def test_parses_converged_scf_run():
    result = QELogParser.parse_scf_output(CONVERGED_LOG)
    assert result["converged"] is True
    assert result["total_energy_Ry"] == pytest.approx(-456.32918273)
    assert result["fermi_energy_eV"] == pytest.approx(3.8172)
    assert result["bandgap_eV"] == pytest.approx(5.0123 - 3.4521)


def test_flags_unconverged_run():
    result = QELogParser.parse_scf_output(UNCONVERGED_LOG)
    assert result["converged"] is False
    assert result["total_energy_Ry"] is None
    assert result["bandgap_eV"] is None


def test_bandgap_never_negative():
    """If HOMO/LUMO were ever mis-ordered in a log, the parser should
    floor bandgap at 0 rather than report a nonsensical negative gap."""
    weird_log = """
!    total energy              =    -100.0 Ry
     highest occupied level (ev):     5.0
     lowest unoccupied level (ev):     3.0
"""
    result = QELogParser.parse_scf_output(weird_log)
    assert result["bandgap_eV"] == 0.0
