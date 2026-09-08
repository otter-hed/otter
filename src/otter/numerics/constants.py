"""
otter/numerics/constants.py

Purpose
-------
Define common physical constants and unit conversions (atomic units).

Methods
-------
- Use fixed scalars for common energy, length, and temperature conversions.

References
----------
- CODATA 2018 values :cite:p:`TiesingaEtAl2021` (rounded for convenience);
  these constants are supplied as fixed atomic-unit conversion factors by
  Otter.
"""

EV_TO_HA = 0.03674932217565499
HA_TO_EV = 1.0 / EV_TO_HA
BOHR_TO_CM = 5.29177210903e-9
CM_TO_BOHR = 1.0 / BOHR_TO_CM
BOHR_TO_ANGSTROM = BOHR_TO_CM * 1.0e8
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
KELVIN_TO_EV = 8.617333262145e-5
EV_TO_KELVIN = 1.0 / KELVIN_TO_EV
AVOGADRO_CONSTANT_MOL = 6.02214076e23
ATOMIC_MASS_UNIT_TO_G = 1.66053906660e-24
