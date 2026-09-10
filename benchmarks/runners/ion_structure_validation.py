"""Shared numerical acceptance settings for the ion-structure library."""
from __future__ import annotations

import numpy as np

# This is the existing library archive-test limit, not a new solver tolerance.
MAX_RAW_SCREENING_CHARGE_ERROR = 7.0e-3


def validate_screening_charge(ion: dict) -> None:
    """Reject a failed native-grid sum rule before publishing a curve.

    ``zbar_screening_integral_raw`` is the native electronic radial integral,
    not ``charge_fix.q_scr_raw`` on the remapped QOZ/DST lattice.
    """
    raw = float(ion["zbar_screening_integral_raw"])
    target = float(ion["zbar_partition"])
    error = abs(raw-target)
    if not np.isfinite(raw) or not np.isfinite(target) or error > MAX_RAW_SCREENING_CHARGE_ERROR:
        raise RuntimeError(
            f"Native screening-charge error {error:.6e} e exceeds the library "
            f"limit {MAX_RAW_SCREENING_CHARGE_ERROR:.6e} e; refine and validate "
            "the electronic calculation before publishing this state."
        )
