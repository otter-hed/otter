"""Fast checks for the cache-only CH2 HNC/MD comparison launcher."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "applications/ch2_xrts_dataset/compare_hnc_md.py"
SPEC = importlib.util.spec_from_file_location("compare_hnc_md", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_temperature_selection_uses_nearest_distinct_saved_states() -> None:
    available = [float(value) for value in range(5, 121, 2)]
    assert module.select_temperatures((10.0, 30.0, 100.0), available) == [
        9.0,
        29.0,
        99.0,
    ]


def test_md_composition_and_clock_are_physical() -> None:
    class Prepared:
        n_i = np.asarray([0.01, 0.02])
        zbar = np.asarray([4.0, 1.0])
        r = np.linspace(0.01, 20.0, 256)

    omega, timestep_ps, damping_ps = module.plasma_clock(Prepared())
    assert omega > 0.0
    assert timestep_ps > 0.0
    assert damping_ps > timestep_ps
    assert module.CH2_FORMULA_UNITS * 3 == 3072


def test_high_temperature_md_guard_reduces_step_and_extends_table() -> None:
    class Prepared:
        n_i = np.asarray([0.01, 0.02])
        zbar = np.asarray([4.0, 1.0])
        r = np.linspace(0.01, 20.0, 256)

    cold = module.md_controls(Prepared(), 5.0)
    intermediate = module.md_controls(Prepared(), 29.0)
    hot = module.md_controls(Prepared(), 100.0)
    assert not cold["thermal_limited"]
    assert not intermediate["thermal_limited"]
    assert cold["r_min_bohr"] == module.DEFAULT_MD_R_MIN_BOHR
    assert hot["thermal_limited"]
    assert hot["thermal_step_bohr"] <= module.MAX_THERMAL_STEP_BOHR
    assert hot["timestep_omega_p_inv"] < module.TIMESTEP_OMEGA_P_INV
    assert hot["r_min_bohr"] == module.HOT_MD_R_MIN_BOHR


def test_coulomb_core_repair_removes_ringing_and_preserves_outer_potential() -> None:
    r = np.linspace(0.01, 2.0, 400)
    zbar = np.asarray([4.0, 1.0])
    matrix = np.empty((2, 2, r.size))
    for left in range(2):
        for right in range(2):
            coefficient = zbar[left] * zbar[right]
            ringing = 0.4 * np.sin(80.0 * r) * np.exp(-8.0 * r)
            matrix[left, right] = coefficient / r - 0.3 + ringing

    repaired = module.regularize_coulomb_core(r, matrix, zbar)
    inner = r < module.COULOMB_CORE_BOHR
    outer = r >= module.COULOMB_BLEND_END_BOHR
    for left in range(2):
        for right in range(2):
            coefficient = zbar[left] * zbar[right]
            remainder = repaired[left, right, inner] - coefficient / r[inner]
            np.testing.assert_allclose(remainder, remainder[0], atol=1.0e-12)
            np.testing.assert_allclose(
                repaired[left, right, outer], matrix[left, right, outer]
            )


def test_script_does_not_mislabel_one_component_vmhnc_as_a_ch2_bridge() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert module.BRIDGE_STATUS == "not_implemented_for_multicomponent_ch2"
    assert "scalar hard-sphere bridge" in source
    assert 'hnc_bridge_model="rosenfeld_ashcroft"' not in source
    assert "run_otter_lammps_md" in source
