"""LAMMPS-free physical checks for the cache-independent public CH2 driver."""
import importlib.util
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "tools/reproduce_ch2_hnc_md.py"
SPEC = importlib.util.spec_from_file_location("public_ch2_md_inputs", PATH)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


class Prepared:
    n_i = np.asarray([0.01, 0.02])
    zbar = np.asarray([4., 1.])
    r = np.linspace(.01, 20., 256)


def test_composition_clock_and_hot_step_limit():
    omega, timestep, damping = driver.plasma_clock(Prepared())
    assert omega > 0 and 0 < timestep < damping
    assert driver.CH2_FORMULA_UNITS*3 == 3072
    cold = driver.md_controls(Prepared(), 5.)
    hot = driver.md_controls(Prepared(), 100.)
    assert not cold["thermal_limited"] and hot["thermal_limited"]
    assert hot["thermal_step_bohr"] <= driver.MAX_THERMAL_STEP_BOHR
    assert hot["timestep_omega_p_inv"] < driver.TIMESTEP_OMEGA_P_INV
    assert hot["r_min_bohr"] == driver.HOT_MD_R_MIN_BOHR


def test_core_regularization_leaves_outer_potential_unchanged():
    r = np.linspace(.01, 2., 400)
    charges = np.asarray([4., 1.])
    coefficients = charges[:, None, None]*charges[None, :, None]
    potential = coefficients/r-.3+.4*np.sin(80*r)*np.exp(-8*r)
    regularized = driver.regularize_coulomb_core(r, potential, charges)
    inner = r < driver.COULOMB_CORE_BOHR
    outer = r >= driver.COULOMB_BLEND_END_BOHR
    remainder = regularized[..., inner]-coefficients/r[inner]
    np.testing.assert_allclose(remainder, np.broadcast_to(remainder[..., :1], remainder.shape), atol=1e-12)
    np.testing.assert_array_equal(regularized[..., outer], potential[..., outer])


def test_no_one_component_bridge_is_claimed_for_ch2():
    assert driver.BRIDGE_STATUS == "not_implemented_for_multicomponent_ch2"
    assert 'hnc_bridge_model="rosenfeld_ashcroft"' not in PATH.read_text()
