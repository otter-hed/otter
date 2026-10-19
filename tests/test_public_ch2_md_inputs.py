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


def test_composition_clock_and_fixed_physical_step():
    omega, timestep, damping = driver.plasma_clock(Prepared())
    assert omega > 0 and 0 < timestep < damping
    assert driver.CH2_FORMULA_UNITS*3 == 3072
    cold = driver.md_controls(Prepared(), 5.)
    hot = driver.md_controls(Prepared(), 100.)
    assert cold["timestep_ps"] == hot["timestep_ps"] == 0.01e-3
    assert not cold["resolve_coulomb_core"] and hot["resolve_coulomb_core"]
    assert hot["r_min_bohr"] == driver.HOT_MD_R_MIN_BOHR
    assert hot["thermal_step_bohr"] > cold["thermal_step_bohr"]
    duration = driver.PRODUCTION_OMEGA_P_INV / omega
    atomic_time_ps = driver.physical_constants["atomic unit of time"][0] * 1e12
    steps = round(driver.PRODUCTION_OMEGA_P_INV / hot["timestep_omega_p_inv"])
    np.testing.assert_allclose(steps * timestep, duration * atomic_time_ps,
                               atol=timestep/2, rtol=0)


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


def test_timestep_change_invalidates_completed_case_signature(monkeypatch):
    controls = driver.md_controls(Prepared(), 99.)
    before = driver._case_signature(99., 99., controls)
    monkeypatch.setattr(driver, "MD_TIMESTEP_FS", .025)
    after = driver._case_signature(99., 99., driver.md_controls(Prepared(), 99.))
    assert len(before) == 64 and before != after


def test_no_one_component_bridge_is_claimed_for_ch2():
    assert driver.BRIDGE_STATUS == "not_implemented_for_multicomponent_ch2"
    assert 'hnc_bridge_model="rosenfeld_ashcroft"' not in PATH.read_text()
