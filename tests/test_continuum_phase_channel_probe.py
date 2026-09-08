"""A single-channel probe must reproduce the original selected phase exactly."""
import numpy as np
import pytest

from otter.electronic.continuum import scattering as sc
from otter.numerics.grids import create_sqrt_grid


@pytest.mark.parametrize("mode", ["unreduced", "accepted_trial", "recovery", "soft", "fallback", "coulomb"])
def test_phase_probe_retains_matching_plan_and_angular_decisions(monkeypatch, mode):
    grid = create_sqrt_grid(27.5, 1000, rmin=1e-5)
    r = grid.r
    potential = 0.3*np.exp(-r/8.)
    options = dict(density_rmax=9.2)
    asymptotic, v_tol = "auto", .1
    if mode == "unreduced":
        options = dict(l_cap_strategy="none")
    elif mode == "accepted_trial":
        potential = np.zeros_like(r)
    elif mode == "recovery":
        # Force a too-small density trial, not a changed acceptance threshold.
        monkeypatch.setattr(sc, "_density_domain_l_cap", lambda *a: 8)
    elif mode == "soft":
        options["l_max_soft"] = 8
    elif mode == "fallback":
        potential = -2./r
        v_tol = 0.
    elif mode == "coulomb":
        potential = -2./r
        asymptotic, v_tol = "coulomb", None
    args = (potential, r, 0., 1., 2., 100, "sqrt", grid.dxi,
            2, .3, None, 23.4, "r", 4.1, 3., v_tol, 16, asymptotic, .1, False, "free", 1e6)
    plans, work = [], []
    plan = sc._prepare_match_plan_for_energy
    batch, suffix = sc._numerov_propagate_sqrt_wbase_batch_numba, sc._numerov_propagate_sqrt_wbase_suffix_numba
    def observed_plan(*a, **kw):
        plans.append(a[3])
        return plan(*a, **kw)
    def observed(fn):
        def propagate(*a, **kw):
            work.extend(a[4].tolist())
            return fn(*a, **kw)
        return propagate
    monkeypatch.setattr(sc, "_prepare_match_plan_for_energy", observed_plan)
    monkeypatch.setattr(sc, "_numerov_propagate_sqrt_wbase_batch_numba", observed(batch))
    monkeypatch.setattr(sc, "_numerov_propagate_sqrt_wbase_suffix_numba", observed(suffix))
    density, expected = sc._scattering_density_and_phase(*args, **options)
    original_plans, original_work = plans.copy(), len(work)
    for l in (0, 1, 30, 70, 100):
        plans.clear()
        work.clear()
        actual_density, phases = sc._scattering_density_and_phase(*args, **options, _phase_channel=l)
        assert phases[l] == expected[l]
        assert plans == original_plans  # Includes soft-limit and domain retries.
        if actual_density is not None:
            np.testing.assert_array_equal(actual_density, density)
            np.testing.assert_array_equal(phases, expected)
        if mode == "recovery":
            assert len(plans) == 2 and actual_density is None
            assert len(work) < original_work
        if mode == "soft":
            assert len(plans) == 3  # Soft cap -> density cap -> matching-box cap.


def test_low_cap_scalar_propagation_is_preserved():
    grid = create_sqrt_grid(3., 256, rmin=1e-5)
    args = (-np.exp(-grid.r), grid.r, 0., 1., .1, 1, "sqrt", grid.dxi,
            2, .3, None, None, "r", None, None, None, 8, "free", .1, False, "free", 1e6)
    _, expected = sc._scattering_density_and_phase(*args, l_cap_strategy="none")
    density, phase = sc._scattering_density_and_phase(*args, l_cap_strategy="none", _phase_channel=1)
    assert density is None and phase[1] == expected[1]
