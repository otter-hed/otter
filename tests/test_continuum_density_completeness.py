"""Evanescent channels are needed even when their oscillations start outside A3."""
import numpy as np
import pytest
from otter.electronic.continuum import scattering as sc
from otter.numerics.grids import create_sqrt_grid


@pytest.mark.parametrize("energy", [.001, .01, .02])
def test_low_energy_free_density_obeys_bessel_completeness(energy):
    grid = create_sqrt_grid(27.5, 3200, rmin=1e-5)
    r = grid.r
    args = (np.zeros_like(r), r, -6., 3.7, energy, 32, "sqrt", grid.dxi,
            2, .3, None, 23.4, "r", 4.1, 3., .1, 16, "auto", .1, False, "free", 1e6)
    actual, _ = sc._scattering_density_and_phase(*args, apply_occ=False, density_rmax=9.2)
    reference, _ = sc._scattering_density_and_phase(*args, apply_occ=False, l_cap_strategy="none")
    mask = (r >= 1.) & (r <= 9.2)
    # Actual interacting-wave solver, not a mocked l-selector. Independent
    # analytic normalization: NIST DLMF 10.60.12, k/pi^2 including spin.
    expected = np.sqrt(2*energy)/np.pi**2
    np.testing.assert_allclose(actual[mask], expected, rtol=2e-8)
    np.testing.assert_allclose(actual[mask], reference[mask], rtol=1e-10)


def test_fixed_small_angular_range_is_not_silently_extended():
    grid = create_sqrt_grid(27.5, 1200, rmin=1e-5)
    args = (np.zeros_like(grid.r), grid.r, -6., 3.7, .01, 2, "sqrt", grid.dxi,
            2, .3, None, 23.4, "r", 4.1, 3., .1, 16, "auto", .1, False, "free", 1e6)
    n, phase = sc._scattering_density_and_phase(*args, apply_occ=False, l_cap_strategy="none")
    assert len(phase) == 3
    assert abs(n[np.searchsorted(grid.r, 9.)]/(np.sqrt(.02)/np.pi**2)-1) > .001


@pytest.mark.parametrize("strength", [1., 6.])
def test_interacting_density_agrees_with_explicit_large_angular_sum(strength):
    grid = create_sqrt_grid(27.5, 2400, rmin=1e-5)
    r = grid.r
    args = (-strength*np.exp(-r)/r, r, -6., 3.7, .01, 32, "sqrt", grid.dxi,
            2, .3, None, 23.4, "r", 4.1, 3., .1, 16, "auto", .1, False, "free", 1e6)
    automatic, phase = sc._scattering_density_and_phase(*args, apply_occ=False, density_rmax=9.2)
    reference, reference_phase = sc._scattering_density_and_phase(
        *args, apply_occ=False, l_cap_strategy="none")
    np.testing.assert_allclose(automatic[r <= 9.2], reference[r <= 9.2], rtol=1e-10)
    np.testing.assert_allclose(np.sin(phase[:8]-reference_phase[:8]), 0., atol=1e-10)
