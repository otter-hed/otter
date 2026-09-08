"""The analytic IS cavity must not leave a uniform-background grid charge."""
import numpy as np
import pytest

from otter.electronic.ks_dft import _electron_count, _source_background_charge, _trapz
from otter.electronic.full_external import FullExternalConfig, _build_ks_config
from otter.electronic.potential import (
    effective_potential_external,
    effective_potential_full,
)


@pytest.mark.parametrize("n_points", [128, 4096])
@pytest.mark.parametrize("r_box", [20.0, 80.0])
def test_neutral_uniform_sphere_has_no_exterior_potential(n_points, r_box):
    r_ws = 5.0
    z = 6.0
    n0 = z / (4.0 * np.pi * r_ws**3 / 3.0)
    r = np.geomspace(1e-5, r_box, n_points)
    n = np.full_like(r, n0)
    g = (r >= r_ws).astype(float)
    full = effective_potential_full(r, n, n0, g, z, ion_sphere_radius=r_ws)
    ext = effective_potential_external(r, n, n0, g, ion_sphere_radius=r_ws)
    outside = r >= r_ws
    np.testing.assert_allclose(full[outside], 0.0, atol=1e-14)
    np.testing.assert_allclose(ext[outside], z / r[outside], atol=1e-14)
    electron_count = 4.0 * np.pi * _trapz(r**2 * n, r)
    background = _source_background_charge(r, n0, g, ion_sphere_radius=r_ws)
    assert electron_count + background == pytest.approx(z, abs=1e-10)


def test_analytic_cavity_does_not_replace_a_correlated_background():
    r = np.geomspace(1e-4, 20.0, 128)
    n = np.full_like(r, 0.01)
    with pytest.raises(ValueError, match="literal sharp ion-sphere"):
        effective_potential_full(r, n, 0.01, np.ones_like(r), 6.0,
                                 ion_sphere_radius=5.0)


@pytest.mark.parametrize("r_box", [30., 60.])
@pytest.mark.parametrize("exact", [True, False])
def test_ws_quadrature_policy_does_not_depend_on_geometry(r_box, exact):
    cfg = FullExternalConfig(element="C", temperature_ev=100., rho_g_cc=5.2,
                             exact_ws_boundary_quadrature=exact)
    low = _build_ks_config(cfg, z_nuc=6, temperature_ha=3.7, n_i=.03,
        r_ws=2., rmax=r_box, mu_guess=-6., mu_bounds=(-20., 10.),
        max_iter=3, cont_params={}, compute_external=False)
    assert low.exact_ws_boundary_quadrature is exact
    assert low.analytic_ion_sphere_background is exact
    assert low.rmax == r_box  # Accuracy policy must not redefine R_geometry.


def test_exact_ws_is_default_but_does_not_replace_correlated_background():
    cfg = FullExternalConfig(element="C", temperature_ev=100., rho_g_cc=5.2,
                             g_ii_override=np.ones(16))
    assert cfg.exact_ws_boundary_quadrature is True
    low = _build_ks_config(cfg, z_nuc=6, temperature_ha=3.7, n_i=.03,
        r_ws=2., rmax=30., mu_guess=-6., mu_bounds=(-20., 10.),
        max_iter=3, cont_params={}, compute_external=False)
    assert low.exact_ws_boundary_quadrature is True
    assert low.analytic_ion_sphere_background is False


def test_ws_count_includes_partial_last_cell():
    # Constant radial charge integrand gives an exact endpoint oracle,
    # independent of Poisson, SCF, or the quadrature implementation.
    r = np.array([.1, 1., 1.9, 2.2, 3.])
    n = 1/r**2
    assert _electron_count(r, n, 2.) == pytest.approx(4*np.pi*(2.-.1))
    assert _electron_count(r, n, 2., interpolate_boundary=False) == pytest.approx(
        4*np.pi*(1.9-.1))
