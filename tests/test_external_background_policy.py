"""Full and external must use the same explicitly selected IS quadrature."""
from dataclasses import replace

import numpy as np
import pytest

from otter.electronic import full_external as fe, mixture


class HandoffChecked(Exception):
    """Stop a cheap integration test at the actual external potential call."""


@pytest.fixture
def constant_continuum(monkeypatch):
    class Continuum:
        @staticmethod
        def density(r, mu, temperature, params):
            return np.full_like(r, 0.02)

    monkeypatch.setattr("otter.electronic.ks_dft._select_continuum_model",
                        lambda model: Continuum())


def external_kwargs(r, **params):
    return dict(r=r, mu=0.1, temperature_ha=0.2, n0=0.02,
                ext_params_base={"r_ws": 1.3, "source_closure": False, **params},
                mix=0.25, dn_tol=1e-4, dv_tol=1e-4, max_iter=1,
                adaptive_mix=False, mixing_scheme="linear", mixing_m=2,
                mixing_w0=5e-4, ext_b3_tail_mode="off", verbose=False,
                print_every=1)


@pytest.mark.parametrize("analytic", [True, False])
@pytest.mark.parametrize("explicit", [True, False])
def test_explicit_default_profile_preserves_policy(
    monkeypatch, constant_continuum, analytic, explicit,
):
    r = np.linspace(1e-3, 4., 64)
    g = fe.IonSphereStepModel(r_ws=1.3).g_ii(r)

    def check_potential(r_arg, density, n0, g_arg, **kwargs):
        np.testing.assert_array_equal(g_arg, g)
        assert kwargs["ion_sphere_radius"] == (1.3 if analytic else None)
        raise HandoffChecked

    monkeypatch.setattr(fe, "effective_potential_external", check_potential)
    with pytest.raises(HandoffChecked):
        fe._external_fixed_mu_scf(
            **external_kwargs(r, analytic_ion_sphere_background=analytic),
            g_ii=g if explicit else None,
        )


@pytest.mark.parametrize("profile", ["correlated", "wrong_radius", "nan"])
def test_conflicting_analytic_policy_fails_before_scf(constant_continuum, profile):
    r = np.linspace(1e-3, 4., 64)
    if profile == "correlated":
        g = 1 - np.exp(-r)
    else:
        g = fe.IonSphereStepModel(r_ws=1.4 if profile == "wrong_radius" else 1.3).g_ii(r)
        if profile == "nan":
            g[-1] = np.nan
    with pytest.raises(ValueError, match="literal sharp ion-sphere"):
        fe._external_fixed_mu_scf(
            **external_kwargs(r, analytic_ion_sphere_background=True), g_ii=g,
        )


@pytest.mark.parametrize("analytic", [True, False])
def test_source_closure_and_potential_share_policy(monkeypatch, constant_continuum, analytic):
    r = np.linspace(1e-3, 4., 64)
    g = fe.IonSphereStepModel(r_ws=1.3).g_ii(r)
    seen = []

    def check_closure(r_arg, bound, density, *args, **kwargs):
        seen.append(kwargs["ion_sphere_radius"])
        return density, {"applied": False}

    def check_potential(*args, **kwargs):
        assert seen == [1.3 if analytic else None]
        assert kwargs["ion_sphere_radius"] == seen[0]
        raise HandoffChecked

    monkeypatch.setattr(fe, "_enforce_source_charge_closure", check_closure)
    monkeypatch.setattr(fe, "effective_potential_external", check_potential)
    with pytest.raises(HandoffChecked):
        fe._external_fixed_mu_scf(**external_kwargs(
            r, analytic_ion_sphere_background=analytic,
            source_closure=True, source_charge_closure=True,
        ), g_ii=g)


@pytest.mark.parametrize("via_mixture", [False, True])
@pytest.mark.parametrize("policy", ["analytic", "sampled", "correlated", "custom_step"])
def test_real_workflow_forwards_background_policy(
    monkeypatch, constant_continuum, via_mixture, policy,
):
    """Run the real builder/splitter/caller/callee, mocking only KS density.

    Reuse a synthetic full state to avoid an expensive SCF. Stop at the first
    external potential call; the synthetic state is never a physical result.
    Also exercise the mixture species dispatcher, not a duplicated code path.
    """
    cfg = fe.FullExternalConfig(
        element="C", temperature_ev=50., rho_g_cc=20., n_points=64,
        r_ws_override_bohr=1.3, ext_b3_tail_mode="off",
        exact_ws_boundary_quadrature=policy != "sampled",
    )
    rmax = fe._resolve_outer_geometry(cfg, r_ws=1.3)["rmax"]
    r = fe._target_radial_grid(rmax=rmax, n_points=cfg.n_points)
    g = fe.IonSphereStepModel(r_ws=1.3).g_ii(r)
    if policy == "correlated":
        g = 1 - np.exp(-r)
    if policy in ("correlated", "custom_step"):
        cfg = replace(cfg, g_ii_override=g, g_ii_override_r=r)
    cfg = replace(cfg, full_result_init=dict(
        r=r, r_ws=1.3, mu=0.1, n0=0.02, g_ii=g,
        n_full=np.full_like(r, 0.02), n_ion=np.zeros_like(r),
        v_full=-6*np.exp(-r)/r, converged=True, stage2_converged=True,
        threshold_state_status="resolved", scf_convergence_steps=cfg.scf_convergence_steps,
        meta={"ws_charge_quadrature": "sampled_step_grid" if policy == "sampled"
              else "exact_boundary_linear"},
    ))

    def check_potential(r_arg, density, n0, g_arg, **kwargs):
        np.testing.assert_array_equal(g_arg, g)
        assert kwargs["ion_sphere_radius"] == (1.3 if policy == "analytic" else None)
        raise HandoffChecked

    monkeypatch.setattr(fe, "effective_potential_external", check_potential)
    solve = mixture._solve_species_from_config if via_mixture else fe.solve_full_then_external
    with pytest.raises(HandoffChecked):
        solve(cfg)
