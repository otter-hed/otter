"""Consecutive SCF gates reset on a failed pass; defaults stay compatible."""
import numpy as np
import pytest

from otter.electronic import ks_dft as ks
from otter.electronic.full_external import FullExternalConfig
from otter.electronic import full_external as fe


@pytest.mark.parametrize("mode", ["fixed", "neutral"])
@pytest.mark.parametrize("steps,budget,expected_iterations,converged", [
    (1, 8, 2, True), (3, 8, 6, True), (3, 5, 5, False)])
def test_actual_scf_loop_requires_consecutive_passes(
    monkeypatch, mode, steps, budget, expected_iterations, converged,
):
    # Keep actual density/potential updates. The third map residual fails;
    # earlier successful checks cannot count toward the later three-pass run.
    # A one-pass run exits before iteration 3; its final-refresh check stays
    # valid. The independent test below exercises a failure during refresh.
    errors = iter([0., 0., np.inf, 0., 0., 0., 0., 0.] if steps > 1 else [0.] * 8)
    monkeypatch.setattr(ks, "_gauge_aligned_map_error", lambda *a: next(errors))
    cfg = ks.KSDTFConfig(Z=1, temperature=.5, mu=0., mu_mode=mode,
        r_ws=1., rmax=4., n_points=48, l_list=np.array([0]), n_states=1,
        continuum_model="ideal", compute_external=False, mu_bounds=(-10., 10.), mu_max_iter=100,
        max_iter=budget, tol=1e6, dn_tol=1e6, dv_tol=1e6,
        convergence_steps=steps)
    result = ks.solve_ks_dft_is(cfg)
    assert result["iters"] == expected_iterations
    assert result["converged"] is converged
    assert result["scf_convergence_steps"] == steps
    if steps == 3:
        assert result["history"][2]["convergence_streak"] == 0


@pytest.mark.parametrize("value", [0, -1, 1.5, True, np.nan])
def test_reject_invalid_confirmation_count(value):
    with pytest.raises(ValueError, match="positive integer"):
        ks._validate_scf_convergence_steps(value)
    with pytest.raises(ValueError, match="positive integer"):
        FullExternalConfig(element="C", temperature_ev=10., rho_g_cc=1.,
                           scf_convergence_steps=value)


def test_default_is_one_and_tf_does_not_silently_ignore_option():
    cfg = FullExternalConfig(element="C", temperature_ev=10., rho_g_cc=1.)
    assert cfg.scf_convergence_steps == 1
    with pytest.raises(ValueError, match="only for QM"):
        FullExternalConfig(element="C", temperature_ev=10., rho_g_cc=1.,
                           electronic_model="tf", scf_convergence_steps=3)


@pytest.mark.parametrize("steps", [1, 3])
def test_external_loop_honours_confirmation_count(monkeypatch, steps):
    class UniformContinuum:
        def density(self, r, *args, **kwargs):
            return np.full_like(r, .02)

    monkeypatch.setattr(ks, "_select_continuum_model", lambda _: UniformContinuum())
    monkeypatch.setattr(fe, "effective_potential_external", lambda r, *a, **kw: np.zeros_like(r))
    r = np.linspace(.01, 3., 48)
    _, _, status = fe._external_fixed_mu_scf(r=r, mu=0., temperature_ha=.5, n0=.02,
        g_ii=np.ones_like(r), ext_params_base=dict(source_closure=False),
        mix=.2, dn_tol=1e-5, dv_tol=1e-5, max_iter=4, adaptive_mix=False,
        mixing_scheme="linear", mixing_m=2, mixing_w0=1e-4, ext_b3_tail_mode="off",
        verbose=False, print_every=1, convergence_steps=steps)
    assert status["converged"] and status["iters"] == steps
    assert status["history"][-1]["convergence_streak"] == steps


def test_reused_full_cannot_claim_a_confirmation_it_never_performed():
    candidate = {key: 1. for key in ("r", "r_ws", "mu", "n0", "g_ii", "n_full", "n_ion", "v_full")}
    candidate["converged"] = True
    cfg = FullExternalConfig(element="H", temperature_ev=10., rho_g_cc=1.,
        scf_convergence_steps=3, full_result_init=candidate)
    with pytest.raises(ValueError, match="confirmation count"):
        fe.solve_full_then_external(cfg)


@pytest.mark.parametrize("refresh_scale,converged", [(1., True), (2., False)])
def test_final_refresh_must_still_satisfy_physical_map(monkeypatch, refresh_scale, converged):
    calls = 0

    def potential(r, *args, **kwargs):
        nonlocal calls
        calls += 1
        return -(refresh_scale if calls > 2 else 1.) / r

    monkeypatch.setattr(ks, "effective_potential_full", potential)
    cfg = ks.KSDTFConfig(Z=1, temperature=.5, mu=0., mu_mode="neutral",
        r_ws=1., rmax=4., n_points=48, l_list=np.array([0]), n_states=1,
        continuum_model="ideal", compute_external=False, mu_bounds=(-10., 10.),
        mu_max_iter=100, max_iter=2, mix=1., mixing_scheme="linear",
        tol=1e-6, dn_tol=1e6, dv_tol=1e6, bound_spectrum_check=False)
    result = ks.solve_ks_dft_is(cfg)
    assert calls == 3  # Two iterations, then the refreshed density's map.
    assert result["history"][-1]["convergence_streak"] == 1
    assert result["converged"] is converged
    assert (result["final_state_map_error"] < cfg.tol) == converged
    assert result["scf_stop_reason"] == ("converged" if converged else "final_refresh_map_residual")
