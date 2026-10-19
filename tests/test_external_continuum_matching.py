"""External SCF must retain interacting matching and a physical fixed point."""

import numpy as np
import pytest

from otter.electronic import full_external as fe
from otter.electronic import ks_dft as ks


@pytest.fixture
def external_loop(monkeypatch):
    r = np.linspace(.01, 12., 240)
    params = dict(source_closure=False, solve_rmax=6., match_r_cut=4.,
                  match_width=1., match_v_tol=.1, match_min_points=8)
    evaluated = []

    class UniformContinuum:
        def density(self, r, mu, temperature, params):
            evaluated.append(params["v_eff"].copy())
            return np.full_like(r, .02)

    monkeypatch.setattr(ks, "_select_continuum_model", lambda _: UniformContinuum())

    def run(**overrides):
        settings = dict(r=r, mu=0., temperature_ha=.5, n0=.02,
                        g_ii=np.ones_like(r), ext_params_base=params,
                        mix=.5, dn_tol=1e-5, dv_tol=1e-4, max_iter=40,
                        adaptive_mix=False, mixing_scheme="linear", mixing_m=2,
                        mixing_w0=1e-4, ext_b3_tail_mode="off", verbose=False,
                        print_every=1)
        return fe._external_fixed_mu_scf(**(settings | overrides))

    return r, params, evaluated, run


@pytest.mark.parametrize("scheme", ["linear", "eyert"])
def test_external_backtracks_before_leaving_interacting_branch(
    monkeypatch, external_loop, scheme,
):
    r, params, evaluated, run = external_loop
    calls = 0

    def potential(r, *args, **kwargs):
        nonlocal calls
        calls += 1
        return (1. if calls == 1 else .05) * (1. - r / r[-1])

    monkeypatch.setattr(fe, "effective_potential_external", potential)
    _, v, status = run(mixing_scheme=scheme)
    assert status["converged"]
    assert status["history"][0]["continuum_mix_backtracks"] > 0
    assert all(ks._continuum_matching_window_valid(r, v, params) for v in evaluated)
    assert ks._continuum_matching_window_valid(r, v, params)
    assert status["map_err"] < 1e-4


def test_blocked_external_step_cannot_pass_on_stationary_density_and_potential(
    monkeypatch, external_loop,
):
    r, _, _, run = external_loop
    monkeypatch.setattr(fe, "effective_potential_external",
                        lambda r, *a, **kw: 1e12 * (1. - r / r[-1]))
    _, v, status = run(max_iter=3)
    np.testing.assert_array_equal(v, np.zeros_like(r))
    assert not status["converged"]
    for row in status["history"]:
        assert row["dn_rel"] == row["dv_rel"] == 0.
        assert row["continuum_mix_backtracks"] == 14
        assert row["map_err"] > 1.
        assert row["convergence_streak"] == 0


@pytest.mark.parametrize("scheme", ["linear", "eyert"])
def test_fixed_mu_external_preserves_coulomb_boundary_value(
    monkeypatch, external_loop, scheme,
):
    """A finite source has V(R)=Q/R, even though V(infinity)=0.

    The potential convention in Starrett--Saumon (2014), Eq. (7), shares
    the full system's chemical potential; it cannot be reset at the box edge.
    """
    r, _, evaluated, run = external_loop
    radius, charge = 1., .02
    potential = np.where(
        r < radius,
        charge * (3 - (r / radius) ** 2) / (2 * radius),
        charge / r,
    )
    monkeypatch.setattr(fe, "effective_potential_external",
                        lambda *a, **kw: potential.copy())
    _, v, status = run(mixing_scheme=scheme, dv_tol=1e-9, max_iter=80)
    assert status["converged"]
    np.testing.assert_allclose(v, potential, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(evaluated[-1], potential, rtol=1e-9, atol=1e-12)
    assert v[-1] == pytest.approx(charge / r[-1], rel=1e-9)


@pytest.mark.parametrize("invalid_check", [0, 1])
def test_invalid_input_or_output_window_resets_external_confirmation(
    monkeypatch, external_loop, invalid_check,
):
    _, _, _, run = external_loop
    monkeypatch.setattr(fe, "effective_potential_external",
                        lambda r, *a, **kw: np.zeros_like(r))
    checks = iter([i != invalid_check for i in range(6)])
    monkeypatch.setattr(fe, "_continuum_matching_window_valid", lambda *a: next(checks))
    _, _, status = run(max_iter=3, convergence_steps=2)
    assert status["converged"] and status["iters"] == 3
    assert status["history"][0]["map_err"] == 0.
    assert [row["convergence_streak"] for row in status["history"]] == [0, 1, 2]
