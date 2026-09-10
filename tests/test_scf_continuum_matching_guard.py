"""SCF must not converge on a free-wave substitute for an interacting state."""
import numpy as np
import pytest

from otter.electronic import ks_dft as ks
from otter.electronic.continuum.scattering import QuantumContinuumScattering
from otter.electronic.continuum.ideal import IdealContinuum


@pytest.fixture
def window():
    r = np.linspace(.01, 12., 240)
    params = dict(solve_rmax=6., match_r_cut=4., match_width=1.,
                  match_v_tol=.1, match_min_points=8)
    return r, params


def test_guard_uses_propagation_window_not_large_box_or_ws(window):
    r, params = window
    v = np.where(r < 6., .3, 0.)
    assert not ks._continuum_matching_window_valid(r, v, params)
    # An asymptotic region beyond the actual propagation domain does not help.
    params = params | dict(match_r_cut=9., solve_rmax=12.)
    assert ks._continuum_matching_window_valid(r, v, params)


def test_valid_step_and_invalid_initial_bootstrap_are_unchanged(window):
    r, params = window
    old, proposed = np.zeros_like(r), np.full_like(r, .05)
    output, steps = ks._guard_continuum_potential_mix(r, old, proposed, proposed, .15, params)
    assert output is proposed and steps == 0
    invalid = np.full_like(r, .3)
    output, steps = ks._guard_continuum_potential_mix(r, invalid, invalid, invalid, .15, params)
    assert output is invalid and steps == 0


def test_guard_damps_whole_potential_without_clipping_or_tail_edit(window):
    r, params = window
    old = -.1/r
    raw = old + 4./r
    proposal = old + 8./r
    output, steps = ks._guard_continuum_potential_mix(r, old, raw, proposal, .15, params)
    assert steps > 0
    assert ks._continuum_matching_window_valid(r, output, params)
    np.testing.assert_allclose(output, old + .15 * 2.**(-(steps-1)) * (raw-old))
    np.testing.assert_array_equal(old, -.1/r)
    np.testing.assert_array_equal(proposal, old+8./r)


def test_blocked_guard_is_bounded_and_does_not_certify_convergence(window):
    r, params = window
    old, raw = np.zeros_like(r), np.full_like(r, 1e12)
    output, steps = ks._guard_continuum_potential_mix(r, old, raw, raw, .15, params)
    np.testing.assert_array_equal(output, old)
    assert steps == 14


@pytest.mark.parametrize("mu_mode", ["fixed", "neutral"])
def test_final_audit_rejects_invalid_window_even_with_small_scf_residual(window, mu_mode):
    r, params = window
    config = ks.KSDTFConfig(Z=4, temperature=1., mu=1., r_ws=1.,
                           mu_mode=mu_mode, continuum_params=params, compute_external=False)
    result = dict(r=r, v_full=np.full_like(r, .3), converged=True,
                  final_state_map_error=1e-10, history=[dict(err=1e-10)])
    ks._audit_continuum_matching(result, config, QuantumContinuumScattering())
    assert not result["converged"]
    assert result["scf_stop_reason"] == "continuum_matching_window"
    assert result["final_state_map_error"] == 1e-10
    # A later valid potential does not certify a density evaluated on fallback.
    result.update(v_full=np.zeros_like(r), converged=True,
                  history=[dict(continuum_matching_window_full_valid=False)])
    ks._audit_continuum_matching(result, config, QuantumContinuumScattering())
    assert not result["converged"]


def test_audit_checks_external_but_leaves_ideal_model_unchanged(window):
    r, params = window
    config = ks.KSDTFConfig(Z=4, temperature=1., mu=1., r_ws=1.,
                           continuum_params=params, compute_external=True)
    result = dict(r=r, v_full=np.zeros_like(r), v_ext=np.full_like(r, .3),
                  converged=True, history=[])
    ks._audit_continuum_matching(result, config, IdealContinuum())
    assert result["converged"]
    ks._audit_continuum_matching(result, config, QuantumContinuumScattering())
    assert result["continuum_matching_window_full_valid"]
    assert not result["continuum_matching_window_ext_valid"]
    assert not result["converged"]


def test_be65_default_full_scf_does_not_use_free_wave_branch():
    from otter.electronic import FullExternalConfig, solve_full_only

    config = FullExternalConfig(element="Be", rho_g_cc=65., temperature_ev=50., run_mode="full")
    result = solve_full_only(config)
    assert result["stage2_converged"]
    assert result["threshold_state_status"] == "resolved"
    assert result["continuum_matching_window_full_valid"]
    assert result["final_state_map_error"] < config.scf_tol
    # A better extrapolation may avoid backtracking altogether. Once the
    # interacting branch is reached, no evaluated input may leave it.
    valid = [h["continuum_matching_window_full_valid"] for h in result["history"]]
    assert any(valid) and all(valid[valid.index(True):])
    assert result["zbar"] == pytest.approx(3.42173, abs=.002)
    assert result["meta"]["n0_final_bohr3"] / result["meta"]["n_i_bohr3"] == pytest.approx(2.83033, abs=.002)
