"""Unit tests for the opt-in charge-constrained B3 tail fit."""

from __future__ import annotations

import numpy as np
import pytest

import otter.electronic.ks_dft as ks_dft
import otter.electronic.full_external as full_external
from otter.electronic.ks_dft import (
    KSDTFConfig,
    _apply_charge_constrained_b3_tail,
    _charge_constraint_failure_meta,
    _source_background_charge,
    _source_electron_charge_target,
    solve_ks_dft_is,
)
from otter.electronic.continuum.tail import (
    _solve_tail_fit_system,
    apply_tail_match,
    linear_response_tail,
)
from otter.data.helpers import trapz_integral
from otter.electronic.full_external import (
    FullExternalConfig,
    _apply_paired_pseudoatom_b3_charge_closure,
    _build_continuum_params,
    _needs_threshold_state_refine_retry,
    _reclose_legacy_diffuse_threshold_pseudoatom_for_qoz,
    _threshold_refine_cont_rmax_mult,
    _threshold_energy_refinement,
    solve_full_then_external,
)


def _charge(r: np.ndarray, density: np.ndarray) -> float:
    return float(4.0 * np.pi * trapz_integral((r**2) * density, r))


@pytest.mark.parametrize("already_fine", [False, True])
def test_threshold_energy_recovery_is_consistent_and_preserves_finer_requests(
    already_fine: bool,
) -> None:
    from otter.electronic.mixture import _threshold_refine_config

    cfg = FullExternalConfig(element="H", temperature_ev=9., rho_g_cc=.946)
    if already_fine:
        cfg.bound_zero_tail_min_binding_ha = 1e-14
        cfg.cont_e_min, cfg.cont_e_tol, cfg.cont_dE_min = 1e-12, 1e-6, 1e-12
        cfg.cont_near_zero_log_points_per_decade = 16
        cfg.cont_near_zero_log_max_nodes = 160
    refined = _threshold_energy_refinement(cfg)
    mixture = _threshold_refine_config(cfg, l_max=0)
    for field in ("bound_zero_tail_min_binding_ha", "cont_e_min", "cont_e_tol", "cont_dE_min"):
        assert getattr(refined, field) <= getattr(cfg, field)
        assert getattr(mixture, field) == getattr(refined, field)
    for field in ("bound_zero_tail_scan_points", "cont_near_zero_log_points_per_decade",
                  "cont_near_zero_log_max_nodes"):
        assert getattr(refined, field) >= getattr(cfg, field)
        assert getattr(mixture, field) == getattr(refined, field)
    assert (refined.rmax_mult, refined.n_points, refined.scf_dn_tol, refined.scf_dv_tol) == (
        cfg.rmax_mult, cfg.n_points, cfg.scf_dn_tol, cfg.scf_dv_tol)
    if not already_fine:
        assert cfg.cont_e_min == 1e-6  # Ordinary AA and the input object stay unchanged.
        assert refined.cont_e_min == 1e-10


def test_b3_charge_constraint_includes_hermite_bridge() -> None:
    r = np.linspace(0.08, 18.0, 1400)
    n0 = 0.035
    mu = 0.42
    temperature = 0.24
    idx_cut = 520
    n_raw = linear_response_tail(
        r,
        n0,
        mu,
        temperature,
        A=0.18,
        B=0.07,
        delta=0.35,
    )
    # Add an inner-only perturbation to verify that the retained prefix enters
    # the finite-box charge equation but is not modified by the tail fit.
    n_raw[:idx_cut] += 0.004 * np.exp(-0.6 * r[:idx_cut])
    target = _charge(r, n_raw) + 0.025

    constrained, meta = apply_tail_match(
        r,
        n_raw,
        n0,
        mu,
        temperature,
        idx_cut,
        fit_points=36,
        r_fit_max=float(r[idx_cut + 180]),
        fit_window_mode="physical",
        blend_points=12,
        model="full",
        charge_target=target,
    )

    assert np.array_equal(constrained[:idx_cut], n_raw[:idx_cut])
    assert abs(_charge(r, constrained) - target) < 2.0e-9
    assert bool(meta["charge_constraint_applied"])
    assert abs(float(meta["charge_constraint_residual"])) < 2.0e-9
    assert np.isfinite(float(meta["charge_constraint_unconstrained_fit_rms"]))
    assert np.isfinite(float(meta["charge_constraint_fit_rms_ratio"]))
    assert np.isfinite(float(meta["charge_constraint_coeff_delta_rel"]))
    assert np.isfinite(float(meta["charge_constraint_profile_delta_rel"]))
    assert bool(meta["charge_constraint_accepted"])
    assert float(meta["charge_constraint_tail_min"]) >= 0.0


@pytest.mark.parametrize("exact_ws", [True, False])
def test_paired_b3_closure_repairs_canonical_pseudoatom_profiles(exact_ws) -> None:
    r_ws = 1.5
    r = np.linspace(0.02, 15.0 * r_ws, 2400)
    n0 = 0.02
    g_ii = np.asarray(r >= r_ws, dtype=float)

    def normalized_profile(values: np.ndarray, charge: float) -> np.ndarray:
        return np.asarray(values, dtype=float) * float(charge) / _charge(r, values)

    n_bound = normalized_profile(np.exp(-(r / 0.45) ** 2), 0.8)
    n_ion = normalized_profile(np.exp(-(r / 0.35) ** 2), 0.2)
    free_inside = normalized_profile(
        np.where(r < r_ws, np.exp(-(r / 0.65) ** 4), 0.0), 0.2
    )
    r_cut = 4.0 * r_ws
    b3_response = linear_response_tail(
        r, n0, 0.4, 5.0 / 27.211386245988, 1.0e-3, 5.0e-4, 0.3
    ) - n0
    b3_response[r < r_cut] = 0.0
    compensation_shell = np.asarray((r >= r_ws) & (r < r_cut), dtype=float)
    # Construct a neutral external source under the SAME background policy
    # being tested; a sampled step is not an exactly integrated sharp cavity.
    external_charge = _source_electron_charge_target(
        r, n0, g_ii, 0., ion_sphere_radius=r_ws if exact_ws else None)
    n_ext_pre_tail = (
        n0 * g_ii
        + b3_response
        + normalized_profile(compensation_shell,
            external_charge - _charge(r, n0*g_ii + b3_response))
    )
    n_cont_pre_tail = n_ext_pre_tail + free_inside

    # Mimic two independently extrapolated B3 tails whose difference carries
    # a spurious 0.1 electron even though both saved pre-tail profiles are
    # individually compatible with their physical source charges.
    tail_artifact = normalized_profile(
        np.exp(-((r - 8.0 * r_ws) / (1.5 * r_ws)) ** 2), 0.1
    )
    n_ext_raw = n_ext_pre_tail + tail_artifact
    n_full_raw = n_bound + n_cont_pre_tail
    result = {
        "r": r,
        "r_ws": r_ws,
        "Z": 1,
        "mu": 0.4,
        "n0": n0,
        "g_ii": g_ii,
        "n_bound": n_bound,
        "n_ion": n_ion,
        "n_cont_pre_tail": n_cont_pre_tail,
        "n_ext_pre_tail": n_ext_pre_tail,
        "n_cont": n_cont_pre_tail.copy(),
        "n_full": n_full_raw,
        "n_ext": n_ext_raw,
        "n_pa": n_full_raw - n_ext_raw,
        "n_scr": n_full_raw - n_ext_raw - n_ion,
        "stage2_converged": True,
        "ext_status": {"converged": True},
    }
    cfg = FullExternalConfig(
        element="H",
        temperature_ev=5.0,
        rho_g_cc=1.0,
        r_ws_override_bohr=r_ws,
        b3_tail_target="cont",
        exact_ws_boundary_quadrature=exact_ws,
    )

    closed, meta = _apply_paired_pseudoatom_b3_charge_closure(
        result,
        cfg,
        r_ws=r_ws,
        rmax=float(r[-1]),
    )

    assert bool(meta["applied"]), meta
    assert float(meta["q_scr_rel_raw"]) > cfg.b3_pseudoatom_charge_rel_tol
    assert float(meta["q_scr_rel_closed"]) < 1.0e-9
    assert (
        float(meta["full_tail_meta"]["charge_constraint_fit_rms_ratio"]) <= 10.0
    )
    assert float(meta["ext_tail_meta"]["charge_constraint_fit_rms_ratio"]) <= 10.0
    np.testing.assert_allclose(
        closed["n_full"], closed["n_bound"] + closed["n_cont"]
    )
    np.testing.assert_allclose(
        closed["n_pa"], closed["n_full"] - closed["n_ext"]
    )
    np.testing.assert_allclose(
        closed["n_scr"], closed["n_pa"] - closed["n_ion"]
    )
    assert abs(_charge(r, closed["n_full"] - closed["n_ext"]) - 1.0) < 1.0e-9


def test_diffuse_threshold_closure_fits_total_density_even_when_charge_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scalar charge check must not hide a nonlocal pressure-ionization tail."""
    r_ws = 1.5
    r = np.linspace(0.02, 15.0 * r_ws, 1200)
    n0 = 0.02
    g_ii = np.asarray(r >= r_ws, dtype=float)
    n_bound = np.exp(-r / 3.0)
    n_bound /= _charge(r, n_bound)
    n_ext_pre = n0 * g_ii
    n_full_pre = n_ext_pre + n_bound
    result = {
        "r": r,
        "r_ws": r_ws,
        "Z": 1,
        "mu": 0.4,
        "n0": n0,
        "g_ii": g_ii,
        "n_bound": n_bound,
        "n_ion": np.zeros_like(r),
        "n_cont_pre_tail": n_ext_pre.copy(),
        "n_full_pre_tail": n_full_pre.copy(),
        "n_ext_pre_tail": n_ext_pre.copy(),
        "n_cont": n_ext_pre.copy(),
        "n_full": n_full_pre.copy(),
        "n_ext": n_ext_pre.copy(),
        "n_pa": n_bound.copy(),
        "n_scr": n_bound.copy(),
        "stage2_converged": True,
        "ext_status": {"converged": True},
        "threshold_state_localization": "diffuse",
        "zero_tail_bound_meta": {"applied": True, "states": [{}]},
    }
    calls: list[dict[str, object]] = []

    def fake_tail_match(
        r_arg: np.ndarray,
        density: np.ndarray,
        n0_arg: float,
        *args: object,
        **kwargs: object,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del args
        target = float(kwargs["charge_target"])
        background = float(n0_arg) * g_ii
        local = np.exp(-(np.asarray(r_arg) / 0.4) ** 2)
        candidate = background + local * (
            target - _charge(r_arg, background)
        ) / _charge(r_arg, local)
        calls.append(
            {
                "density": np.asarray(density, dtype=float).copy(),
                "fit_rms_ratio_max": kwargs[
                    "charge_constraint_fit_rms_ratio_max"
                ],
            }
        )
        return candidate, {
            "charge_constraint_applied": True,
            "charge_constraint_accepted": True,
            "charge_constraint_residual": 0.0,
        }

    monkeypatch.setattr(full_external, "apply_tail_match", fake_tail_match)
    cfg = FullExternalConfig(
        element="H",
        temperature_ev=5.0,
        rho_g_cc=1.0,
        r_ws_override_bohr=r_ws,
        b3_tail_target="cont",
    )

    preserved, preserved_meta = _apply_paired_pseudoatom_b3_charge_closure(
        result,
        cfg,
        r_ws=r_ws,
        rmax=float(r[-1]),
    )
    assert not bool(preserved_meta["applied"]), preserved_meta
    np.testing.assert_allclose(preserved["n_scr"], result["n_scr"])

    closed, meta = _apply_paired_pseudoatom_b3_charge_closure(
        result,
        cfg,
        r_ws=r_ws,
        rmax=float(r[-1]),
        allow_diffuse_threshold_full_b3=True,
    )

    assert bool(meta["applied"]), meta
    assert meta["density_target"] == "full"
    assert float(meta["q_scr_rel_raw"]) < 1.0e-12
    np.testing.assert_allclose(calls[0]["density"], n_full_pre)
    assert calls[0]["fit_rms_ratio_max"] == 100.0
    assert closed["n_full_tail_meta"]["target"] == "full"
    np.testing.assert_allclose(closed["n_cont"], result["n_cont"])
    np.testing.assert_allclose(closed["n_pa"], closed["n_full"] - closed["n_ext"])
    np.testing.assert_allclose(closed["n_scr"], closed["n_pa"])
    assert abs(_charge(r, closed["n_scr"]) - 1.0) < 1.0e-9
    assert abs(float(closed["n_scr"][-1])) < 1.0e-14

    result["meta"] = {
        "temperature_ev": 5.0,
        "rho_g_cc": 1.0,
        "b3_tail_stage2_mode": "in_scf",
        "b3_tail_target": "cont",
        "b3_tail_fit_points": 20,
        "b3_tail_local_fit_width_mult": 0.064,
        "b3_tail_fit_window_mode": "local",
        "b3_tail_blend_points": 10,
        "b3_tail_model": "full",
        "b3_tail_auto_rel_improve_tol": 0.2,
        "b3_tail_auto_signal_rel_tol": 5.0e-5,
        "full_r_fit_max_bohr": 5.0 * r_ws,
        "full_r_cut_bohr": 4.0 * r_ws,
        "ext_b3_tail_mode": "in_scf",
        "ext_b3_tail_model": "",
        "b3_charge_constraint_fit_rms_ratio_max": 10.0,
        "b3_charge_constraint_profile_delta_rel_max": 10.0,
        "b3_pseudoatom_charge_rel_tol": 5.0e-2,
    }
    upgraded, upgrade_meta = (
        _reclose_legacy_diffuse_threshold_pseudoatom_for_qoz(result)
    )
    assert bool(upgrade_meta["legacy_qoz_compatibility_reclosure"])
    assert upgrade_meta["density_target"] == "full"
    assert abs(_charge(r, upgraded["n_scr"]) - 1.0) < 1.0e-9

    result["meta"]["b3_diffuse_threshold_policy"] = (
        "self_consistent_explicit_target"
    )
    preserved_current, current_meta = (
        _reclose_legacy_diffuse_threshold_pseudoatom_for_qoz(result)
    )
    assert not bool(current_meta["applied"]), current_meta
    np.testing.assert_allclose(preserved_current["n_scr"], result["n_scr"])


def test_total_density_b3_is_the_default_target() -> None:
    cfg = FullExternalConfig(
        element="C",
        temperature_ev=30.0,
        rho_g_cc=3.51538,
    )
    assert cfg.b3_tail_target == "full"
    assert cfg.threshold_state_refine_retry
    assert cfg.cont_parallel_mode == "shard"


@pytest.mark.parametrize("status", ["none", "resolved", "marginal"])
def test_failed_full_b3_gets_a_domain_check_even_without_unresolved_state(status) -> None:
    cfg = FullExternalConfig(element="Al", temperature_ev=5.0, rho_g_cc=2.7)
    result = {"stage2_converged": False, "threshold_state_status": status}
    assert _needs_threshold_state_refine_retry(result, cfg)
    assert not _needs_threshold_state_refine_retry({**result, "stage2_converged": True}, cfg)
    cfg.cont_rmax_mult = cfg.rmax_mult
    assert not _needs_threshold_state_refine_retry(result, cfg)
    cfg.cont_rmax_mult = 7.0
    cfg.b3_tail_target = "cont"
    assert not _needs_threshold_state_refine_retry(result, cfg)


def test_unresolved_threshold_state_gets_one_zero_tail_retry() -> None:
    result = {
        "stage2_converged": True,
        "threshold_state_status": "unresolved",
    }
    cfg = FullExternalConfig(
        element="C",
        temperature_ev=10.0,
        rho_g_cc=3.51,
    )
    zero_tail_cfg = FullExternalConfig(
        element="C",
        temperature_ev=10.0,
        rho_g_cc=3.51,
        bound_zero_tail_refine=True,
        cont_rmax_mult=15.0,
    )
    disabled_cfg = FullExternalConfig(
        element="C",
        temperature_ev=10.0,
        rho_g_cc=3.51,
        threshold_state_refine_retry=False,
    )

    assert _needs_threshold_state_refine_retry(result, cfg)
    assert _needs_threshold_state_refine_retry({**result, "stage2_converged": False}, cfg)
    assert not _needs_threshold_state_refine_retry(result, zero_tail_cfg)
    zero_tail_cfg.cont_rmax_mult = 7.0
    assert _needs_threshold_state_refine_retry(result, zero_tail_cfg)
    assert not _needs_threshold_state_refine_retry(result, disabled_cfg)
    assert not _needs_threshold_state_refine_retry(
        {**result, "threshold_state_status": "marginal"}, cfg
    )


def test_threshold_retry_matches_the_continuum_to_the_existing_bound_domain() -> None:
    cfg = FullExternalConfig(
        element="H",
        temperature_ev=7.0,
        rho_g_cc=0.946,
    )

    assert cfg.cont_rmax_mult == pytest.approx(7.0)
    assert _threshold_refine_cont_rmax_mult(cfg) == pytest.approx(15.0)
    assert (cfg.rmax_mult, cfg.n_points) == (15.0, 4096)
    assert (cfg.b3_r_cut_mult, cfg.b3_r_fit_max_mult) == (4.0, 5.0)


def test_high_level_threshold_retry_is_cold_bounded_and_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_full(cfg):
        calls.append(cfg)
        result = _synthetic_full_result(cfg)
        refined = bool(cfg.bound_zero_tail_refine)
        result.update(
            {
                "threshold_state_status": "resolved" if refined else "unresolved",
                "threshold_state_localization": "diffuse",
                "threshold_state_representation": (
                    "zero_tail_matched" if refined else "finite_dirichlet_box"
                ),
                "bound_state_diagnostics": {
                    "states": [],
                    "shallowest": {
                        "l": 1,
                        "binding_below_continuum_edge_ha": 2.0e-2,
                    },
                    "shallowest_status": "resolved" if refined else "unresolved",
                },
            }
        )
        return result

    monkeypatch.setattr(full_external, "solve_ks_dft_is", fake_full)
    monkeypatch.setattr(
        full_external, "_build_bound_tables_and_dos", lambda **kwargs: {}
    )
    monkeypatch.setattr(
        full_external, "_build_scattering_continuum_dos", lambda **kwargs: {}
    )
    result = solve_full_then_external(
        FullExternalConfig(
            element="C",
            temperature_ev=10.0,
            rho_g_cc=3.51,
            n_points=48,
            rmax_mult=4.0,
            run_mode="full",
            ext_scf_enabled=False,
            save_data=False,
            show_scf_progress=False,
            verbose=False,
        )
    )

    assert len(calls) == 4  # retry stage 1 is a zero-iteration initialization
    assert calls[2].max_iter == 0
    assert all(not bool(call.bound_zero_tail_refine) for call in calls[:2])
    assert all(bool(call.bound_zero_tail_refine) for call in calls[2:])
    assert int(calls[-1].bound_zero_tail_l_max) == 1
    assert float(calls[-1].bound_zero_tail_max_binding) >= 2.5e-2
    assert float(calls[-1].rmax / calls[-1].r_ws) == pytest.approx(4.0)
    assert int(calls[-1].n_points) == 48
    assert calls[-1].continuum_params["solve_rmax"] == pytest.approx(calls[-1].rmax)
    assert calls[-1].continuum_params["adaptive_mode"] == "phase-root"
    assert result["threshold_state_status"] == "resolved"
    assert bool(result["threshold_state_refine_retry"]["applied"])


@pytest.mark.parametrize("retry_converges", [True, False])
@pytest.mark.parametrize("fixed_mu", [None, 0.2])
@pytest.mark.parametrize("bound_search_lmax", [0, 1])
def test_scf_domain_retry_does_not_match_deep_shells_or_accept_failed_retry(
    monkeypatch, retry_converges, fixed_mu, bound_search_lmax,
) -> None:
    calls = []

    def fake_full(cfg):
        calls.append(cfg)
        if fixed_mu is not None:
            assert cfg.max_iter > 0  # No uninitialized zero-step fixed-mu SCF.
            assert cfg.mu == fixed_mu
        result = _synthetic_full_result(cfg)
        refined = cfg.continuum_params["solve_rmax"] == cfg.rmax
        result.update(
            converged=bool(refined and retry_converges),
            stage2_converged=bool(refined and retry_converges),
            threshold_state_status="resolved",
            bound_state_diagnostics={"states": [], "shallowest": {
                "l": 2, "binding_below_continuum_edge_ha": 2.0,
            }},
        )
        return result

    monkeypatch.setattr(full_external, "solve_ks_dft_is", fake_full)
    monkeypatch.setattr(full_external, "_build_bound_tables_and_dos", lambda **kw: {})
    monkeypatch.setattr(full_external, "_build_scattering_continuum_dos", lambda **kw: {})
    result = solve_full_then_external(FullExternalConfig(
        element="Al", temperature_ev=5.0, rho_g_cc=2.7,
        n_points=48, rmax_mult=4.0, cont_rmax_mult=2.0,
        full_fixed_mu_ha=fixed_mu,
        bound_zero_tail_l_max=bound_search_lmax,
        run_mode="full", ext_scf_enabled=False, show_scf_progress=False,
    ))
    assert len(calls) == (4 if fixed_mu is None else 3)
    if fixed_mu is not None:
        np.testing.assert_allclose(
            calls[-1].v_full_init,
            -calls[-1].Z / full_external._target_radial_grid(
                rmax=calls[-1].rmax, n_points=calls[-1].n_points,
            ),
        )
    assert calls[-1].bound_zero_tail_l_max == bound_search_lmax
    assert calls[-1].continuum_params["adaptive_parallel_mode"] == "batch"
    assert calls[-1].bound_zero_tail_max_binding == pytest.approx(0.01)
    assert calls[-1].continuum_params["adaptive_mode"] == "simpson"
    assert result["stage2_converged"] is retry_converges
    assert result["threshold_state_refine_retry"]["reason"] == "unconverged full-B3 SCF"


@pytest.mark.parametrize("energy_outcome", ["success", "unconverged", "error"])
@pytest.mark.parametrize("early_handoff", [False, True])
def test_energy_refinement_precedes_cold_domain_retry(monkeypatch, energy_outcome, early_handoff):
    """Try the same-state energy check once; keep the cold recovery on failure."""
    calls = []
    cfg = FullExternalConfig(
        element="Al", temperature_ev=1., rho_g_cc=8.1, n_points=48,
        run_mode="full", show_scf_progress=False,
        scf_stagnation_recovery=early_handoff,
    )

    def fake_full(low):
        calls.append(low)
        energy_refined = low.continuum_params["e_tol"] < cfg.cont_e_tol
        domain_refined = low.continuum_params["solve_rmax"] == low.rmax
        if energy_refined and not domain_refined and energy_outcome == "error":
            raise RuntimeError("synthetic quadrature failure")
        out = _synthetic_full_result(low)
        converged = domain_refined or (energy_refined and energy_outcome == "success")
        out.update(converged=converged, stage2_converged=converged,
                   scf_stop_reason="converged" if converged else "stagnation",
                   threshold_state_status="resolved",
                   history=[{"dn_rel": 1e-3, "dv_rel": 1e-3} for _ in range(4)])
        return out

    monkeypatch.setattr(full_external, "solve_ks_dft_is", fake_full)
    monkeypatch.setattr(full_external, "_build_bound_tables_and_dos", lambda **kw: {})
    monkeypatch.setattr(full_external, "_build_scattering_continuum_dos", lambda **kw: {})
    result = solve_full_then_external(cfg)
    # Isolate the cheap same-domain retry. The later cold-domain recovery
    # now also resolves the threshold energy mesh, rather than reverting it.
    refined = [c for c in calls if c.continuum_params["e_tol"] < cfg.cont_e_tol
               and c.continuum_params["solve_rmax"] < c.rmax]
    assert len(refined) == 1  # Direct stage 2, no repeated initialization.
    local = refined[0]
    assert local.continuum_params["solve_rmax"] == pytest.approx(cfg.cont_rmax_mult*local.r_ws)
    assert local.n_points == cfg.n_points
    assert local.max_iter <= 60
    assert local.continuum_params["e_tol"] == pytest.approx(cfg.cont_e_tol/10)
    assert local.continuum_params["e_min_width"] == pytest.approx(cfg.cont_dE_min/10)
    assert local.continuum_params["n_e_base"] == 2*cfg.cont_n_e_base
    assert local.mix == cfg.scf_mix
    assert calls[1].stop_on_stagnation is early_handoff
    assert not calls[0].stop_on_stagnation
    assert all(not c.stop_on_stagnation for c in calls[2:])
    target = full_external._target_radial_grid(rmax=local.rmax, n_points=local.n_points)
    np.testing.assert_allclose(
        local.v_full_init, np.interp(target, result["r"], -np.exp(-result["r"])),
    )
    assert result["stage2_converged"]
    assert result["scf_energy_refine_retry"]["accepted"] == (energy_outcome == "success")
    assert result["scf_energy_refine_retry"]["first_pass_stop_reason"] == "stagnation"
    assert ("threshold_state_refine_retry" in result) == (energy_outcome != "success")


def test_density_domain_cannot_fall_back_to_unconverged_raw_outer_density():
    ks_dft._require_density_domain_tail({}, {}, {})
    ks_dft._require_density_domain_tail({"density_rmax": 5.}, {"applied": True}, {})
    ks_dft._require_density_domain_tail({"density_rmax": 5.}, {}, {"applied": True})
    with pytest.raises(RuntimeError, match="requires successful tail replacement"):
        ks_dft._require_density_domain_tail({"density_rmax": 5.}, {}, {})


def test_b3_charge_row_survives_tiny_yukawa_basis_without_cancellation() -> None:
    """A tiny A column must not vanish when the response is added to n0."""
    r = np.linspace(0.08, 30.0, 3000)
    n0 = 0.035
    mu = 10.0
    temperature = 1.0e-3
    idx_cut = int(np.searchsorted(r, 20.0))
    n_raw = linear_response_tail(
        r,
        n0,
        mu,
        temperature,
        A=0.18,
        B=0.07,
        delta=0.35,
    )
    target = _charge(r, n_raw) + 1.0e-3

    out, meta = apply_tail_match(
        r,
        n_raw,
        n0,
        mu,
        temperature,
        idx_cut,
        fit_points=48,
        r_fit_max=24.0,
        fit_window_mode="physical",
        blend_points=12,
        model="full",
        charge_target=target,
        charge_constraint_fit_rms_ratio_max=None,
        charge_constraint_profile_delta_rel_max=None,
    )

    # The former splice(unit)-splice(zero) construction rounded this response
    # to exactly zero because the local basis is about 1e-22 while n0 is 0.035.
    assert 0.0 < abs(float(meta["charge_constraint_row_A"])) < 1.0e-15
    assert abs(_charge(r, out) - target) < 2.0e-9
    assert bool(meta["charge_constraint_accepted"])


def test_b3_charge_constraint_rejects_degraded_fit_and_can_disable_quality_guard() -> None:
    r = np.linspace(0.08, 18.0, 1400)
    n0 = 0.035
    mu = 0.42
    temperature = 0.24
    idx_cut = 520
    n_raw = linear_response_tail(
        r, n0, mu, temperature, A=0.18, B=0.07, delta=0.35
    )
    n_raw[:idx_cut] += 0.004 * np.exp(-0.6 * r[:idx_cut])
    target = _charge(r, n_raw) - 1.0
    kwargs = dict(
        fit_points=36,
        r_fit_max=float(r[idx_cut + 180]),
        fit_window_mode="physical",
        blend_points=12,
        model="full",
        charge_target=target,
    )

    with pytest.raises(ValueError, match="fit RMS ratio"):
        apply_tail_match(
            r,
            n_raw,
            n0,
            mu,
            temperature,
            idx_cut,
            **kwargs,
        )

    out, meta = apply_tail_match(
        r,
        n_raw,
        n0,
        mu,
        temperature,
        idx_cut,
        **kwargs,
        charge_constraint_fit_rms_ratio_max=None,
        charge_constraint_profile_delta_rel_max=None,
    )
    assert bool(meta["charge_constraint_accepted"])
    assert np.min(out[idx_cut:]) >= 0.0
    assert abs(_charge(r, out) - target) < 2.0e-9


def test_b3_charge_constraint_always_rejects_negative_tail() -> None:
    r = np.linspace(0.08, 18.0, 1400)
    n0 = 0.035
    mu = 0.42
    temperature = 0.24
    idx_cut = 520
    n_raw = linear_response_tail(
        r, n0, mu, temperature, A=0.18, B=0.07, delta=0.35
    )
    target = _charge(r, n_raw) - 100.0

    with pytest.raises(ValueError, match="minimum density"):
        apply_tail_match(
            r,
            n_raw,
            n0,
            mu,
            temperature,
            idx_cut,
            fit_points=36,
            r_fit_max=float(r[idx_cut + 180]),
            fit_window_mode="physical",
            blend_points=12,
            model="full",
            charge_target=target,
            charge_constraint_fit_rms_ratio_max=None,
            charge_constraint_profile_delta_rel_max=None,
        )


def test_b3_charge_constraint_is_opt_in() -> None:
    r = np.linspace(0.1, 12.0, 800)
    n0 = 0.02
    n_raw = linear_response_tail(r, n0, 0.25, 0.3, 0.1, 0.03, -0.2)
    out, meta = apply_tail_match(
        r,
        n_raw,
        n0,
        0.25,
        0.3,
        300,
        fit_points=24,
        blend_points=6,
        model="full",
    )
    assert np.all(np.isfinite(out))
    assert not bool(meta["charge_constraint_applied"])
    assert np.isnan(float(meta["charge_constraint_residual"]))


def test_apply_tail_match_preserves_unconstrained_auto_selection() -> None:
    """The orchestration layer must report the model resolved by the local fit."""
    r = np.linspace(1.0, 8.0, 500)
    n0 = 0.05
    n_raw = linear_response_tail(
        r,
        n0,
        0.8,
        0.08,
        A=-0.03,
        B=0.8,
        delta=0.4,
    )
    _, meta = apply_tail_match(
        r,
        n_raw,
        n0,
        0.8,
        0.08,
        80,
        fit_points=40,
        model="auto",
    )
    assert meta["model_requested"] == "auto"
    assert meta["model_selected"] == "full"


def test_auto_model_is_selected_before_charge_constraint() -> None:
    """A global charge equality must not erase a locally resolved Friedel term."""
    r = np.linspace(0.08, 18.0, 1400)
    n0 = 0.035
    mu = 0.42
    temperature = 0.03
    idx_cut = int(np.searchsorted(r, 6.0))
    n_raw = linear_response_tail(
        r,
        n0,
        mu,
        temperature,
        A=0.18,
        B=1.0,
        delta=0.35,
    )
    n_raw[:idx_cut] += 0.004 * np.exp(-0.6 * r[:idx_cut])

    out, meta = apply_tail_match(
        r,
        n_raw,
        n0,
        mu,
        temperature,
        idx_cut,
        fit_points=60,
        r_fit_max=10.0,
        fit_window_mode="physical",
        blend_points=12,
        model="auto",
        charge_target=_charge(r, n_raw) - 1.0,
        charge_constraint_fit_rms_ratio_max=None,
        charge_constraint_profile_delta_rel_max=None,
    )

    assert np.all(np.isfinite(out))
    assert meta["model_requested"] == "auto"
    assert meta["model_selection_basis"] == "unconstrained_local_fit"
    assert meta["model_selection_unconstrained_model"] == "full"
    assert meta["model_selected"] == "full"
    # This is the regression condition: before the fix, the constrained
    # residual comparison was below the 15% auto threshold and selected
    # ``a_only`` even though the unconstrained data clearly required B3.
    assert float(meta["model_selection_unconstrained_fit_rel_improve_full"]) > 0.15
    assert float(meta["charge_constrained_fit_rel_improve_full"]) < 0.15
    assert abs(_charge(r, out) - (_charge(r, n_raw) - 1.0)) < 2.0e-9


def test_zero_integral_row_with_zero_target_is_redundant() -> None:
    r = np.linspace(0.1, 1.0, 6)
    coeffs, _, _, diag = _solve_tail_fit_system(
        np.zeros((4, 1)),
        np.zeros(4),
        r=r,
        n_r=np.zeros_like(r),
        idx_cut=1,
        r_fit=r[1:5],
        deriv_row=np.zeros(1),
        match_value_weight=0.0,
        match_slope_weight=0.0,
        integral_row=np.zeros(1),
        integral_target=0.0,
    )
    np.testing.assert_array_equal(coeffs, np.zeros(1))
    assert bool(diag["integral_constraint_applied"])
    assert bool(diag["integral_constraint_redundant"])
    assert float(diag["integral_constraint_residual"]) == 0.0


def test_extreme_singular_value_ratio_reports_infinity_without_warning() -> None:
    from otter.electronic.continuum.tail import (
        _finite_singular_value_condition_number,
    )

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        condition = _finite_singular_value_condition_number(
            np.asarray((1.0, np.nextafter(0.0, 1.0)))
        )

    assert condition == float("inf")


def test_default_tail_match_works_without_numpy_trapezoid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opt-out/default path remains compatible with NumPy 1.x."""
    r = np.linspace(0.1, 12.0, 800)
    n0 = 0.02
    n_raw = linear_response_tail(r, n0, 0.25, 0.3, 0.1, 0.03, -0.2)
    monkeypatch.delattr(np, "trapezoid", raising=False)

    out, meta = apply_tail_match(
        r,
        n_raw,
        n0,
        0.25,
        0.3,
        300,
        fit_points=24,
        blend_points=6,
        model="full",
    )
    assert np.all(np.isfinite(out))
    assert not bool(meta["charge_constraint_applied"])


def test_source_charge_targets_use_full_Z_and_external_zero() -> None:
    r = np.linspace(0.05, 10.0, 900)
    n0 = 0.04
    g_ii = np.ones_like(r)
    g_ii[r < 1.7] = 0.0
    q_background = _source_background_charge(r, n0, g_ii)
    electron_full = _source_electron_charge_target(r, n0, g_ii, 6.0)
    electron_ext = _source_electron_charge_target(r, n0, g_ii, 0.0)

    assert abs((electron_full + q_background) - 6.0) < 1.0e-11
    assert abs(electron_ext + q_background) < 1.0e-11


def test_full_external_config_keeps_constraint_disabled_by_default() -> None:
    cfg = FullExternalConfig(element="C", temperature_ev=10.0, rho_g_cc=3.0)
    params_default = _build_continuum_params(
        cfg,
        l_max=4,
        r_ws=2.0,
        rmax=20.0,
        adaptive_mode="shared",
        b3_stage_mode="in_scf",
        e_max_mode="fixed",
    )
    assert params_default["b3_source_charge_constraint"] is False

    cfg.b3_source_charge_constraint = True
    params_enabled = _build_continuum_params(
        cfg,
        l_max=4,
        r_ws=2.0,
        rmax=20.0,
        adaptive_mode="shared",
        b3_stage_mode="in_scf",
        e_max_mode="fixed",
    )
    assert params_enabled["b3_source_charge_constraint"] is True
    assert params_enabled["b3_charge_constraint_fit_rms_ratio_max"] == 10.0
    assert params_enabled["b3_charge_constraint_profile_delta_rel_max"] == 10.0

    cfg.b3_charge_constraint_fit_rms_ratio_max = None
    cfg.b3_charge_constraint_profile_delta_rel_max = None
    params_disabled_guards = _build_continuum_params(
        cfg,
        l_max=4,
        r_ws=2.0,
        rmax=20.0,
        adaptive_mode="shared",
        b3_stage_mode="in_scf",
        e_max_mode="fixed",
    )
    assert params_disabled_guards["b3_charge_constraint_fit_rms_ratio_max"] is None
    assert params_disabled_guards["b3_charge_constraint_profile_delta_rel_max"] is None


def test_charge_constrained_helper_respects_fixed_tail_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_apply_tail_match(
        r: np.ndarray,
        density: np.ndarray,
        n0: float,
        mu_id: float,
        temperature: float,
        idx_cut: int,
        **kwargs: object,
    ) -> tuple[np.ndarray, dict[str, object]]:
        captured.update(
            n0=n0,
            mu_id=mu_id,
            fit_max=kwargs["charge_constraint_fit_rms_ratio_max"],
            profile_max=kwargs["charge_constraint_profile_delta_rel_max"],
        )
        return np.asarray(density, dtype=float), {
            "charge_constraint_applied": True,
            "charge_constraint_accepted": True,
        }

    monkeypatch.setattr(
        "otter.electronic.continuum.tail.apply_tail_match", fake_apply_tail_match
    )
    r = np.linspace(0.1, 10.0, 100)
    density = np.full_like(r, 0.03)
    _, meta = _apply_charge_constrained_b3_tail(
        r,
        density,
        n0=0.03,
        mu_id=0.2,
        temperature=0.1,
        params={
            "tail_r_cut": 6.0,
            "tail_n0_fixed": 0.031,
            "tail_mu_id_fixed": 0.24,
            "b3_charge_constraint_fit_rms_ratio_max": 7.0,
            "b3_charge_constraint_profile_delta_rel_max": None,
        },
        electron_charge_target=15.0,
    )

    assert captured == {
        "n0": 0.031,
        "mu_id": 0.24,
        "fit_max": 7.0,
        "profile_max": None,
    }
    assert meta["tail_n0_used"] == 0.031
    assert meta["tail_mu_id_used"] == 0.24
    assert bool(meta["charge_constraint_requested"])


def test_charge_constraint_failure_meta_is_explicit() -> None:
    meta = _charge_constraint_failure_meta(
        {"applied": True, "model_selected": "full"},
        ValueError("quality guard"),
    )
    assert bool(meta["charge_constraint_requested"])
    assert not bool(meta["charge_constraint_applied"])
    assert not bool(meta["charge_constraint_accepted"])
    assert meta["charge_constraint_failure_reason"] == "quality guard"


def _empty_bound_solution(
    potential: np.ndarray,
    r: np.ndarray,
    step: float,
    l_list: np.ndarray,
    **kwargs: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a shape-correct no-bound-state spectrum for lightweight SCF tests."""
    del potential, step, kwargs
    n_l = int(np.asarray(l_list).size)
    return (
        np.empty((n_l, 0), dtype=float),
        np.empty((n_l, 0, np.asarray(r).size), dtype=float),
    )


def _patch_lightweight_bound_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the sparse eigen solve without changing continuum/B3 control flow."""
    monkeypatch.setattr(
        ks_dft, "solve_bound_states_sparse_numerov", _empty_bound_solution
    )
    monkeypatch.setattr(
        ks_dft,
        "_refine_shallow_bound_states_zero_tail",
        lambda *args, **kwargs: (
            args[2],
            args[3],
            {"applied": False, "states": []},
        ),
    )
    monkeypatch.setattr(
        ks_dft,
        "bound_state_reliability_diagnostics",
        lambda *args, **kwargs: {
            "states": [],
            "shallowest": None,
            "shallowest_status": "none",
        },
    )


def test_combined_full_external_exact_b3_failure_is_not_converged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful full constraint must not hide a failed external constraint."""
    _patch_lightweight_bound_channel(monkeypatch)
    exact_calls = 0

    def fake_exact_tail(
        r: np.ndarray,
        density: np.ndarray,
        *,
        n0: float,
        mu_id: float,
        temperature: float,
        params: dict[str, object],
        electron_charge_target: float,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del n0, mu_id, temperature, params, electron_charge_target
        nonlocal exact_calls
        exact_calls += 1
        # With target="cont", each SCF iteration constrains the full
        # continuum first and the external continuum second.
        if exact_calls % 2 == 0:
            raise ValueError("synthetic external quality rejection")
        return np.asarray(density, dtype=float).copy(), {
            "charge_constraint_requested": True,
            "charge_constraint_applied": True,
            "charge_constraint_accepted": True,
            "charge_constraint_residual": 0.0,
        }

    monkeypatch.setattr(
        ks_dft, "_apply_charge_constrained_b3_tail", fake_exact_tail
    )
    cfg = KSDTFConfig(
        Z=1,
        temperature=0.5,
        mu=0.2,
        r_ws=1.0,
        rmax=4.0,
        rmax_mult=None,
        n_points=48,
        l_list=np.array([0]),
        n_states=1,
        continuum_model="ideal",
        compute_external=True,
        mu_mode="fixed",
        n0_mode="ideal",
        mixing_scheme="linear",
        mix=1.0,
        max_iter=3,
        tol=1.0e6,
        dn_tol=1.0e6,
        dv_tol=1.0e6,
        bound_zero_tail_refine=False,
        continuum_params={
            "tail_match": True,
            "tail_mode": "in_scf",
            "tail_match_target": "cont",
            "tail_r_cut": 2.5,
            "tail_fit_points": 8,
            "tail_blend_points": 0,
            "tail_model": "a_only",
            "tail_fallback_on_error": True,
            "b3_source_charge_constraint": True,
        },
    )

    with pytest.warns(
        RuntimeWarning, match="Charge-constrained external B3 fit failed"
    ):
        result = solve_ks_dft_is(cfg)

    assert exact_calls == 2 * cfg.max_iter
    assert not bool(result["converged"])
    assert all(
        bool(item["b3_charge_constraint_full_applied"])
        and not bool(item["b3_charge_constraint_ext_applied"])
        for item in result["history"]
    )
    assert bool(result["b3_charge_constraint_full_applied"])
    assert not bool(result["b3_charge_constraint_ext_applied"])
    assert (
        result["n_ext_tail_meta"]["charge_constraint_failure_reason"]
        == "synthetic external quality rejection"
    )


def test_final_refresh_constraint_failure_revokes_prior_convergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refreshed return state, not the preceding iterate, owns convergence."""
    _patch_lightweight_bound_channel(monkeypatch)
    potential_updates = 0

    def fake_effective_potential(
        r: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> np.ndarray:
        del args, kwargs
        nonlocal potential_updates
        potential_updates += 1
        # Iteration 1 hands iteration 2 a recognizable nonzero map. Iteration
        # 2 then hands the final-refresh evaluation the zero map.
        value = -0.5 if potential_updates == 1 else 0.0
        return np.full_like(np.asarray(r, dtype=float), value)

    def fake_exact_tail(
        r: np.ndarray,
        density: np.ndarray,
        *,
        params: dict[str, object],
        **kwargs: object,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del r, kwargs
        v_eff = np.asarray(params["v_eff"], dtype=float)
        if np.all(v_eff == 0.0):
            raise ValueError("synthetic final-refresh rejection")
        return np.asarray(density, dtype=float).copy(), {
            "charge_constraint_requested": True,
            "charge_constraint_applied": True,
            "charge_constraint_accepted": True,
            "charge_constraint_residual": 0.0,
        }

    monkeypatch.setattr(
        ks_dft, "effective_potential_full", fake_effective_potential
    )
    monkeypatch.setattr(
        ks_dft, "_apply_charge_constrained_b3_tail", fake_exact_tail
    )
    cfg = KSDTFConfig(
        Z=1,
        temperature=0.5,
        mu=0.0,
        mu_mode="neutral",
        mu_strategy="inner",
        mu_bounds=(-2.0, 2.0),
        mu_tol=1.0e-8,
        mu_max_iter=30,
        r_ws=1.0,
        rmax=4.0,
        rmax_mult=None,
        n_points=48,
        l_list=np.array([0]),
        n_states=1,
        continuum_model="ideal",
        compute_external=False,
        n0_mode="ideal",
        mixing_scheme="linear",
        mix=1.0,
        max_iter=2,
        tol=1.0e6,
        dn_tol=1.0e6,
        dv_tol=1.0e6,
        bound_zero_tail_refine=False,
        continuum_params={
            "tail_match": True,
            "tail_mode": "in_scf",
            "tail_match_target": "cont",
            "tail_r_cut": 2.5,
            "tail_fit_points": 8,
            "tail_blend_points": 0,
            "tail_model": "a_only",
            "tail_fallback_on_error": True,
            "b3_source_charge_constraint": True,
        },
    )

    with pytest.warns(
        RuntimeWarning, match="synthetic final-refresh rejection"
    ):
        result = solve_ks_dft_is(cfg)

    assert potential_updates == cfg.max_iter
    assert len(result["history"]) == cfg.max_iter
    assert bool(result["history"][-1]["b3_charge_constraint_full_applied"])
    assert not bool(result["converged"])
    assert not bool(result["b3_charge_constraint_full_applied"])
    assert (
        result["n_cont_tail_meta"]["charge_constraint_failure_reason"]
        == "synthetic final-refresh rejection"
    )


def _synthetic_full_result(cfg: KSDTFConfig) -> dict[str, object]:
    """Minimal, internally consistent high-level payload for metadata tests."""
    r = np.linspace(0.05, float(cfg.rmax), int(cfg.n_points))
    n0 = 0.025
    n_bound = np.zeros_like(r)
    n_cont = np.full_like(r, n0)
    n_full = n_bound + n_cont
    n_ext = np.full_like(r, n0)
    n_ion = np.zeros_like(r)
    n_pa = n_full - n_ext
    return {
        "Z": float(cfg.Z),
        "r": r,
        "r_bound": r.copy(),
        "g_ii": (r >= float(cfg.r_ws)).astype(float),
        "n0": float(n0),
        "n_bound": n_bound,
        "n_ion": n_ion,
        "n_cont": n_cont,
        "n_free": n_cont.copy(),
        "n_cont_dft_raw": n_cont.copy(),
        "n_cont_pre_tail": n_cont.copy(),
        "n_cont_tail_meta": {
            "charge_constraint_requested": False,
            "charge_constraint_applied": False,
        },
        "n_full": n_full,
        "n_full_pre_tail": n_full.copy(),
        "n_full_source": n_full.copy(),
        "n_full_source_provenance": "synthetic_fixed_point_candidate",
        "n_full_tail_meta": {
            "charge_constraint_requested": True,
            "charge_constraint_applied": True,
            "charge_constraint_accepted": True,
            "charge_constraint_residual": 0.0,
        },
        "n_ext": n_ext,
        "n_pa": n_pa,
        "n_scr": n_pa - n_ion,
        "v_full": -np.exp(-r),
        "v_ext": np.zeros_like(r),
        "history": [],
        "converged": True,
        "r_ws": float(cfg.r_ws),
        "mu": float(cfg.mu),
        "zbar": 1.0,
        "zero_tail_bound_meta": {"applied": False, "states": []},
        "bound_state_diagnostics": {
            "states": [],
            "shallowest": None,
            "shallowest_status": "none",
        },
    }


def test_legacy_b3_aliases_are_resolved_in_high_level_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata must describe the controls actually passed to the KS solver."""
    monkeypatch.setattr(
        full_external, "solve_ks_dft_is", lambda cfg: _synthetic_full_result(cfg)
    )
    monkeypatch.setattr(
        full_external, "_build_bound_tables_and_dos", lambda **kwargs: {}
    )
    monkeypatch.setattr(
        full_external, "_build_scattering_continuum_dos", lambda **kwargs: {}
    )

    cfg = FullExternalConfig(
        element="H",
        temperature_ev=10.0,
        rho_g_cc=1.0,
        n_points=48,
        rmax_mult=4.0,
        run_mode="full",
        ext_scf_enabled=False,
        save_data=False,
        show_scf_progress=False,
        verbose=False,
        # Exercise both historical aliases: the empty explicit controls
        # resolve from cont_tail_match and cont_tail_match_target.
        b3_tail_stage1_mode="",
        b3_tail_stage2_mode="",
        b3_tail_target="",
        cont_tail_match=True,
        cont_tail_match_target="full",
        b3_source_charge_constraint=True,
        ext_b3_tail_mode="off",
    )
    result = solve_full_then_external(cfg)
    meta = result["meta"]

    assert meta["b3_tail_stage2_mode_raw"] == ""
    assert meta["b3_tail_stage2_mode"] == "in_scf"
    assert meta["b3_tail_target_raw"] == ""
    assert meta["b3_tail_target"] == "full"
    assert bool(meta["b3_charge_constraint_requested"])
    assert bool(meta["b3_charge_constraint_applied"])
    assert float(meta["b3_charge_constraint_residual"]) == 0.0


def test_high_level_external_constraint_state_replaces_full_only_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public result-level external metadata must describe the ext SCF."""
    monkeypatch.setattr(
        full_external, "solve_ks_dft_is", lambda cfg: _synthetic_full_result(cfg)
    )
    monkeypatch.setattr(
        full_external, "_build_bound_tables_and_dos", lambda **kwargs: {}
    )
    monkeypatch.setattr(
        full_external, "_build_scattering_continuum_dos", lambda **kwargs: {}
    )

    ext_tail_meta = {
        "charge_constraint_requested": True,
        "charge_constraint_applied": True,
        "charge_constraint_accepted": True,
        "charge_constraint_residual": 0.0,
    }

    def fake_external_scf(**kwargs):
        r = np.asarray(kwargs["r"], dtype=float)
        n0 = float(kwargs["n0"])
        return (
            np.full_like(r, n0),
            np.zeros_like(r),
            {
                "iters": 1,
                "err": 0.0,
                "converged": True,
                "history": [],
                "final_ph_kappa": 0.0,
                "tail_meta": dict(ext_tail_meta),
            },
        )

    monkeypatch.setattr(
        full_external, "_external_fixed_mu_scf", fake_external_scf
    )
    cfg = FullExternalConfig(
        element="H",
        temperature_ev=10.0,
        rho_g_cc=1.0,
        n_points=48,
        rmax_mult=4.0,
        run_mode="full+ext",
        ext_scf_enabled=True,
        save_data=False,
        show_scf_progress=False,
        verbose=False,
        b3_tail_stage1_mode="in_scf",
        b3_tail_stage2_mode="in_scf",
        ext_b3_tail_mode="in_scf",
        b3_tail_target="full",
        b3_source_charge_constraint=True,
    )

    result = solve_full_then_external(cfg)

    assert result["n_ext_tail_meta"] == ext_tail_meta
    assert bool(result["b3_charge_constraint_ext_applied"])
    assert bool(result["meta"]["ext_b3_charge_constraint_applied"])


def test_full_external_can_reuse_a_converged_full_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adding the external branch must not rerun an accepted full-AA state."""
    full_calls = 0

    def fake_full(cfg):
        nonlocal full_calls
        full_calls += 1
        return _synthetic_full_result(cfg)

    def fake_external(**kwargs):
        r = np.asarray(kwargs["r"], dtype=float)
        n0 = float(kwargs["n0"])
        return (
            np.full_like(r, n0),
            np.zeros_like(r),
            {
                "iters": 1,
                "err": 0.0,
                "converged": True,
                "history": [],
                "final_ph_kappa": 0.0,
                "tail_meta": {},
            },
        )

    monkeypatch.setattr(full_external, "solve_ks_dft_is", fake_full)
    monkeypatch.setattr(full_external, "_external_fixed_mu_scf", fake_external)
    monkeypatch.setattr(
        full_external, "_build_bound_tables_and_dos", lambda **kwargs: {}
    )
    monkeypatch.setattr(
        full_external, "_build_scattering_continuum_dos", lambda **kwargs: {}
    )

    common = {
        "element": "H",
        "temperature_ev": 10.0,
        "rho_g_cc": 1.0,
        "n_points": 48,
        "rmax_mult": 4.0,
        "save_data": False,
    }
    full = solve_full_then_external(
        FullExternalConfig(**common, run_mode="full", ext_scf_enabled=False)
    )
    calls_after_root = full_calls
    result = solve_full_then_external(
        FullExternalConfig(
            **common,
            run_mode="full+ext",
            ext_scf_enabled=True,
            full_fixed_mu_ha=float(full["mu"]),
            full_result_init=full,
        )
    )

    assert full_calls == calls_after_root
    assert result["full_result_reused"] is True
    assert result["stage2_converged"] is True
    assert result["ext_status"]["converged"] is True

    # Changing a discretization policy is not an external-only continuation.
    # Never reinterpret a saved full state as if it used the new WS boundary.
    with pytest.raises(ValueError, match="WS charge quadrature"):
        solve_full_then_external(FullExternalConfig(
            **common, exact_ws_boundary_quadrature=False, full_result_init=full))
    with pytest.raises(ValueError, match="WS charge quadrature"):
        solve_full_then_external(FullExternalConfig(
            **common, full_result_init={**full, "meta": {}}))
    assert full_calls == calls_after_root
