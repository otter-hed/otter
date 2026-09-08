"""Fast guards for mixture AA convergence, caches, and downstream use."""

from __future__ import annotations

import numpy as np
import pytest

import otter.electronic.mixture as mixmod
from otter.electronic.mixture import MixtureConfig, solve_mixture_full
from otter.workflows import (
    PlasmaWorkflowConfig,
    _electronic_convergence_issues,
    continue_plasma_workflow_from_electronic_result,
    solve_plasma_workflow,
)


def _fake_species_result(cfg_species, *, mu: float, converged: bool) -> dict:
    r = np.linspace(1.0e-3, 4.0, 32)
    return {
        "Z": float(mixmod.element_info(cfg_species.element).z),
        "mu": float(mu),
        "r": r,
        "r_ws": float(cfg_species.r_ws_override_bohr),
        "n0": 0.1,
        "zbar": float(mixmod.element_info(cfg_species.element).z),
        "n_full": np.exp(-r),
        "n_cont": 0.5 * np.exp(-r),
        "n_ion": 0.25 * np.exp(-r),
        "v_full": -np.exp(-r),
        "v_scf": -np.exp(-r),
        "stage2_converged": bool(converged),
        "history": [{"err": 1.0e-6}],
    }


def test_mixture_defaults_to_one_parallel_worker_per_species() -> None:
    cfg = MixtureConfig(
        species=["C", "H"],
        counts=[1.0, 1.36],
        temperature_ev=10.0,
        rho_g_cc=2.94,
    )
    assert cfg.species_parallel_jobs == 2


def test_unconverged_species_result_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exact-theta revisit should retry a failed species AA solve."""
    calls = {"H": 0, "C": 0}

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        calls[symbol] += 1
        return _fake_species_result(
            cfg_species,
            mu=float(1.0 / cfg_species.n_i_override_bohr3),
            converged=not (symbol == "H" and calls[symbol] == 1),
        )

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    theta = np.asarray([0.0], dtype=float)
    try:
        first = evaluator.evaluate(theta)
        second = evaluator.evaluate(theta)
    finally:
        evaluator.close()

    assert not mixmod._record_species_results_are_converged(first)
    assert mixmod._record_species_results_are_converged(second)
    assert calls == {"H": 2, "C": 1}


def test_unresolved_threshold_species_is_not_a_root_or_cache_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A box-unresolved shallow level must not manufacture a dmu root."""
    calls = {"H": 0, "C": 0}

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        calls[symbol] += 1
        result = _fake_species_result(cfg_species, mu=0.0, converged=True)
        if symbol == "H" and calls[symbol] == 1:
            result["threshold_state_status"] = "unresolved"
        else:
            result["threshold_state_status"] = "resolved"
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
        root_threshold_refine_retry=False,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    theta = np.asarray([0.0], dtype=float)
    try:
        first = evaluator.evaluate(theta)
        assert not mixmod._record_species_results_are_converged(first)
        assert np.all(np.isfinite(evaluator.residual(theta)))
        second = evaluator.evaluate(theta)
    finally:
        evaluator.close()

    # ``residual`` retried the rejected point, and only the resolved revisit
    # became an exact-theta cache hit.
    assert mixmod._record_species_results_are_converged(second)
    assert calls == {"H": 2, "C": 1}


def test_unresolved_threshold_warm_start_is_retried_cold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A branch-specific shallow pole must not abort an otherwise valid root point."""
    calls: dict[str, list[bool]] = {"H": [], "C": []}

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        used_warm_start = cfg_species.v_full_init is not None
        calls[symbol].append(bool(used_warm_start))
        volume = 1.0 / float(cfg_species.n_i_override_bohr3)
        result = _fake_species_result(
            cfg_species,
            mu=float(np.log(volume)),
            converged=True,
        )
        result["threshold_state_status"] = (
            "unresolved" if symbol == "H" and used_warm_start else "none"
        )
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    try:
        first = evaluator.evaluate(np.asarray([0.0], dtype=float))
        second = evaluator.evaluate(np.asarray([0.1], dtype=float))
    finally:
        evaluator.close()

    assert mixmod._record_species_results_are_converged(first)
    assert mixmod._record_species_results_are_converged(second)
    h_result = dict(second["results"][0])
    assert h_result["threshold_state_status"] == "none"
    assert h_result["mixture_threshold_cold_retry_attempted"] is True
    assert h_result["mixture_threshold_cold_retry_selected"] is True
    assert h_result["mixture_threshold_cold_retry_initial_reasons"] == (
        "threshold_state_unresolved",
    )
    assert calls["H"] == [False, True, False]
    assert calls["C"] == [False, True]
    assert evaluator._species_threshold_cold_retries == 1


def test_latched_threshold_continuation_failure_is_retried_cold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A path failure is safe to retry after the threshold branch is fixed."""
    calls: dict[str, list[bool]] = {"H": [], "C": []}

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        used_warm_start = cfg_species.v_full_init is not None
        calls[symbol].append(bool(used_warm_start))
        result = _fake_species_result(
            cfg_species,
            mu=float(np.log(1.0 / cfg_species.n_i_override_bohr3)),
            converged=not (symbol == "H" and used_warm_start),
        )
        result["threshold_state_status"] = "none"
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    try:
        evaluator.evaluate(np.asarray([0.0], dtype=float))
        evaluator._species_threshold_refine_latched["H"] = 3
        second = evaluator.evaluate(np.asarray([0.1], dtype=float))
    finally:
        evaluator.close()

    assert mixmod._record_species_results_are_converged(second)
    assert calls["H"] == [False, True, False]
    assert calls["C"] == [False, True]
    assert evaluator._species_threshold_cold_retries == 1


def test_unlatched_continuation_failure_preserves_invalid_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not splice a cold shallow-state branch into a Brent bracket."""
    calls: dict[str, list[bool]] = {"H": [], "C": []}

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        used_warm_start = cfg_species.v_full_init is not None
        calls[symbol].append(bool(used_warm_start))
        result = _fake_species_result(
            cfg_species,
            mu=float(np.log(1.0 / cfg_species.n_i_override_bohr3)),
            converged=not (symbol == "H" and used_warm_start),
        )
        result["threshold_state_status"] = "none"
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    try:
        evaluator.evaluate(np.asarray([0.0], dtype=float))
        second = evaluator.evaluate(np.asarray([0.1], dtype=float))
    finally:
        evaluator.close()

    assert not mixmod._record_species_results_are_converged(second)
    assert calls["H"] == [False, True]
    assert calls["C"] == [False, True]
    assert evaluator._species_threshold_cold_retries == 0


def test_threshold_failure_is_retried_with_physical_matching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shallow full-AA failure gets one cold, physical-domain retry."""
    calls: dict[str, list[dict[str, float | bool | int]]] = {"H": [], "C": []}

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        calls[symbol].append({
            "refine": bool(cfg_species.bound_zero_tail_refine),
            "energy_cut_mode": str(cfg_species.bound_energy_cut_mode),
            "max_binding": float(cfg_species.bound_zero_tail_max_binding_ha),
            "scan_points": int(cfg_species.bound_zero_tail_scan_points),
            "l_max": int(cfg_species.bound_zero_tail_l_max),
            "adaptive_mode": str(cfg_species.cont_adaptive_mode_stage2),
            "edge_tol": float(cfg_species.bound_zero_tail_edge_rel_tol),
            "mix": float(cfg_species.scf_mix),
            "w0": float(cfg_species.scf_mixing_w0),
            "stage1_max_iter": int(cfg_species.stage1_max_iter),
            "max_iter": int(cfg_species.stage2_max_iter),
            "dn_tol": float(cfg_species.scf_dn_tol),
            "dv_tol": float(cfg_species.scf_dv_tol),
            "warm": bool(cfg_species.v_full_init is not None),
            "rmax_mult": float(cfg_species.rmax_mult),
            "n_points": int(cfg_species.n_points),
        })
        recovered = symbol != "H" or bool(cfg_species.bound_zero_tail_refine)
        result = _fake_species_result(cfg_species, mu=0.0, converged=recovered)
        result["threshold_state_status"] = "resolved" if recovered else "unresolved"
        result["threshold_state_localization"] = "localized"
        result["shallowest_bound_energy_ha"] = -2.0e-3
        result["bound_basis_l_list"] = np.asarray([0, 1, 2])
        result["bound_state_diagnostics"] = {"shallowest": {"l": 0}}
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    try:
        record = evaluator.evaluate(np.asarray([0.0], dtype=float))
    finally:
        evaluator.close()

    assert mixmod._record_species_results_are_converged(record)
    h_result = dict(record["results"][0])
    assert h_result["mixture_threshold_refine_retry_attempted"] is True
    assert h_result["mixture_threshold_refine_retry_selected"] is True
    assert h_result["mixture_threshold_refine_retry_initial_reasons"] == (
        "stage2_unconverged",
        "threshold_state_unresolved",
    )
    assert len(calls["H"]) == 2
    retry = calls["H"][1]
    assert retry["refine"] is True
    assert retry["energy_cut_mode"] == "zero"
    assert retry["max_binding"] >= 1.0e-2
    assert retry["scan_points"] == 64
    assert retry["l_max"] == 0
    assert retry["adaptive_mode"] == "simpson"
    assert retry["edge_tol"] == pytest.approx(0.25)
    assert retry["mix"] == pytest.approx(0.15)
    assert retry["w0"] == pytest.approx(5.0e-4)
    assert retry["stage1_max_iter"] == 0
    assert retry["max_iter"] >= 300
    assert retry["dn_tol"] <= 1.0e-6
    assert retry["dv_tol"] <= 1.0e-6
    assert retry["warm"] is False
    assert retry["rmax_mult"] == pytest.approx(15.0)
    assert retry["n_points"] == 4096
    assert evaluator._species_threshold_refine_retries == 1


def test_heavy_threshold_retry_scouts_a_missing_p_channel() -> None:
    """A vanished C p state is scouted without the empty d padding channel."""
    cfg = mixmod.FullExternalConfig(
        element="C", temperature_ev=23.0, rho_g_cc=0.946,
    )
    refined = mixmod._threshold_refine_config(
        cfg,
        threshold_result={
            "Z": 6,
            "bound_basis_l_list": np.asarray([0, 1, 2]),
            "bound_state_diagnostics": {"shallowest": {"l": 0}},
        },
    )

    assert refined.bound_zero_tail_l_max == 1
    assert refined.bound_energy_cut_mode == "zero"
    assert refined.cont_adaptive_mode_stage2 == "phase-root"


def test_diffuse_threshold_retry_extends_a3_without_changing_b3_or_box() -> None:
    cfg = mixmod.FullExternalConfig(
        element="H", temperature_ev=7.0, rho_g_cc=0.946,
    )
    refined = mixmod._threshold_refine_config(
        cfg,
        threshold_result={
            "Z": 1,
            "threshold_state_status": "unresolved",
            "threshold_state_localization": "diffuse",
            "bound_basis_l_list": np.asarray([0, 1]),
            "bound_state_diagnostics": {"shallowest": {"l": 0}},
        },
    )

    assert refined.rmax_mult == cfg.rmax_mult
    assert refined.n_points == cfg.n_points
    assert refined.cont_rmax_mult == cfg.rmax_mult
    assert refined.b3_r_cut_mult == cfg.b3_r_cut_mult
    assert refined.b3_r_fit_max_mult == cfg.b3_r_fit_max_mult
    assert refined.bound_zero_tail_edge_rel_tol == cfg.bound_zero_tail_edge_rel_tol
    assert refined.stage1_max_iter == 0
    assert refined.scf_dn_tol == refined.scf_dv_tol == 1.0e-6


def test_nonthreshold_scf_failure_does_not_use_threshold_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guarded retry must not modify an ordinary failed AA point."""
    calls = {"H": 0, "C": 0}

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        calls[symbol] += 1
        result = _fake_species_result(
            cfg_species, mu=0.0, converged=symbol != "H"
        )
        result["threshold_state_status"] = "resolved"
        result["shallowest_bound_energy_ha"] = -1.0
        result["bound_state_diagnostics"] = {"shallowest": {"l": 1}}
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    try:
        record = evaluator.evaluate(np.asarray([0.0], dtype=float))
    finally:
        evaluator.close()

    assert not mixmod._record_species_results_are_converged(record)
    assert calls == {"H": 1, "C": 1}
    assert evaluator._species_threshold_refine_retries == 0


def test_bound_charge_branch_flips_trigger_threshold_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A final continuum frame must not hide repeated threshold crossings."""
    calls = {"H": 0, "C": 0}

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        calls[symbol] += 1
        recovered = symbol != "H" or bool(cfg_species.bound_zero_tail_refine)
        result = _fake_species_result(cfg_species, mu=0.0, converged=recovered)
        result["threshold_state_status"] = "resolved"
        result["shallowest_bound_energy_ha"] = -1.0
        if symbol == "H" and not recovered:
            # Reproduce the Te=9 failure shape: early pressure-ionization
            # crossings poison the mixer, followed by more than 80 iterations
            # on the all-continuum branch.  The occupied revisits also sit
            # below half of the largest transient often enough that one
            # midpoint split is insufficient.
            charges = [
                0.0,
                0.24,
                0.0,
                0.08,
                0.0,
                0.11,
                0.0,
                0.09,
                0.0,
                0.12,
                0.0,
            ] + [0.0] * 90
            result["history"] = [
                {"err": 1.0e-2, "charge_bound": value}
                for value in charges
            ]
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    try:
        record = evaluator.evaluate(np.asarray([0.0], dtype=float))
    finally:
        evaluator.close()

    assert mixmod._record_species_results_are_converged(record)
    assert calls == {"H": 2, "C": 1}
    h_result = dict(record["results"][0])
    assert h_result["mixture_threshold_refine_retry_selected"] is True


def test_converged_bound_charge_branch_flips_trigger_threshold_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A converged SCF residual must not hide two threshold branches."""
    calls = {"H": 0, "C": 0}

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        calls[symbol] += 1
        refined = bool(cfg_species.bound_zero_tail_refine)
        result = _fake_species_result(cfg_species, mu=0.0, converged=True)
        result["threshold_state_status"] = "resolved"
        result["shallowest_bound_energy_ha"] = -1.0
        result["bound_state_diagnostics"] = {"shallowest": {"l": 1}}
        if symbol == "H" and not refined:
            result["history"] = [
                {"err": 1.0e-7, "charge_bound": value}
                for value in (0.0, 0.2, 0.0, 0.1, 0.0, 0.15, 0.0, 0.1)
            ]
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    try:
        record = evaluator.evaluate(np.asarray([0.0], dtype=float))
    finally:
        evaluator.close()

    assert mixmod._record_species_results_are_converged(record)
    assert calls == {"H": 2, "C": 1}
    h_result = dict(record["results"][0])
    assert h_result["mixture_threshold_refine_retry_attempted"] is True
    assert h_result["mixture_threshold_refine_retry_selected"] is True
    assert h_result["mixture_threshold_refine_latched"] is True
    assert h_result["mixture_threshold_refine_retry_initial_reasons"] == (
        "bound_charge_branch_flips",
    )


def test_threshold_refine_representation_is_latched_across_root_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a diagnosed branch flip, later root points start on that representation."""
    h_calls: list[tuple] = []

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        refined = bool(cfg_species.bound_zero_tail_refine)
        if symbol == "H":
            h_calls.append(
                (
                    refined,
                    cfg_species.v_full_init is not None,
                    str(cfg_species.bound_energy_cut_mode),
                    float(cfg_species.rmax_mult),
                    int(cfg_species.n_points),
                    float(cfg_species.cont_rmax_mult),
                    (
                        0
                        if cfg_species.v_full_init is None
                        else int(np.asarray(cfg_species.v_full_init).size)
                    ),
                )
            )
        result = _fake_species_result(
            cfg_species,
            mu=float(cfg_species.r_ws_override_bohr),
            converged=symbol != "H" or refined,
        )
        result["threshold_state_status"] = (
            "resolved" if symbol != "H" or refined else "unresolved"
        )
        result["threshold_state_localization"] = (
            "diffuse" if symbol == "H" and not refined else "localized"
        )
        result["shallowest_bound_energy_ha"] = -1.0
        result["bound_state_diagnostics"] = {"shallowest": {"l": 1}}
        if symbol == "H" and not refined:
            result["history"] = [
                {"err": 1.0e-2, "charge_bound": value}
                for value in (0.0, 0.2, 0.0, 0.1, 0.0, 0.15, 0.0, 0.1)
            ]
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    try:
        first = evaluator.evaluate(np.asarray([0.0], dtype=float))
        second = evaluator.evaluate(np.asarray([0.1], dtype=float))
    finally:
        evaluator.close()

    assert mixmod._record_species_results_are_converged(first)
    assert mixmod._record_species_results_are_converged(second)
    # The first diagnosed retry must be cold, while the latched representation
    # at the next root point reuses the previous converged potential.
    assert h_calls == [
        (False, False, "zero", 15.0, 4096, 7.0, 0),
        (True, False, "zero", 15.0, 4096, 15.0, 0),
        (True, True, "zero", 15.0, 4096, 15.0, 4096),
    ]
    h_second = dict(second["results"][0])
    assert h_second["mixture_threshold_refine_latched"] is True
    assert h_second["mixture_threshold_refine_l_max"] == 1
    assert h_second["mixture_threshold_refine_cont_rmax_mult"] == pytest.approx(15.0)
    assert h_second["mixture_threshold_refine_retry_attempted"] is False


@pytest.mark.parametrize("automatic", [False, True])
def test_threshold_latch_invalidates_old_radius_cache(monkeypatch, automatic) -> None:
    """Both AA-level and mixture-level recovery must replace old cached spectra."""
    h_calls = []

    def fake_full(cfg):
        symbol = mixmod.element_info(cfg.element).symbol
        refined = cfg.bound_zero_tail_refine
        radius = float(cfg.r_ws_override_bohr)
        needs_retry = bool(symbol == "H" and h_calls and radius != h_calls[0][0])
        recovered = not needs_retry or refined or automatic
        result = _fake_species_result(cfg, mu=radius, converged=bool(recovered))
        if symbol == "H":
            h_calls.append((radius, cfg.cont_rmax_mult))
        result["threshold_state_status"] = "resolved" if recovered else "unresolved"
        result["bound_state_diagnostics"] = {"shallowest": {"l": 0}}
        if needs_retry and automatic and not refined:
            result["threshold_state_refine_retry"] = {"applied": True, "shallow_l": 0}
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", fake_full)
    evaluator = mixmod._MixtureEvaluator(MixtureConfig(
        species=["H", "C"], counts=[1, 1], temperature_ev=7, rho_g_cc=0.946,
        species_parallel_jobs=1, save_data=False,
    ))
    try:
        first = evaluator.evaluate(np.array([0.0]))
        second = evaluator.evaluate(np.array([0.1]))
        assert [entry["r_ws_bohr"] for entry in evaluator._species_init_cache["H"]] == [
            second["r_ws_bohr"][0]]
        revisit = evaluator.evaluate(np.array([0.0]))
    finally:
        evaluator.close()
    assert first["threshold_refine_generation"] == 0
    assert second["threshold_refine_generation"] == 1
    assert revisit is not first
    assert h_calls[-1] == (h_calls[0][0], 15.0)
    assert revisit["results"][0]["mixture_threshold_refine_latched"]


def test_threshold_latch_rebuilds_the_outer_mu_table(monkeypatch) -> None:
    calls = {"H": 0, "C": 0}
    samples_seen = []
    original_record = mixmod._record_species_samples

    def fake_full(cfg):
        symbol = mixmod.element_info(cfg.element).symbol
        calls[symbol] += 1
        refined = bool(cfg.bound_zero_tail_refine)
        failed = symbol == "H" and calls[symbol] > 1 and not refined
        result = _fake_species_result(
            cfg, mu=cfg.r_ws_override_bohr + (0.05 if refined else 0.0),
            converged=not failed,
        )
        result["threshold_state_status"] = "unresolved" if failed else "resolved"
        result["bound_state_diagnostics"] = {"shallowest": {"l": 0}}
        return result

    def record_samples(samples, record):
        samples_seen.append((record["threshold_refine_generation"], len(samples[0])))
        return original_record(samples, record)

    monkeypatch.setattr(mixmod, "solve_full_only", fake_full)
    monkeypatch.setattr(mixmod, "_record_species_samples", record_samples)
    result = mixmod.solve_mixture_full_only(MixtureConfig(
        species=["H", "C"], counts=[1, 1], temperature_ev=7, rho_g_cc=0.946,
        species_parallel_jobs=1, save_data=False,
    ))
    assert result["meta"]["root_success"]
    assert {generation for generation, _ in samples_seen} == {0, 1}
    assert next(size for generation, size in samples_seen if generation == 1) == 0


def _install_binary_seed_recovery_model(
    monkeypatch,
    *,
    recovered_root=.8,
    recover=True,
    invalid_hint_residual=None,
    seed_thetas=(1., 0., -1., 2.),
    repeated_recovery=False,
):
    """Install cheap AA states whose H spectrum changes at the second seed."""
    cfg = MixtureConfig(
        species=["C", "H"], counts=[1, 1], temperature_ev=9, rho_g_cc=.946,
        species_parallel_jobs=1, save_data=False, root_maxfev=8,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    vbar = evaluator.vbar_bohr3
    evaluator.close()
    calls = {"C": [], "H": []}
    records = []
    monkeypatch.setattr(mixmod, "_initial_weight_guesses", lambda **kwargs: [
        mixmod._theta_to_weights(np.asarray([theta])) for theta in seed_thetas
    ])
    original_evaluate = mixmod._MixtureEvaluator.evaluate

    def evaluate(self, theta):
        history_before = len(self.history)
        record = original_evaluate(self, theta)
        if len(self.history) > history_before:
            records.append(record)
        return record

    def fake_full(settings):
        symbol = mixmod.element_info(settings.element).symbol
        w = .5 / (float(settings.n_i_override_bohr3) * vbar)
        theta = float(np.log(w / (1. - w)) if symbol == "C" else np.log((1. - w) / w))
        refined = bool(settings.bound_zero_tail_refine)
        calls[symbol].append((theta, refined))
        trigger = bool(recover and symbol == "H" and abs(theta) < 1e-10)
        if repeated_recovery and symbol == "H":
            trigger = True
        root = recovered_root if refined or trigger else .6
        residual = float(root - theta)
        invalid = bool(
            symbol == "H" and refined and abs(theta - 1.) < 1e-10
            and invalid_hint_residual is not None
        )
        if invalid:
            residual = float(invalid_hint_residual)
        result = _fake_species_result(
            settings, mu=-residual if symbol == "H" else 0., converged=not invalid,
        )
        if trigger and (not refined or repeated_recovery):
            result["threshold_state_refine_retry"] = {
                "applied": True,
                "shallow_l": len(calls["H"]) if repeated_recovery else 0,
            }
        return result

    monkeypatch.setattr(mixmod._MixtureEvaluator, "evaluate", evaluate)
    monkeypatch.setattr(mixmod, "solve_full_only", fake_full)
    return cfg, calls, records


@pytest.mark.parametrize("recover", [False, True])
def test_binary_seed_spectrum_recovery_revisits_only_old_coordinates(monkeypatch, recover) -> None:
    """Recovery refreshes H at the old seed; C is reusable and easy roots are unchanged."""
    cfg, calls, records = _install_binary_seed_recovery_model(monkeypatch, recover=recover)
    result = mixmod.solve_mixture_full_only(cfg)
    expected_root = .8 if recover else .6
    assert result["meta"]["root_success"]
    assert result["theta"][0] == pytest.approx(expected_root, abs=1e-4)
    assert [float(record["theta"][0]) for record in records] == pytest.approx(
        [1., 0., 1., expected_root] if recover else [1., 0., expected_root], abs=1e-4,
    )
    assert result["meta"]["root_n_seed_evals"] == (3 if recover else 2)
    assert result["meta"]["root_nfev"] == (4 if recover else 3)
    assert result["meta"]["root_representation_restarts"] == 0
    assert sum(abs(theta - 1.) < 1e-10 for theta, _ in calls["C"]) == 1
    h_initial_radius = [refined for theta, refined in calls["H"] if abs(theta - 1.) < 1e-10]
    assert h_initial_radius == ([False, True] if recover else [False])
    if recover:
        assert records[0]["threshold_refine_generation"] == 0
        assert all(record["threshold_refine_generation"] == 1 for record in records[1:])
        # The refreshed endpoint has new H physics, not its old -.4 Ha value.
        assert records[2]["mu_residual_ha"][0] == pytest.approx(-.2)


@pytest.mark.parametrize("invalid_hint_residual", [0., -10.])
def test_binary_seed_recovery_rejects_invalid_hint_and_uses_original_seeds(
    monkeypatch, invalid_hint_residual,
) -> None:
    """An invalid refreshed zero/sign is neither convergence nor a Brent endpoint."""
    cfg, _, records = _install_binary_seed_recovery_model(
        monkeypatch, recovered_root=1.5, invalid_hint_residual=invalid_hint_residual,
    )
    scipy_brent = mixmod.brentq
    brackets = []

    def brent(function, left, right, **kwargs):
        brackets.append((left, right))
        return scipy_brent(function, left, right, **kwargs)

    monkeypatch.setattr(mixmod, "brentq", brent)
    result = mixmod.solve_mixture_full_only(cfg)
    assert result["meta"]["root_success"]
    assert result["theta"][0] == pytest.approx(1.5, abs=1e-4)
    assert [float(record["theta"][0]) for record in records[:5]] == pytest.approx(
        [1., 0., 1., -1., 2.],
    )
    assert not mixmod._record_species_results_are_converged(records[2])
    assert result["meta"]["root_n_invalid_inner"] == 1
    assert result["meta"]["root_n_seed_evals"] == 5
    assert brackets == pytest.approx([(0., 2.)])


def test_binary_seed_recovery_does_not_reuse_an_old_negative_residual(monkeypatch) -> None:
    """A historical sign change disappears when both new endpoints are positive."""
    cfg, _, records = _install_binary_seed_recovery_model(
        monkeypatch, recovered_root=3., seed_thetas=(1., 0.),
    )

    def no_surrogate(*args, **kwargs):
        raise RuntimeError("Synthetic seed set has no in-range common-mu root")

    def forbidden_brent(*args, **kwargs):
        pytest.fail("A discarded historical residual manufactured a Brent bracket")

    monkeypatch.setattr(mixmod, "_surrogate_common_mu", no_surrogate)
    monkeypatch.setattr(mixmod, "brentq", forbidden_brent)
    with pytest.raises(RuntimeError, match="common-mu solve did not converge"):
        mixmod.solve_mixture_full_only(cfg)
    assert [float(record["theta"][0]) for record in records] == pytest.approx([1., 0., 1.])
    assert records[0]["mu_residual_ha"][0] < 0.
    assert all(record["mu_residual_ha"][0] > 0. for record in records[1:])


def test_binary_seed_recovery_generation_churn_stops_at_primary_budget(monkeypatch) -> None:
    """Repeated basis changes cannot create an unbounded old-coordinate cycle."""
    cfg, _, records = _install_binary_seed_recovery_model(
        monkeypatch, recovered_root=3., repeated_recovery=True,
    )
    cfg.root_maxfev = 4
    with pytest.raises(RuntimeError, match="common-mu solve did not converge"):
        mixmod.solve_mixture_full_only(cfg)
    assert len(records) == cfg.root_maxfev
    assert [float(record["theta"][0]) for record in records] == pytest.approx([1., 0., 1., 0.])
    assert [record["threshold_refine_generation"] for record in records] == [1, 2, 3, 4]


@pytest.mark.parametrize("recovered_root", [.8, 1.2])
def test_binary_brent_restarts_after_interior_spectrum_recovery(
    monkeypatch, recovered_root,
) -> None:
    """A running Brent call must not retain residuals from the old AA basis."""
    cfg = MixtureConfig(species=["C", "H"], counts=[1, 1],
                        temperature_ev=23, rho_g_cc=0.946,
                        species_parallel_jobs=1, save_data=False)
    probe = mixmod._MixtureEvaluator(cfg)
    vbar = probe.vbar_bohr3
    probe.close()
    monkeypatch.setattr(mixmod, "_initial_weight_guesses", lambda **kwargs: [
        mixmod._theta_to_weights(np.array([theta])) for theta in (0., 1.)])

    def fake_full(settings):
        symbol = mixmod.element_info(settings.element).symbol
        w = .5 / (settings.n_i_override_bohr3 * vbar)
        theta = np.log(w / (1-w)) if symbol == "C" else np.log((1-w) / w)
        recovered = symbol == "C" and (
            settings.bound_zero_tail_refine or abs(theta - .5) < 1e-10)
        mu = (recovered_root if recovered else .6) - theta if symbol == "C" else 0.
        result = _fake_species_result(settings, mu=mu, converged=True)
        if recovered and not settings.bound_zero_tail_refine:
            result["threshold_state_refine_retry"] = {"applied": True, "shallow_l": 0}
        return result

    scipy_brent = mixmod.brentq
    brackets = []

    def brent(function, left, right, **kwargs):
        brackets.append((left, right))
        if len(brackets) == 1:
            function(left)
            function(right)
            function(.75)  # Negative on the old basis, positive on the new one.
            function(.5)   # Recovery must interrupt this Brent invocation.
            pytest.fail("A changed-basis residual was returned to the old Brent call")
        return scipy_brent(function, left, right, **kwargs)

    monkeypatch.setattr(mixmod, "solve_full_only", fake_full)
    monkeypatch.setattr(mixmod, "brentq", brent)
    if recovered_root > 1.:
        # No current-basis sign bracket exists in the supplied seed set. A
        # historical negative sample must not manufacture a successful root.
        with pytest.raises(RuntimeError, match="spectral recovery invalidated"):
            mixmod.solve_mixture_full_only(cfg)
        return
    result = mixmod.solve_mixture_full_only(cfg)
    assert result["meta"]["root_success"]
    assert result["theta"][0] == pytest.approx(.8, abs=1e-4)
    assert len(brackets) == 2
    assert result["meta"]["root_representation_restarts"] == 1


def test_common_mu_evaluator_uses_full_aa_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External AA is not part of the common-chemical-potential root."""
    calls: list[tuple[str, bool]] = []

    def _fake_full(cfg_species):
        calls.append((str(cfg_species.run_mode), bool(cfg_species.ext_scf_enabled)))
        return _fake_species_result(cfg_species, mu=0.0, converged=True)

    def _forbid_external(_cfg_species):
        raise AssertionError("common-mu evaluation called external AA")

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    monkeypatch.setattr(mixmod, "solve_full_then_external", _forbid_external)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    try:
        record = evaluator.evaluate(np.asarray([0.0], dtype=float))
    finally:
        evaluator.close()

    assert mixmod._record_species_results_are_converged(record)
    assert calls == [("full", False), ("full", False)]


def test_unresolved_auto_b3_threshold_is_retried_with_a_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unresolved oscillatory auto-B3 pole gets one audited simpler retry."""
    calls: dict[str, list[str]] = {"H": [], "C": []}

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        model = str(cfg_species.b3_tail_model)
        calls[symbol].append(model)
        result = _fake_species_result(cfg_species, mu=0.0, converged=True)
        selected = "full" if model == "auto" else model
        result["n_full_tail_meta"] = {"model_selected": selected}
        result["n_cont_tail_meta"] = {"applied": False}
        result["threshold_state_status"] = (
            "unresolved" if symbol == "H" and selected == "full" else "none"
        )
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
        aa_overrides={
            "b3_tail_model": "auto",
            "b3_tail_target": "full",
        },
        root_threshold_refine_retry=False,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    try:
        record = evaluator.evaluate(np.asarray([0.0], dtype=float))
    finally:
        evaluator.close()

    assert mixmod._record_species_results_are_converged(record)
    h_result = dict(record["results"][0])
    assert h_result["threshold_state_status"] == "none"
    assert h_result["mixture_threshold_b3_a_only_retry_attempted"] is True
    assert h_result["mixture_threshold_b3_a_only_retry_selected"] is True
    assert h_result["mixture_threshold_b3_tail_meta_source"] == "n_full_tail_meta"
    assert h_result["mixture_threshold_b3_a_only_retry_initial_reasons"] == (
        "threshold_state_unresolved",
    )
    assert calls["H"] == ["auto", "a_only"]
    assert calls["C"] == ["auto"]
    assert evaluator._species_threshold_b3_a_only_retries == 1


def test_b3_tail_meta_router_prefers_target_owner_then_compatibility_fallback() -> None:
    """Full/cont targets read their own fit metadata without breaking old files."""
    result = {
        "n_full_tail_meta": {"model_selected": "full", "target": "full"},
        "n_cont_tail_meta": {"model_selected": "a_only", "target": "cont"},
    }

    meta, source = mixmod._b3_tail_meta_for_target(
        result, b3_tail_target="full"
    )
    assert source == "n_full_tail_meta"
    assert meta["model_selected"] == "full"

    meta, source = mixmod._b3_tail_meta_for_target(
        result, b3_tail_target="both"
    )
    assert source == "n_full_tail_meta"
    assert meta["model_selected"] == "full"

    meta, source = mixmod._b3_tail_meta_for_target(
        result, b3_tail_target="cont"
    )
    assert source == "n_cont_tail_meta"
    assert meta["model_selected"] == "a_only"

    meta, source = mixmod._b3_tail_meta_for_target(
        {
            "n_full_tail_meta": {"applied": False},
            "n_cont_tail_meta": {"model_selected": "full"},
        },
        b3_tail_target="full",
    )
    assert source == "n_cont_tail_meta"
    assert meta["model_selected"] == "full"


def test_explicit_full_b3_threshold_uses_root_only_a_only_surrogate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full-B3 threshold failure may guide theta without changing the model."""
    calls: dict[str, list[str]] = {"H": [], "C": []}

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        model = str(cfg_species.b3_tail_model)
        calls[symbol].append(model)
        result = _fake_species_result(cfg_species, mu=0.0, converged=True)
        result["n_full_tail_meta"] = {"model_selected": model}
        result["n_cont_tail_meta"] = {"applied": False}
        result["threshold_state_status"] = (
            "unresolved" if symbol == "H" and model == "full" else "none"
        )
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
        aa_overrides={
            "b3_tail_model": "full",
            "b3_tail_target": "full",
        },
        root_threshold_refine_retry=False,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    try:
        record = evaluator.evaluate(np.asarray([0.0], dtype=float))
    finally:
        evaluator.close()

    assert mixmod._record_species_results_are_converged(record)
    assert record["root_uses_b3_surrogate"] is True
    h_result = dict(record["results"][0])
    assert h_result["mixture_threshold_b3_a_only_retry_attempted"] is True
    assert h_result["mixture_threshold_b3_a_only_retry_selected"] is True
    assert h_result["mixture_threshold_b3_a_only_retry_role"] == "root_surrogate"
    assert h_result["mixture_threshold_b3_tail_meta_source"] == "n_full_tail_meta"
    assert calls == {"H": ["full", "a_only"], "C": ["full"]}
    # The A-only surrogate must not enter either requested-model species cache.
    assert not any(key[0] == "H" for key in evaluator._species_result_cache)
    assert evaluator._species_init_cache["H"] == []


def test_explicit_full_b3_threshold_surrogate_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The strict policy preserves the old all-full root eligibility guard."""
    calls = {"H": 0, "C": 0}

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        calls[symbol] += 1
        result = _fake_species_result(cfg_species, mu=0.0, converged=True)
        result["n_full_tail_meta"] = {"model_selected": "full"}
        result["n_cont_tail_meta"] = {"applied": False}
        result["threshold_state_status"] = (
            "unresolved" if symbol == "H" else "none"
        )
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
        aa_overrides={
            "b3_tail_model": "full",
            "b3_tail_target": "full",
        },
        root_threshold_b3_surrogate_mode="off",
        root_threshold_refine_retry=False,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    try:
        record = evaluator.evaluate(np.asarray([0.0], dtype=float))
    finally:
        evaluator.close()

    assert not mixmod._record_species_results_are_converged(record)
    h_result = dict(record["results"][0])
    assert h_result["mixture_threshold_b3_a_only_retry_attempted"] is False
    assert calls == {"H": 1, "C": 1}


def test_full_b3_root_surrogate_is_verified_with_requested_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A proxy root is returned only after an independent all-full AA solve."""
    h_full_calls = 0

    def _fake_full(cfg_species):
        nonlocal h_full_calls
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        model = str(cfg_species.b3_tail_model)
        result = _fake_species_result(cfg_species, mu=0.0, converged=True)
        result["test_b3_model"] = model
        result["n_full_tail_meta"] = {"model_selected": model}
        result["n_cont_tail_meta"] = {"applied": False}
        if symbol == "H" and model == "full":
            h_full_calls += 1
            result["threshold_state_status"] = (
                "unresolved" if h_full_calls == 1 else "none"
            )
        else:
            result["threshold_state_status"] = "none"
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
        aa_overrides={
            "b3_tail_model": "full",
            "b3_tail_target": "full",
        },
        volume_weights_init=[0.5, 0.5],
        root_threshold_refine_retry=False,
    )

    result = mixmod.solve_mixture_full_only(cfg)

    assert result["meta"]["root_threshold_b3_surrogate_used"] is True
    assert result["meta"]["root_threshold_b3_full_verification_success"] is True
    assert result["meta"]["root_threshold_b3_surrogate_retries"] == 1
    assert result["meta"]["root_threshold_b3_a_only_retries"] == 1
    assert all(
        sp["result"]["test_b3_model"] == "full"
        for sp in result["species"]
    )


def test_unverified_full_b3_root_surrogate_is_never_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A converged A-only proxy cannot cross the electronic/QOZ boundary."""

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        model = str(cfg_species.b3_tail_model)
        result = _fake_species_result(cfg_species, mu=0.0, converged=True)
        result["n_full_tail_meta"] = {"model_selected": model}
        result["n_cont_tail_meta"] = {"applied": False}
        result["threshold_state_status"] = (
            "unresolved" if symbol == "H" and model == "full" else "none"
        )
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
        aa_overrides={
            "b3_tail_model": "full",
            "b3_tail_target": "full",
        },
        volume_weights_init=[0.5, 0.5],
    )

    with pytest.raises(RuntimeError, match="surrogate was not returned"):
        mixmod.solve_mixture_full_only(cfg)


def test_species_eligibility_reasons_cover_threshold_and_nonfinite_mu() -> None:
    eligible, reasons = mixmod._species_result_eligibility({
        "stage2_converged": True,
        "mu": np.nan,
        "threshold_state_status": "unresolved",
    })
    assert not eligible
    assert reasons == ("mu_nonfinite", "threshold_state_unresolved")

    assert mixmod._species_result_eligibility({
        "stage2_converged": True,
        "mu": 0.0,
        "threshold_state_status": "marginal",
    }) == (True, ())

    # Old analytic evaluators with no production-only metadata remain usable.
    assert mixmod._species_result_eligibility({"value": 1.0}) == (True, ())


def test_production_external_eligibility_rejects_post_scf_b3() -> None:
    result = {
        "stage2_converged": True,
        "mu": 0.0,
        "threshold_state_status": "none",
        "n_scr": np.ones(8),
        "ext_status": {"enabled": True, "converged": True},
        "b3_post_self_consistent": False,
        "meta": {"b3_tail_stage2_mode": "post"},
    }

    eligible, reasons = mixmod._species_result_eligibility(
        result, require_external=True
    )

    assert not eligible
    assert reasons == ("b3_post_diagnostic_only",)


def test_production_eligibility_rejects_unapplied_requested_full_b3() -> None:
    result = {
        "stage2_converged": True,
        "mu": 0.0,
        "threshold_state_status": "none",
        "n_scr": np.ones(8),
        "n_full_tail_meta": {"applied": False},
        "n_ext_tail_meta": {"applied": False},
        "ext_status": {"enabled": True, "converged": True},
        "meta": {
            "b3_tail_stage2_mode": "in_scf",
            "b3_tail_target": "full",
        },
    }

    root_eligible, root_reasons = mixmod._species_result_eligibility(result)
    qoz_eligible, qoz_reasons = mixmod._species_result_eligibility(
        result, require_external=True
    )

    assert not root_eligible
    assert root_reasons == ("b3_full_tail_unapplied",)
    assert not qoz_eligible
    assert qoz_reasons == (
        "b3_full_tail_unapplied",
        "b3_external_tail_unapplied",
    )


def test_final_species_config_preserves_selected_a_only_threshold_retry() -> None:
    """The post-root full+external solve must not recreate the rejected tail."""
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, final_run_mode="full+ext", save_data=False,
        aa_overrides={"b3_tail_model": "auto"},
    )
    full_result = {
        "r": np.asarray([0.1, 1.0, 2.0]),
        "v_full": np.asarray([-1.0, -0.1, 0.0]),
        "mu": 0.15,
        "mixture_threshold_b3_a_only_retry_selected": True,
    }

    species_cfg = mixmod._final_species_config(
        cfg,
        element_key="H",
        r_ws_bohr=2.0,
        n_i_bohr3=3.0 / (4.0 * np.pi * 2.0**3),
        extra_overrides={},
        full_result_init=full_result,
    )

    assert species_cfg.b3_tail_model == "a_only"
    assert species_cfg.v_full_init is not None


def test_final_species_config_continues_from_accepted_root_state() -> None:
    """The external rerun must retain the finite-tolerance root state."""
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, final_run_mode="full+ext", save_data=False,
    )
    full_result = {
        "r": np.asarray([0.1, 1.0, 2.0]),
        "v_full": np.asarray([-1.0, -0.1, 0.0]),
        "mu": 0.15,
        "stage2_converged": True,
    }

    species_cfg = mixmod._final_species_config(
        cfg,
        element_key="H",
        r_ws_bohr=2.0,
        n_i_bohr3=3.0 / (4.0 * np.pi * 2.0**3),
        extra_overrides={},
        full_result_init=full_result,
        root_mu_ha=0.15,
    )

    assert species_cfg.stage1_max_iter == 0
    assert species_cfg.continuation_stage2_from_init is True
    assert species_cfg.continuation_mu_init == pytest.approx(0.15)
    assert species_cfg.full_fixed_mu_ha == pytest.approx(0.15)
    assert species_cfg.full_result_init is not full_result
    assert species_cfg.full_result_init["mu"] == pytest.approx(0.15)


@pytest.mark.parametrize(
    "marker",
    (
        "mixture_threshold_refine_retry_selected",
        "mixture_threshold_refine_latched",
    ),
)
def test_final_species_config_preserves_selected_threshold_refinement(
    marker: str,
) -> None:
    """The post-root full+external solve must retain the resolved shallow pole."""
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, final_run_mode="full+ext", save_data=False,
    )
    full_result = {
        "r": np.asarray([0.1, 1.0, 2.0]),
        "v_full": np.asarray([-1.0, -0.1, 0.0]),
        "mu": 0.25,
        marker: True,
        "mixture_threshold_refine_l_max": 1,
        "mixture_threshold_refine_cont_rmax_mult": 15.0,
    }

    species_cfg = mixmod._final_species_config(
        cfg,
        element_key="C",
        r_ws_bohr=2.0,
        n_i_bohr3=3.0 / (4.0 * np.pi * 2.0**3),
        extra_overrides={},
        full_result_init=full_result,
    )

    assert species_cfg.bound_zero_tail_refine is True
    assert species_cfg.bound_energy_cut_mode == "zero"
    assert species_cfg.bound_zero_tail_max_binding_ha >= 1.0e-2
    assert species_cfg.bound_zero_tail_l_max == 1
    assert species_cfg.rmax_mult == pytest.approx(15.0)
    assert species_cfg.n_points == 4096
    assert species_cfg.cont_rmax_mult == pytest.approx(15.0)
    assert species_cfg.cont_adaptive_mode_stage2 == "phase-root"
    assert species_cfg.stage2_max_iter >= 300
    assert species_cfg.scf_dn_tol <= 1.0e-6
    assert species_cfg.scf_dv_tol <= 1.0e-6
    assert species_cfg.stage1_max_iter == 0
    assert species_cfg.continuation_stage2_from_init is True
    assert species_cfg.continuation_mu_init == pytest.approx(0.25)
    assert species_cfg.scf_mix == pytest.approx(0.15)
    assert species_cfg.scf_mixing_w0 == pytest.approx(5.0e-4)


def test_final_species_config_keeps_screening_tail_fit_off_by_default() -> None:
    """A mixture must not silently replace one species' canonical n_scr tail."""
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=2.0,
        rho_g_cc=0.94, final_run_mode="full+ext", save_data=False,
    )

    species_cfg = mixmod._final_species_config(
        cfg,
        element_key="H",
        r_ws_bohr=2.0,
        n_i_bohr3=3.0 / (4.0 * np.pi * 2.0**3),
        extra_overrides={},
        full_result_init=None,
    )

    assert species_cfg.screening_tail_repair_mode == "off"


def test_direct_residual_hides_unresolved_threshold_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_full(cfg_species):
        result = _fake_species_result(cfg_species, mu=0.1, converged=True)
        if str(mixmod.element_info(cfg_species.element).symbol) == "H":
            result["threshold_state_status"] = "unresolved"
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    try:
        assert np.all(np.isnan(evaluator.residual(np.asarray([0.0]))))
    finally:
        evaluator.close()


def test_unconverged_species_result_is_not_used_by_mu_surrogate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed inner AA point must not enter the tabulated mu_i(V_i) data."""
    calls = {"H": 0, "C": 0}
    recorded_points: list[dict] = []
    original_record = mixmod._record_species_samples

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        calls[symbol] += 1
        volume = 1.0 / float(cfg_species.n_i_override_bohr3)
        return _fake_species_result(
            cfg_species,
            mu=float(np.log(volume) + (0.025 if symbol == "H" else 0.0)),
            converged=not (symbol == "H" and calls[symbol] == 1),
        )

    def _record_only_converged(samples, record):
        assert mixmod._record_species_results_are_converged(record)
        recorded_points.append(record)
        original_record(samples, record)

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    monkeypatch.setattr(mixmod, "_record_species_samples", _record_only_converged)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, root_maxfev=12, save_data=False,
    )
    result = solve_mixture_full(cfg)

    assert bool(result["meta"]["root_success"])
    assert recorded_points


def test_unconverged_common_mu_raises_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production mixture solves must not silently return an invalid closure."""

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        return _fake_species_result(
            cfg_species,
            mu=0.0 if symbol == "H" else 1.0,
            converged=True,
        )

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, root_maxfev=4, save_data=False,
    )
    with pytest.raises(RuntimeError, match="common-mu solve did not converge"):
        solve_mixture_full(cfg)


def test_best_effort_common_mu_requires_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Diagnostics can still request the former best-available behavior."""

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        return _fake_species_result(
            cfg_species,
            mu=0.0 if symbol == "H" else 1.0,
            converged=True,
        )

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, root_maxfev=4, allow_unconverged_root=True,
        save_data=False,
    )
    result = solve_mixture_full(cfg)
    assert not bool(result["meta"]["root_success"])
    assert np.isclose(float(result["meta"]["mu_residual_max_ha"]), 1.0)


def test_adaptive_local_root_tolerances_tighten_by_decades() -> None:
    """The default local fallback tightens only as far as its bounded floor."""
    assert mixmod._adaptive_local_root_tolerances(1.0e-4) == (
        1.0e-5,
        1.0e-6,
    )
    assert mixmod._adaptive_local_root_tolerances(1.0e-5) == (1.0e-6,)
    assert mixmod._adaptive_local_root_tolerances(1.0e-7) == ()


def test_local_refinement_stops_at_physical_mu_tolerance() -> None:
    """Do not over-solve theta after the true AA residual is acceptable."""

    class _Evaluator:
        calls = 0

        def residual(self, theta):
            self.calls += 1
            return np.asarray([5.0e-5, -2.0e-5])

    evaluator = _Evaluator()
    theta = np.asarray([0.2, -0.3])
    refined = mixmod._local_theta_refine(
        evaluator,
        theta_init=theta,
        root_tol=1.0e-6,
        max_nfev=12,
        residual_tol=1.0e-4,
    )

    np.testing.assert_array_equal(refined, theta)
    assert evaluator.calls == 1


def test_easy_multicomponent_root_skips_adaptive_refinement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A root that already meets mu_e_tol must incur no local AA evaluations."""

    def _fake_full(cfg_species):
        return _fake_species_result(cfg_species, mu=0.0, converged=True)

    def _unexpected_refinement(*args, **kwargs):
        raise AssertionError("adaptive local refinement should not run")

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    monkeypatch.setattr(mixmod, "_local_theta_refine", _unexpected_refinement)
    result = solve_mixture_full(
        MixtureConfig(
            species=["C", "H", "O"],
            counts=[1.0, 1.0, 1.0],
            temperature_ev=10.0,
            rho_g_cc=1.0,
            save_data=False,
        )
    )

    assert bool(result["meta"]["root_success"])
    assert result["meta"]["root_local_refinement_tolerances"] == []
    assert result["meta"]["root_local_refinement_nfev"] == []
    assert not bool(result["meta"]["root_local_refinement_success"])


def test_tf_three_species_root_tightens_only_after_default_stalls() -> None:
    """C10H8O4 TF common-mu recovers without globally tightening root_tol."""
    result = solve_plasma_workflow(
        PlasmaWorkflowConfig(
            elements=["C", "H", "O"],
            counts=[10.0, 8.0, 4.0],
            temperature_ev=8.617333262,
            rho_g_cc=1.3,
            electronic_model="tf",
            run_mode="full",
            show_progress=False,
        )
    )["electronic"]["result"]
    meta = result["meta"]

    assert bool(meta["root_success"])
    assert float(meta["mu_residual_max_ha"]) <= float(meta["mu_e_tol_ha"])
    assert meta["root_method"] == "tabulated_mu_adaptive_local_refine"
    tolerances = list(meta["root_local_refinement_tolerances"])
    assert tolerances[0] == pytest.approx(1.0e-5)
    assert tolerances[-1] <= 1.0e-5
    assert tolerances == sorted(tolerances, reverse=True)
    assert bool(meta["root_local_refinement_success"])


def test_binary_explicit_seed_expands_when_local_points_do_not_bracket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A binary warm start must not disable the remaining global root budget."""

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        volume = 1.0 / float(cfg_species.n_i_override_bohr3)
        # For equal fractions, theta=log(V_H/V_C).  The true root is at
        # theta=0.4, outside the explicit seed neighborhood +/-0.1 but inside
        # the bracket supplied by the broader physical seed set.
        mu = float(np.log(volume) + (0.4 if symbol == "H" else 0.0))
        return _fake_species_result(cfg_species, mu=mu, converged=True)

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"],
        counts=[1.0, 1.0],
        temperature_ev=10.0,
        rho_g_cc=1.0,
        volume_weights_init=[0.5, 0.5],
        root_maxfev=12,
        save_data=False,
    )
    result = solve_mixture_full(cfg)

    assert bool(result["meta"]["root_success"])
    assert float(result["meta"]["mu_residual_max_ha"]) <= float(cfg.mu_e_tol)
    assert int(result["meta"]["root_nfev"]) > 5
    assert result["meta"]["root_method"] == "binary_observed_bracket_mu_tolerance"


def test_binary_root_probes_valid_side_of_unconverged_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed seed may guide edge probes but must never enter the bracket."""
    root_theta = 0.70
    invalid_above = 0.80
    x_c = 0.5
    x_h = 0.5
    avg_mass = 0.5 * (
        float(mixmod.element_info("C").atomic_mass)
        + float(mixmod.element_info("H").atomic_mass)
    )
    vbar = 1.0 / float(mixmod.ion_density_bohr3(1.0, avg_mass))

    def _theta_from_species_volume(symbol: str, volume: float) -> float:
        if symbol == "C":
            w_c = float(np.clip(x_c * volume / vbar, 1.0e-12, 1.0 - 1.0e-12))
            return float(np.log(w_c / (1.0 - w_c)))
        w_h = float(np.clip(x_h * volume / vbar, 1.0e-12, 1.0 - 1.0e-12))
        return float(np.log((1.0 - w_h) / w_h))

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        volume = 1.0 / float(cfg_species.n_i_override_bohr3)
        theta = _theta_from_species_volume(symbol, volume)
        residual = root_theta - theta
        converged = not (symbol == "C" and theta > invalid_above)
        return _fake_species_result(
            cfg_species,
            mu=float(residual if symbol == "C" else 0.0),
            converged=converged,
        )

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["C", "H"],
        counts=[1.0, 1.0],
        temperature_ev=10.0,
        rho_g_cc=1.0,
        root_maxfev=20,
        root_brent_maxiter=16,
        save_data=False,
    )

    result = solve_mixture_full(cfg)

    assert bool(result["meta"]["root_success"])
    assert abs(float(result["theta"][0]) - root_theta) < 1.0e-3
    assert int(result["meta"]["root_n_invalid_inner"]) >= 1
    assert all(
        not (
            bool(row.get("root_eligible", False))
            and np.log(
                float(row["weight_C"]) / float(row["weight_H"])
            ) > invalid_above
        )
        for row in result["history"]
    )


def test_binary_seed_loop_stops_as_soon_as_a_bracket_is_observed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not spend expensive AA evaluations on unused seeds after bracketing."""

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        volume = 1.0 / float(cfg_species.n_i_override_bohr3)
        mu = float(np.log(volume) + (0.025 if symbol == "H" else 0.0))
        return _fake_species_result(cfg_species, mu=mu, converged=True)

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"],
        counts=[1.0, 1.0],
        temperature_ev=10.0,
        rho_g_cc=1.0,
        volume_weights_init=[0.5, 0.5],
        root_maxfev=12,
        save_data=False,
    )
    result = solve_mixture_full(cfg)

    assert bool(result["meta"]["root_success"])
    assert int(result["meta"]["root_n_seed_evals"]) == 2
    assert int(result["meta"]["root_nfev"]) == 3


def test_binary_brent_has_a_separate_budget_after_last_primary_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bracket found on root_maxfev must still receive Brent iterations."""
    root_theta = -2.0

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        volume = 1.0 / float(cfg_species.n_i_override_bohr3)
        mu = float(np.log(volume) + (0.0 if symbol == "H" else root_theta))
        return _fake_species_result(cfg_species, mu=mu, converged=True)

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"],
        counts=[1.0, 1.0],
        temperature_ev=10.0,
        rho_g_cc=1.0,
        root_maxfev=4,
        root_brent_maxiter=12,
        save_data=False,
    )

    result = solve_mixture_full(cfg)

    assert bool(result["meta"]["root_success"])
    assert abs(float(result["theta"][0]) - root_theta) < 2.0e-4
    assert int(result["meta"]["root_n_seed_evals"]) == 4
    assert int(result["meta"]["root_nfev"]) > int(cfg.root_maxfev)
    assert int(result["meta"]["root_brent_maxiter"]) == 12


def test_binary_brent_recovers_valid_subbracket_beside_invalid_aa_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A narrow rejected pressure-ionization interval need not hide a valid root."""
    root_theta = 0.2605
    x_c = 1.0 / 3.0
    x_h = 2.0 / 3.0
    avg_mass = (
        float(mixmod.element_info("C").atomic_mass)
        + 2.0 * float(mixmod.element_info("H").atomic_mass)
    ) / 3.0
    vbar = 1.0 / float(mixmod.ion_density_bohr3(0.94, avg_mass))

    def _theta_from_species_volume(symbol: str, volume: float) -> float:
        if symbol == "C":
            w_c = float(np.clip(x_c * volume / vbar, 1.0e-12, 1.0 - 1.0e-12))
            return float(np.log(w_c / (1.0 - w_c)))
        w_h = float(np.clip(x_h * volume / vbar, 1.0e-12, 1.0 - 1.0e-12))
        return float(np.log((1.0 - w_h) / w_h))

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        volume = 1.0 / float(cfg_species.n_i_override_bohr3)
        theta = _theta_from_species_volume(symbol, volume)
        # Positive scale with unequal endpoint values makes the first secant
        # land inside the rejected interval, while preserving one true root.
        scale = 0.363 + 0.11945 * (theta - 0.202733)
        residual = (root_theta - theta) * scale
        invalid = bool(symbol == "H" and 0.245 < theta < 0.252)
        result = _fake_species_result(
            cfg_species,
            mu=float(residual if symbol == "C" else 0.0),
            converged=not invalid,
        )
        result["threshold_state_status"] = "unresolved" if invalid else "none"
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["C", "H"],
        counts=[1.0, 2.0],
        temperature_ev=10.0,
        rho_g_cc=0.94,
        root_maxfev=8,
        root_brent_maxiter=16,
        save_data=False,
    )

    result = solve_mixture_full(cfg)

    assert bool(result["meta"]["root_success"])
    assert float(result["meta"]["mu_residual_max_ha"]) <= float(cfg.mu_e_tol)
    assert abs(float(result["theta"][0]) - root_theta) < 1.0e-3
    assert int(result["meta"]["root_n_invalid_inner"]) >= 1
    assert str(result["meta"]["root_method"]).startswith("binary_invalid_gap")


def test_final_electronic_failures_are_identified_before_qoz() -> None:
    """Both the full and external final stages are part of QOZ provenance."""
    issues = _electronic_convergence_issues([
        {
            "element": "C",
            "result": {
                "stage2_converged": False,
                "ext_status": {"converged": False},
            },
        },
        {
            "element": "H",
            "result": {
                "stage2_converged": True,
                "ext_status": {"converged": True},
            },
        },
    ])
    assert issues == [
        "C: full AA stage-2 unconverged",
        "C: external fixed-mu SCF unconverged",
    ]


def test_unresolved_threshold_state_is_identified_before_qoz() -> None:
    issues = _electronic_convergence_issues(
        [{
            "element": "H",
            "result": {
                "stage2_converged": True,
                "mu": -0.01,
                "threshold_state_status": "unresolved",
                "ext_status": {"converged": True},
            },
        }],
        require_external=True,
    )
    assert issues == ["H: unresolved threshold bound state"]


def test_diffuse_zero_tail_state_requires_self_consistent_full_b3_for_qoz() -> None:
    issues = _electronic_convergence_issues(
        [
            {
                "element": "C",
                "result": {
                    "stage2_converged": True,
                    "mu": 0.1,
                    "threshold_state_status": "resolved",
                    "threshold_state_localization": "diffuse",
                    "zero_tail_bound_meta": {"applied": True},
                    "ext_status": {"enabled": True, "converged": True},
                    "meta": {
                        "b3_tail_stage2_mode": "in_scf",
                        "b3_tail_target": "cont",
                    },
                },
            }
        ],
        require_external=True,
    )
    assert issues == [
        "C: diffuse threshold state requires a self-consistent "
        "b3_tail_target='full' calculation"
    ]


def test_raw_screening_charge_mismatch_is_identified_before_qoz() -> None:
    issues = _electronic_convergence_issues(
        [{
            "element": "H",
            "result": {
                "stage2_converged": True,
                "mu": 0.0,
                "threshold_state_status": "none",
                "ext_status": {"converged": True},
                "q_scr_all": 1.2246369664,
                "zbar_partition": 0.9989254380,
            },
        }],
        require_external=True,
        screening_charge_rel_tol=5.0e-2,
    )
    assert len(issues) == 1
    assert issues[0].startswith("H: raw screening charge relative error")
    assert "exceeds 5.000000e-02" in issues[0]


def test_final_rerun_mu_residual_is_rejected_before_qoz() -> None:
    cfg = PlasmaWorkflowConfig(
        elements=["C", "H"],
        counts=[1.0, 1.0],
        temperature_ev=10.0,
        rho_g_cc=1.0,
        ion_temperature_ev=10.0,
    )
    electronic = {
        "meta": {
            "root_success": True,
            "mu_residual_max_ha": 1.0e-8,
            "final_mu_root_success": False,
            "final_mu_residual_max_ha": 2.0e-2,
        },
        "species": [
            {"element": "C", "result": {}},
            {"element": "H", "result": {}},
        ],
    }
    with pytest.raises(RuntimeError, match=r"final full\+external rerun lost common-mu"):
        continue_plasma_workflow_from_electronic_result(
            cfg,
            electronic_kind="mixture",
            electronic_result=electronic,
        )


def test_final_unresolved_threshold_state_is_rejected_before_qoz() -> None:
    cfg = PlasmaWorkflowConfig(
        elements=["C", "H"],
        counts=[1.0, 1.0],
        temperature_ev=10.0,
        rho_g_cc=1.0,
        ion_temperature_ev=10.0,
    )
    electronic = {
        "meta": {
            "root_success": True,
            "final_mu_root_success": True,
            "final_mu_residual_max_ha": 1.0e-8,
        },
        "species": [
            {
                "element": "C",
                "result": {
                    "stage2_converged": True,
                    "mu": 0.0,
                    "threshold_state_status": "resolved",
                    "ext_status": {"converged": True},
                },
            },
            {
                "element": "H",
                "result": {
                    "stage2_converged": True,
                    "mu": 0.0,
                    "threshold_state_status": "unresolved",
                    "ext_status": {"converged": True},
                },
            },
        ],
    }
    with pytest.raises(RuntimeError, match="unresolved threshold bound state"):
        continue_plasma_workflow_from_electronic_result(
            cfg,
            electronic_kind="mixture",
            electronic_result=electronic,
        )


def test_raw_screening_charge_mismatch_is_rejected_before_qoz() -> None:
    cfg = PlasmaWorkflowConfig(
        elements=["C", "H"],
        counts=[1.0, 1.0],
        temperature_ev=10.0,
        rho_g_cc=1.0,
        ion_temperature_ev=10.0,
    )
    electronic = {
        "meta": {
            "root_success": True,
            "final_mu_root_success": True,
            "final_mu_residual_max_ha": 1.0e-8,
        },
        "species": [
            {
                "element": "C",
                "result": {
                    "stage2_converged": True,
                    "mu": 0.0,
                    "threshold_state_status": "resolved",
                    "ext_status": {"converged": True},
                    "q_scr_all": 4.0,
                    "zbar_partition": 4.0,
                },
            },
            {
                "element": "H",
                "result": {
                    "stage2_converged": True,
                    "mu": 0.0,
                    "threshold_state_status": "none",
                    "ext_status": {"converged": True},
                    "q_scr_all": 1.2246369664,
                    "zbar_partition": 0.9989254380,
                },
            },
        ],
    }
    with pytest.raises(RuntimeError, match="raw screening charge relative error"):
        continue_plasma_workflow_from_electronic_result(
            cfg,
            electronic_kind="mixture",
            electronic_result=electronic,
        )


def test_mixture_final_rerun_records_electronic_eligibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_result = {
        "r": np.asarray([0.1, 1.0]),
        "v_full": np.asarray([-1.0, 0.0]),
        "stage2_converged": True,
        "mu": 0.0,
        "threshold_state_status": "resolved",
    }
    mixture_full = {
        "mu_common_ha": 1.0e-5,
        "theta": np.asarray([0.0]),
        "volume_weights": np.asarray([0.5, 0.5]),
        "history": [],
        "species": [
            {
                "element": "H", "Z": 1, "atomic_mass": 1.0,
                "count": 1.0, "x": 0.5, "volume_bohr3": 1.0,
                "r_ws_bohr": 1.0, "mu_ha": 2.0e-5,
                "result": {**base_result, "mu": 2.0e-5},
            },
            {
                "element": "C", "Z": 6, "atomic_mass": 12.0,
                "count": 1.0, "x": 0.5, "volume_bohr3": 1.0,
                "r_ws_bohr": 1.0, "mu_ha": 0.0, "result": dict(base_result),
            },
        ],
        "meta": {"root_success": True, "mu_residual_max_ha": 0.0},
    }

    monkeypatch.setattr(mixmod, "_mixture_full_only_payload", lambda cfg: mixture_full)
    root_mu_by_z: dict[int, float] = {}

    def _fake_final_config(cfg, **kwargs):
        z = int(kwargs["element_key"])
        root_mu_by_z[z] = float(kwargs["root_mu_ha"])
        return z

    monkeypatch.setattr(mixmod, "_final_species_config", _fake_final_config)

    def _fake_final(z):
        return {
            "stage2_converged": True,
            "mu": 0.02 if int(z) == 1 else 0.0,
            "threshold_state_status": "unresolved" if int(z) == 1 else "resolved",
            "ext_status": {"converged": True},
        }

    monkeypatch.setattr(mixmod, "_solve_species_from_config", _fake_final)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, final_run_mode="full+ext", species_parallel_jobs=1,
        save_data=False,
    )
    result = mixmod.solve_mixture_full_then_ext(cfg)

    assert root_mu_by_z[1] == pytest.approx(2.0e-5)
    assert root_mu_by_z[6] == pytest.approx(0.0)
    assert not bool(result["meta"]["final_mu_root_success"])
    assert not bool(result["meta"]["final_electronic_eligible"])
    assert "H:threshold_state_unresolved" in result["meta"]["final_electronic_issues"]
    assert "mixture:final_mu_residual_above_tolerance" in result["meta"]["final_electronic_issues"]
