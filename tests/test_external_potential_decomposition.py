"""External Hartree reporting must subtract the correlation potential used."""
from dataclasses import replace

import numpy as np
import pytest

from otter.electronic import full_external as fe


@pytest.mark.parametrize("correlation", ["none", "zero", "native", "resampled"])
@pytest.mark.parametrize("route", ["fresh", "reuse", "full_only"])
def test_external_hartree_uses_applied_correlation(monkeypatch, correlation, route):
    cfg = fe.FullExternalConfig(
        element="H", temperature_ev=10., rho_g_cc=1., n_points=48,
        r_ws_override_bohr=1.3, full_fixed_mu_ha=.1,
        b3_tail_stage1_mode="off", b3_tail_stage2_mode="off", ext_b3_tail_mode="off",
    )
    rmax = fe._resolve_outer_geometry(cfg, r_ws=1.3)["rmax"]
    r = fe._target_radial_grid(rmax=rmax, n_points=cfg.n_points)
    source_r = np.linspace(r[0], r[-1], 17) if correlation == "resampled" else r
    potential = (.03*np.cos(source_r)*np.exp(-source_r)
                 if correlation in {"native", "resampled"} else np.zeros_like(source_r))
    if correlation != "none":
        cfg = replace(cfg, v_corr_ext=potential.copy(), v_corr_ext_r=source_r)
    expected_corr = np.interp(r, source_r, potential)
    n0 = .02
    n_ext = n0*(1+.1*np.exp(-r))
    v_xc_ext = fe.xc_potential(n_ext, model=cfg.xc_model, r=r) - fe.xc_potential(
        np.full_like(r, n0), model=cfg.xc_model, r=r,
    )
    expected_hartree = .04*np.exp(-r)
    full = dict(
        r=r, r_ws=1.3, mu=.1, n0=n0,
        g_ii=fe.IonSphereStepModel(r_ws=1.3).g_ii(r),
        n_full=np.full_like(r, n0), n_ion=np.zeros_like(r),
        n_ext=n_ext.copy(), v_ext=expected_hartree+v_xc_ext,
        v_full=-np.exp(-r)/r, converged=True, stage2_converged=True,
        threshold_state_status="resolved", cont_e_max_final=37.,
        scf_convergence_steps=cfg.scf_convergence_steps,
        history=[{"cont_e_max": 37.}],
        meta={"ws_charge_quadrature": "exact_boundary_linear"},
    )
    if route == "reuse":
        # A cached/legacy annotation must not override the actual input.
        full["v_corr_ext"] = np.full_like(r, 99.)
        cfg = replace(cfg, full_result_init=full)
    elif route == "full_only":
        cfg = replace(cfg, run_mode="full")

    def solve_external(**kwargs):
        actual = kwargs["v_corr_ext"]
        np.testing.assert_allclose(0. if actual is None else actual, expected_corr)
        return (n_ext.copy(), expected_hartree+v_xc_ext+expected_corr,
                {"converged": True, "history": [{"cont_e_max": 37.}]})

    class ReportingChecked(Exception):
        pass

    def check_reporting(result, config):
        applied = expected_corr if route != "full_only" else np.zeros_like(r)
        np.testing.assert_allclose(result["v_H_ext"], expected_hartree, atol=1e-14)
        np.testing.assert_allclose(
            result["v_H_ext"]+result["v_xc_ext"]+applied, result["v_ext"], atol=1e-14,
        )
        np.testing.assert_array_equal(result["v_full"], full["v_full"])
        np.testing.assert_array_equal(result["n_full"], full["n_full"])
        np.testing.assert_array_equal(result["n_ext"], n_ext)
        raise ReportingChecked

    monkeypatch.setattr(fe, "solve_ks_dft_is", lambda _: full.copy())
    monkeypatch.setattr(fe, "_external_fixed_mu_scf", solve_external)
    monkeypatch.setattr(fe, "_apply_paired_pseudoatom_b3_charge_closure",
                        lambda result, *args, **kwargs: (result, {}))
    monkeypatch.setattr(fe, "_final_ion_gamma_for_reporting", check_reporting)
    with pytest.raises(ReportingChecked):
        fe.solve_full_then_external(cfg)
    if correlation != "none":
        np.testing.assert_array_equal(cfg.v_corr_ext, potential)
