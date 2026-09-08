"""A3 must approach the same infinite-energy background used by B3."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit

from otter.electronic.full_external import FullExternalConfig, _build_continuum_params
from otter.electronic.ks_dft import (
    _resolve_iteration_continuum_e_max, _resolve_iteration_continuum_l_max,
    _occupied_simpson_weights,
)


@pytest.mark.parametrize("eta", [-100.0, -4.82, 0.0, 1.0, 10.0, 1000.0])
def test_energy_cutoff_bounds_relative_ideal_density_tail(eta):
    tol = 1e-5
    params = {"e_max_mode": "prev_mu_fd", "e_max_occ_tol": tol}
    cutoff = _resolve_iteration_continuum_e_max(params, mu_ref=eta, temperature=1.0)
    # Scale out Boltzmann suppression so the dilute integral does not underflow.
    scale = np.exp(-min(eta, 0.0))
    integrand = lambda e: np.sqrt(e)*expit(eta-e)*scale
    total = quad(integrand, 0.0, cutoff, epsabs=1e-10)[0]
    tail = quad(integrand, cutoff, np.inf, epsabs=1e-12)[0]
    assert tail/(total+tail) <= tol*(1+1e-8)
    assert expit(eta-cutoff) <= tol*(1+1e-8)


def test_explicit_energy_cutoff_is_not_silently_changed():
    assert _resolve_iteration_continuum_e_max(
        {"e_max_mode": "fixed", "e_max": 12.0}, mu_ref=-10.0, temperature=3.0
    ) == 12.0


def test_dilute_a3_box_is_not_silently_capped_at_150_waves():
    cfg = FullExternalConfig(element="C", temperature_ev=100.0, rho_g_cc=0.2)
    params = _build_continuum_params(
        cfg, l_max=2, r_ws=5.436, rmax=15*5.436,
        adaptive_mode="simpson", b3_stage_mode="in_scf", e_max_mode="prev_mu_fd",
    )
    energy = _resolve_iteration_continuum_e_max(params, mu_ref=-17.7, temperature=3.675)
    angular_cap = _resolve_iteration_continuum_l_max(
        params, e_max=energy, r_eval_max=params["solve_rmax"],
    )
    assert angular_cap > 150
    assert angular_cap >= np.sqrt(2*energy)*params["solve_rmax"]
    assert params["density_rmax"] == pytest.approx(5*5.436)
    assert params["l_max_soft"] == 250


@pytest.mark.parametrize("limit", [None, 8, 250, 400])
def test_soft_angular_limit_is_configurable(limit):
    cfg = FullExternalConfig(element="C", temperature_ev=100., rho_g_cc=0.2,
                             cont_l_max_soft=limit)
    params = _build_continuum_params(
        cfg, l_max=2, r_ws=5.436, rmax=15*5.436, adaptive_mode="simpson",
        b3_stage_mode="in_scf", e_max_mode="prev_mu_fd",
    )
    assert params["l_max_soft"] == limit


@pytest.mark.parametrize("limit", [0, 7, -1, 250.5, "250"])
def test_invalid_soft_angular_limit_is_rejected(limit):
    with pytest.raises(ValueError, match="l_max_soft"):
        FullExternalConfig(element="C", temperature_ev=100., rho_g_cc=0.2,
                           cont_l_max_soft=limit)


@pytest.mark.parametrize("mode", ["adaptive", "linear"])
def test_continuum_model_forwards_the_soft_limit(monkeypatch, mode):
    from otter.electronic.continuum import scattering as sc

    def evaluate(*args, **kwargs):
        assert kwargs["l_max_soft"] == 250
        assert kwargs["partial_wave_tol"] == 1e-7
        density = np.ones_like(args[1])
        return (density, {}) if mode == "adaptive" else density

    target = "continuum_density_scattering" + ("_adaptive" if mode == "adaptive" else "")
    monkeypatch.setattr(sc, target, evaluate)
    r = np.linspace(0.1, 2., 40)
    sc.QuantumContinuumScattering().density(
        r, 0., 1., params={"v_eff": np.zeros_like(r), "energy_mode": mode,
                          "l_max_soft": 250, "tail_match": False},
    )


def test_density_domain_covers_a_user_widened_local_b3_stencil():
    cfg = FullExternalConfig(element="C", temperature_ev=100., rho_g_cc=0.2,
                             b3_tail_local_fit_width_mult=2.0)
    params = _build_continuum_params(
        cfg, l_max=2, r_ws=5.436, rmax=15*5.436, adaptive_mode="simpson",
        b3_stage_mode="in_scf", e_max_mode="prev_mu_fd",
    )
    assert params["density_rmax"] >= 6*5.436
    cfg.cont_b3_density_domain = False
    params = _build_continuum_params(
        cfg, l_max=2, r_ws=5.436, rmax=15*5.436, adaptive_mode="simpson",
        b3_stage_mode="in_scf", e_max_mode="prev_mu_fd",
    )
    assert "density_rmax" not in params


@pytest.mark.parametrize("r_ws, cap, geometry", [(0.2, None, 0.8), (5., 2., 2.)])
def test_density_domain_uses_geometry_scale_not_physical_ws(r_ws, cap, geometry):
    cfg = FullExternalConfig(element="C", temperature_ev=100., rho_g_cc=0.2,
                             geometry_r_ws_cap_bohr=cap)
    params = _build_continuum_params(
        cfg, l_max=2, r_ws=r_ws, rmax=15*geometry, adaptive_mode="simpson",
        b3_stage_mode="in_scf", e_max_mode="prev_mu_fd",
    )
    assert params["solve_rmax"] == pytest.approx(7*geometry)
    assert params["density_rmax"] == pytest.approx(5*geometry)


@pytest.mark.parametrize("temperature", [1.0, 0.01, 1e-6])
def test_basis_occupation_quadrature_resolves_a_moving_fermi_edge(temperature):
    nodes = np.array([0., 0.5, 1., 1.5, 2.])
    panels = [(0., 1.), (1., 2.)]
    for mu in [-0.4, 0.71, 1.03, 2.4]:
        weights = _occupied_simpson_weights(nodes, panels, mu, temperature)
        expected = quad(lambda e: (1 + e*e)*expit((mu-e)/temperature), 0, 2,
                        points=[np.clip(mu, 0, 2)], epsabs=1e-10)[0]
        assert weights @ (1+nodes**2) == pytest.approx(expected, abs=2e-9)


def _trace_wave_batches(monkeypatch, calls):
    from otter.electronic.continuum import scattering as sc
    for name in ("_numerov_propagate_sqrt_wbase_batch_numba", "_numerov_propagate_sqrt_wbase_suffix_numba"):
        original = getattr(sc, name)
        def traced(*args, _original=original, **kwargs):
            calls.append(len(args[4]))
            return _original(*args, **kwargs)
        monkeypatch.setattr(sc, name, traced)


def test_partial_wave_domain_keeps_inner_density_and_falls_back_if_needed(monkeypatch):
    from otter.electronic.continuum import scattering as sc
    from otter.numerics.grids import create_sqrt_grid

    grid = create_sqrt_grid(rmax=30, N=1600)
    args = (np.zeros_like(grid.r), grid.r, 0., 1., 10., 140, "sqrt", grid.dxi,
            2, .2, None, None, "r", None, 4., None, 12, "auto", .1, False, "free", 1e6)
    calls = []
    _trace_wave_batches(monkeypatch, calls)
    full, _ = sc._scattering_density_and_phase(*args, apply_occ=False)
    limited, _ = sc._scattering_density_and_phase(*args, apply_occ=False, density_rmax=6.)
    # A finite Numerov grid may also trigger the stricter phase-tail guard;
    # either a verified reduction or a complete fallback is legitimate.
    assert calls[1] < calls[0]
    np.testing.assert_allclose(limited[grid.r <= 6], full[grid.r <= 6], rtol=1e-7)
    # Deliberately propose a wrong cutoff: the actual wave-tail guard, not
    # the heuristic turning estimate, must prevent acceptance of lost charge.
    monkeypatch.setattr(sc, "_density_domain_l_cap", lambda *a: 2)
    guarded, _ = sc._scattering_density_and_phase(*args, apply_occ=False, density_rmax=6.)
    assert calls[-2:] == [3, calls[0]-3]
    np.testing.assert_allclose(guarded, full, rtol=1e-11, atol=1e-13)


def test_soft_250_limit_recovers_waves_and_respects_explicit_strategy(monkeypatch):
    from otter.electronic.continuum import scattering as sc
    from otter.numerics.grids import create_sqrt_grid

    grid = create_sqrt_grid(rmax=35, N=1600)
    args = (np.zeros_like(grid.r), grid.r, 0., 1., 50., 360, "sqrt", grid.dxi,
            2, .2, None, None, "r", None, 4., None, 12, "auto", .1, False, "free", 1e6)
    calls = []
    _trace_wave_batches(monkeypatch, calls)
    full, phases = sc._scattering_density_and_phase(*args, apply_occ=False)
    original_count = calls[-1]
    limited, limited_phases = sc._scattering_density_and_phase(
        *args, apply_occ=False, l_max_soft=250,
    )
    # l=0..250 is a trial, not a hard ceiling. High-l density near the
    # boundary forces recovery of the full automatic range in this case.
    assert original_count > 251
    assert calls[-2:] == [251, original_count-251]
    # Scalar suffix and vectorized initial batch differ only in JIT rounding;
    # compare physical amplitudes/phases far below the 1e-7 angular tolerance.
    np.testing.assert_allclose(limited, full, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(np.sin(limited_phases-phases), 0., atol=1e-10)
    # Exercise BOTH recovery levels, including an increment smaller than
    # the eight-channel tail stencil. No channel may be propagated twice.
    monkeypatch.setattr(sc, "_density_domain_l_cap", lambda *a: 252)
    perf = sc._init_scatter_perf_accum()
    recovered, recovered_phases = sc._scattering_density_and_phase(
        *args, apply_occ=False, l_max_soft=250, density_rmax=30., perf_accum=perf,
    )
    assert calls[-3:] == [251, 2, original_count-253]
    np.testing.assert_allclose(recovered, full, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(np.sin(recovered_phases-phases), 0., atol=1e-10)
    assert perf["eval_total_s"] >= sum(perf[k] for k in (
        "plan_s", "propagate_s", "match_s", "accumulate_s",
    ))
    sc._scattering_density_and_phase(
        *args, apply_occ=False, l_max_soft=250, l_cap_strategy="none",
    )
    assert calls[-1] == 361


@pytest.mark.parametrize("rescale_limit", [0., 2., 1e6])
@pytest.mark.parametrize("waves", [np.array([7, 0, 3]), np.arange(65), np.full(8, 12)])
@pytest.mark.parametrize("kernel", ["batch", "suffix"])
def test_batched_wave_rows_preserve_scalar_origin_and_rescaling(rescale_limit, waves, kernel):
    from otter.electronic.continuum import scattering as sc
    from otter.numerics.grids import create_sqrt_grid

    grid = create_sqrt_grid(rmax=20., N=512)
    geom = sc._prepare_numerov_geometry(grid.r, -6*np.exp(-grid.r)/grid.r)
    w_base = geom["r8"]*3. + geom["v_term"]
    # Recovery batches begin at nonzero physical l and can have any length.
    # Identical channels trigger simultaneous rescaling; all rows must still
    # agree with independent scalar propagation of the original recurrence.
    args = (grid.r, geom["r_quarter"], geom["inv_r"], w_base)
    rows = getattr(sc, f"_numerov_propagate_sqrt_wbase_{kernel}_numba")(
        *args, waves, grid.dxi, rescale_limit, float(geom["origin_charge"]),
    )
    assert rows.flags.c_contiguous
    assert np.all(np.isfinite(rows))
    for i, l in enumerate(waves):
        single = sc._numerov_propagate_sqrt_wbase_numba(
            *args, l, grid.dxi, rescale_limit, float(geom["origin_charge"]),
        )
        # Scalar and vectorized JIT arithmetic can round differently; compare
        # against the wave amplitude, not pointwise relative error at its nodes.
        scale = np.max(np.abs(single))
        np.testing.assert_allclose(rows[i]/scale, single/scale, rtol=0., atol=1e-12)
