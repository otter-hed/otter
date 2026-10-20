"""Free angular completion preserves native wave and phase selection."""

import inspect
import pickle

import numpy as np
import pytest
from scipy.special import spherical_jn

from otter.electronic import full_external as fe
from otter.electronic import ks_dft as ks
from otter.electronic.continuum import scattering as sc
from otter.numerics.grids import create_sqrt_grid


def settings():
    grid = create_sqrt_grid(27.5, 1600, rmin=1e-5)
    return dict(
        v_eff=np.zeros_like(grid.r), r=grid.r, mu=-0.2, temperature=0.7,
        energy=0.02, l_max=2, grid_kind="sqrt", grid_step=grid.dxi,
        l_pad=2, match_fraction=0.3, match_slice=None, match_r_cut=23.4,
        match_fraction_mode="r", match_width=4.1, match_kr_min=3.0,
        match_v_tol=0.1, match_min_points=16, match_asymptotic="auto",
        match_coulomb_tol=0.1, match_allow_shift=False, match_fallback="free",
        prop_rescale_limit=1e6, apply_occ=False, l_cap_strategy="none",
    )


@pytest.mark.parametrize("cap", [0, 1, 4, 32, 257])
@pytest.mark.parametrize("energy", [0.001, 0.2, 20.0])
def test_closed_remainder_matches_independent_partial_sum(cap, energy):
    z = np.r_[0.0, np.geomspace(1e-8, 1000.0, 120)]
    partial = sum(
        (2 * ell + 1) * spherical_jn(ell, z)**2 for ell in range(cap + 1)
    )
    k = np.sqrt(2 * energy)
    actual = sc._free_channel_remainder(z / k, energy, cap)
    np.testing.assert_allclose(
        actual / (k / np.pi**2), 1 - partial, rtol=2e-11, atol=2e-13
    )


@pytest.mark.parametrize("energy", [0.0, -0.1])
def test_nonpositive_energy_has_no_free_remainder(energy):
    np.testing.assert_array_equal(
        sc._free_channel_remainder(np.arange(4.0), energy, 3), np.zeros(4)
    )


@pytest.mark.parametrize("occupied", [False, True])
def test_zero_potential_completion_preserves_phases_and_input(occupied):
    kw = settings()
    kw["apply_occ"] = occupied
    r_before, v_before = kw["r"].copy(), kw["v_eff"].copy()
    raw, phases = sc._scattering_density_and_phase(**kw)
    disabled, disabled_phases = sc._scattering_density_and_phase(
        **kw, free_completion=False
    )
    completed, completed_phases = sc._scattering_density_and_phase(
        **kw, free_completion=True
    )
    occ = sc.fermi_dirac(np.array([kw["energy"]]), kw["mu"], kw["temperature"])[0]
    expected = np.sqrt(2 * kw["energy"]) / np.pi**2 * (occ if occupied else 1)
    np.testing.assert_allclose(completed[kw["r"] >= 1], expected, rtol=2e-7)
    assert abs(raw[-1] / expected - 1) > 0.1
    np.testing.assert_array_equal(raw, disabled)
    np.testing.assert_array_equal(phases, disabled_phases)
    np.testing.assert_array_equal(phases, completed_phases)
    np.testing.assert_array_equal(kw["r"], r_before)
    np.testing.assert_array_equal(kw["v_eff"], v_before)


def test_retries_complete_only_the_final_native_angular_range(monkeypatch):
    kw = settings()
    kw.update(energy=2.0, l_max=64, l_max_soft=8,
              density_rmax=9.2, l_cap_strategy="match")
    plans, completions = [], []
    original_plan = sc._prepare_match_plan_for_energy
    original_remainder = sc._free_channel_remainder

    def plan(*args, **kwargs):
        plans.append(args[3])
        return original_plan(*args, **kwargs)

    def remainder(r, energy, cap):
        completions.append(cap)
        return original_remainder(r, energy, cap)

    monkeypatch.setattr(sc, "_prepare_match_plan_for_energy", plan)
    monkeypatch.setattr(sc, "_free_channel_remainder", remainder)
    raw, phases = sc._scattering_density_and_phase(**kw)
    raw_plans = plans.copy()
    plans.clear()
    completed, completed_phases = sc._scattering_density_and_phase(
        **kw, free_completion=True
    )
    assert len(raw_plans) > 1
    assert plans == raw_plans
    assert completions == [plans[-1]]
    np.testing.assert_allclose(
        completed, raw + original_remainder(kw["r"], kw["energy"], plans[-1]),
        rtol=0, atol=0,
    )
    np.testing.assert_array_equal(phases, completed_phases)


def test_phase_only_probe_does_not_construct_a_density(monkeypatch):
    def unexpected(*args):
        raise AssertionError("A phase-only probe must not complete density.")

    monkeypatch.setattr(sc, "_free_channel_remainder", unexpected)
    density, phases = sc._scattering_density_and_phase(
        **settings(), free_completion=True, _phase_channel=1
    )
    assert density is None
    assert np.isfinite(phases[1])


@pytest.mark.parametrize("strength", [-6.0, 3.0, -0.05])
@pytest.mark.parametrize("energy", [0.02, 0.2, 2.0])
def test_interacting_spectrum_matches_independent_free_sum(strength, energy):
    kw = settings()
    kw.update(energy=energy, l_max=8)
    kw["v_eff"] = strength * np.exp(-kw["r"]) / kw["r"]
    native, phases = sc._scattering_density_and_phase(**kw)
    completed, completed_phases = sc._scattering_density_and_phase(
        **kw, free_completion=True
    )
    k = np.sqrt(2 * energy)
    free_partial = sum(
        (2 * ell + 1) * spherical_jn(ell, k * kw["r"])**2
        for ell in range(9)
    )
    expected = native + k / np.pi**2 * (1 - free_partial)
    np.testing.assert_allclose(
        completed, expected, rtol=2e-12, atol=2e-13 * k / np.pi**2
    )
    np.testing.assert_array_equal(phases, completed_phases)


@pytest.mark.parametrize("mode", ["fixed", "basis", "adaptive"])
@pytest.mark.parametrize("jobs", [1, 2])
def test_integration_cache_and_parallel_paths(mode, jobs):
    kw = settings()
    common = {key: value for key, value in kw.items()
              if key not in ("energy", "mu", "temperature", "apply_occ")}
    common.update(free_completion=True, n_jobs=jobs)
    energies = np.linspace(0.01, 0.1, 7)

    def evaluate(n_jobs, cache, enabled=True):
        params = dict(common, n_jobs=n_jobs, energy_cache=cache,
                      free_completion=enabled)
        if mode == "basis":
            return sc.continuum_density_scattering_basis(
                **params, e_grid=energies, return_meta=True
            )
        if mode == "fixed":
            value = sc.continuum_density_scattering(
                **params, mu=-0.2, temperature=0.7, e_grid=energies
            )
            return value, {}
        return sc.continuum_density_scattering_adaptive(
            **params, mu=-0.2, temperature=0.7, e_min=0.01, e_max=0.1,
            n_e_base=4, e_tol=1e-3, e_max_depth=2, delta_tol=None,
        )

    cache = {}
    value, meta = evaluate(jobs, cache)
    reference, _ = evaluate(1, {})
    np.testing.assert_allclose(value, reference, rtol=1e-13, atol=1e-15)
    cached, cached_meta = evaluate(1, cache)
    # Parallel and serial Simpson paths accumulate accepted panels in a
    # different order; cached spectra remain identical.
    np.testing.assert_allclose(value, cached, rtol=1e-14, atol=1e-16)
    if mode != "fixed":
        assert meta["free_completion"] is True
        assert cached_meta["n_eval"] == 0
    restored = pickle.loads(pickle.dumps(cache))
    assert all(entry.free_completion for entry in restored.values())
    with pytest.raises(ValueError, match="different free_completion"):
        evaluate(1, cache, False)
    with pytest.raises(ValueError, match="different free_completion"):
        evaluate(1, {1.0: (np.zeros(2), np.zeros(2))}, True)


@pytest.mark.parametrize("enabled", [False, True])
def test_full_external_and_basis_discovery_receive_same_switch(enabled):
    cfg = fe.FullExternalConfig(
        element="Be", temperature_ev=10.0, rho_g_cc=10.0,
        cont_free_completion=enabled,
    )
    for external in (False, True):
        params = fe._build_continuum_params(
            cfg, l_max=50, r_ws=1.0, rmax=15.0,
            adaptive_mode="simpson", b3_stage_mode="in_scf",
            e_max_mode="fixed", for_external=external,
        )
        assert params["free_completion"] is enabled
        controls = ks._adaptive_reuse_spectral_controls(params)
        assert controls["free_completion"] is enabled
    assert cfg.cont_rmax_mult == 7
    assert cfg.b3_r_cut_mult == 4
    assert cfg.b3_r_fit_max_mult == 5


@pytest.mark.parametrize("value", ["false", 1, None])
def test_invalid_completion_switch_is_rejected(value):
    with pytest.raises(ValueError, match="boolean"):
        sc._scattering_density_and_phase(**settings(), free_completion=value)
    with pytest.raises(ValueError, match="boolean"):
        fe.FullExternalConfig(
            element="Be", temperature_ev=10.0, rho_g_cc=10.0,
            cont_free_completion=value,
        )


def test_tf_cannot_silently_accept_quantum_completion():
    with pytest.raises(ValueError, match="only for QM"):
        fe.FullExternalConfig(
            element="Be", temperature_ev=10.0, rho_g_cc=10.0,
            electronic_model="tf", cont_free_completion=True,
        )


def test_completion_option_preserves_existing_positional_config_arguments():
    parameters = inspect.signature(fe.FullExternalConfig).parameters
    assert parameters["cont_free_completion"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["cont_l_cap_strategy"].kind is (
        inspect.Parameter.POSITIONAL_OR_KEYWORD
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_reused_full_rejects_a_different_completion_mode(enabled):
    candidate = {
        key: np.ones(2)
        for key in ("r", "r_ws", "mu", "n0", "g_ii", "n_full", "n_ion", "v_full")
    }
    candidate.update(
        converged=True, scf_convergence_steps=3,
        meta={"cont_free_completion": not enabled,
              "ws_charge_quadrature": "exact_boundary_linear"},
    )
    cfg = fe.FullExternalConfig(
        element="Be", temperature_ev=10.0, rho_g_cc=10.0,
        cont_free_completion=enabled, full_result_init=candidate,
    )
    with pytest.raises(ValueError, match="different cont_free_completion"):
        fe.solve_full_then_external(cfg)


def test_direct_ks_rejects_completion_for_an_ideal_continuum():
    cfg = ks.KSDTFConfig(
        Z=1, temperature=0.5, mu=0.0, r_ws=1.0, continuum_model="ideal",
        continuum_params={"free_completion": True},
    )
    with pytest.raises(ValueError, match="scattering continuum"):
        ks._validate_continuum_completion(cfg)


def test_bessel_identity_is_included_in_enabled_model_citations():
    cfg = fe.FullExternalConfig(
        element="Be", temperature_ev=10.0, rho_g_cc=10.0,
        cont_free_completion=True,
    )
    assert "NISTDLMF" in cfg.citation_keys
