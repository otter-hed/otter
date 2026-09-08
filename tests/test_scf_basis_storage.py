"""SCF-owned density storage changes must leave quadrature and phases intact."""
import weakref

import numpy as np
import pytest

from otter.electronic import ks_dft as ks


def test_adaptive_basis_consumes_density_allocations_but_keeps_all_phases():
    cache = {
        energy: (np.arange(16, dtype=float) + energy, np.array([energy, energy / 3]))
        for energy in (0.1, 0.2, 0.3, 0.4)
    }
    original_rows = [weakref.ref(entry[0]) for entry in cache.values()]
    phases = {energy: entry[1] for energy, entry in cache.items()}
    # Preserve the supplied order rather than sorting or using cache order.
    energies = np.array([0.4, 0.1, 0.3])
    expected = np.vstack([cache[float(energy)][0] for energy in energies])
    weights = np.array([0.3, 0.1, 0.2])
    gamma_args = dict(mu=0.2, temperature=0.1, n_i=0.01, n0=0.02)
    expected_gamma = ks.gamma_from_phase_shift_cache(cache, **gamma_args)

    basis = ks._consume_adaptive_scf_basis(cache, energies)

    np.testing.assert_array_equal(basis, expected)
    np.testing.assert_array_equal(
        ks._weighted_energy_sum_numba(basis, weights),
        ks._weighted_energy_sum_numba(expected, weights),
    )
    assert basis.flags.c_contiguous
    assert all(row() is None for row in original_rows)
    assert list(cache) == [0.1, 0.2, 0.3, 0.4]
    assert cache[0.2][0].size == 0  # Scout density discarded, not its phase.
    for index, energy in enumerate(energies):
        assert np.shares_memory(cache[float(energy)][0], basis[index])
    assert all(cache[energy][1] is phase for energy, phase in phases.items())
    assert ks.gamma_from_phase_shift_cache(cache, **gamma_args) == expected_gamma

    basis_ref = weakref.ref(basis)
    ks._release_scf_cache_densities(cache)
    del basis
    assert basis_ref() is None  # A returned attempt cannot pin the dense basis.
    assert all(entry[0].size == 0 for entry in cache.values())
    assert all(cache[energy][1] is phase for energy, phase in phases.items())
    assert ks.gamma_from_phase_shift_cache(cache, **gamma_args) == expected_gamma
    ks._release_scf_cache_densities(None)


@pytest.mark.parametrize("gamma_mode", ["fixed", "scattering"])
@pytest.mark.parametrize("use_panels", [False, True])
def test_inner_scf_compaction_preserves_densities_phases_and_caller_cache(
    monkeypatch, gamma_mode, use_panels,
):
    caches = []
    energies = np.array([0.1, 0.4, 0.7])
    weights = np.array([0.15, 0.3, 0.15])

    def adaptive(potential, r, *args, energy_cache, **kwargs):
        assert not energy_cache
        for energy in (0.1, 0.25, 0.4, 0.7):
            energy_cache[energy] = (
                np.full_like(r, 1.0 + energy),
                np.array([0.5 * energy, 0.2 * energy]),
            )
        caches.append(energy_cache)
        meta = {
            "quadrature_energies": energies.tolist(),
            "quadrature_weights": weights.tolist(),
        }
        if use_panels:
            meta["quadrature_panels"] = [[0.1, 0.7]]
        return np.zeros_like(r), meta

    def empty_bound(potential, r, step, l_list, **kwargs):
        return np.empty((len(l_list), 0)), np.empty((len(l_list), r.size, 0))

    monkeypatch.setattr(ks, "continuum_density_scattering_adaptive", adaptive)
    monkeypatch.setattr(ks, "solve_bound_states_sparse_numerov", empty_bound)
    caller_density = np.arange(8, dtype=float)
    caller_phase = np.array([0.2, 0.1])
    caller_cache = {9.0: (caller_density, caller_phase)}
    config = ks.KSDTFConfig(
        Z=1, temperature=0.5, mu=0.0, mu_mode="neutral", mu_strategy="inner",
        r_ws=1.0, rmax=4.0, n_points=48, l_list=np.array([0]), n_states=1,
        continuum_model="scattering", compute_external=True, neutrality_mode="ws",
        mu_bounds=(-10.0, 10.0), mu_max_iter=100, max_iter=1,
        bound_spectrum_check=False, ion_gamma_mode=gamma_mode,
        continuum_params={
            "energy_mode": "adaptive", "e_max": 0.7, "l_max": 1,
            "energy_cache": caller_cache,
        },
    )
    actual = ks.solve_ks_dft_is(config)
    # Full and external caches are built in both the iteration and final refresh.
    assert len(caches) == 4
    assert all(list(cache) == [0.1, 0.25, 0.4, 0.7] for cache in caches)
    if gamma_mode == "scattering":
        assert all(entry[0].size == 0 for cache in caches for entry in cache.values())
        np.testing.assert_array_equal(actual["cont_phase_energy_ha"], [0.1, 0.25, 0.4, 0.7])
    assert list(caller_cache) == [9.0]
    assert caller_cache[9.0][0] is caller_density
    assert caller_cache[9.0][1] is caller_phase
    np.testing.assert_array_equal(caller_density, np.arange(8, dtype=float))

    # Compare the real inner-mu/assembly control flow against its old storage.
    monkeypatch.setattr(ks, "_consume_adaptive_scf_basis", lambda cache, nodes:
                        np.vstack([np.asarray(cache[float(e)][0], dtype=float) for e in nodes]))
    monkeypatch.setattr(ks, "_release_scf_cache_densities", lambda cache: None)
    expected = ks.solve_ks_dft_is(config)
    keys = ["mu", "n_full", "n_cont", "n_ext", "n_pa", "n_scr", "ion_gamma", "charge_ws"]
    if gamma_mode == "scattering":
        keys.extend(["cont_phase_energy_ha", "cont_phase_shift_rad"])
    for key in keys:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


def test_outer_brent_returns_cached_root_without_another_scf(monkeypatch):
    calls = []
    results = {}

    def fixed_mu(config, mu, **kwargs):
        calls.append(mu)
        result = {
            "mu": mu, "charge_ws": config.Z + mu,
            "v_full": np.array([mu]), "v_ext": np.array([0.0]),
        }
        results[mu] = result
        return result

    monkeypatch.setattr(ks, "_scf_fixed_mu", fixed_mu)
    config = ks.KSDTFConfig(
        Z=1, temperature=0.5, mu=0.2, mu_mode="neutral", mu_strategy="outer",
        mu_solver="brent", mu_bounds=(-1.0, 1.0),
    )
    result = ks.solve_ks_dft_is(config)
    assert calls == [-1.0, 1.0, 0.0]
    assert result is results[0.0]
    assert [entry["mu"] for entry in result["mu_history"]] == [-1.0, 1.0, 0.0]


def test_outer_brent_evaluates_an_uncached_returned_root(monkeypatch):
    import scipy.optimize

    calls = []

    def fixed_mu(config, mu, **kwargs):
        calls.append(mu)
        return {"charge_ws": config.Z + mu, "v_full": np.zeros(1), "v_ext": np.zeros(1)}

    monkeypatch.setattr(ks, "_scf_fixed_mu", fixed_mu)
    monkeypatch.setattr(scipy.optimize, "brentq", lambda *args, **kwargs: 0.0)
    config = ks.KSDTFConfig(
        Z=1, temperature=0.5, mu=0.2, mu_mode="neutral", mu_strategy="outer",
        mu_solver="brent", mu_bounds=(-1.0, 1.0),
    )
    result = ks.solve_ks_dft_is(config)
    assert calls == [-1.0, 1.0, 0.0]
    assert result["charge_ws"] == config.Z
