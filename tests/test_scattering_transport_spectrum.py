"""Reuse fixed-potential transport work without freezing mu or mutable caches."""
import numpy as np
import pytest

from otter.electronic import ks_dft as ks
from otter.electronic.continuum import scattering as scat


def _legacy_gamma(cache, mu, temperature, n_i, n0, n0_floor=0.0):
    """Pre-snapshot implementation, keeping its arithmetic and ordering."""
    n0_eff = max(float(n0), max(float(n0_floor), 0.0))
    if n_i <= 0.0 or n0_eff <= 0.0 or not cache:
        return 0.0
    energies, deltas = [], []
    for energy, (_, delta) in cache.items():
        if energy <= 0.0:
            continue
        energies.append(float(energy))
        deltas.append(np.asarray(delta, dtype=float))
    if len(energies) < 2:
        return 0.0
    order = np.argsort(energies)
    energies = np.asarray(energies, dtype=float)[order]
    deltas = [deltas[i] for i in order]
    k = np.sqrt(2.0 * energies)
    g = scat.fermi_dirac(energies, mu, temperature)
    dgde = g * (1.0 - g) / max(float(temperature), 1e-12)
    tau = np.zeros_like(k)
    for i, k_val in enumerate(k):
        sigma = scat.transport_cross_section_from_deltas(k_val, deltas[i])
        if sigma <= 0.0 or k_val <= 0.0:
            tau[i] = 0.0
        else:
            tau[i] = 1.0 / (n_i * k_val * sigma)
    integrand = (k ** 4) * dgde * tau
    sigma_dc = (1.0 / (3.0 * np.pi ** 2)) * scat._trapz(integrand, k)
    if sigma_dc <= 0.0:
        return 0.0
    tau = sigma_dc / n0_eff
    if tau <= 0.0:
        return 0.0
    return float(1.0 / tau)


def _phase_cache():
    return {
        energy: (np.empty(0), np.sin(np.arange(count, dtype=float) + energy))
        for energy, count in ((0.7, 5), (0.1, 3), (-0.2, 0), (0.4, 1), (0.25, 7), (0.0, 0))
    }


@pytest.mark.parametrize("temperature", [0.0, 1e-13, 0.1, 2.0])
@pytest.mark.parametrize("n0,n0_floor", [(0.02, 0.0), (0.0, 0.03), (-0.1, 0.0)])
def test_prepared_transport_matches_legacy_exactly_for_repeated_mus(temperature, n0, n0_floor):
    cache = _phase_cache()
    spectrum = scat._prepare_phase_shift_transport_spectrum(cache, n_i=0.01)
    np.testing.assert_array_equal(spectrum[0], [0.1, 0.25, 0.4, 0.7])
    assert all(not array.flags.writeable for array in spectrum)
    for mu in np.linspace(-1.0, 2.0, 20):
        # Both n0 and FD occupation must remain mu-dependent at evaluation time.
        n0_eval = n0 * np.exp(mu)
        expected = _legacy_gamma(cache, mu, temperature, 0.01, n0_eval, n0_floor)
        actual = scat._gamma_from_transport_spectrum(spectrum, mu, temperature, n0_eval, n0_floor)
        assert actual == expected
        assert scat.gamma_from_phase_shift_cache(
            cache, mu, temperature, 0.01, n0_eval, n0_floor,
        ) == expected


@pytest.mark.parametrize("mutation", ["add", "replace", "in_place", "remove"])
def test_public_gamma_rereads_mutated_cache_and_new_snapshot_rebuilds(mutation):
    cache = _phase_cache()
    args = dict(mu=0.3, temperature=0.2, n_i=0.01, n0=0.02)
    original = scat.gamma_from_phase_shift_cache(cache, **args)
    spectrum = scat._prepare_phase_shift_transport_spectrum(cache, args["n_i"])
    if mutation == "add":
        cache[0.9] = (np.empty(0), np.array([0.6, -0.2]))
    elif mutation == "replace":
        cache[0.25] = (np.empty(0), np.array([0.6, -0.2]))
    elif mutation == "in_place":
        cache[0.25][1][:] *= 0.3
    else:
        del cache[0.25]
    expected = _legacy_gamma(cache, **args)
    assert expected != original
    assert scat.gamma_from_phase_shift_cache(cache, **args) == expected
    assert scat._gamma_from_transport_spectrum(spectrum, 0.3, 0.2, 0.02) == original
    refreshed = scat._prepare_phase_shift_transport_spectrum(cache, args["n_i"])
    assert scat._gamma_from_transport_spectrum(refreshed, 0.3, 0.2, 0.02) == expected


@pytest.mark.parametrize("phase", [[], [0.1], [0.2, 0.2], [np.nan, 0.1], [np.inf, 0.1]])
def test_empty_zero_and_invalid_phase_behavior_is_unchanged(phase):
    cache = {energy: (np.empty(0), np.array(phase)) for energy in (0.1, 0.5)}
    with np.errstate(invalid="ignore"):
        expected = _legacy_gamma(cache, 0.2, 0.1, 0.01, 0.02)
        spectrum = scat._prepare_phase_shift_transport_spectrum(cache, 0.01)
        np.testing.assert_equal(
            scat._gamma_from_transport_spectrum(spectrum, 0.2, 0.1, 0.02), expected,
        )
        np.testing.assert_equal(
            scat.gamma_from_phase_shift_cache(cache, 0.2, 0.1, 0.01, 0.02), expected,
        )


@pytest.mark.parametrize("cache", [{}, {0.1: (np.empty(0), np.array([0.2, 0.1]))}])
@pytest.mark.parametrize("n_i", [0.0, -0.01, 0.01])
def test_missing_transport_spectrum_keeps_zero_gamma(cache, n_i):
    spectrum = scat._prepare_phase_shift_transport_spectrum(cache, n_i)
    assert spectrum is None
    assert scat._gamma_from_transport_spectrum(spectrum, 0.2, 0.1, 0.02) == 0.0
    assert scat.gamma_from_phase_shift_cache(cache, 0.2, 0.1, n_i, 0.02) == 0.0


def _stub_scf_config(**overrides):
    params = dict(
        Z=1, temperature=0.5, mu=0.0, mu_mode="neutral", mu_strategy="inner",
        r_ws=1.0, rmax=4.0, n_points=48, l_list=np.array([0]), n_states=1,
        continuum_model="scattering", compute_external=False, neutrality_mode="ws",
        mu_bounds=(-10.0, 10.0), mu_max_iter=100, max_iter=2,
        dn_tol=0.0, dv_tol=0.0, bound_spectrum_check=False,
        ion_gamma_mode="scattering", continuum_params={"energy_mode": "adaptive", "e_max": 0.7, "l_max": 1},
    )
    params.update(overrides)
    return ks.KSDTFConfig(**params)


def _empty_bound(potential, r, step, l_list, **kwargs):
    return np.empty((len(l_list), 0)), np.empty((len(l_list), r.size, 0))


@pytest.mark.parametrize("force_retry", [False, True])
def test_inner_scf_prepares_once_per_potential_retry_and_final_refresh(monkeypatch, force_retry):
    builds, preparations, evaluations = [], [], []
    prepare = ks._prepare_phase_shift_transport_spectrum
    evaluate = ks._gamma_from_transport_spectrum

    def adaptive(potential, r, *args, energy_cache, **kwargs):
        assert not energy_cache
        scale = 0.1 + 0.03 * len(builds)
        for energy in (0.1, 0.25, 0.4, 0.7):
            energy_cache[energy] = (np.full_like(r, 1.0 + energy), np.array([scale * energy, 0.0]))
        builds.append((energy_cache, potential.copy()))
        return np.zeros_like(r), {
            "quadrature_energies": [0.1, 0.4, 0.7],
            "quadrature_weights": [0.15, 0.3, 0.15],
        }

    def record_prepare(cache, n_i):
        spectrum = prepare(cache, n_i)
        assert cache is builds[-1][0]
        np.testing.assert_array_equal(spectrum[0], [0.1, 0.25, 0.4, 0.7])
        preparations.append((spectrum, cache, n_i))
        return spectrum

    def record_evaluate(spectrum, mu, temperature, n0, n0_floor=0.0):
        expected_spectrum, cache, n_i = preparations[-1]
        assert spectrum is expected_spectrum
        result = evaluate(spectrum, mu, temperature, n0, n0_floor)
        assert result == _legacy_gamma(cache, mu, temperature, n_i, n0, n0_floor)
        evaluations.append((len(preparations), mu, n0))
        return result

    def unexpected_public(*args, **kwargs):
        pytest.fail("A frozen continuum basis must reuse its prepared transport spectrum")

    monkeypatch.setattr(ks, "continuum_density_scattering_adaptive", adaptive)
    monkeypatch.setattr(ks, "solve_bound_states_sparse_numerov", _empty_bound)
    monkeypatch.setattr(ks, "_prepare_phase_shift_transport_spectrum", record_prepare)
    monkeypatch.setattr(ks, "_gamma_from_transport_spectrum", record_evaluate)
    monkeypatch.setattr(ks, "gamma_from_phase_shift_cache", unexpected_public)
    if force_retry:
        monkeypatch.setattr(ks, "_CONT_E_MAX_RETRY_MAX_TRIES", 2)
        monkeypatch.setattr(ks, "_CONT_E_MAX_RETRY_CHARGE_REL_TOL", -1.0)
    result = ks.solve_ks_dft_is(_stub_scf_config())
    expected_attempts = 5 if force_retry else 3
    assert len(builds) == len(preparations) == expected_attempts
    assert len({id(spectrum) for spectrum, _, _ in preparations}) == expected_attempts
    for attempt in range(1, expected_attempts + 1):
        samples = [(mu, n0) for count, mu, n0 in evaluations if count == attempt]
        assert len(samples) > 2
        assert len({mu for mu, _ in samples}) == len(samples)
        assert len({n0 for _, n0 in samples}) > 1
    if force_retry:
        np.testing.assert_array_equal(builds[0][1], builds[1][1])
        np.testing.assert_array_equal(builds[2][1], builds[3][1])
    assert not np.array_equal(builds[0][1], builds[-1][1])
    np.testing.assert_array_equal(result["cont_phase_energy_ha"], [0.1, 0.25, 0.4, 0.7])


def test_inner_scf_density_fallback_rereads_growing_and_mutating_phase_cache(monkeypatch):
    density_calls, gamma_calls = [], []

    def density(self, r, mu, temperature, *, params):
        cache = params["energy_cache"]
        energy = 0.1 + 0.01 * len(cache)
        cache[energy] = (np.zeros_like(r), np.array([0.4 + energy, 0.1]))
        cache[0.1][1][0] += 0.001
        density_calls.append((cache, len(cache)))
        return np.full_like(r, ks.ideal_unbound_density(mu, temperature))

    def public_gamma(cache, mu, temperature, n_i, n0, n0_floor=0.0):
        assert cache is density_calls[-1][0]
        expected = _legacy_gamma(cache, mu, temperature, n_i, n0, n0_floor)
        actual = scat.gamma_from_phase_shift_cache(cache, mu, temperature, n_i, n0, n0_floor)
        assert actual == expected
        gamma_calls.append((cache, len(cache)))
        return actual

    def unexpected_prepared(*args, **kwargs):
        pytest.fail("The mutable density fallback must not reuse a transport snapshot")

    monkeypatch.setattr(ks.QuantumContinuumScattering, "density", density)
    monkeypatch.setattr(ks, "solve_bound_states_sparse_numerov", _empty_bound)
    monkeypatch.setattr(ks, "gamma_from_phase_shift_cache", public_gamma)
    monkeypatch.setattr(ks, "_prepare_phase_shift_transport_spectrum", unexpected_prepared)
    monkeypatch.setattr(ks, "_gamma_from_transport_spectrum", unexpected_prepared)
    config = _stub_scf_config(continuum_params={
        "energy_mode": "adaptive", "adaptive_reuse_basis": False, "e_max": 0.7, "l_max": 1,
    })
    ks.solve_ks_dft_is(config)
    assert len(gamma_calls) == len(density_calls) > 6
    assert len({id(cache) for cache, _ in gamma_calls}) == 3
    assert any(count > 2 for _, count in gamma_calls)
