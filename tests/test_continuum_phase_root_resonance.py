"""Regression tests for threshold and sub-grid continuum energy features.

The phase-root scout is deliberately tested with a controlled Breit-Wigner
line shape.  That isolates energy-mesh behavior from radial Numerov and tail
matching errors while retaining the phase/density relation of a shape
resonance.  It is inspired by the supplementary resonance treatment in
Wilson et al., JQSRT 99, 658-679 (2006), but does not claim to reproduce their
relativistic Theta construction.  The Breit-Wigner phase leaves a bracketed
sign change outside the narrow peak; this regression therefore does not claim
that a finite scout mesh catches every possible rise-and-fall phase feature.
"""
from __future__ import annotations

import numpy as np
import pytest

import otter.electronic.continuum.scattering as quantum
from otter.numerics.grids import create_sqrt_grid


def _run_synthetic(monkeypatch: pytest.MonkeyPatch, evaluator, **overrides):
    grid = create_sqrt_grid(rmax=2.0, N=48)

    def _fake_scattering(*args, **kwargs):
        energy = float(args[4])
        l_max = int(args[5])
        density, phases = evaluator(energy, l_max)
        if kwargs.get("_phase_channel") is not None:
            selected = np.zeros(l_max + 1)
            selected[kwargs["_phase_channel"]] = phases[kwargs["_phase_channel"]]
            return None, selected
        return np.full_like(grid.r, float(density)), np.asarray(phases, dtype=float)

    monkeypatch.setattr(quantum, "_scattering_density_and_phase", _fake_scattering)
    options = {
        "v_eff": np.zeros_like(grid.r),
        "r": grid.r,
        "mu": 0.0,
        "temperature": 1.0,
        "e_min": 0.1,
        "e_max": 1.1,
        "l_max": 2,
        "grid_kind": "sqrt",
        "grid_step": grid.dxi,
        "l_cap_strategy": "none",
        "e_tol": 1.0e-3,
        "e_max_depth": 8,
        "e_min_width": 1.0e-3,
        "n_e_base": 5,
        "e_base_grid": "linear",
        "delta_tol": None,
        "resonance_tol": None,
        "near_zero_log_grid": False,
        "apply_occ": False,
        "energy_cache": {},
    }
    options.update(overrides)
    return quantum.continuum_density_scattering_adaptive(**options)


@pytest.mark.parametrize("adaptive_mode", ["simpson", "bisection", "phase-root"])
def test_workers_preserve_serial_adaptive_nodes_and_density(monkeypatch, adaptive_mode) -> None:
    """Worker count must change scheduling, not the discretized AA map.

    Inline workers exercise the batching path without OS-specific fork or
    pickling requirements. Real parallel AA comparisons live in the runner.
    """
    from types import SimpleNamespace

    class InlinePool:
        def __init__(self, *, processes, initializer, initargs):
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, values):
            return list(map(function, values))

    monkeypatch.setattr(quantum.mp, "get_context", lambda method: SimpleNamespace(Pool=InlinePool))

    def spectrum(energy, l_max):
        return 1.0 + np.exp(-((energy - 0.63) / 0.06)**2), np.zeros(l_max + 1)

    cache = {}
    serial, serial_meta = _run_synthetic(
        monkeypatch, spectrum, energy_cache=cache, n_jobs=1, adaptive_mode=adaptive_mode
    )
    for mode, workers, shards, policy in (
        ("batch", 2, 12, "egrid"), ("batch", 4, 12, "egrid"),
        ("shard", 2, 1, "egrid"), ("shard", 4, 12, "egrid"),
        ("shard", 2, 256, "cost"), ("shard", 4, 3, "cost"),
    ):
        parallel_cache = {}
        parallel, parallel_meta = _run_synthetic(
            monkeypatch, spectrum, energy_cache=parallel_cache,
            n_jobs=workers, adaptive_parallel_mode=mode, adaptive_mode=adaptive_mode,
            adaptive_shards=shards, adaptive_shard_policy=policy,
        )
        assert sorted(cache) == sorted(parallel_cache)
        np.testing.assert_allclose(parallel, serial, rtol=1e-14, atol=1e-14)
        assert serial_meta["quadrature_energies"] == parallel_meta["quadrature_energies"]
        np.testing.assert_allclose(serial_meta["quadrature_weights"],
                                   parallel_meta["quadrature_weights"], rtol=1e-14)


@pytest.mark.parametrize("mode", ["simpson", "bisection", "phase-root"])
def test_reused_basis_retains_the_accepted_quadrature(monkeypatch, mode):
    """Basis reuse must integrate with the rule actually tested for accuracy."""
    def spectrum(energy, l_max):
        return 1 + energy**2, np.zeros(l_max + 1)

    cache = {}
    value, meta = _run_synthetic(
        monkeypatch, spectrum, adaptive_mode=mode, energy_cache=cache, e_tol=0.1,
    )
    nodes = np.asarray(meta["quadrature_energies"])
    weights = np.asarray(meta["quadrature_weights"])
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights > 0)
    reused = weights @ np.stack([cache[e][0] for e in nodes])
    exact = 1 + (1.1**3 - 0.1**3)/3
    np.testing.assert_allclose(reused, value, rtol=1e-14)
    np.testing.assert_allclose(reused, exact, rtol=1e-14)
    # This is the former bug: merely reusing the same nodes is not enough.
    assert abs(np.trapezoid(1 + nodes**2, nodes) - exact) > 1e-5


def test_simpson_does_not_refine_arbitrary_pi_phase_sign(monkeypatch):
    def phases(energy, l_max):
        delta = np.full(l_max + 1, -0.1 if energy < 0.53 else np.pi - 0.1)
        return 1.0, delta

    _, meta = _run_synthetic(monkeypatch, phases, delta_tol=np.pi/2)
    assert meta["delta_hits"] == 0
    assert meta["max_depth"] == 0


def test_bisection_retains_a_sampled_physical_pi_wide_resonance(monkeypatch):
    def resonance(energy, l_max):
        delta = np.zeros(l_max+1)
        delta[1] = np.arctan2(0.02, 0.725-energy)
        return 1.0, delta

    _, meta = _run_synthetic(monkeypatch, resonance, adaptive_mode="bisection",
                             delta_tol=np.pi/2)
    assert meta["bisection_intervals"] > 0
    assert meta["n_windows"] > 0


def test_phase_root_scout_integrates_subgrid_l1_resonance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A narrow l=1 peak between all base nodes agrees with a dense reference."""
    e_root = 0.531234
    gamma = 1.0e-5
    strength = 0.4

    def evaluator(energy: float, l_max: int):
        lorentzian = (gamma / np.pi) / ((energy - e_root) ** 2 + gamma**2)
        phases = np.zeros(l_max + 1)
        # sin(2 delta_1) changes sign smoothly at the Breit-Wigner centre.
        phases[1] = np.arctan2(gamma, e_root - energy)
        return 1.0 + strength * lorentzian, phases

    plain, plain_meta = _run_synthetic(
        monkeypatch,
        evaluator,
        adaptive_mode="simpson",
    )
    caught, caught_meta = _run_synthetic(
        monkeypatch,
        evaluator,
        adaptive_mode="phase-root",
        resonance_theta_root_tol=1.0e-10,
        resonance_theta_refine_depth=22,
    )

    dense_e = np.linspace(0.1, 1.1, 400_001)
    dense_y = 1.0 + strength * (gamma / np.pi) / (
        (dense_e - e_root) ** 2 + gamma**2
    )
    reference = float(np.trapezoid(dense_y, dense_e))
    plain_value = float(plain[0])
    caught_value = float(caught[0])

    assert abs(plain_value - reference) / reference > 0.1
    assert abs(caught_value - reference) / reference < 3.0e-3
    assert int(caught_meta["theta_candidates"]) >= 1
    assert len(caught_meta["theta_roots"]) == 1
    assert int(caught_meta["theta_roots"][0]["l"]) == 1
    assert abs(float(caught_meta["theta_roots"][0]["energy"]) - e_root) < 1.0e-8
    assert int(caught_meta["n_eval"]) > int(plain_meta["n_eval"])


def test_phase_root_scout_rejects_broad_crossing_and_excludes_l0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Broad phase crossings and s-wave threshold crossings are not shape peaks."""
    e_root = 0.531234

    def broad_l1(energy: float, l_max: int):
        phases = np.zeros(l_max + 1)
        phases[1] = 0.5 * np.pi + 0.25 * (energy - e_root)
        return 1.0, phases

    _, broad_meta = _run_synthetic(
        monkeypatch,
        broad_l1,
        adaptive_mode="phase-root",
    )
    assert broad_meta["theta_roots"] == []
    assert bool(broad_meta["theta_fallback"])
    assert int(broad_meta["theta_rejected"]) >= 1

    def narrow_l0(energy: float, l_max: int):
        phases = np.zeros(l_max + 1)
        phases[0] = np.arctan2(1.0e-5, e_root - energy)
        return 1.0, phases

    _, s_meta = _run_synthetic(
        monkeypatch,
        narrow_l0,
        adaptive_mode="phase-root",
    )
    assert s_meta["theta_roots"] == []
    assert int(s_meta["theta_candidates"]) == 0


def test_phase_root_scout_rejects_equivalent_pi_phase_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An arbitrary sign flip of the regular solution is not a resonance."""
    e_jump = 0.531234

    def sign_flipped_solution(energy: float, l_max: int):
        phases = np.zeros(l_max + 1)
        # These phases differ by pi but describe the same S matrix.  The small
        # background phase keeps the two sides physically continuous modulo pi.
        phases[1] = -0.1 if energy < e_jump else np.pi - 0.1
        return 1.0, phases

    value, meta = _run_synthetic(
        monkeypatch,
        sign_flipped_solution,
        adaptive_mode="phase-root",
        resonance_theta_root_tol=1.0e-10,
    )

    assert float(value[0]) == pytest.approx(1.0)
    assert meta["theta_roots"] == []
    assert bool(meta["theta_fallback"])


def test_density_only_zero_potential_omits_noisy_regular_roots(monkeypatch):
    def noisy_free(energy, l_max):
        phases = np.full(l_max + 1, 1e-10*np.sin(31*energy))
        return 1.0 + 0.05*energy**2, phases

    baseline, old = _run_synthetic(monkeypatch, noisy_free, adaptive_mode="phase-root")
    actual, new = _run_synthetic(monkeypatch, noisy_free, adaptive_mode="phase-root",
                                 _density_only=True)
    np.testing.assert_array_equal(actual, baseline)
    assert old["theta_candidates"] > 0 and not old["theta_free_reference_skipped"]
    assert new["theta_free_reference_skipped"] and new["theta_root_evals"] == 0
    assert new["n_eval"] < old["n_eval"]
    for key in ("quadrature_energies", "quadrature_weights", "quadrature_panels", "theta_roots"):
        assert new[key] == old[key]


@pytest.mark.parametrize("potential", [-1e-100, 1e-100])
def test_density_only_hint_never_suppresses_nonzero_potential_resonance(monkeypatch, potential):
    # Even a tiny but nonzero V must not be classified as a free reference.
    def resonant(energy, l_max):
        phases = np.zeros(l_max + 1)
        phases[1] = np.arctan2(1e-5, 0.531234-energy)
        return 1.0 + 0.2e-5/np.pi/((energy-0.531234)**2 + 1e-10), phases

    options = dict(v_eff=np.full(48, potential), adaptive_mode="phase-root",
                   resonance_theta_root_tol=1e-10, resonance_theta_refine_depth=22)
    expected, old = _run_synthetic(monkeypatch, resonant, **options)
    actual, new = _run_synthetic(monkeypatch, resonant, **options,
                                 _density_only=True)
    np.testing.assert_array_equal(actual, expected)
    assert not new["theta_free_reference_skipped"]
    assert len(new["theta_roots"]) == 1 and new == old


@pytest.mark.parametrize("cache", [None, {}])
def test_density_wrapper_preserves_requested_transport_cache(monkeypatch, cache):
    seen = []
    def adaptive(v, r, *args, **kwargs):
        seen.append(kwargs)
        return np.ones_like(r), {}
    monkeypatch.setattr(quantum, "continuum_density_scattering_adaptive", adaptive)
    r = create_sqrt_grid(rmax=2., N=48).r
    quantum.QuantumContinuumScattering().density(r, 0., 1., params={
        "v_eff": np.zeros_like(r), "energy_cache": cache, "tail_match": False,
    })
    assert seen[0]["_density_only"] is (cache is None)
    assert seen[0]["energy_cache"] is cache


def test_partial_probe_cache_cannot_replace_density_or_transport(monkeypatch):
    def resonant(energy, l_max):
        phases = np.zeros(l_max + 1)
        phases[1] = np.arctan2(1e-5, .531234-energy)
        return 1. + .2e-5/np.pi/((energy-.531234)**2 + 1e-10), phases
    options = dict(v_eff=np.full(48, .1), adaptive_mode="phase-root",
                   resonance_theta_root_tol=1e-10, resonance_theta_refine_depth=22)
    expected, old = _run_synthetic(monkeypatch, resonant, **options)
    actual, new = _run_synthetic(monkeypatch, resonant, **options,
                                 energy_cache=None, _density_only=True)
    np.testing.assert_array_equal(actual, expected)
    assert new["n_phase_eval"] > 0 and len(new["theta_roots"]) == 1
    for key in ("quadrature_energies", "quadrature_weights", "quadrature_panels",
                "theta_roots", "theta_candidates", "theta_rejected", "theta_root_evals"):
        assert new[key] == old[key]
    # An explicit cache promises complete spectra, regardless of the hint.
    cache = {}
    complete, cached = _run_synthetic(monkeypatch, resonant, **options,
                                      energy_cache=cache, _density_only=True)
    np.testing.assert_array_equal(complete, expected)
    assert cached["n_phase_eval"] == 0 and cached["n_eval"] == old["n_eval"]
    assert all(n is not None for n, _ in cache.values())


def test_multiresolution_scout_resolves_off_anchor_even_root_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two roots hidden between coarse scouts require an explicit scan scale.

    This is the adversarial case an isolated Breit-Wigner regression does not
    exercise: the phase advances by two pi, so ``sin(2 delta)`` has the same
    sign on both sides of the pair.  A finite scout still has no arbitrary
    sub-grid guarantee, but the nested depth and achieved spacing make the
    resolved scale measurable rather than accidental.
    """
    roots = (0.514, 0.526)
    gamma = 1.0e-5
    strength_each = 0.25

    def evaluator(energy: float, l_max: int):
        phases = np.zeros(l_max + 1)
        phases[1] = sum(np.arctan2(gamma, root - energy) for root in roots)
        density = 1.0
        for root in roots:
            density += strength_each * (gamma / np.pi) / (
                (energy - root) ** 2 + gamma**2
            )
        return density, phases

    common = {
        "adaptive_mode": "phase-root",
        "resonance_theta_root_tol": 1.0e-10,
        "resonance_theta_refine_depth": 22,
        "resonance_theta_scout_max_extra_nodes": 160,
    }
    coarse, coarse_meta = _run_synthetic(
        monkeypatch,
        evaluator,
        resonance_theta_scan_depth=4,
        **common,
    )
    caught, caught_meta = _run_synthetic(
        monkeypatch,
        evaluator,
        resonance_theta_scan_depth=5,
        **common,
    )

    e_lo, e_hi = 0.1, 1.1
    line_area = sum(
        (
            np.arctan((e_hi - root) / gamma)
            - np.arctan((e_lo - root) / gamma)
        )
        / np.pi
        for root in roots
    )
    reference = (e_hi - e_lo) + strength_each * line_area

    assert len(coarse_meta["theta_roots"]) == 1
    assert abs(float(caught[0]) - reference) / reference < 4.0e-3
    assert len(caught_meta["theta_roots"]) == 2
    assert int(caught_meta["theta_scout_completed_depth"]) == 5
    assert int(caught_meta["theta_scout_extra_node_count"]) <= 160
    assert not bool(caught_meta["theta_scout_budget_exhausted"])
    assert float(caught_meta["theta_scout_min_spacing"]) > 0.0
    assert float(caught_meta["theta_scout_max_spacing"]) <= 0.25 / 32.0 + 1.0e-14
    assert caught_meta["theta_scout_limitation"] == (
        "finite_mesh_no_arbitrary_subgrid_guarantee"
    )

    _, limited_meta = _run_synthetic(
        monkeypatch,
        evaluator,
        resonance_theta_scan_depth=6,
        resonance_theta_scout_max_extra_nodes=20,
        resonance_theta_root_tol=1.0e-10,
        resonance_theta_refine_depth=22,
        adaptive_mode="phase-root",
    )
    assert int(limited_meta["theta_scout_extra_node_count"]) <= 20
    assert bool(limited_meta["theta_scout_budget_exhausted"])
    assert int(limited_meta["theta_scout_completed_depth"]) < 6


def test_coincident_multichannel_roots_share_one_usable_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coincident roots in different l channels must not collapse the panel."""
    e_root = 0.531234
    gamma = 1.0e-5
    strength = 0.4

    def evaluator(energy: float, l_max: int):
        phases = np.zeros(l_max + 1)
        phase = np.arctan2(gamma, e_root - energy)
        phases[1] = phase
        phases[2] = phase
        lorentzian = (gamma / np.pi) / ((energy - e_root) ** 2 + gamma**2)
        return 1.0 + strength * lorentzian, phases

    caught, meta = _run_synthetic(
        monkeypatch,
        evaluator,
        adaptive_mode="phase-root",
        resonance_theta_root_tol=1.0e-10,
        resonance_theta_refine_depth=22,
        resonance_theta_scan_depth=1,
    )
    e_lo, e_hi = 0.1, 1.1
    line_area = (
        np.arctan((e_hi - e_root) / gamma)
        - np.arctan((e_lo - e_root) / gamma)
    ) / np.pi
    reference = (e_hi - e_lo) + strength * line_area

    assert abs(float(caught[0]) - reference) / reference < 4.0e-3
    assert len(meta["theta_roots"]) == 2
    assert len(meta["theta_root_clusters"]) == 1
    cluster = meta["theta_root_clusters"][0]
    assert cluster["channels"] == [1, 2]
    assert int(cluster["root_count"]) == 2
    assert int(meta["n_windows"]) == 1
    assert not bool(meta["theta_fallback"])
    assert {int(item["cluster_id"]) for item in meta["theta_roots"]} == {0}


def test_phase_root_shards_keep_boundary_centred_global_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A root at an old shard edge still owns a symmetric global panel."""
    e_root = 0.6  # exactly on the requested two-shard boundary
    gamma = 2.0e-5

    def evaluator(energy: float, l_max: int):
        phases = np.zeros(l_max + 1)
        phases[1] = np.arctan2(gamma, e_root - energy)
        lorentzian = (gamma / np.pi) / ((energy - e_root) ** 2 + gamma**2)
        return 1.0 + 0.2 * lorentzian, phases

    value, meta = _run_synthetic(
        monkeypatch,
        evaluator,
        adaptive_mode="phase-root",
        n_jobs=2,
        adaptive_parallel_mode="shard",
        adaptive_shards=2,
        resonance_theta_root_tol=1.0e-10,
        resonance_theta_refine_depth=22,
    )

    assert meta["adaptive_parallel_mode_requested"] == "shard"
    assert meta["adaptive_parallel_mode"] == "shard"
    assert meta["adaptive_mesh_policy"] == "global_shared"
    assert not meta["theta_shard_mode_forced_batch"]
    assert float(value[0]) == pytest.approx(1.2, rel=3e-3)
    assert len(meta["theta_roots"]) == 1
    assert int(meta["n_windows"]) == 1


def test_near_zero_log_nodes_recover_s_wave_threshold_integral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Log anchors expose a threshold feature hidden inside the first sqrt panel."""
    e_root = 1.0e-5
    sigma = 1.0e-7
    strength = 0.4

    def threshold_feature(energy: float, l_max: int):
        phases = np.zeros(l_max + 1)
        density = 1.0 + strength * np.exp(-((energy - e_root) / sigma) ** 2) / (
            np.sqrt(np.pi) * sigma
        )
        return density, phases

    common = {
        "e_min": 1.0e-6,
        "e_max": 1.0,
        "l_max": 0,
        "n_e_base": 6,
        "e_base_grid": "sqrt",
        "e_tol": 1.0e-3,
        "e_min_width": 1.0e-8,
        "e_max_depth": 14,
        "adaptive_mode": "simpson",
    }
    missed, missed_meta = _run_synthetic(
        monkeypatch,
        threshold_feature,
        near_zero_log_grid=False,
        **common,
    )
    resolved, resolved_meta = _run_synthetic(
        monkeypatch,
        threshold_feature,
        near_zero_log_grid=True,
        near_zero_log_points_per_decade=4,
        near_zero_log_max_nodes=24,
        near_zero_log_max_energy=1.0e-2,
        **common,
    )

    # The Gaussian is over 90 widths from the lower limit and effectively all
    # of its normalized area lies inside the integration domain.
    reference = (1.0 - 1.0e-6) + strength
    assert abs(float(missed[0]) - reference) / reference > 0.1
    assert abs(float(resolved[0]) - reference) / reference < 3.0e-3
    assert 1 <= int(resolved_meta["near_zero_log_anchor_count"]) <= 24
    assert resolved_meta["near_zero_log_anchors"][0] > 1.0e-6
    # Resolving an actual 1e-7-Ha feature necessarily triggers deep local
    # refinement.  In a smooth threshold channel, the guard itself costs only
    # endpoints and one Simpson midpoint per added panel.
    def flat_threshold(energy: float, l_max: int):
        return 1.0, np.zeros(l_max + 1)

    _, flat_off = _run_synthetic(
        monkeypatch,
        flat_threshold,
        near_zero_log_grid=False,
        **common,
    )
    _, flat_on = _run_synthetic(
        monkeypatch,
        flat_threshold,
        near_zero_log_grid=True,
        near_zero_log_points_per_decade=4,
        near_zero_log_max_nodes=24,
        near_zero_log_max_energy=1.0e-2,
        **common,
    )
    extra_flat = int(flat_on["n_eval"]) - int(flat_off["n_eval"])
    assert extra_flat <= 2 * int(flat_on["near_zero_log_anchor_count"]) + 2
