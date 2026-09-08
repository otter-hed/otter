"""Small cache-lifetime regressions without radial scattering computations."""

import gc
import weakref

import numpy as np
import pytest

import otter.electronic.continuum.scattering as quantum


@pytest.fixture
def cyclic_gc_disabled():
    """Expose reference cycles instead of depending on collection timing."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def _stub_integration(monkeypatch, *, fail_after=None):
    r = np.linspace(0.1, 1.0, 8) ** 2
    references = []

    def scattering(*args, **kwargs):
        if fail_after is not None and len(references) == fail_after:
            raise RuntimeError("synthetic scattering failure")
        energy = float(args[4])
        density = np.full_like(r, 1.0 + energy**2)
        phases = np.full(int(args[5]) + 1, 0.1 * energy)
        references.append((energy, weakref.ref(density), weakref.ref(phases)))
        return density, phases

    monkeypatch.setattr(quantum, "_scattering_density_and_phase", scattering)

    def run(**overrides):
        options = dict(
            v_eff=np.zeros_like(r), r=r, mu=0.0, temperature=1.0,
            e_min=0.1, e_max=1.1, l_max=2, l_cap_strategy="none",
            e_tol=0.01, e_max_depth=3, e_min_width=1.0e-3, n_e_base=3,
            near_zero_log_grid=False, delta_tol=None, apply_occ=False,
        )
        options.update(overrides)
        return quantum.continuum_density_scattering_adaptive(**options)

    return run, references


@pytest.mark.parametrize("mode", ["simpson", "bisection", "phase-root"])
def test_internal_cache_released_on_return(monkeypatch, cyclic_gc_disabled, mode):
    run, references = _stub_integration(monkeypatch)
    density, meta = run(adaptive_mode=mode)

    assert len(references) == meta["n_cache_total"] == meta["n_eval"]
    assert references
    assert all(ref() is None for _, *refs in references for ref in refs)
    energies = np.asarray(meta["quadrature_energies"])
    weights = np.asarray(meta["quadrature_weights"])
    np.testing.assert_allclose(density, weights @ (1.0 + energies**2), rtol=1e-15)
    np.testing.assert_allclose(density, 1.0 + (1.1**3 - 0.1**3) / 3.0, rtol=1e-15)


@pytest.mark.parametrize("mode", ["simpson", "bisection", "phase-root"])
def test_internal_cache_released_on_failure(monkeypatch, cyclic_gc_disabled, mode):
    run, references = _stub_integration(monkeypatch, fail_after=6)
    # Do not retain the exception/traceback: those intentionally own active
    # frame locals. Once discarded, no integration closure may keep the cache.
    try:
        run(adaptive_mode=mode)
    except RuntimeError as exc:
        assert str(exc) == "synthetic scattering failure"
    else:
        pytest.fail("synthetic failure was not reached")

    assert len(references) == 6
    assert all(ref() is None for _, *refs in references for ref in refs)


@pytest.mark.parametrize("mode", ["simpson", "bisection", "phase-root"])
def test_caller_cache_retains_density_phases_and_quadrature(
    monkeypatch, cyclic_gc_disabled, mode,
):
    run, references = _stub_integration(monkeypatch)
    cache = {}
    first, first_meta = run(adaptive_mode=mode, energy_cache=cache)
    second, second_meta = run(adaptive_mode=mode, energy_cache=cache)

    assert len(cache) == len(references) == first_meta["n_cache_total"]
    assert second_meta["n_eval"] == 0
    for energy, density_ref, phases_ref in references:
        assert cache[energy][0] is density_ref()
        assert cache[energy][1] is phases_ref()
        np.testing.assert_array_equal(cache[energy][0], np.full(8, 1.0 + energy**2))
        np.testing.assert_array_equal(cache[energy][1], np.full(3, 0.1 * energy))
    np.testing.assert_array_equal(first, second)
    for key in ("quadrature_energies", "quadrature_weights", "quadrature_panels"):
        assert first_meta[key] == second_meta[key]


def test_caller_cache_retained_on_failure(monkeypatch, cyclic_gc_disabled):
    run, references = _stub_integration(monkeypatch, fail_after=6)
    cache = {}
    try:
        run(energy_cache=cache)
    except RuntimeError as exc:
        assert str(exc) == "synthetic scattering failure"
    else:
        pytest.fail("synthetic failure was not reached")

    assert len(cache) == len(references) == 6
    for energy, density_ref, phases_ref in references:
        assert cache[energy][0] is density_ref()
        assert cache[energy][1] is phases_ref()
