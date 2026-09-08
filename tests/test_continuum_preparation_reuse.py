"""Preparation reuse must not reuse interacting states or alter quadrature."""
import numpy as np
import pytest

from otter.electronic.continuum import scattering as sc
from otter.numerics.grids import create_sqrt_grid


@pytest.mark.parametrize("l_max", [0, 1, 8, 64, 250])
def test_bessel_scratch_reuse_preserves_each_scalar_recurrence(l_max):
    # Alternate long/short Miller recurrences and include the exact zero
    # branch, evanescent high-l channels, rescaling and oscillatory arguments.
    z = np.array([500., 1e-9, 0., 1., 100., .01, 250., 1e-6, -3., 5.])
    j, y = sc._free_bessel_tables_numba(z, l_max)
    for i, x in enumerate(z):
        scalar_j, scalar_y = sc._spherical_jn_yn_all_numba(l_max, x)
        np.testing.assert_array_equal(j[:, i], scalar_j)
        np.testing.assert_array_equal(y[:, i], scalar_y)
    empty = sc._free_bessel_tables_numba(np.array([]), l_max)
    assert all(a.shape == (l_max+1, 0) for a in empty)
    # A new call cannot mutate arrays returned by an earlier call.
    saved = j.copy(), y.copy()
    sc._free_bessel_tables_numba(z[::-1].copy(), l_max)
    np.testing.assert_array_equal(j, saved[0])
    np.testing.assert_array_equal(y, saved[1])


def test_free_table_cache_keys_eviction_and_scope(monkeypatch):
    build = sc._free_bessel_tables_numba
    calls = []

    def counted(z, l):
        calls.append(l)
        return build(z, l)

    monkeypatch.setattr(sc, "_free_bessel_tables_numba", counted)
    z = np.linspace(.01, 10., 32)
    # Space for exactly one l=3 entry, including its exact input key.
    cache = sc._FreeBasisCache(z.nbytes * (1 + 2*4))
    first = cache.tables(z, 3)
    assert cache.tables(z.copy(), 3) is first
    assert calls == [3]
    for changed_z, l in [(z, 2), (np.nextafter(z, np.inf), 2), (z, 3), (z, 4)]:
        actual = cache.tables(changed_z, l)
        for a, b in zip(actual, build(changed_z, l)):
            np.testing.assert_array_equal(a, b)
        assert cache.nbytes <= cache.max_bytes
    assert calls == [3, 2, 2, 3, 4]
    assert len(cache.entries) == 1  # oversized l=4 entry was not retained
    assert not first[0].flags.writeable
    assert sc._FREE_BASIS_CACHE.get() is None
    with pytest.raises(RuntimeError), sc._free_basis_cache_scope():
        outer = sc._FREE_BASIS_CACHE.get()
        with sc._free_basis_cache_scope():
            assert sc._FREE_BASIS_CACHE.get() is outer
        raise RuntimeError("test cleanup on a failed AA")
    assert sc._FREE_BASIS_CACHE.get() is None


@pytest.mark.parametrize("mode", ["linear", "simpson", "bisection"])
def test_geometry_reused_only_within_one_potential(mode, monkeypatch):
    # Leave an actual potential-valid matching window. The old short/coarse
    # grid fell back to free waves; its apparent density change came only
    # from changing the old angular cutoff, not from reusing the potential.
    grid = create_sqrt_grid(rmax=12., N=256)
    v = -np.exp(-grid.r)/grid.r
    original = sc._scattering_density_and_phase
    seen = []

    def traced(*args, **kwargs):
        seen.append(kwargs.get("numerov_geom"))
        return original(*args, **kwargs)

    def evaluate():
        params = dict(v_eff=v, energy_mode="linear" if mode == "linear" else "adaptive",
                      adaptive_mode=mode, e_min=.001, e_max=1., n_e=12, n_e_base=8,
                      e_tol=.01, e_max_depth=2, l_max=8, tail_match=False)
        return sc.QuantumContinuumScattering().density(grid.r, .1, .3, params=params)

    monkeypatch.setattr(sc, "_scattering_density_and_phase", traced)
    with sc._free_basis_cache_scope():
        first = evaluate()
        geometry = seen[0]
        assert len(seen) > 1 and geometry is not None
        assert all(g is geometry for g in seen)
        seen.clear()
        v *= 1.2  # same array identity, different potential on next SCF iterate
        changed = evaluate()
        assert all(g is seen[0] for g in seen) and seen[0] is not geometry
    assert not np.array_equal(first, changed)

    def unprepared(*args, **kwargs):
        kwargs.pop("numerov_geom", None)
        return original(*args, **kwargs)

    monkeypatch.setattr(sc, "_scattering_density_and_phase", unprepared)
    np.testing.assert_array_equal(changed, evaluate())
