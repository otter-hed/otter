"""Batched channel matching preserves scalar fits and adaptive decisions."""
import numpy as np
import pytest

from otter.electronic.continuum import scattering as sc
from otter.numerics.grids import create_sqrt_grid


@pytest.mark.parametrize("tail", ["disjoint", "invalid", "coulomb", "shifted"])
def test_match_plan_preserves_scalar_windows_and_original_bessel_order(tail):
    """Batch only metadata; preserve kr relaxation, tail rejection and bases."""
    r = create_sqrt_grid(rmax=20., N=256).r
    v = np.where(np.sin(r) > .2, 0., -.1)
    mode, tol = "auto", .01
    if tail == "invalid":
        v[:] = -.1
    elif tail == "coulomb":
        v, mode, tol = -.1/r, "coulomb", None
    elif tail == "shifted":
        v, mode, tol = np.full_like(r, .03), "shifted", None
    for energy in (.001, .1, 5.):
        slices, flags, metadata, cache = sc._prepare_match_plan_for_energy(
            r, v, energy, 36, .7, None, None, "r", None,
            4., tol, 6, mode, .1, True)
        free = []
        for l in range(37):
            window, selection = sc._select_match_window(
                r, v, energy, l, None, .7, None, None, "r", 4., tol, 6)
            assert slices[l] == window and flags[l] == selection["fallback"]
            if flags[l]:
                assert metadata[l] is None
                continue
            i0, i1 = window
            expected = sc._resolve_asymptotic_basis_meta(
                r[i0:i1], v[i0:i1], energy, mode, tol or 0., .1, True)
            assert metadata[l] == expected
            if expected["kind"] == "free":
                free.append(l)
        if free or any(flags):
            i0 = min(slices[l][0] for l in free) if free else 0
            i1 = max(slices[l][1] for l in free) if free else r.size
            z = np.sqrt(2*energy)*r[i0:i1]
            assert cache["i0"] == i0
            np.testing.assert_array_equal(cache["z"], z)
            j, y = sc._free_bessel_tables_numba(z, max(free) if free else 36)
            np.testing.assert_array_equal(cache["j_tab"], j)
            np.testing.assert_array_equal(cache["y_tab"], y)
        else:
            assert cache is None


@pytest.mark.parametrize("mode", ["simpson", "bisection", "phase-root"])
@pytest.mark.parametrize("recover", [False, True])
def test_batched_matching_preserves_entire_adaptive_integral(mode, recover, monkeypatch):
    grid = create_sqrt_grid(rmax=12. if recover else 8., N=160 if recover else 128)
    original = sc._match_free_scales_batch
    calls = []

    def tracked(*args):
        result = original(*args)
        if result is not None:
            calls.append((result[0].copy(), isinstance(args[0], list),
                          args[6] if len(args) > 6 else 0))
        return result

    monkeypatch.setattr(sc, "_match_free_scales_batch", tracked)

    def evaluate():
        cache = {}
        result = sc.continuum_density_scattering_adaptive(
            -np.exp(-grid.r)/grid.r, grid.r, .1, .3, .001, 1.,
            20 if recover else 8, "sqrt", grid.dxi,
            adaptive_mode=mode, n_e_base=8, e_tol=.01, e_max_depth=2,
            energy_cache=cache, match_v_tol=1e-3,
            l_max_soft=8 if recover else None,
        )
        return *result, cache

    n, meta, cache = evaluate()
    assert calls and any(np.any(status == 0) for status, _, _ in calls)
    if recover:
        assert any(prefix and np.any(status == 0) for status, prefix, _ in calls)
        assert any(offset > 0 and np.any(status == 0) for status, _, offset in calls)
    monkeypatch.setattr(sc, "_match_free_scales_batch", lambda *args: None)
    plain_n, plain_meta, plain_cache = evaluate()
    np.testing.assert_array_equal(n, plain_n)
    assert cache.keys() == plain_cache.keys()
    for energy in cache:
        for a, b in zip(cache[energy], plain_cache[energy]):
            np.testing.assert_array_equal(a, b)
    for key in ("n_eval", "delta_hits", "resonance_hits", "theta_roots",
                "quadrature_energies", "quadrature_weights", "quadrature_panels"):
        assert meta[key] == plain_meta[key]


def test_batch_declines_nonfree_bases_and_rejects_degenerate_fit():
    r = create_sqrt_grid(rmax=8., N=128).r
    energy = .3
    k = np.sqrt(2*energy)
    windows = [(95, 128), (100, 128), (103, 128)]
    z = k*r[95:]
    j_tab, y_tab = sc._free_bessel_tables_numba(z, 2)
    cache = dict(i0=95, z=z, j_tab=j_tab, y_tab=y_tab)
    meta = [dict(kind="free", k_use=k)]*3
    waves = np.asarray([np.sin(k*r)]*3)
    waves[1] = 0.  # must reach scalar least-squares/free fallback, not be accepted
    status, phases, scales = sc._match_free_scales_batch(waves, windows, [False]*3, meta, cache, energy)
    assert status[1] != 0
    for l in (0, 2):
        override, phase, scale = sc._match_scattering_scale_preplanned(
            waves[l], r, energy, l, windows[l], False, meta[l], free_basis_cache=cache,
        )
        assert status[l] == 0 and override is None
        assert phases[l] == phase and scales[l] == scale
    assert sc._match_free_scales_batch(waves, windows, [True]*3, meta, cache, energy) is None
    for kind in ("coulomb", "shifted", "shifted_forbidden"):
        other = [dict(kind=kind, k_use=k)]*3
        assert sc._match_free_scales_batch(waves, windows, [False]*3, other, cache, energy) is None


@pytest.mark.parametrize("recover", [False, True])
def test_unsuccessful_batch_fit_uses_existing_scalar_fallback(recover, monkeypatch):
    grid = create_sqrt_grid(rmax=12. if recover else 8., N=160 if recover else 128)
    original = sc._match_free_scales_batch
    scalar = sc._match_scattering_scale_preplanned
    failures, calls = [], []

    def rejected(*args):
        result = original(*args)
        if result is not None:
            result[0][::2] = 3
            offset = args[6] if len(args) > 6 else 0
            failures.extend(offset + np.flatnonzero(result[0] != 0))
        return result

    def tracked(*args, **kwargs):
        calls.append(args[3])
        return scalar(*args, **kwargs)

    monkeypatch.setattr(sc, "_match_free_scales_batch", rejected)
    monkeypatch.setattr(sc, "_match_scattering_scale_preplanned", tracked)
    sc._scattering_density_and_phase(
        np.zeros_like(grid.r), grid.r, 0., 1., 1., 20 if recover else 4, "sqrt", grid.dxi,
        2, .2, None, None, "r", None, 4., None, 12, "auto", .1, True, "free", 1e6,
        l_max_soft=8 if recover else None,
    )
    assert failures and calls == failures


@pytest.mark.parametrize("offset", [0, 3])
@pytest.mark.parametrize("layout", ["matrix", "retained", "fragmented"])
def test_raw_batch_matches_original_bessel_rows_without_wave_copies(offset, layout, monkeypatch):
    r = create_sqrt_grid(rmax=12., N=160).r
    energy = .3
    k = np.sqrt(2*energy)
    windows = [(110 + channel, 160 - channel) for channel in range(6)]
    z = k*r[110:]
    j_tab, y_tab = sc._free_bessel_tables_numba(z, offset + 5)
    cache = dict(i0=110, z=z, j_tab=j_tab, y_tab=y_tab)
    meta = [dict(kind="free", k_use=k)]*6
    waves = np.asarray([np.sin(k*r + .1*channel) for channel in range(6)])
    if layout == "matrix":
        inputs = waves
        owners = [waves]
    else:
        # Two retained batches, each beginning at an interior row, exercise
        # recovery through more than one cap without stacking radial arrays.
        owners = [np.vstack((waves[:1], waves[:3], waves[:1])),
                  np.vstack((waves[:1], waves[3:], waves[:1]))]
        inputs = [*owners[0][1:4], *owners[1][1:4]]
        if layout == "fragmented":
            inputs[2] = inputs[2].copy()  # scalar-propagated row has no batch
    original = sc._match_free_scales_batch_numba
    blocks = []

    def tracked(*args):
        assert any(np.shares_memory(args[0], owner) for owner in owners)
        blocks.append((len(args[0]), args[-1]))
        return original(*args)

    monkeypatch.setattr(sc, "_match_free_scales_batch_numba", tracked)
    status, phases, scales = sc._match_free_scales_batch(
        inputs, windows, [False]*6, meta, cache, energy, offset,
    )
    for channel in range(6):
        if layout == "fragmented" and channel == 2:
            assert status[channel] == -1
            continue
        override, phase, scale = sc._match_scattering_scale_preplanned(
            inputs[channel], r, energy, offset + channel, windows[channel], False,
            meta[channel], free_basis_cache=cache,
        )
        assert status[channel] == 0 and override is None
        assert phases[channel] == phase and scales[channel] == scale
    expected = ([(6, offset)] if layout == "matrix" else
                [(2 if layout == "fragmented" else 3, offset), (3, offset + 3)])
    assert blocks == expected


def test_both_angular_recovery_levels_preserve_scalar_matching(monkeypatch):
    grid = create_sqrt_grid(rmax=12., N=160)
    monkeypatch.setattr(sc, "_density_domain_l_cap", lambda *args: 10)
    match = sc._match_free_scales_batch
    calls = []

    def tracked(*args):
        result = match(*args)
        if result is not None:
            calls.append((len(args[0]), isinstance(args[0], list),
                          args[6] if len(args) > 6 else 0))
        return result

    monkeypatch.setattr(sc, "_match_free_scales_batch", tracked)

    def evaluate():
        return sc._scattering_density_and_phase(
            -np.exp(-grid.r)/grid.r, grid.r, 0., 1., 1., 20, "sqrt", grid.dxi,
            2, .2, None, None, "r", None, 4., 1e-3, 12, "auto", .1, False, "free", 1e6,
            apply_occ=False, l_max_soft=8, density_rmax=12.,
        )

    actual = evaluate()
    assert [count for count, prefix, _ in calls if prefix] == [9, 11]
    assert [offset for _, _, offset in calls if offset] == [9, 11]
    monkeypatch.setattr(sc, "_match_free_scales_batch", lambda *args: None)
    expected = evaluate()
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
