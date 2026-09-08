"""Prepared angular-cap geometry must preserve density and adaptive decisions."""
import pickle

import numpy as np
import pytest

from otter.electronic.continuum import scattering as sc
from otter.numerics.grids import create_sqrt_grid


@pytest.mark.parametrize("mode", ["simpson", "bisection", "phase-root"])
def test_cap_preparation_lifetime_and_unprepared_equivalence(mode, monkeypatch):
    grid = create_sqrt_grid(rmax=8.0, N=128)
    # Loaded float64 arrays need not retain object identity through asarray.
    r = pickle.loads(pickle.dumps(grid.r))
    v = -np.exp(-r) / r
    prepare = sc._matching_l_cap_radius
    preparations = []

    def tracked(*args):
        preparations.append(args[-1].copy())
        return prepare(*args)

    monkeypatch.setattr(sc, "_matching_l_cap_radius", tracked)

    def evaluate():
        cache = {}
        n, meta = sc.continuum_density_scattering_adaptive(
            v, r, 0.1, 0.3, .001, 1.0, 8, "sqrt", grid.dxi,
            adaptive_mode=mode, n_e_base=8, e_tol=.01, e_max_depth=2,
            energy_cache=cache, match_v_tol=1e-3,
        )
        return n, meta, cache

    evaluate()
    assert len(preparations) == 1
    v *= 1.2  # unchanged array identity, a new SCF potential
    expected, meta, cache = evaluate()
    assert len(preparations) == 2
    np.testing.assert_array_equal(preparations[-1], v)
    assert not np.array_equal(preparations[0], v)

    compute = sc._compute_l_cap
    scatter = sc._scattering_density_and_phase

    def unprepared_cap(*args, **kwargs):
        kwargs.pop("_match_radius", None)
        return compute(*args, **kwargs)

    def unprepared_scatter(*args, **kwargs):
        kwargs.pop("_matching_l_cap", None)
        return scatter(*args, **kwargs)

    monkeypatch.setattr(sc, "_compute_l_cap", unprepared_cap)
    monkeypatch.setattr(sc, "_scattering_density_and_phase", unprepared_scatter)
    actual, plain_meta, plain_cache = evaluate()
    assert len(preparations) > 3
    np.testing.assert_array_equal(actual, expected)
    assert cache.keys() == plain_cache.keys()
    for energy in cache:
        for a, b in zip(cache[energy], plain_cache[energy]):
            np.testing.assert_array_equal(a, b)  # density and phase at every node
    for key in ("n_eval", "delta_hits", "resonance_hits", "theta_roots",
                "quadrature_energies", "quadrature_weights", "quadrature_panels"):
        assert meta[key] == plain_meta[key]


def test_prepared_cap_keeps_energy_constraints_and_window_variants():
    r = create_sqrt_grid(rmax=20., N=256).r
    v = np.where(np.sin(r) > .5, 0., -.1)  # disjoint potential-valid regions
    for window in [(None, None, "r"), ((40, 100), None, "index"), (None, 12., "r")]:
        match_slice, cut, mode = window
        radius = sc._matching_l_cap_radius(r, match_slice, cut, .2, mode, 4., 12, .01, v)
        for energy in [0., 1e-12, .001, .1, 1., 100.]:
            for strategy in ["match", "rmax", "none"]:
                args = (energy, 60, r, 2, match_slice, cut, .2, mode, 4., 12, 4., .01, v, strategy)
                assert sc._compute_l_cap(*args, _match_radius=radius) == sc._compute_l_cap(*args)


@pytest.mark.parametrize("input_kind", ["float32", "zero_origin"])
def test_cap_handoff_respects_numerov_input_sanitization(input_kind, monkeypatch):
    """The phase scout's input grid may differ from Numerov's float64 grid."""
    grid = create_sqrt_grid(rmax=8.0, N=128)
    r = grid.r.astype(np.float32) if input_kind == "float32" else grid.r.copy()
    if input_kind == "zero_origin":
        r[0] = 0.0
    original = sc._scattering_density_and_phase
    calls = []

    def checked(*args, **kwargs):
        # Preserve the original per-energy cap on the sanitized wave grid.
        assert kwargs.get("_matching_l_cap") is None
        calls.append(args[4])
        return original(*args, **kwargs)

    monkeypatch.setattr(sc, "_scattering_density_and_phase", checked)
    density, _ = sc.continuum_density_scattering_adaptive(
        -np.exp(-r), r, .1, .3, .001, .1, 8, "sqrt", grid.dxi,
        n_e_base=4, e_tol=.01, e_max_depth=1,
    )
    assert calls and np.all(np.isfinite(density))
