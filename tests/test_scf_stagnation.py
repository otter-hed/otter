"""Stagnation may request recovery but must never imply convergence."""
import numpy as np
import pytest
from otter.electronic import ks_dft as ks

from otter.electronic.ks_dft import _scf_stagnated


def history(scores):
    return [dict(dn_rel=x*1e-5, dv_rel=x*1e-5, err=x*1e-3) for x in scores]


def stagnant(rows):
    return _scf_stagnated(rows, dn_tol=1e-5, dv_tol=1e-5, tol=1e-3)


@pytest.mark.parametrize("scores,expected", [
    (np.full(80, 10.), True),
    (np.tile([10., 1000.], 40), True),
    (np.full(79, 10.), False),
    (np.full(81, 10.), False),
    (np.geomspace(1000., 2., 80), False),
    (np.full(80, .5), False),
    # Either an improving best residual or improving typical residual defers recovery.
    (np.r_[np.full(79, 10.), 2.], False),
    (np.r_[np.full(60, 10.), 10., np.full(19, 5.)], False),
])
def test_window_decision(scores, expected):
    assert stagnant(history(scores)) is expected


def test_small_mixed_updates_do_not_hide_large_unmixed_error():
    rows = [dict(dn_rel=1e-9, dv_rel=1e-9, err=.1) for _ in range(80)]
    assert stagnant(rows)
    rows[-1]["err"] = np.nan
    assert not stagnant(rows)
    assert not _scf_stagnated(history([10.]*80), dn_tol=None, dv_tol=1e-5, tol=1e-3)


@pytest.mark.parametrize("enabled", [False, True])
def test_real_inner_loop_returns_unconverged_and_preserves_disabled_budget(monkeypatch, enabled):
    # Force the detector's decision, not the density/potential calculation.
    # A tiny ideal-continuum problem checks the actual break/final-refresh path.
    monkeypatch.setattr(ks, "_scf_stagnated", lambda *a, **kw: True)
    cfg = ks.KSDTFConfig(Z=1, temperature=.5, mu=0., mu_mode="neutral",
        r_ws=1., rmax=4., n_points=48, l_list=np.array([0]), n_states=1,
        continuum_model="ideal", mu_bounds=(-10., 10.), mu_max_iter=100,
        max_iter=22, tol=1e-30, dn_tol=1e-30, dv_tol=1e-30,
        stop_on_stagnation=enabled)
    result = ks.solve_ks_dft_is(cfg)
    assert result["iters"] == (1 if enabled else cfg.max_iter)
    assert not result["converged"]
    assert result["scf_stop_reason"] == ("stagnation" if enabled else "not_converged")
    assert np.isfinite(result["mu"]) and np.all(np.isfinite(result["n_full"]))
