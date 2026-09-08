"""Initial potentials must be sampled on the grid actually consumed by SCF."""
from __future__ import annotations

import numpy as np
import pytest

from otter.electronic import full_external as fe, ks_dft as ks


@pytest.mark.parametrize("r_ws,n_points", [(0.3, 128), (2.1, 4096), (10.0, 4096)])
def test_same_geometry_restart_preserves_potential(monkeypatch, r_ws, n_points):
    """Intercept the real workflow handoff, without solving any KS states.

    Include geometry-floor activation and a large domain: equal array lengths
    alone must never be interpreted as equal physical radial coordinates.
    """
    cfg = fe.FullExternalConfig(
        element="Al", temperature_ev=1.0, rho_g_cc=8.1,
        r_ws_override_bohr=r_ws, n_points=n_points,
        stage1_max_iter=0, continuation_stage2_from_init=True,
        continuation_mu_init=0.5,
    )
    rmax = fe._resolve_outer_geometry(cfg, r_ws=r_ws)["rmax"]
    grid_cfg = ks.KSDTFConfig(Z=13, temperature=1.0, mu=0.5,
                             r_ws=r_ws, rmax=rmax, n_points=n_points)
    source_r = ks._build_grid_pair(grid_cfg)[0]
    potential = -13.0 * np.exp(-source_r) / source_r
    cfg.v_full_init_r = source_r.copy()
    cfg.v_full_init = potential.copy()

    class HandoffChecked(Exception):
        pass

    def check_handoff(local):
        actual_r = ks._build_grid_pair(local)[0]
        np.testing.assert_array_equal(actual_r, source_r)
        np.testing.assert_array_equal(local.v_full_init, potential)
        raise HandoffChecked

    monkeypatch.setattr(fe, "solve_ks_dft_is", check_handoff)
    with pytest.raises(HandoffChecked):
        fe.solve_full_then_external(cfg)
    np.testing.assert_array_equal(cfg.v_full_init, potential)
    np.testing.assert_array_equal(cfg.v_full_init_r, source_r)


def test_changed_geometry_interpolates_in_physical_radius():
    """Keep interpolation/extrapolation and the no-source-grid API unchanged."""
    target = fe._target_radial_grid(rmax=20.0, n_points=128)
    source = np.linspace(1e-4, 15.0, 80)
    values = 2.0 * source - 3.0
    mapped, meta = fe._resample_initial_potential(
        values=values, source_r=source, target_r=target, name="v_full_init",
    )
    np.testing.assert_allclose(mapped, 2.0 * np.clip(target, source[0], source[-1]) - 3.0)
    assert meta["used"] and meta["interpolated"]
    direct, meta = fe._resample_initial_potential(
        values=mapped, source_r=None, target_r=target, name="v_full_init",
    )
    np.testing.assert_array_equal(direct, mapped)
    assert not np.shares_memory(direct, mapped)
    assert not meta["interpolated"]
