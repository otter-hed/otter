"""Regression tests for interacting continuum states below the old kr gate.

The asymptotic free/Coulomb basis is mathematically defined at small ``k r``.
``match_kr_min`` is therefore a conditioning preference, not a reason to
replace a propagated interacting state by a free wave.  The latter used to
erase the low-energy continuum precisely where a weak bound state crosses the
continuum threshold.
"""

from __future__ import annotations

import numpy as np
import pytest

from otter.electronic.continuum import scattering as qmod
from otter.numerics.grids import create_sqrt_grid


def _phase_error_mod_pi(delta: float, reference: float) -> float:
    """Return the absolute scattering-phase error modulo pi."""
    return float(abs(0.5 * np.angle(np.exp(2.0j * (delta - reference)))))


@pytest.mark.parametrize("energy", [1e-6, 0.5])
@pytest.mark.parametrize("tail", ["fragmented", "rejected", "potential_only",
                                  "coulomb", "shifted", "unconstrained"])
def test_match_plan_agrees_with_scalar_windows_and_basis(energy, tail):
    """Batching must preserve kr relaxation, potential rejection and basis unions."""
    # At E=0.5, integer radii lie exactly on kr=l+1 boundaries. The final
    # one-point valid run is too short and must not hide the earlier long run.
    r = np.arange(1., 21.)
    v = np.full_like(r, 0.05)
    if tail in ("fragmented", "potential_only"):
        v[2:7] = v[10:17] = v[19:] = 0.
    elif tail == "coulomb":
        v = -0.2/r
    elif tail == "shifted":
        v = 0.1 + 0.005*r
    elif tail == "unconstrained":
        v = None
    options = dict(
        match_fraction=0.2, match_slice=(1, 20), match_r_cut=None,
        match_fraction_mode="r", match_width=None, match_kr_min=4.,
        match_v_tol=1e-4, match_min_points=3, match_asymptotic="auto",
        match_coulomb_tol=0.01, match_allow_shift=True,
    )
    if tail in ("coulomb", "shifted", "unconstrained"):
        options["match_v_tol"] = None
    if tail in ("potential_only", "unconstrained"):
        options["match_kr_min"] = None
    slices, flags, bases, cache = qmod._prepare_match_plan_for_energy(
        r, v, energy, 24, **options,
    )
    window_options = {k: value for k, value in options.items() if k not in (
        "match_asymptotic", "match_coulomb_tol", "match_allow_shift",
    )}
    for l in range(25):
        window, meta = qmod._select_match_window(r, v, energy, l, **window_options)
        assert slices[l] == window
        assert flags[l] == meta["fallback"]
        i0, i1 = window
        expected = None if meta["fallback"] else qmod._resolve_asymptotic_basis_meta(
            r[i0:i1], None if v is None else v[i0:i1], energy, "auto",
            options["match_v_tol"] or 0., 0.01, True,
        )
        assert bases[l] == expected

    free_ls = [l for l, basis in enumerate(bases) if basis and basis["kind"] == "free"]
    if free_ls:
        i0 = min(slices[l][0] for l in free_ls)
        i1 = max(slices[l][1] for l in free_ls)
        last_l = max(free_ls)
    elif any(flags):
        i0, i1, last_l = 0, r.size, 24
    else:
        assert cache is None
        return
    z = np.sqrt(2*energy)*r[i0:i1]
    j, y = qmod._free_bessel_tables_numba(z, last_l)
    assert cache["i0"] == i0
    np.testing.assert_array_equal(cache["z"], z)
    np.testing.assert_array_equal(cache["j_tab"], j)
    np.testing.assert_array_equal(cache["y_tab"], y)


def _match_l0(
    r: np.ndarray,
    v_eff: np.ndarray,
    energy: float,
    dxi: float,
) -> tuple[np.ndarray, float, dict]:
    u_raw = qmod._numerov_propagate_sqrt(r, v_eff, energy, 0, dxi)
    u_norm, delta, _, meta = qmod._match_scattering_u(
        u_raw,
        r,
        v_eff,
        energy,
        0,
        match_fraction=0.4,
        match_slice=None,
        match_r_cut=0.6 * float(r[-1]),
        match_fraction_mode="r",
        match_width=None,
        match_kr_min=3.0,
        match_v_tol=1.0e-8,
        match_min_points=16,
        match_asymptotic="auto",
        match_coulomb_tol=0.1,
        match_allow_shift=True,
        match_fallback="free",
    )
    return u_norm, float(delta), dict(meta)


def test_low_energy_free_state_is_matched_without_fallback() -> None:
    grid = create_sqrt_grid(rmax=30.0, N=500)
    r = np.asarray(grid.r, dtype=float)
    v_eff = np.zeros_like(r)
    energy = 1.0e-8

    u_norm, delta, meta = _match_l0(r, v_eff, energy, grid.dxi)

    k = np.sqrt(2.0 * energy)
    amp_target = np.sqrt(2.0 / np.pi) / np.sqrt(k)
    u_exact = amp_target * np.sin(k * r)
    rel = np.linalg.norm(u_norm - u_exact) / np.linalg.norm(u_exact)

    assert meta["status"] == "ok"
    assert bool(meta.get("kr_constraint_relaxed", False))
    assert not bool(meta.get("fallback", False))
    assert abs(delta) < 1.0e-8
    assert rel < 1.0e-7


def test_scattering_phase_unwrap_uses_pi_period() -> None:
    raw = np.asarray([[1.45], [-1.45], [-1.20]])
    unwrapped = qmod.unwrap_scattering_phases(raw, axis=0)[:, 0]

    assert np.max(np.abs(np.diff(unwrapped))) < 0.3
    assert np.allclose(np.exp(2.0j * unwrapped), np.exp(2.0j * raw[:, 0]))


def test_coulomb_asymptotic_phase_has_standard_log_sign() -> None:
    """The fallback basis must solve the Coulomb equation to leading order."""
    rho = np.linspace(100.0, 200.0, 4001)
    eta = 0.7
    l_value = 0
    regular, _ = qmod._coulomb_asymptotic_basis(
        rm=rho,
        k=1.0,
        l=l_value,
        eta=eta,
    )

    first = np.gradient(regular, rho, edge_order=2)
    second = np.gradient(first, rho, edge_order=2)
    residual = second + (
        1.0 - 2.0 * eta / rho - l_value * (l_value + 1.0) / rho**2
    ) * regular
    interior = slice(4, -4)

    assert np.sqrt(np.mean(residual[interior] ** 2)) < 3.0e-4


def test_low_energy_square_well_phase_matches_analytic_result() -> None:
    """The interacting s wave must survive when the whole tail has kr < 3."""
    grid = create_sqrt_grid(rmax=40.0, N=600)
    r = np.asarray(grid.r, dtype=float)
    radius = 1.5
    depth = 0.8
    v_eff = np.where(r < radius, -depth, 0.0)

    for energy in (1.0e-5, 1.0e-4, 1.0e-3):
        _, delta, meta = _match_l0(r, v_eff, energy, grid.dxi)
        k = np.sqrt(2.0 * energy)
        q = np.sqrt(2.0 * (energy + depth))
        # From q*cot(q a) = k*cot(k a + delta).
        reference = np.arctan2(
            k * np.sin(q * radius),
            q * np.cos(q * radius),
        ) - k * radius

        assert meta["status"] == "ok"
        assert not bool(meta.get("fallback", False))
        assert _phase_error_mod_pi(delta, reference) < 5.0e-3


def test_production_preplanned_match_preserves_low_energy_interaction() -> None:
    """Exercise the fast path used by the A3 continuum integrator."""
    grid = create_sqrt_grid(rmax=40.0, N=600)
    r = np.asarray(grid.r, dtype=float)
    radius = 1.5
    depth = 0.8
    energy = 1.0e-4
    v_eff = np.where(r < radius, -depth, 0.0)

    n_energy, delta_vec = qmod._scattering_density_and_phase(
        v_eff=v_eff,
        r=r,
        mu=0.0,
        temperature=1.0,
        energy=energy,
        l_max=0,
        grid_kind="sqrt",
        grid_step=grid.dxi,
        l_pad=2,
        match_fraction=0.4,
        match_slice=None,
        match_r_cut=0.6 * float(r[-1]),
        match_fraction_mode="r",
        match_width=None,
        match_kr_min=3.0,
        match_v_tol=1.0e-8,
        match_min_points=16,
        match_asymptotic="auto",
        match_coulomb_tol=0.1,
        match_allow_shift=True,
        match_fallback="free",
        prop_rescale_limit=1.0e6,
        apply_occ=False,
        l_cap_strategy="none",
    )

    k = np.sqrt(2.0 * energy)
    q = np.sqrt(2.0 * (energy + depth))
    reference = np.arctan2(
        k * np.sin(q * radius),
        q * np.cos(q * radius),
    ) - k * radius

    assert np.all(np.isfinite(n_energy))
    assert np.max(n_energy) > 0.0
    assert _phase_error_mod_pi(float(delta_vec[0]), reference) < 5.0e-3


def test_low_energy_free_production_path_is_stable_for_high_partial_waves() -> None:
    """Relaxing the kr preference must not destabilize nonoscillatory l>0 waves."""
    grid = create_sqrt_grid(rmax=30.0, N=500)
    r = np.asarray(grid.r, dtype=float)
    energy = 1.0e-8

    n_energy, delta_vec = qmod._scattering_density_and_phase(
        v_eff=np.zeros_like(r),
        r=r,
        mu=0.0,
        temperature=1.0,
        energy=energy,
        l_max=8,
        grid_kind="sqrt",
        grid_step=grid.dxi,
        l_pad=2,
        match_fraction=0.4,
        match_slice=None,
        match_r_cut=0.6 * float(r[-1]),
        match_fraction_mode="r",
        match_width=None,
        match_kr_min=3.0,
        match_v_tol=1.0e-8,
        match_min_points=16,
        match_asymptotic="auto",
        match_coulomb_tol=0.1,
        match_allow_shift=True,
        match_fallback="free",
        prop_rescale_limit=1.0e6,
        apply_occ=False,
        l_cap_strategy="none",
    )

    # The complete free partial-wave sum is the spatially uniform local DOS
    # k/pi**2 per unit energy (including spin).  At this very small k the
    # l>0 channels are deeply nonoscillatory throughout the match window.
    expected = np.sqrt(2.0 * energy) / np.pi**2
    phase_mod_pi = 0.5 * np.angle(np.exp(2.0j * delta_vec))

    assert np.all(np.isfinite(n_energy))
    assert np.max(np.abs(phase_mod_pi)) < 1.0e-8
    assert np.max(np.abs(n_energy / expected - 1.0)) < 2.0e-3


def test_match_l_cap_retains_low_partial_waves_at_threshold() -> None:
    """The match cap must not erase a p resonance before it can be scouted."""
    grid = create_sqrt_grid(rmax=30.0, N=500)
    r = np.asarray(grid.r, dtype=float)
    cap = qmod._compute_l_cap(
        1.0e-8,
        8,
        r,
        2,
        None,
        0.6 * float(r[-1]),
        0.4,
        "r",
        None,
        16,
        3.0,
        1.0e-8,
        np.zeros_like(r),
        "match",
    )

    # Matching is possible before oscillation. Keep evanescent channels as
    # well as the mandatory low-l threshold channels, within the given ceiling.
    assert 2 <= cap <= 8


def test_failed_production_match_never_uses_raw_origin_normalization() -> None:
    """A bad match may fall back to free, but not to an arbitrary raw scale."""
    grid = create_sqrt_grid(rmax=10.0, N=200)
    r = np.asarray(grid.r, dtype=float)
    energy = 0.2
    raw = np.linspace(1.0, 3.0, r.size)

    override, delta, scale = qmod._match_scattering_scale_preplanned(
        u=raw,
        r=r,
        energy=energy,
        l=0,
        match_slice=(10, 11),  # Deliberately too short.
        fallback_free=False,
        basis_meta=None,
        match_fallback="free",
        free_basis_cache=None,
    )
    expected, _ = qmod._normalized_free_scattering_state(r, energy, 0)

    assert override is not None
    assert delta == 0.0
    assert scale == 1.0
    assert np.allclose(override, expected)
