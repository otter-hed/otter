"""Post-process the final AA ion-orbital densities without another solve."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from otter.numerics.transforms import (
    precompute_dst_lattice_transform_like,
    radial_forward,
)


def bound_wavefunctions(aa: Mapping[str, Any], *, multiply_fd: bool = False) -> np.ndarray:
    r"""Return radial :math:`R_{nl}(r)` on ``aa['r_bound']``.

    The ``(l, n_index, r_bound)`` result is unweighted by default. If requested,
    multiply the amplitude by f_FD(E), **not** sqrt(f_FD), as a display/analysis
    convention. Neither variant applies M(E), f_cut, or degeneracy. The FD
    variant is not a new orbital normalization and must not be used to
    reconstruct the ionic density (which already includes f_FD once).
    """
    if "bound_wavefunction_r" not in aa:
        raise ValueError("This electronic result has no retained bound wavefunctions (e.g. TF or an older result).")
    wave = np.asarray(aa["bound_wavefunction_r"], dtype=float)
    energies = np.asarray(aa["bound_energy_ha"], dtype=float)
    r = np.asarray(aa["r_bound"], dtype=float)
    if energies.ndim != 2 or r.ndim != 1 or wave.shape != (*energies.shape, r.size):
        raise ValueError("Bound wavefunctions must align with levels and r_bound.")
    if not multiply_fd:
        return wave
    fd = np.asarray(aa["bound_fd"], dtype=float)
    if fd.shape != energies.shape or not np.all(np.isfinite(fd)):
        raise ValueError("FD occupations must align with bound levels and be finite.")
    return wave * fd[..., None]


def ion_orbital_form_factors(
    aa: Mapping[str, Any], *, r: np.ndarray, k: np.ndarray
) -> np.ndarray:
    r"""Transform each existing :math:`n^{\rm ion}_{nl}(r)` on the QOZ grid.

    Pass one species' electronic result and the *complete*, uncropped ``r``
    and ``k`` arrays from the ionic result. The returned ``(l, n_index, k)``
    array uses the same level ordering as ``aa['bound_energy_ha']``; unused
    slots are zero. It includes FD occupations, degeneracy, M(E), and f_cut
    already present in ``ion_orbital_density_r``. No new orbital, SCF, or HNC
    calculation is performed. TF has no orbitals and is not supported here.

    The interpolation and DST-I are the same linear operations as the total
    ``n_ion`` -> ``f_k`` workflow, so summing over levels recovers that total
    to floating-point precision. Do not normalize individual form factors to
    an integer occupation or apply a separate screening-charge correction.
    """
    if "ion_orbital_density_r" not in aa:
        raise ValueError("This electronic result has no ion-orbital densities (e.g. TF).")
    r_native = np.asarray(aa["r"], dtype=float)
    density = np.asarray(aa["ion_orbital_density_r"], dtype=float)
    energies = np.asarray(aa["bound_energy_ha"], dtype=float)
    if (
        r_native.ndim != 1 or r_native.size < 2
        or not np.all(np.isfinite(r_native)) or np.any(np.diff(r_native) <= 0)
        or energies.ndim != 2 or density.shape != (*energies.shape, r_native.size)
        or not np.all(np.isfinite(density))
    ):
        raise ValueError("Ion-orbital densities must have aligned (l, n_index, native_r) shape.")
    r_grid, k_grid = np.asarray(r, dtype=float), np.asarray(k, dtype=float)
    transform = precompute_dst_lattice_transform_like(r_grid)
    for name, supplied, expected in (("r", r_grid, transform.r), ("k", k_grid, transform.k)):
        if supplied.shape != expected.shape or not np.allclose(
            supplied, expected, rtol=1.0e-12, atol=1.0e-14
        ):
            raise ValueError(f"{name} must be the complete matching QOZ DST grid.")
    profiles = np.empty((*energies.shape, r_grid.size), dtype=float)
    for index in np.ndindex(energies.shape):
        # Match workflows._interp_profile_linear, including its endpoint rules.
        profiles[index] = np.interp(
            r_grid, r_native, density[index], left=float(density[index][0]), right=0.0
        )
    return radial_forward(profiles, transform)
