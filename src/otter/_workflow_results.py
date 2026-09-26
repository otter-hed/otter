"""Species-axis conventions for assembled workflow results.

These helpers change representation only; native AA and ionic solver outputs
retain their numerical values. Profile reshaping uses NumPy views.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def aa_ion_density(entry: Mapping[str, Any]) -> float:
    """Return the AA-cell ion density, never the mixture's partial density."""
    aa = entry["result"]
    for key in ("n_i", "n_i_bohr3"):
        if key in aa:
            return float(aa[key])
    meta = aa.get("meta", {})
    if isinstance(meta, Mapping) and "n_i_bohr3" in meta:
        return float(meta["n_i_bohr3"])
    if "volume_bohr3" in entry:
        return 1.0 / float(entry["volume_bohr3"])
    r_ws = aa.get("r_ws", entry.get("r_ws_bohr"))
    if r_ws is not None and np.isfinite(float(r_ws)) and float(r_ws) > 0.0:
        return float(3.0 / (4.0 * np.pi * float(r_ws)**3))
    raise ValueError(
        f"Species {entry.get('element', '?')!r} lacks n_i, volume, and Rws data."
    )


def electronic_species_vectors(entries: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Collect available AA summary fields in the workflow species order."""
    results = [entry["result"] for entry in entries]
    out = {"n_i_aa": np.asarray([aa_ion_density(entry) for entry in entries])}
    for key in ("zbar_partition", "n0", "mu"):
        if all(key in aa for aa in results):
            out[key] = np.asarray([aa[key] for aa in results], dtype=float)
    if all("zbar_ws" in aa or "zbar" in aa for aa in results):
        out["zbar_aa_ws"] = np.asarray([
            aa.get("zbar_ws", aa.get("zbar")) for aa in results
        ], dtype=float)
    if all("zstar" in aa or "n0" in aa for aa in results):
        out["zstar"] = np.asarray([
            float(aa["zstar"]) if "zstar" in aa
            else float(aa["n0"]) / out["n_i_aa"][i]
            for i, aa in enumerate(results)
        ])
    return out


_SPECIES_PROFILES = (
    "n_scr_r", "n_scr_k", "q_k", "n_ion_r", "n_ion_k", "f_k",
    "v_ie_k", "v_ei_k", "v_ie_r", "v_ei_r", "c_ie_k", "c_ie_r",
    "gii_r", "sii_k",
)
_SPECIES_SCALARS = (
    "zbar", "zbar_qoz", "zbar_partition", "zbar_aa_ws", "zbar_electronic",
    "zbar_screening_integral_raw", "n_i",
)
_PAIR_PROFILES = (
    "gij_r", "sij_k", "hij_r", "cij_r", "vij_r", "vij_k",
    "bridge_r", "hnc_effective_potential_r",
)
_CHARGE_SCALARS = (
    "q_scr_raw", "q_scr_used", "q_scr_rel", "scale_factor", "total_scale_factor",
    "q_scr_trapz_raw", "q_scr_trapz_used", "q_scr_dst_raw", "q_scr_dst_used",
    "zbar_target", "zbar_partition", "q_scr_native_raw", "pa_charge_residual_native",
    "zbar_aa_ws", "zbar_electronic",
)


def _species_array(value: Any, n_species: int, rank: int, key: str) -> np.ndarray:
    array = np.asarray(value)
    # Scalars have one species axis; profiles additionally have a grid axis.
    if n_species == 1 and array.ndim == (0 if rank == 1 else 1):
        array = array.reshape((1,) * (rank - array.ndim) + array.shape)
    prefix = array.shape if rank == 1 else array.shape[:-1]
    expected = (n_species,) * (1 if rank == 1 else rank - 1)
    if array.ndim != rank or prefix != expected:
        raise ValueError(f"{key} has shape {array.shape}; expected species axes {expected} "
                         f"and {rank} dimensions.")
    return array


def ionic_species_axes(
    ion: dict[str, Any],
    *,
    n_species: int,
    electronic_vectors: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Return QOZ/HNC fields with explicit species axes, including Ns=1."""
    out = dict(ion)
    for keys, rank in ((_SPECIES_SCALARS, 1), (_SPECIES_PROFILES, 2), (_PAIR_PROFILES, 3)):
        for key in keys:
            if key in ion:
                out[key] = _species_array(ion[key], n_species, rank, key)
    # AA-cell densities and bulk partial ion densities are distinct in mixtures.
    for key in ("zstar", "n_i_aa"):
        if key in electronic_vectors:
            out[key] = electronic_vectors[key]
    if "charge_fix" in ion:
        charge = dict(ion["charge_fix"])
        for key in _CHARGE_SCALARS:
            if key in charge:
                charge[key] = _species_array(charge[key], n_species, 1, key)
        out["charge_fix"] = charge
    return out
