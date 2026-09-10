r"""Portable, machine-readable plasma-state exports.

The public state format can store selected converged average-atom,
pseudoatom, and QOZ/HNC quantities needed for analysis or downstream XRTS
calculations.  The default ``complete`` profile preserves the historical
full-state behaviour; smaller profiles omit unrelated arrays:

``q(k)``
    The charge-closed pseudoatom screening cloud
    :math:`n_\mathrm{scr}(k)`.
``f(k)``
    The ion-associated electron density :math:`n_\mathrm{ion}(k)`.
``g_ij(r)``, ``S_ij(k)``
    Ashcroft--Langreth partial pair distributions and static structure
    factors.
``V_Ie(k)``, ``V_ee(k)``, ``C_Ie(k)``, ``C_ee(k)``
    Electron--ion and electron--electron potentials and direct-correlation
    channels used by the QOZ construction.
``chi0_k``, ``chi_ee_k``, ``G_ee_k``
    Ideal and interacting electron responses and the selected local-field
    correction :math:`G_{ee}(k)`.  ``gee_k`` and ``g_ee_k`` are temporary
    compatibility aliases for ``G_ee_k``.
``species_<i>_*``
    Native-grid electronic densities, potential components, mean-ionization
    definitions, bound levels, and per-level density contributions.

Only numeric arrays and fixed-width Unicode strings are written, so files can
always be loaded with ``allow_pickle=False``.  The default public window is
``r < 20 Bohr`` and ``k < 20 Bohr**-1``.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from otter.electronic.orbitals import ion_orbital_form_factors
from otter.io._npz import save_npz_atomic
from otter._version import __version__
from otter.numerics.transforms import (
    precompute_dst_lattice_transform_like,
    radial_forward,
    radial_inverse,
)


STATE_SCHEMA_VERSION = "otter_state_v5"
_SUPPORTED_STATE_SCHEMA_VERSIONS = {
    "otter_state_v1",
    "otter_state_v2",
    "otter_state_v3",
    "otter_state_v4",
    STATE_SCHEMA_VERSION,
}

STATE_EXPORT_GROUPS = (
    "electronic_summary",
    "bound_levels",
    "electronic_profiles",
    "electronic_potentials",
    "orbital_densities",
    "electronic_spectra",
    "ion_structure",
    "qoz_response",
    "pair_potential",
    "solver_history",
)
STATE_EXPORT_PROFILES = {
    "electronic_summary": frozenset({"electronic_summary", "bound_levels"}),
    "electronic_levels": frozenset({"electronic_summary", "bound_levels"}),
    "ion_structure": frozenset(
        {"electronic_summary", "bound_levels", "ion_structure"}
    ),
    "complete": frozenset(STATE_EXPORT_GROUPS),
}
_ION_EXPORT_GROUPS = frozenset(
    {"ion_structure", "qoz_response", "pair_potential", "solver_history"}
)


@dataclass(frozen=True)
class StateExportOptions:
    """Controls for one portable state export."""

    profile: str = "complete"
    include_groups: tuple[str, ...] = ()
    r_max_bohr: float = 20.0
    k_max_bohr_inv: float = 20.0
    require_converged_hnc: bool = True
    compressed: bool = True
    # Compatibility controls from state v4.  ``None`` follows ``profile``;
    # explicit booleans retain the old opt-in/opt-out behaviour.
    include_electronic_profiles: bool | None = None
    include_orbital_densities: bool | None = None

    def __post_init__(self) -> None:
        profile = str(self.profile).strip().lower().replace("-", "_")
        if profile not in STATE_EXPORT_PROFILES:
            allowed = ", ".join(STATE_EXPORT_PROFILES)
            raise ValueError(f"profile must be one of: {allowed}.")
        object.__setattr__(self, "profile", profile)
        groups = tuple(
            str(value).strip().lower().replace("-", "_")
            for value in self.include_groups
        )
        unknown = sorted(set(groups).difference(STATE_EXPORT_GROUPS))
        if unknown:
            raise ValueError("Unknown export groups: " + ", ".join(unknown))
        object.__setattr__(self, "include_groups", groups)
        if not np.isfinite(float(self.r_max_bohr)) or float(self.r_max_bohr) <= 0.0:
            raise ValueError("r_max_bohr must be finite and positive.")
        if (
            not np.isfinite(float(self.k_max_bohr_inv))
            or float(self.k_max_bohr_inv) <= 0.0
        ):
            raise ValueError("k_max_bohr_inv must be finite and positive.")

    @property
    def groups(self) -> frozenset[str]:
        """Resolved data groups after profile defaults and compatibility flags."""
        groups = set(STATE_EXPORT_PROFILES[self.profile])
        groups.update(self.include_groups)
        if self.include_electronic_profiles is True:
            groups.update({"electronic_profiles", "electronic_potentials"})
        elif self.include_electronic_profiles is False:
            groups.difference_update({"electronic_profiles", "electronic_potentials"})
        if self.include_orbital_densities is True:
            groups.add("orbital_densities")
        elif self.include_orbital_densities is False:
            groups.discard("orbital_densities")
        if "orbital_densities" in groups:
            groups.add("bound_levels")
        return frozenset(groups)

    @property
    def requires_ion_stage(self) -> bool:
        """Whether the resolved export requires a completed QOZ/HNC stage."""
        return bool(self.groups.intersection(_ION_EXPORT_GROUPS))


def _json_safe(value: Any) -> Any:
    """Convert nested NumPy-rich diagnostics to ordinary JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _nondefault_configuration(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Return required inputs and resolved values differing from defaults.

    The archive also stores the full resolved configuration.  This compact
    view is for readers: it highlights the thermodynamic/composition inputs
    and intentional overrides without making reproducibility depend on the
    defaults of the Otter version used to read the file.
    """
    # Imported lazily because workflows imports this module only when an
    # export is requested.
    from otter.workflows import PlasmaWorkflowConfig

    compact: dict[str, Any] = {}
    for definition in fields(PlasmaWorkflowConfig):
        if definition.name not in configuration:
            continue
        value = configuration[definition.name]
        if definition.default is not MISSING:
            default = definition.default
        elif definition.default_factory is not MISSING:
            default = definition.default_factory()
        else:
            compact[definition.name] = _json_safe(value)
            continue
        if _json_safe(value) != _json_safe(default):
            compact[definition.name] = _json_safe(value)
    return compact


def _package_version() -> str:
    """Return the version of the Otter source that produced the archive."""
    return __version__


def _as_species_axis(values: Any, *, n_species: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if n_species == 1 and arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2 or arr.shape[0] != n_species:
        raise ValueError(
            f"{name} must have shape (n_species, n_grid); got {arr.shape}."
        )
    return arr


def _as_pair_axes(values: Any, *, n_species: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if n_species == 1 and arr.ndim == 1:
        arr = arr[np.newaxis, np.newaxis, :]
    if arr.ndim != 3 or arr.shape[:2] != (n_species, n_species):
        raise ValueError(
            f"{name} must have shape (n_species, n_species, n_grid); "
            f"got {arr.shape}."
        )
    return arr


def _species_entries(workflow: Mapping[str, Any]) -> list[dict[str, Any]]:
    electronic = dict(workflow["electronic"])
    electronic_result = dict(electronic["result"])
    if str(electronic["kind"]) == "single_species":
        symbol = str(
            electronic_result.get(
                "element",
                list(workflow.get("species_symbols", ["?"]))[0],
            )
        )
        return [{
            "element": symbol,
            "Z": dict(electronic_result.get("meta", {})).get("Z", np.nan),
            "result": electronic_result,
        }]
    return [
        {
            **dict(entry),
            "Z": entry.get("Z", dict(entry["result"].get("meta", {})).get("Z", np.nan)),
            "result": dict(entry["result"]),
        }
        for entry in electronic_result["species"]
    ]


def _interpolate_ion_density(
    *,
    entries: list[dict[str, Any]],
    r_target: np.ndarray,
) -> np.ndarray:
    out = np.zeros((len(entries), r_target.size), dtype=float)
    for idx, entry in enumerate(entries):
        result = dict(entry["result"])
        if "r" not in result or "n_ion" not in result:
            raise ValueError(
                f"Species {entry.get('element', idx)!r} lacks r/n_ion data "
                "required to construct f(k)."
            )
        r_src = np.asarray(result["r"], dtype=float)
        n_ion_src = np.asarray(result["n_ion"], dtype=float)
        if (
            r_src.ndim != 1
            or n_ion_src.shape != r_src.shape
            or r_src.size < 2
            or np.any(np.diff(r_src) <= 0.0)
        ):
            raise ValueError("Each native n_ion profile must use a monotone 1D grid.")
        out[idx] = np.interp(
            r_target,
            r_src,
            n_ion_src,
            left=float(n_ion_src[0]),
            right=0.0,
        )
    return out


_ELECTRON_DENSITY_FIELDS = {
    "n_e": "n_e_r",
    "n_e_base": "n_e_base_r",
    "n_full": "n_full_r",
    "n_bound": "n_bound_r",
    "n_cont": "n_cont_r",
    "n_ext": "n_ext_r",
    "n_ext_pre_tail": "n_ext_pre_tail_r",
    "n_pa": "n_pa_r",
    "n_pa_repaired": "n_pa_repaired_r",
    "n_scr": "n_scr_r_native",
    "n_scr_repaired": "n_scr_repaired_r",
    "n_ion": "n_ion_r_native",
    "n_full_source": "n_full_source_r",
    "n_cont_dft_raw": "n_cont_dft_raw_r",
    "n_positive_energy_tf": "n_positive_energy_tf_r",
    "n_negative_tf": "n_negative_tf_r",
}

_ELECTRON_POTENTIAL_FIELDS = {
    "v_full": "v_full_r_ha",
    "v_scf": "v_scf_r_ha",
    "v_ext": "v_ext_r_ha",
    "v_nuc": "v_nuc_r_ha",
    "v_H": "v_hartree_r_ha",
    "v_xc": "v_xc_r_ha",
    "v_H_ext": "v_hartree_ext_r_ha",
    "v_xc_ext": "v_xc_ext_r_ha",
    "v_corr_full": "v_corr_full_r_ha",
    "v_corr_ext": "v_corr_ext_r_ha",
}

_ELECTRON_SPECTRAL_FIELDS = (
    "dos_energy_ha",
    "dos_bound",
    "dos_bound_fd",
    "dos_cont_ideal",
    "dos_cont_ideal_fd",
    "dos_cont_energy_ha",
    "dos_cont_scattering",
    "dos_cont_scattering_fd",
    "dos_cont_fd",
    "cont_phase_energy_ha",
    "cont_phase_shift_rad",
)


def _finite_numeric(value: Any) -> np.ndarray | None:
    """Return one finite numeric array, or ``None`` when unavailable."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.dtype.kind not in "biufc" or np.any(~np.isfinite(array)):
        return None
    return array


def _entry_n_i(entry: Mapping[str, Any]) -> float:
    """Return one species ion density without evaluating an absent fallback."""
    result = dict(entry["result"])
    if "n_i" in result:
        return float(result["n_i"])
    if "n_i_bohr3" in result:
        return float(result["n_i_bohr3"])
    meta = result.get("meta", {})
    if isinstance(meta, Mapping) and "n_i_bohr3" in meta:
        return float(meta["n_i_bohr3"])
    if "volume_bohr3" in entry:
        return 1.0 / float(entry["volume_bohr3"])
    r_ws = result.get("r_ws", entry.get("r_ws_bohr"))
    if r_ws is not None:
        r_ws_value = float(r_ws)
        if np.isfinite(r_ws_value) and r_ws_value > 0.0:
            return float(3.0 / (4.0 * np.pi * r_ws_value**3))
    raise ValueError(
        f"Species {entry.get('element', '?')!r} lacks n_i, volume, and Rws data."
    )


def _species_vector(
    ion: Mapping[str, Any],
    key: str,
    *,
    entries: list[dict[str, Any]],
    fallback_keys: tuple[str, ...],
) -> np.ndarray:
    """Return one finite scalar per species from the ion or AA result."""
    n_species = len(entries)
    if key in ion:
        value = np.atleast_1d(np.asarray(ion[key], dtype=float))
        if value.shape == (n_species,) and np.all(np.isfinite(value)):
            return value
    values = []
    for entry in entries:
        result = dict(entry["result"])
        value = np.nan
        for fallback_key in fallback_keys:
            if fallback_key in result:
                value = float(result[fallback_key])
                break
        values.append(value)
    return np.asarray(values, dtype=float)


def _entry_zstar(entry: Mapping[str, Any]) -> float:
    """Use the explicit AA value, with n0/n_i fallback for older results."""
    result = entry["result"]
    if "zstar" in result:
        return float(result["zstar"])
    return float(result.get("n0", np.nan)) / _entry_n_i(entry)


def _add_species_electronic_arrays(
    arrays: dict[str, np.ndarray],
    *,
    entries: list[dict[str, Any]],
    r_max_bohr: float,
    groups: frozenset[str],
    ion: Mapping[str, Any],
    k_max_bohr_inv: float,
) -> None:
    """Add native-grid AA fields using stable species-index prefixes."""
    for species_index, entry in enumerate(entries):
        result = dict(entry["result"])
        prefix = f"species_{species_index}_"
        r_native = np.asarray(result.get("r", ()), dtype=float)
        valid_r = not (
            r_native.ndim != 1
            or r_native.size < 1
            or np.any(~np.isfinite(r_native))
            or np.any(np.diff(r_native) <= 0.0)
        )
        r_mask = r_native < float(r_max_bohr) if valid_r else np.zeros(0, dtype=bool)
        native_groups = {
            "bound_levels",
            "electronic_profiles",
            "electronic_potentials",
            "orbital_densities",
            "electronic_spectra",
        }
        needs_native_detail = bool(groups.intersection(native_groups))
        needs_native_grid = bool(
            groups.intersection(
                {
                    "electronic_profiles",
                    "electronic_potentials",
                    "orbital_densities",
                }
            )
        )
        if needs_native_grid:
            if not valid_r or not np.any(r_mask):
                raise ValueError(
                    f"Species {entry.get('element', species_index)!r} lacks a "
                    "valid native radial grid required by the selected export groups."
                )
            arrays[prefix + "r_bohr"] = r_native[r_mask]

        n_i = _entry_n_i(entry)
        n0 = float(result.get("n0", np.nan))
        scalar_values = {
            "nuclear_charge": result.get("Z", entry.get("Z", np.nan)),
            "r_ws_bohr": result.get("r_ws", entry.get("r_ws_bohr", np.nan)),
            "mu_ha": result.get("mu", entry.get("mu_ha", np.nan)),
            "n_i_bohr3": n_i,
            "n0_bohr3": n0,
            "zbar_aa": result.get("zbar", np.nan),
            "zbar_partition": result.get("zbar_partition", np.nan),
            "zbar_ws": result.get("zbar_ws", result.get("zbar", np.nan)),
            "zstar": _entry_zstar(entry),
            "bound_energy_cut_ha": result.get("bound_energy_cut_ha", np.nan),
            "shallowest_bound_energy_ha": result.get(
                "shallowest_bound_energy_ha", np.nan
            ),
            "runtime_s": result.get("runtime_s", np.nan),
        }
        if needs_native_detail:
            for name, value in scalar_values.items():
                value_array = _finite_numeric(value)
                if value_array is not None and value_array.size == 1:
                    arrays[prefix + name] = np.asarray(float(value_array.item()))

        if "electronic_profiles" in groups:
            for source, target in _ELECTRON_DENSITY_FIELDS.items():
                value = _finite_numeric(result.get(source))
                if value is not None and value.shape == r_native.shape:
                    arrays[prefix + target] = np.asarray(value[r_mask], dtype=float)
            if "n_free" in result:
                n_free = np.asarray(result["n_free"], dtype=float)
                if n_free.shape == r_native.shape:
                    free_mask = r_mask & np.isfinite(n_free)
                    if np.any(free_mask):
                        arrays[prefix + "n_free_r_bohr"] = r_native[free_mask]
                        arrays[prefix + "n_free_r"] = n_free[free_mask]
            g_background = _finite_numeric(result.get("g_ii_background"))
            if g_background is not None and g_background.shape == r_native.shape:
                arrays[prefix + "g_ii_background_r"] = np.asarray(
                    g_background[r_mask], dtype=float
                )

        if "electronic_potentials" in groups:
            for source, target in _ELECTRON_POTENTIAL_FIELDS.items():
                value = _finite_numeric(result.get(source))
                if value is not None and value.shape == r_native.shape:
                    arrays[prefix + target] = np.asarray(value[r_mask], dtype=float)

        if "bound_levels" in groups:
            try:
                energies = np.asarray(result.get("bound_energy_ha"), dtype=float)
            except (TypeError, ValueError):
                energies = np.empty((0, 0), dtype=float)
            angular = _finite_numeric(result.get("bound_l_list"))
            if energies.ndim == 2 and energies.size and angular is not None:
                l_values = np.asarray(angular, dtype=int).reshape(-1)
                if l_values.shape == (energies.shape[0],):
                    energy_cut = float(result.get("bound_energy_cut_ha", 0.0))
                    selected = np.isfinite(energies) & (energies < energy_cut)
                    l_grid = np.broadcast_to(l_values[:, None], energies.shape)
                    n_grid = np.broadcast_to(
                        np.arange(1, energies.shape[1] + 1, dtype=int)[None, :],
                        energies.shape,
                    )
                    arrays[prefix + "bound_l"] = l_grid[selected]
                    arrays[prefix + "bound_n_index"] = n_grid[selected]
                    arrays[prefix + "bound_principal_n"] = (
                        n_grid[selected] + l_grid[selected]
                    )
                    arrays[prefix + "bound_energy_ha"] = energies[selected]
                    for source in (
                        "bound_fd",
                        "bound_m",
                        "bound_fdm",
                        "bound_occ_deg_fd",
                        "bound_occ_deg_fdm",
                        "bound_q_ion_ws",
                    ):
                        try:
                            value = np.asarray(result.get(source), dtype=float)
                        except (TypeError, ValueError):
                            continue
                        if value.shape == energies.shape and np.all(
                            np.isfinite(value[selected])
                        ):
                            arrays[prefix + source] = np.asarray(
                                value[selected], dtype=float
                            )
                    if "orbital_densities" in groups:
                        for source in (
                            "bound_orbital_density_r",
                            "ion_orbital_density_r",
                        ):
                            value = _finite_numeric(result.get(source))
                            if value is not None and value.shape == (
                                *energies.shape,
                                r_native.size,
                            ):
                                arrays[prefix + source] = np.asarray(
                                    value[selected][:, r_mask], dtype=float
                                )
                        wave = _finite_numeric(result.get("bound_wavefunction_r"))
                        if wave is not None:
                            r_bound = np.asarray(result["r_bound"], dtype=float)
                            if wave.shape != (*energies.shape, r_bound.size):
                                raise ValueError("Bound wavefunctions must align with r_bound and levels.")
                            bound_mask = r_bound < float(r_max_bohr)
                            arrays[prefix + "r_bound_bohr"] = r_bound[bound_mask]
                            arrays[prefix + "bound_wavefunction_r"] = wave[selected][:, bound_mask]
                        if "ion_orbital_density_r" in result and "r" in ion and "k" in ion:
                            k_full = np.asarray(ion["k"], dtype=float)
                            k_mask = k_full < float(k_max_bohr_inv)
                            # Transform the full profiles before cropping the archive.
                            per_level_k = ion_orbital_form_factors(result, r=ion["r"], k=k_full)
                            arrays[prefix + "orbital_k_bohr_inv"] = k_full[k_mask]
                            arrays[prefix + "ion_orbital_density_k"] = per_level_k[selected][:, k_mask]

            for source in (
                "bound_occ_mode",
                "threshold_state_status",
                "threshold_state_representation",
                "threshold_spectral_representation_status",
            ):
                value = result.get(source, dict(result.get("meta", {})).get(source))
                if value is not None:
                    arrays[prefix + source] = np.asarray(str(value), dtype="<U96")

        if "electronic_spectra" in groups:
            for source in _ELECTRON_SPECTRAL_FIELDS:
                value = _finite_numeric(result.get(source))
                if value is not None:
                    arrays[prefix + source] = np.asarray(value)


def _metadata(
    workflow: Mapping[str, Any],
    ion: Mapping[str, Any],
    options: StateExportOptions,
) -> dict[str, Any]:
    electronic = dict(workflow.get("electronic", {}))
    electronic_result = dict(electronic.get("result", {}))
    entries = _species_entries(workflow)
    electronic_convergence = []
    for entry in entries:
        result = dict(entry["result"])
        meta = dict(result.get("meta", {}))
        ext_status = dict(result.get("ext_status", {}))
        electronic_convergence.append(
            {
                "species": entry.get("element"),
                "stage1_converged": result.get(
                    "stage1_converged", meta.get("stage1_converged")
                ),
                "stage2_converged": result.get(
                    "stage2_converged", meta.get("stage2_converged")
                ),
                "external_converged": ext_status.get(
                    "converged", meta.get("ext_converged")
                ),
                "threshold_state_status": result.get(
                    "threshold_state_status", meta.get("threshold_state_status")
                ),
                "final_state_map_error": result.get(
                    "final_state_map_error", meta.get("final_state_map_error")
                ),
                "bound_spectrum_check": dict(result.get("bound_state_diagnostics", {})).get(
                    "spectrum_check", meta.get("bound_spectrum_check", {})
                ),
                "threshold_state_representation": result.get(
                    "threshold_state_representation",
                    meta.get("threshold_state_representation"),
                ),
                "runtime_s": result.get("runtime_s", meta.get("runtime_s")),
            }
        )
    mixture_meta = dict(electronic_result.get("meta", {}))
    groups = options.groups
    computed_stages = ["electronic.full"]
    external_enabled = []
    for entry in entries:
        final = dict(entry["result"])
        status = dict(final.get("ext_status", {}))
        meta = dict(final.get("meta", {}))
        external_enabled.append(bool(status.get(
            "enabled", meta.get("ext_enabled", "n_ext" in final)
        )))
    if entries and all(external_enabled):
        computed_stages.append("electronic.external")
    if ion:
        computed_stages.append("qoz")
        if ion.get("hnc_converged") is not None:
            computed_stages.append("hnc")
    analysis_complete_for = ["electronic_summary"]
    if "bound_levels" in groups:
        analysis_complete_for.append("electronic_levels")
    if "ion_structure" in groups:
        analysis_complete_for.append("ion_structure")
    resolved_configuration = dict(workflow.get("configuration", {}))
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "producer": "otter",
        "producer_version": _package_version(),
        # Store both views.  Saving only overrides would make old archives
        # ambiguous after a future release changes a default.
        "configuration": resolved_configuration,
        "configuration_nondefault": _nondefault_configuration(
            resolved_configuration
        ),
        "citation_keys": workflow.get("citation_keys", ()),
        "units": {
            "r_bohr": "Bohr",
            "k_bohr_inv": "Bohr^-1",
            "density_r": "Bohr^-3",
            "q_k": "electron number",
            "f_k": "electron number",
            "gij_r": "dimensionless",
            "sij_k": "dimensionless",
            "hij_r": "dimensionless",
            "cij_r": "dimensionless",
            "bridge_r": "dimensionless",
            "vij_r": "Hartree",
            "hnc_effective_potential_r": "Hartree",
            "vij_k": "Hartree Bohr^3",
            "v_ie_k": "Hartree Bohr^3",
            "v_ei_k": "Hartree Bohr^3",
            "v_ee_k": "Hartree Bohr^3",
            "c_ie_k": "Bohr^3",
            "c_ee_k": "Bohr^3",
            "v_ie_r": "Hartree",
            "v_ee_r": "Hartree",
            "c_ie_r": "dimensionless",
            "c_ee_r": "dimensionless",
            "chi0_k": "Bohr^-3 Hartree^-1",
            "chi_ee_k": "Bohr^-3 Hartree^-1",
            "G_ee_k": "dimensionless",
            "gee_k": "dimensionless",
            "g_ee_k": "dimensionless",
            "electronic_density_r": "Bohr^-3",
            "electronic_potential_r": "Hartree",
            "bound_energy_ha": "Hartree",
            "mu_ha": "Hartree",
        },
        "array_layout": {
            "common_species": "(species, grid)",
            "pairs": "(species_i, species_j, grid)",
            "native_electronic": "species_<index>_<field>",
            "orbital_density": "(bound_level, native_r)",
            "species_order": workflow.get("species_symbols"),
        },
        "state": {
            "formula": workflow.get("formula"),
            "species_symbols": workflow.get("species_symbols"),
            "species_counts": workflow.get("species_counts"),
            "temperature_ev": workflow.get("temperature_ev"),
            "ion_temperature_ev": workflow.get("ion_temperature_ev"),
            "rho_g_cc": workflow.get("rho_g_cc"),
        },
        "model": {
            "electronic_kind": dict(workflow.get("electronic", {})).get("kind"),
            "structure_model": workflow.get("structure_model", "IS"),
            "sij_convention": ion.get(
                "sij_convention",
                "ashcroft_langreth",
            ),
            "qoz_zbar_mode": ion.get("qoz_zbar_mode"),
            "chi0_model": ion.get("qoz_response_chi0_model"),
            "lfc_model": ion.get("qoz_response_lfc_model"),
            "hnc_bridge_model": ion.get("hnc_bridge_model", "none"),
            "vmhnc_eta": ion.get("vmhnc_eta"),
            "vmhnc_sigma_bohr": ion.get("vmhnc_sigma_bohr"),
        },
        "window": {
            "r_max_bohr_exclusive": float(options.r_max_bohr),
            "k_max_bohr_inv_exclusive": float(options.k_max_bohr_inv),
        },
        "export": {
            "profile": options.profile,
            "included_groups": sorted(groups),
            "omitted_groups": sorted(set(STATE_EXPORT_GROUPS).difference(groups)),
            "computed_stages": computed_stages,
            "analysis_complete_for": analysis_complete_for,
            # Portable analysis archives do not retain every internal object
            # needed to resume an iterative solver exactly.
            "restart_capable": False,
        },
        "convergence": {
            "electronic": electronic_convergence,
            "common_mu_residual_max_ha": mixture_meta.get("final_mu_residual_max_ha"),
            "common_mu_success": mixture_meta.get("final_mu_root_success"),
            "hnc_converged": ion.get("hnc_converged"),
            "hnc_best_residual": ion.get("hnc_best_residual"),
            "hnc_output_residual": ion.get("hnc_output_residual"),
            "closure_transform_max_abs": ion.get("closure_transform_max_abs"),
            "vmhnc_variational_residual": ion.get("vmhnc_variational_residual"),
            "charge_fix": ion.get("charge_fix"),
        },
        "definitions": {
            "zbar": ("QOZ charge selected by qoz_zbar_mode" if ion else
                     "AA zbar_partition, with legacy zbar fallback"),
            "zstar": "n0 / n_i for each species' AA-cell ion density",
            "bound_orbital_density_r": (
                "per-level contribution to n_bound using bound_occ_mode"
            ),
            "ion_orbital_density_r": (
                "per-level contribution to n_ion including M(E) and f_cut(r)"
            ),
            "bound_wavefunction_r": "R_nl(r) on r_bound_bohr, no occupation, M(E), or f_cut; Bohr^(-3/2)",
            "ion_orbital_density_k": "radial Fourier transform of n_ion_nl on the full QOZ grid; electrons",
            "real_space_electron_channels": (
                "inverse transform on the finite QOZ DST lattice"
            ),
        },
    }


def build_state_arrays(
    workflow: Mapping[str, Any],
    *,
    options: StateExportOptions | None = None,
) -> dict[str, np.ndarray]:
    """Build one profile-selected portable payload from a plasma workflow."""
    opts = options or StateExportOptions()
    groups = opts.groups
    ion_raw = workflow.get("ion")
    ion = dict(ion_raw) if isinstance(ion_raw, Mapping) else {}
    if opts.requires_ion_stage and not ion:
        raise ValueError(
            f"The {opts.profile!r} export requires a completed ion-structure "
            "stage. Use electronic_summary or electronic_levels for full-AA-only "
            "results."
        )
    if (
        opts.requires_ion_stage
        and bool(opts.require_converged_hnc)
        and ion.get("hnc_converged") is not True
    ):
        raise ValueError(
            "Refusing to export a missing or unconverged HNC status. Set "
            "require_converged_hnc=False only for an explicit diagnostic file."
        )
    structure_model = str(
        workflow.get("structure_model", ion.get("structure_model", "IS"))
    ).upper()
    if (
        opts.requires_ion_stage
        and structure_model == "SC"
        and bool(opts.require_converged_hnc)
    ):
        feedback = workflow.get("sc_feedback", ion.get("sc_feedback"))
        if not isinstance(feedback, Mapping) or feedback.get("converged") is not True:
            raise ValueError(
                "Refusing to export a missing or unconverged SC-feedback "
                "status. Disable require_converged_hnc only for an explicit "
                "diagnostic file."
            )

    symbols = [str(value) for value in workflow["species_symbols"]]
    counts = np.asarray(workflow["species_counts"], dtype=float)
    if len(symbols) == 0 or counts.shape != (len(symbols),):
        raise ValueError("Workflow species symbols/counts are inconsistent.")
    fractions = counts / float(np.sum(counts))
    n_species = len(symbols)
    entries = _species_entries(workflow)
    if len(entries) != n_species:
        raise ValueError("Electronic and ionic species counts differ.")
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(STATE_SCHEMA_VERSION),
        "species_symbols": np.asarray(symbols, dtype="<U8"),
        "species_counts": counts,
        "species_number_fraction": fractions,
        "species_nuclear_charge": np.asarray(
            [
                dict(entry["result"]).get("Z", entry.get("Z", np.nan))
                for entry in entries
            ],
            dtype=float,
        ),
        "zbar": _species_vector(
            ion, "zbar", entries=entries, fallback_keys=("zbar_partition", "zbar")
        ),
        "zbar_partition": _species_vector(
            ion,
            "zbar_partition",
            entries=entries,
            fallback_keys=("zbar_partition", "zbar"),
        ),
        "zbar_aa_ws": _species_vector(
            ion,
            "zbar_aa_ws",
            entries=entries,
            fallback_keys=("zbar_ws", "zbar"),
        ),
        "n_i_bohr3": np.asarray([_entry_n_i(entry) for entry in entries]),
    }
    mu_values = []
    r_ws_values = []
    n0_values = []
    zstar_values = []
    for entry in entries:
        electronic_result = dict(entry["result"])
        n0_value = float(electronic_result.get("n0", np.nan))
        mu_values.append(float(electronic_result.get("mu", entry.get("mu_ha", np.nan))))
        r_ws_values.append(
            float(electronic_result.get("r_ws", entry.get("r_ws_bohr", np.nan)))
        )
        n0_values.append(n0_value)
        zstar_values.append(_entry_zstar(entry))
    arrays["mu_ha"] = np.asarray(mu_values, dtype=float)
    arrays["r_ws_bohr"] = np.asarray(r_ws_values, dtype=float)
    arrays["n0_bohr3"] = np.asarray(n0_values, dtype=float)
    arrays["zstar"] = np.asarray(zstar_values, dtype=float)

    _add_species_electronic_arrays(
        arrays,
        entries=entries,
        r_max_bohr=float(opts.r_max_bohr),
        groups=groups,
        ion=ion,
        k_max_bohr_inv=float(opts.k_max_bohr_inv),
    )

    if opts.requires_ion_stage:
        arrays["zbar_qoz"] = _species_vector(
            ion,
            "zbar_qoz",
            entries=entries,
            fallback_keys=("zbar_partition", "zbar"),
        )
        r_full = np.asarray(ion["r"], dtype=float)
        k_full = np.asarray(ion["k"], dtype=float)
        if (
            r_full.ndim != 1
            or k_full.ndim != 1
            or r_full.size < 2
            or r_full.size != k_full.size
        ):
            raise ValueError(
                "Ion r/k grids must be equal-length one-dimensional arrays."
            )
        transform = precompute_dst_lattice_transform_like(r_full)
        if not np.allclose(transform.r, r_full, rtol=1.0e-12, atol=1.0e-13):
            raise ValueError(
                "Ion r grid is not the strict DST lattice expected by QOZ."
            )
        if not np.allclose(transform.k, k_full, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError(
                "Ion k grid is inconsistent with its real-space DST lattice."
            )
        r_mask = r_full < float(opts.r_max_bohr)
        k_mask = k_full < float(opts.k_max_bohr_inv)
        if not np.any(r_mask) or not np.any(k_mask):
            raise ValueError("Requested export window contains no grid points.")
        arrays["r_bohr"] = r_full[r_mask]
        arrays["k_bohr_inv"] = k_full[k_mask]

        if groups.intersection({"ion_structure", "qoz_response"}):
            n_ion_r_full = _interpolate_ion_density(
                entries=entries,
                r_target=r_full,
            )
            f_k_full = np.asarray(
                radial_forward(n_ion_r_full, transform),
                dtype=float,
            )
            q_k_full = _as_species_axis(
                ion["n_scr_k"],
                n_species=n_species,
                name="n_scr_k",
            )
            arrays["f_k"] = f_k_full[:, k_mask]
            arrays["q_k"] = q_k_full[:, k_mask]
            arrays["n_ion_k"] = f_k_full[:, k_mask]
            arrays["n_scr_k"] = q_k_full[:, k_mask]

        if "ion_structure" in groups:
            gij_full = _as_pair_axes(
                ion.get("gij_r", ion.get("gii_r")),
                n_species=n_species,
                name="gij_r",
            )
            sij_full = _as_pair_axes(
                ion.get("sij_k", ion.get("sii_k")),
                n_species=n_species,
                name="sij_k",
            )
            arrays["gij_r"] = gij_full[..., r_mask]
            arrays["sij_k"] = sij_full[..., k_mask]

        if "qoz_response" in groups:
            n_ion_r_full = _interpolate_ion_density(
                entries=entries,
                r_target=r_full,
            )
            n_scr_r_full = _as_species_axis(
                ion["n_scr_r"],
                n_species=n_species,
                name="n_scr_r",
            )
            v_ie_k_full = _as_species_axis(
                ion["v_ie_k"],
                n_species=n_species,
                name="v_ie_k",
            )
            c_ie_k_full = _as_species_axis(
                ion["c_ie_k"],
                n_species=n_species,
                name="c_ie_k",
            )
            common_k = {
                "v_ee_k": np.asarray(ion["v_ee_k"], dtype=float),
                "c_ee_k": np.asarray(ion["c_ee_k"], dtype=float),
                "chi0_k": np.asarray(ion["chi0_k"], dtype=float),
                "chi_ee_k": np.asarray(ion["chi_ee_k"], dtype=float),
            }
            lfc_key = next(
                (key for key in ("G_ee_k", "gee_k", "g_ee_k") if key in ion),
                None,
            )
            if lfc_key is None:
                raise KeyError(
                    "Ion structure result has no G_ee_k local-field correction."
                )
            G_ee_k_full = np.asarray(ion[lfc_key], dtype=float)
            if any(value.shape != k_full.shape for value in common_k.values()) or (
                G_ee_k_full.shape != k_full.shape
            ):
                raise ValueError(
                    "Electron response and common interaction channels must share "
                    "the ion reciprocal grid."
                )
            arrays.update(
                {
                    "n_ion_r": n_ion_r_full[:, r_mask],
                    "n_scr_r": n_scr_r_full[:, r_mask],
                    "v_ie_k": v_ie_k_full[:, k_mask],
                    "v_ei_k": v_ie_k_full[:, k_mask],
                    "c_ie_k": c_ie_k_full[:, k_mask],
                    "v_ee_k": common_k["v_ee_k"][k_mask],
                    "c_ee_k": common_k["c_ee_k"][k_mask],
                    "chi0_k": common_k["chi0_k"][k_mask],
                    "chi_ee_k": common_k["chi_ee_k"][k_mask],
                    "G_ee_k": G_ee_k_full[k_mask],
                    "gee_k": G_ee_k_full[k_mask],
                    "g_ee_k": G_ee_k_full[k_mask],
                    "v_ie_r": np.asarray(
                        radial_inverse(v_ie_k_full, transform), dtype=float
                    )[:, r_mask],
                    "c_ie_r": np.asarray(
                        radial_inverse(c_ie_k_full, transform), dtype=float
                    )[:, r_mask],
                    "v_ee_r": np.asarray(
                        radial_inverse(common_k["v_ee_k"], transform), dtype=float
                    )[r_mask],
                    "c_ee_r": np.asarray(
                        radial_inverse(common_k["c_ee_k"], transform), dtype=float
                    )[r_mask],
                }
            )
            arrays["v_ei_r"] = arrays["v_ie_r"]

        if "pair_potential" in groups:
            vij_r_full = _as_pair_axes(
                ion.get("vij_r", ion.get("vii_r")),
                n_species=n_species,
                name="vij_r",
            )
            vij_k_full = _as_pair_axes(
                ion.get("vij_k", ion.get("vii_k")),
                n_species=n_species,
                name="vij_k",
            )
            arrays["vij_r"] = vij_r_full[..., r_mask]
            arrays["vij_k"] = vij_k_full[..., k_mask]
            if str(ion.get("hnc_bridge_model", "none")) != "none":
                bridge_full = _as_pair_axes(
                    ion["bridge_r"],
                    n_species=n_species,
                    name="bridge_r",
                )
                effective_full = _as_pair_axes(
                    ion["hnc_effective_potential_r"],
                    n_species=n_species,
                    name="hnc_effective_potential_r",
                )
                arrays["bridge_r"] = bridge_full[..., r_mask]
                arrays["hnc_effective_potential_r"] = effective_full[..., r_mask]
                arrays["vmhnc_eta"] = np.asarray(float(ion["vmhnc_eta"]))
                arrays["vmhnc_sigma_bohr"] = np.asarray(float(ion["vmhnc_sigma_bohr"]))
                arrays["vmhnc_variational_residual"] = np.asarray(
                    float(ion["vmhnc_variational_residual"])
                )

        if "solver_history" in groups:
            hij_full = _as_pair_axes(
                ion.get("hij_r", ion.get("hii_r")),
                n_species=n_species,
                name="hij_r",
            )
            cij_full = _as_pair_axes(
                ion.get("cij_r", ion.get("cii_r")),
                n_species=n_species,
                name="cij_r",
            )
            arrays["hij_r"] = hij_full[..., r_mask]
            arrays["cij_r"] = cij_full[..., r_mask]
            residual_history = _finite_numeric(ion.get("residual_history"))
            if residual_history is not None:
                arrays["hnc_residual_history"] = np.asarray(
                    residual_history,
                    dtype=float,
                ).reshape(-1)

    metadata = _metadata(workflow, ion, opts)
    metadata["fields"] = sorted(arrays)
    metadata_text = json.dumps(
        _json_safe(metadata),
        sort_keys=True,
        separators=(",", ":"),
    )
    arrays["metadata_json"] = np.asarray(metadata_text)
    validate_state_arrays(arrays)
    return arrays


def validate_state_arrays(arrays: Mapping[str, Any]) -> None:
    """Validate the public state schema without loading pickled objects."""
    base_required = {
        "schema_version",
        "species_symbols",
        "species_counts",
        "species_number_fraction",
        "metadata_json",
    }
    missing = sorted(base_required.difference(arrays))
    if missing:
        raise ValueError(f"State payload is missing fields: {', '.join(missing)}")

    converted = {key: np.asarray(value) for key, value in arrays.items()}
    object_keys = [key for key, value in converted.items() if value.dtype.kind == "O"]
    if object_keys:
        raise ValueError(
            "Portable state payloads cannot contain object arrays: "
            + ", ".join(sorted(object_keys))
        )
    schema_version = str(converted["schema_version"].item())
    if schema_version not in _SUPPORTED_STATE_SCHEMA_VERSIONS:
        raise ValueError("Unsupported state schema version.")

    try:
        metadata = json.loads(str(converted["metadata_json"].item()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("metadata_json is not valid scalar JSON.") from exc
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must decode to an object.")
    if metadata.get("schema_version") != schema_version:
        raise ValueError("metadata_json has an unsupported schema version.")
    groups: frozenset[str]
    if schema_version == STATE_SCHEMA_VERSION:
        if not isinstance(metadata.get("configuration"), dict):
            raise ValueError("metadata_json lacks the workflow configuration.")
        if not isinstance(metadata.get("citation_keys"), list):
            raise ValueError("metadata_json lacks the citation-key list.")
        recorded_fields = metadata.get("fields")
        if not isinstance(recorded_fields, list) or set(recorded_fields) != (
            set(converted) - {"metadata_json"}
        ):
            raise ValueError("metadata_json field inventory is inconsistent.")
        export = metadata.get("export")
        if not isinstance(export, dict):
            raise ValueError("metadata_json lacks export-profile metadata.")
        profile = str(export.get("profile", ""))
        if profile not in STATE_EXPORT_PROFILES:
            raise ValueError("metadata_json has an unknown export profile.")
        included = export.get("included_groups")
        if not isinstance(included, list):
            raise ValueError("metadata_json lacks the included export groups.")
        groups = frozenset(str(value) for value in included)
        if not groups or not groups.issubset(STATE_EXPORT_GROUPS):
            raise ValueError("metadata_json contains invalid export groups.")
        if "electronic_summary" not in groups:
            raise ValueError("Every state export must include electronic_summary.")
        required = {
            "species_nuclear_charge",
            "zbar",
            "zbar_partition",
            "zbar_aa_ws",
            "zstar",
            "mu_ha",
            "r_ws_bohr",
            "n0_bohr3",
            "n_i_bohr3",
        }
        if groups.intersection(_ION_EXPORT_GROUPS):
            required.update({"r_bohr", "k_bohr_inv", "zbar_qoz"})
        if groups.intersection({"ion_structure", "qoz_response"}):
            required.update({"f_k", "q_k", "n_ion_k", "n_scr_k"})
        if "ion_structure" in groups:
            required.update({"gij_r", "sij_k"})
        if "qoz_response" in groups:
            required.update(
                {
                    "n_ion_r",
                    "n_scr_r",
                    "v_ie_k",
                    "v_ei_k",
                    "v_ee_k",
                    "c_ie_k",
                    "c_ee_k",
                    "v_ie_r",
                    "v_ei_r",
                    "v_ee_r",
                    "c_ie_r",
                    "c_ee_r",
                    "chi0_k",
                    "chi_ee_k",
                    "G_ee_k",
                    "gee_k",
                    "g_ee_k",
                }
            )
        if "pair_potential" in groups:
            required.update({"vij_r", "vij_k"})
        if "solver_history" in groups:
            required.update({"hij_r", "cij_r"})
        missing_profile_fields = sorted(required.difference(converted))
        if missing_profile_fields:
            raise ValueError(
                "State payload is missing fields required by its export groups: "
                + ", ".join(missing_profile_fields)
            )
    else:
        groups = frozenset(STATE_EXPORT_GROUPS)
        legacy_required = {
            "r_bohr",
            "k_bohr_inv",
            "n_ion_r",
            "n_scr_r",
            "f_k",
            "q_k",
            "n_ion_k",
            "n_scr_k",
            "gij_r",
            "sij_k",
        }
        missing_legacy = sorted(legacy_required.difference(converted))
        if missing_legacy:
            raise ValueError(
                "Legacy state payload is missing fields: " + ", ".join(missing_legacy)
            )

    n_species = int(converted["species_symbols"].size)
    has_ion_grid = bool(groups.intersection(_ION_EXPORT_GROUPS))
    nr = int(converted["r_bohr"].size) if has_ion_grid else 0
    nk = int(converted["k_bohr_inv"].size) if has_ion_grid else 0
    if n_species < 1 or (has_ion_grid and (nr < 1 or nk < 1)):
        raise ValueError("State species and requested grids must be non-empty.")
    counts = converted["species_counts"]
    fractions = converted["species_number_fraction"]
    if counts.shape != (n_species,) or fractions.shape != (n_species,):
        raise ValueError("Species counts/fractions have inconsistent shapes.")
    if (
        np.any(~np.isfinite(counts))
        or np.any(counts <= 0.0)
        or np.any(~np.isfinite(fractions))
        or np.any(fractions <= 0.0)
        or not np.isclose(float(np.sum(fractions)), 1.0, atol=2.0e-14)
    ):
        raise ValueError("Species counts/fractions must be finite and positive.")

    r_grid = converted.get("r_bohr", np.empty(0))
    k_grid = converted.get("k_bohr_inv", np.empty(0))
    if has_ion_grid and (
        r_grid.ndim != 1
        or k_grid.ndim != 1
        or np.any(~np.isfinite(r_grid))
        or np.any(~np.isfinite(k_grid))
        or np.any(r_grid <= 0.0)
        or np.any(k_grid <= 0.0)
        or np.any(np.diff(r_grid) <= 0.0)
        or np.any(np.diff(k_grid) <= 0.0)
    ):
        raise ValueError("State r/k grids must be finite, positive, and increasing.")

    if "n_ion_r" in converted and converted["n_ion_r"].shape != (
        n_species,
        nr,
    ):
        raise ValueError("n_ion_r shape is inconsistent with species/r grids.")
    if "n_scr_r" in converted and converted["n_scr_r"].shape != (
        n_species,
        nr,
    ):
        raise ValueError("n_scr_r shape is inconsistent with species/r grids.")
    for key in ("f_k", "q_k", "n_ion_k", "n_scr_k"):
        if key in converted and converted[key].shape != (n_species, nk):
            raise ValueError(f"{key} shape is inconsistent with species/k grids.")
    for key in ("v_ie_k", "v_ei_k", "c_ie_k"):
        if key in converted and converted[key].shape != (n_species, nk):
            raise ValueError(f"{key} shape is inconsistent with species/k grids.")
    for key in (
        "v_ee_k",
        "c_ee_k",
        "chi0_k",
        "chi_ee_k",
        "G_ee_k",
        "gee_k",
        "g_ee_k",
    ):
        if key in converted and converted[key].shape != (nk,):
            raise ValueError(f"{key} shape is inconsistent with the k grid.")
    if "gij_r" in converted and converted["gij_r"].shape != (
        n_species,
        n_species,
        nr,
    ):
        raise ValueError("gij_r shape is inconsistent with species/r grids.")
    if "sij_k" in converted and converted["sij_k"].shape != (
        n_species,
        n_species,
        nk,
    ):
        raise ValueError("sij_k shape is inconsistent with species/k grids.")
    for key in ("hij_r", "cij_r", "vij_r"):
        if key in converted and converted[key].shape != (
            n_species,
            n_species,
            nr,
        ):
            raise ValueError(f"{key} shape is inconsistent with species/r grids.")
    if "vij_k" in converted and converted["vij_k"].shape != (
        n_species,
        n_species,
        nk,
    ):
        raise ValueError("vij_k shape is inconsistent with species/k grids.")
    for key in ("v_ie_r", "v_ei_r", "c_ie_r"):
        if key in converted and converted[key].shape != (n_species, nr):
            raise ValueError(f"{key} shape is inconsistent with species/r grids.")
    for key in ("v_ee_r", "c_ee_r"):
        if key in converted and converted[key].shape != (nr,):
            raise ValueError(f"{key} shape is inconsistent with the r grid.")
    for key in (
        "zbar",
        "zbar_qoz",
        "zbar_partition",
        "zbar_aa_ws",
        "zstar",
        "mu_ha",
        "r_ws_bohr",
        "n0_bohr3",
        "n_i_bohr3",
        "species_nuclear_charge",
    ):
        if key in converted and converted[key].shape != (n_species,):
            raise ValueError(f"{key} shape is inconsistent with the species axis.")
    for key in (
        "species_counts",
        "species_number_fraction",
        "species_nuclear_charge",
        "n_ion_r",
        "n_scr_r",
        "f_k",
        "q_k",
        "n_ion_k",
        "n_scr_k",
        "v_ie_k",
        "v_ei_k",
        "v_ee_k",
        "c_ie_k",
        "c_ee_k",
        "chi0_k",
        "chi_ee_k",
        "G_ee_k",
        "gee_k",
        "g_ee_k",
        "v_ie_r",
        "v_ei_r",
        "v_ee_r",
        "c_ie_r",
        "c_ee_r",
        "gij_r",
        "sij_k",
        "hij_r",
        "cij_r",
        "vij_r",
        "zbar",
        "zbar_qoz",
        "zbar_partition",
        "zbar_aa_ws",
        "zstar",
        "mu_ha",
        "r_ws_bohr",
        "n0_bohr3",
        "n_i_bohr3",
        "vij_k",
    ):
        if key in converted and np.any(~np.isfinite(converted[key])):
            raise ValueError(f"{key} contains non-finite values.")
    if "f_k" in converted and not np.array_equal(
        converted["f_k"], converted["n_ion_k"]
    ):
        raise ValueError("f_k must be the explicit alias of n_ion_k.")
    if "q_k" in converted and not np.array_equal(
        converted["q_k"], converted["n_scr_k"]
    ):
        raise ValueError("q_k must be the explicit alias of n_scr_k.")
    if "v_ei_k" in converted and not np.array_equal(
        converted["v_ie_k"], converted["v_ei_k"]
    ):
        raise ValueError("v_ei_k must be the explicit alias of v_ie_k.")
    if "v_ei_r" in converted and not np.array_equal(
        converted["v_ie_r"], converted["v_ei_r"]
    ):
        raise ValueError("v_ei_r must be the explicit alias of v_ie_r.")
    if "G_ee_k" in converted:
        for alias in ("gee_k", "g_ee_k"):
            if alias in converted and not np.array_equal(
                converted["G_ee_k"], converted[alias]
            ):
                raise ValueError(f"{alias} must be an explicit alias of G_ee_k.")
    elif "g_ee_k" in converted and not np.array_equal(
        converted["gee_k"], converted["g_ee_k"]
    ):
        raise ValueError("g_ee_k must be the explicit alias of gee_k.")

    for key, value in converted.items():
        if value.dtype.kind in "biufc" and np.any(~np.isfinite(value)):
            raise ValueError(f"{key} contains non-finite values.")

    if schema_version == STATE_SCHEMA_VERSION:
        needs_native_grid = bool(
            groups.intersection(
                {
                    "electronic_profiles",
                    "electronic_potentials",
                    "orbital_densities",
                }
            )
        )
        for species_index in range(n_species):
            prefix = f"species_{species_index}_"
            r_key = prefix + "r_bohr"
            if needs_native_grid and r_key not in converted:
                raise ValueError(f"State payload is missing {r_key}.")
            r_native = converted.get(r_key, np.empty(0))
            if r_key in converted:
                if (
                    r_native.ndim != 1
                    or r_native.size < 1
                    or np.any(r_native <= 0.0)
                    or np.any(np.diff(r_native) <= 0.0)
                ):
                    raise ValueError(f"{r_key} must be positive and increasing.")
            n_level = int(converted.get(prefix + "bound_energy_ha", np.empty(0)).size)
            for name in ("bound_l", "bound_n_index", "bound_principal_n"):
                key = prefix + name
                if key in converted and converted[key].shape != (n_level,):
                    raise ValueError(f"{key} is not aligned with bound levels.")
            for name in ("bound_orbital_density_r", "ion_orbital_density_r"):
                key = prefix + name
                if key in converted and r_key not in converted:
                    raise ValueError(f"{key} requires {r_key}.")
                if key in converted and converted[key].shape != (
                    n_level,
                    r_native.size,
                ):
                    raise ValueError(f"{key} has an inconsistent orbital/r shape.")
            for field, grid_name in (
                ("bound_wavefunction_r", "r_bound_bohr"),
                ("ion_orbital_density_k", "orbital_k_bohr_inv"),
            ):
                key, grid_key = prefix + field, prefix + grid_name
                if key not in converted:
                    continue
                if grid_key not in converted:
                    raise ValueError(f"{key} requires {grid_key}.")
                grid = converted[grid_key]
                if grid.ndim != 1 or np.any(grid <= 0) or np.any(np.diff(grid) <= 0):
                    raise ValueError(f"{grid_key} must be positive and increasing.")
                if converted[key].shape != (n_level, grid.size):
                    raise ValueError(f"{key} has an inconsistent orbital/grid shape.")

    try:
        r_limit = float(metadata["window"]["r_max_bohr_exclusive"])
        k_limit = float(metadata["window"]["k_max_bohr_inv_exclusive"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("metadata_json lacks valid exclusive r/k windows.") from exc
    if (
        not np.isfinite(r_limit)
        or r_limit <= 0.0
        or not np.isfinite(k_limit)
        or k_limit <= 0.0
    ):
        raise ValueError("metadata_json contains invalid exclusive r/k windows.")
    if np.any(r_grid >= r_limit):
        raise ValueError(f"State export contains r >= {r_limit:g} Bohr.")
    if np.any(k_grid >= k_limit):
        raise ValueError(f"State export contains k >= {k_limit:g} Bohr^-1.")


def save_plasma_state(
    path: str | Path,
    workflow: Mapping[str, Any],
    *,
    options: StateExportOptions | None = None,
) -> Path:
    """Save one completed workflow as a portable compressed NPZ state.

    The file is written beside the destination and atomically replaced only
    after NumPy has completed the archive.  A failed or interrupted export
    therefore cannot expose a partially written production state.
    """
    opts = options or StateExportOptions()
    arrays = build_state_arrays(workflow, options=opts)
    output = Path(path)
    if output.suffix.lower() != ".npz":
        output = output.with_suffix(output.suffix + ".npz")
    return save_npz_atomic(output, arrays, compressed=bool(opts.compressed))


def load_plasma_state(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate one portable state file with pickle disabled."""
    with np.load(Path(path), allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    validate_state_arrays(arrays)
    return arrays


__all__ = [
    "STATE_EXPORT_GROUPS",
    "STATE_EXPORT_PROFILES",
    "STATE_SCHEMA_VERSION",
    "StateExportOptions",
    "build_state_arrays",
    "load_plasma_state",
    "save_plasma_state",
    "validate_state_arrays",
]
