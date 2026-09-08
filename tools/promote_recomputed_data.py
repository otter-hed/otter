"""Promote one complete set of recomputed Otter data into the baselines.

The expensive producers write below ``benchmarks/outputs``.  This maintainer
tool validates the complete candidate set, atomically replaces only the
project-generated baseline NPZ files, refreshes their checksums and numerical
diagnostics, and removes the candidate directories.  Literature reference
data are never modified.

Run ``tools/recompute_all_data.py --fresh`` first.  Promotion is explicit::

    python tools/promote_recomputed_data.py --apply
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

from otter import __version__ as otter_version
from otter.io._npz import save_npz_atomic
from otter.numerics import ATOMIC_MASS_UNIT_TO_G, BOHR_TO_CM


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "benchmarks" / "baselines"
OUTPUTS = ROOT / "benchmarks" / "outputs"


@dataclass(frozen=True)
class Package:
    name: str
    candidate_dir: Path
    manifest_name: str | None = "manifest.json"
    starrett_fig3_names: bool = False
    preserve_md: bool = False
    ch2_hnc: bool = False

    @property
    def baseline_dir(self) -> Path:
        return BASELINES / self.name

    @property
    def candidate_manifest(self) -> Path | None:
        if self.manifest_name is None:
            return None
        return self.candidate_dir / self.manifest_name


PACKAGES = (
    Package(
        "al_full_workflow_1ev",
        OUTPUTS / "al_full_workflow_1ev" / "recomputed",
    ),
    Package(
        "al_is_sc_comparison",
        OUTPUTS / "al_is_sc_comparison" / "recomputed",
    ),
    Package("al_qm_tf", OUTPUTS / "al_qm_tf" / "recomputed"),
    Package(
        "al_rayleigh_weight_10ev",
        OUTPUTS / "al_rayleigh_weight_10ev" / "recomputed",
        manifest_name=None,
    ),
    Package(
        "argha_roy_carbon_sii",
        OUTPUTS / "argha_roy_carbon_sii" / "gallery_recomputed",
        manifest_name=None,
    ),
    Package(
        "carbon_ionization_levels",
        OUTPUTS / "carbon_ionization_levels",
        manifest_name="C_Te100eV_density_scan.manifest.json",
    ),
    Package(
        "carbon_lfc_sensitivity",
        OUTPUTS / "carbon_lfc_sensitivity" / "recomputed",
    ),
    Package(
        "ch136_mixture_workflow_100kk",
        OUTPUTS / "ch136_mixture_workflow_100kk" / "recomputed",
        manifest_name=None,
    ),
    Package(
        "ch2_hnc_md",
        OUTPUTS / "ch2_hnc_md" / "hnc_recomputed",
        manifest_name=None,
        preserve_md=True,
        ch2_hnc=True,
    ),
    Package(
        "ion_structure_library",
        OUTPUTS / "ion_structure_library" / "recomputed",
        preserve_md=True,
    ),
    Package(
        "johnson_et_al_2025_two_temperature_al",
        OUTPUTS
        / "johnson_et_al_2025_two_temperature_al"
        / "gallery_recomputed",
        manifest_name="candidate_manifest.json",
        preserve_md=True,
    ),
    Package(
        "schorner_et_al_2022_al_sii",
        OUTPUTS / "schorner_et_al_2022_al_sii" / "gallery_recomputed",
        manifest_name="candidate_manifest.json",
        preserve_md=True,
    ),
    Package(
        "starrett_et_al_2014_mixtures_fig3",
        OUTPUTS
        / "starrett_et_al_2014_mixtures_fig3"
        / "gallery_recomputed",
        manifest_name=None,
        starrett_fig3_names=True,
    ),
    Package(
        "starrett_saumon_2013_electronic",
        OUTPUTS / "starrett_saumon_2013_electronic",
    ),
    Package(
        "starrett_single_species_2013_2014",
        OUTPUTS
        / "starrett_single_species_2013_2014"
        / "gallery_recomputed",
    ),
)


FORBIDDEN_ARCHIVE_FIELDS = {
    "argha_roy_carbon_sii": {
        "f_k",
        "q_k",
        "vii_k_ha_bohr3",
        "n_scr_k_electrons",
    },
    "al_full_workflow_1ev": {
        "n_free_bohr3",
        "n_cont_bohr3",
        "n_ion_bohr3",
        "n_scr_k_raw_electrons",
        "chi0_k_bohr3_per_ha",
        "gee_k",
    },
    "al_qm_tf": {
        *(f"n_ext_{model}_bohr3" for model in ("qm", "tf")),
        *(f"n_pa_{model}_bohr3" for model in ("qm", "tf")),
        *(f"n_bound_{model}_bohr3" for model in ("qm", "tf")),
        *(f"n_cont_{model}_bohr3" for model in ("qm", "tf")),
        *(f"v_full_{model}_ha" for model in ("qm", "tf")),
        "vii_k_ha_bohr3",
        "hnc_residual",
    },
    "al_rayleigh_weight_10ev": {
        "rayleigh_weight",
    },
    "carbon_lfc_sensitivity": {
        "n_full_bohr3",
        "n_cont_bohr3",
        "n_ext_bohr3",
        "v_full_ha",
        "v_xc_ha",
        "vii_r_ha",
        "n_scr_r_bohr3",
    },
    "ion_structure_library": {
        "n_full_bohr3",
        "n_bound_bohr3",
        "n_cont_bohr3",
        "n_ext_bohr3",
        "n_pa_bohr3",
        "n_ion_bohr3",
        "n_scr_bohr3",
        "v_full_ha",
        "v_ext_ha",
        "v_hartree_ha",
        "v_xc_ha",
        "vii_r_ha",
        "vii_k_ha_bohr3",
        "chi0_k_bohr3_per_ha",
        "f_k",
        "q_k",
        "vmhnc_r_bohr",
        "vmhnc_k_bohr_inv",
        "md_gij_r",
        "md_gij_block_sem",
        "md_snn_k",
        "md_snn_frame_sem",
        "md_sij_k",
        "md_sij_frame_sem",
        "md_vectors_per_k_bin",
        "md_mean_temperature_k",
    },
    "johnson_et_al_2025_two_temperature_al": {
        "vmhnc_r_bohr",
        "k_bohr_inv",
        "sii_k",
        "vmhnc_k_bohr_inv",
        "vmhnc_sii_k",
        "vii_r_ha",
        "ion_density_bohr3",
        "md_k_bohr_inv",
        "md_sii_k",
        "md_sii_block_sem",
        "md_sii_vectors_per_bin",
    },
    "schorner_et_al_2022_al_sii": {
        "r_bohr",
        "gii_r",
        "vmhnc_r_bohr",
        "vmhnc_gii_r",
        "vmhnc_k_bohr_inv",
        "vii_r_ha",
        "ion_density_bohr3",
        "md_r_bohr",
        "md_gii_r",
        "md_gii_block_sem",
    },
    "starrett_et_al_2014_mixtures_fig3": {
        "gij_r",
        "k_bohr_inv",
        "sij_k",
        "f_ik",
        "q_ik",
        "n_full_bohr3",
        "n_bound_bohr3",
        "n_cont_bohr3",
        "n_ext_bohr3",
        "n_pa_bohr3",
        "n_ion_bohr3",
        "n_scr_bohr3",
        "v_full_ha",
        "v_ext_ha",
        "vii_r_ha",
        "vii_k_ha_bohr3",
        "chi0_k_bohr3_per_ha",
        "gee_k",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_name(package: Package, baseline_name: str) -> str:
    if package.starrett_fig3_names:
        return baseline_name.replace("_baseline.npz", "_otter.npz")
    return baseline_name


def _npz_scalar(arrays: dict[str, np.ndarray], key: str) -> Any:
    if key not in arrays:
        return None
    value = np.asarray(arrays[key])
    if value.size != 1:
        return None
    item = value.reshape(()).item()
    if isinstance(item, np.generic):
        item = item.item()
    return item


def _load_candidate(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    if not arrays or "schema_version" not in arrays:
        raise ValueError(f"{path}: missing schema_version.")
    optional_nan_diagnostics = {
        "bound_zero_tail_finite_wall_energy_ha",
        "bound_zero_tail_matched_energy_ha",
        "bound_zero_tail_exterior_probability",
        # The Starrett--Saumon comparison reports the Eq. (81) pressure-width
        # parameter only for the aluminium level table; the Fe ionization rows
        # therefore carry an explicit not-applicable NaN rather than a value.
        "ion_gamma_ha",
        # Rectangular level tables use NaN for orbitals that are absent from a
        # particular state.  Energy and M must share that missing-value mask;
        # the benchmark-specific tests enforce the pairwise contract.
        "level_energy_ha",
        "level_m",
    }
    for key, value in arrays.items():
        if value.dtype.hasobject:
            raise ValueError(f"{path}: object array {key!r} is not portable.")
        if value.dtype.kind in "fiu" and not np.all(np.isfinite(value)):
            if key not in optional_nan_diagnostics or np.any(np.isinf(value)):
                raise ValueError(f"{path}: non-finite numeric values in {key!r}.")
        if value.dtype.kind in "SU":
            text = " ".join(str(item) for item in value.reshape(-1))
            if "/home/" in text or "/tmp/" in text:
                raise ValueError(f"{path}: machine-local path in {key!r}.")

    for key in (
        "aa_stage2_converged",
        "aa_ext_converged",
        "stage2_converged",
        "hnc_converged",
    ):
        if key in arrays and not np.all(np.asarray(arrays[key], dtype=bool)):
            raise ValueError(f"{path}: {key} is not true for every state.")

    for key, limit in (
        ("root_residual_ha", 1.0e-4),
        ("hnc_output_residual", 1.0e-4),
        ("closure_transform_max_abs", 2.5e-3),
    ):
        if key in arrays and float(np.max(np.asarray(arrays[key], dtype=float))) > limit:
            raise ValueError(f"{path}: {key} exceeds {limit:g}.")
    return arrays


def _validate_storage_contract(
    package: Package,
    path: Path,
    arrays: dict[str, np.ndarray],
) -> None:
    """Reject gallery archives that accidentally retain unused solver arrays."""
    forbidden = FORBIDDEN_ARCHIVE_FIELDS.get(package.name, set())
    present = sorted(forbidden.intersection(arrays))
    if present:
        raise ValueError(
            f"{path}: selective storage contract violated by "
            + ", ".join(present)
        )
    profile = _npz_scalar(arrays, "storage_profile")
    if profile not in {
        "electronic_summary",
        "gallery_analysis",
        "benchmark_analysis",
        "hnc_with_preserved_md",
    }:
        raise ValueError(f"{path}: missing selective storage_profile.")


def _records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(manifest.get("states"), list):
        return [dict(item) for item in manifest["states"]]
    if isinstance(manifest.get("state"), dict):
        return [dict(manifest["state"])]
    if isinstance(manifest.get("baseline"), dict):
        return [dict(manifest["baseline"])]
    return []


def _record_filename(record: dict[str, Any]) -> str | None:
    for key in (
        "data_file",
        "baseline_file",
        "candidate_file",
        "result_file",
        "path",
    ):
        value = record.get(key)
        if value:
            return Path(str(value)).name
    return None


def _candidate_record(
    candidate_manifest: dict[str, Any] | None,
    *,
    candidate_name: str,
    baseline_record: dict[str, Any],
) -> dict[str, Any] | None:
    if candidate_manifest is None:
        return None
    records = _records(candidate_manifest)
    state_id = baseline_record.get("state_id")
    for record in records:
        if _record_filename(record) == candidate_name:
            return record
        if state_id is not None and record.get("state_id") == state_id:
            return record
    if len(records) == 1:
        return records[0]
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _manifest_record(
    manifest: dict[str, Any], baseline_name: str
) -> dict[str, Any]:
    for record in _records(manifest):
        if _record_filename(record) == baseline_name:
            return deepcopy(record)
    return {}


def _citation_keys(manifest: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for field in ("method_references", "method_publications"):
        entries = manifest.get(field, [])
        if isinstance(entries, dict):
            entries = list(entries.values())
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            key = entry.get("citation_key") or entry.get("reference_key")
            if key and str(key) not in keys:
                keys.append(str(key))
    return keys


def _method_references(manifest: dict[str, Any]) -> Any:
    for key in (
        "method_references",
        "method_publications",
        "method_publication",
        "publication",
        "literature_source",
    ):
        if key in manifest:
            return deepcopy(manifest[key])
    return []


def _compact_metadata(
    package: Package,
    *,
    manifest: dict[str, Any],
    baseline_name: str,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    scalar_diagnostics: dict[str, Any] = {}
    for key in (
        "aa_stage2_converged",
        "aa_ext_converged",
        "stage2_converged",
        "root_residual_ha",
        "root_residual_final_ha",
        "hnc_converged",
        "hnc_output_residual",
        "hnc_closure_mismatch",
        "closure_transform_max_abs",
        "threshold_state_status",
        "threshold_state_representation",
    ):
        value = _npz_scalar(arrays, key)
        if value is not None:
            scalar_diagnostics[key] = _json_safe(value)

    state = _manifest_record(manifest, baseline_name)
    for key in (
        "baseline_sha256",
        "data_sha256",
        "result_sha256",
        "candidate_sha256",
    ):
        state.pop(key, None)
    producer = deepcopy(manifest.get("producer", {}))
    if not isinstance(producer, dict):
        producer = {}
    producer.setdefault("project", "Otter")
    producer.setdefault("version", otter_version)
    return {
        "schema_version": "otter_compact_archive_metadata_v1",
        "archive_schema_version": str(_npz_scalar(arrays, "schema_version")),
        "archive_role": "project_generated_example_or_benchmark_baseline",
        "package_id": manifest.get(
            "benchmark_id", manifest.get("example_id", package.name)
        ),
        "file": baseline_name,
        "configuration": manifest.get("configuration", {}),
        "state": state,
        "producer": producer,
        "citation_keys": _citation_keys(manifest),
        "method_references": _method_references(manifest),
        "units": manifest.get("units", {}),
        "data_rights": manifest.get("data_rights", {}),
        "convergence": scalar_diagnostics,
        "fields": sorted(arrays),
    }


def _with_compact_metadata(
    package: Package,
    *,
    manifest: dict[str, Any],
    baseline_name: str,
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    payload = dict(arrays)
    # Producer-local metadata may describe a candidate directory.  Rebuild it
    # after promotion so the field inventory and accepted relative path cannot
    # be stale.
    payload.pop("metadata_json", None)
    metadata = _compact_metadata(
        package,
        manifest=manifest,
        baseline_name=baseline_name,
        arrays=payload,
    )
    payload["metadata_json"] = np.asarray(
        json.dumps(_json_safe(metadata), sort_keys=True, separators=(",", ":"))
    )
    return payload


def _update_record(
    baseline_record: dict[str, Any],
    candidate_record: dict[str, Any] | None,
    arrays: dict[str, np.ndarray],
    *,
    baseline_name: str,
    digest: str,
) -> dict[str, Any]:
    record = deepcopy(baseline_record)
    if candidate_record is not None:
        for key, value in candidate_record.items():
            if key not in {
                "status",
                "data_file",
                "data_sha256",
                "baseline_file",
                "baseline_sha256",
                "candidate_file",
                "candidate_sha256",
                "result_file",
                "result_sha256",
            }:
                record[key] = deepcopy(value)

    if "path" in record:
        record["path"] = baseline_name
        record["sha256"] = digest
    elif "data_file" in record:
        record["data_file"] = baseline_name
        record["data_sha256"] = digest
    elif "result_file" in record:
        record["result_file"] = baseline_name
        record["result_sha256"] = digest
    else:
        record["baseline_file"] = baseline_name
        record["baseline_sha256"] = digest
    if "status" in record:
        record["status"] = "accepted"
    if any(key.startswith("md_") for key in arrays):
        record["same_potential_md"] = True
    producer_commit = _npz_scalar(arrays, "otter_git_commit")
    if producer_commit is not None:
        record["producer_git_commit"] = str(producer_commit)

    for key in (
        "zbar_partition",
        "root_residual_ha",
        "hnc_output_residual",
        "hnc_closure_mismatch",
        "closure_transform_max_abs",
        "threshold_state_status",
        "threshold_state_representation",
    ):
        value = _npz_scalar(arrays, key)
        if value is not None:
            if isinstance(value, (np.bool_, bool)):
                value = bool(value)
            elif isinstance(value, (np.integer, int)):
                value = int(value)
            elif isinstance(value, (np.floating, float)):
                value = float(value)
            else:
                value = str(value)
            record[key] = value
    if "r_bohr" in arrays:
        record["n_model_points"] = int(np.asarray(arrays["r_bohr"]).size)
    return record


_STARRETT_SINGLE_REFERENCE_FILES = {
    "fe_10": "gii_Fe_22.5gcc_10.0ev_starrett.csv",
    "h_5": "gii_H_80gcc_5.0ev_starrett.csv",
    "h_172": "gii_H_80gcc_172.0ev_starrett.csv",
    "c_64p64": "gii_C_12.64gcc_64.64ev_starrett.csv",
    "w_10": "gii_W_40gcc_10.0ev_starrett_TF.csv",
    "w_60": "gii_W_40gcc_60.0ev_starrett.csv",
}
_STARRETT_SINGLE_MASS_DENSITY = {
    "fe_10": (55.845, 22.5),
    "h_5": (1.008, 80.0),
    "h_172": (1.008, 80.0),
    "c_64p64": (12.011, 12.64),
    "w_10": (183.84, 40.0),
    "w_60": (183.84, 40.0),
}


def _refresh_starrett_single_species_rmse(
    manifest: dict[str, Any],
    promoted: dict[str, tuple[dict[str, np.ndarray], str]],
) -> None:
    """Recompute recorded RMSE values from each newly promoted curve."""
    reference_dir = (
        ROOT / "benchmarks" / "reference_data" / "starrett_single_species_2013_2014"
    )
    reference_manifest = json.loads(
        (reference_dir / "manifest.json").read_text(encoding="utf-8")
    )
    reference_records = {
        str(record["path"]): record for record in reference_manifest["files"]
    }
    for record in manifest["states"]:
        state_id = str(record["state_id"])
        panel_id = state_id.rsplit("_", 1)[0]
        reference_name = _STARRETT_SINGLE_REFERENCE_FILES[panel_id]
        arrays = promoted[str(record["baseline_file"])][0]
        model_r_ws = float(_npz_scalar(arrays, "r_ws_bohr"))
        model_x = np.asarray(arrays["r_bohr"], dtype=float) / model_r_ws
        model_g = np.asarray(arrays["gii_r"], dtype=float)
        reference = np.asarray(
            np.loadtxt(reference_dir / reference_name, delimiter=","),
            dtype=float,
        )
        reference_x = np.asarray(reference[:, 0], dtype=float)
        if reference_records[reference_name]["column_1_unit"] == "Bohr":
            atomic_mass, rho_g_cc = _STARRETT_SINGLE_MASS_DENSITY[panel_id]
            ion_density_cm3 = rho_g_cc / (
                atomic_mass * ATOMIC_MASS_UNIT_TO_G
            )
            reference_r_ws = (
                (3.0 / (4.0 * np.pi * ion_density_cm3)) ** (1.0 / 3.0)
            ) / BOHR_TO_CM
            reference_x = reference_x / reference_r_ws
        predicted = np.interp(reference_x, model_x, model_g)
        record["reference_rmse"] = float(
            np.sqrt(np.mean((predicted - reference[:, 1]) ** 2))
        )


def _refresh_starrett_saumon_electronic_audit(
    manifest: dict[str, Any],
    promoted: dict[str, tuple[dict[str, np.ndarray], str]],
) -> None:
    """Refresh the selected audit scalars from the promoted IS/SC table."""
    arrays = next(iter(promoted.values()))[0]
    ids = np.asarray(arrays["state_id"], dtype=str)

    def _row(state_id: str) -> int:
        matches = np.flatnonzero(ids == state_id)
        if matches.size != 1:
            raise ValueError(f"Missing electronic audit state {state_id!r}.")
        return int(matches[0])

    al_15 = _row("al_rho2p7_te15_qm")
    fe_10 = _row("fe_rho22.5_te10_tf")
    level_energy = np.asarray(arrays["level_energy_ha"], dtype=float)
    level_m = np.asarray(arrays["level_m"], dtype=float)
    gamma = np.asarray(arrays["ion_gamma_ha"], dtype=float)
    zbar = np.asarray(arrays["zbar_partition"], dtype=float)
    sc_iterations = np.asarray(arrays["sc_iterations"], dtype=int)
    audit = manifest.setdefault("scientific_audit", {})
    audit.update(
        {
            "accepted_state_pairs": int(ids.size),
            "failed_state_pairs": 0,
            "sc_feedback_converged": bool(
                np.all(np.asarray(arrays["sc_converged"], dtype=bool))
            ),
            "sc_outer_iterations": sc_iterations.tolist(),
            "al_15ev_is_3s_status": (
                "resolved"
                if np.isfinite(level_energy[al_15, 0, 3])
                else "no_negative_energy_level"
            ),
            "al_15ev_sc_3s_status": (
                "resolved"
                if np.isfinite(level_energy[al_15, 1, 3])
                else "no_negative_energy_level"
            ),
            "al_15ev_sc_3s_energy_ha": float(level_energy[al_15, 1, 3]),
            "al_15ev_sc_3s_m": float(level_m[al_15, 1, 3]),
            "al_15ev_is_gamma_ha": float(gamma[al_15, 0]),
            "al_15ev_sc_gamma_ha": float(gamma[al_15, 1]),
            "al_15ev_is_zbar_partition": float(zbar[al_15, 0]),
            "al_15ev_sc_zbar_partition": float(zbar[al_15, 1]),
            "fe_10ev_is_zbar_partition": float(zbar[fe_10, 0]),
            "fe_10ev_sc_zbar_partition": float(zbar[fe_10, 1]),
        }
    )


def _candidate_manifest(package: Package) -> dict[str, Any] | None:
    path = package.candidate_manifest
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _accepted_md_arrays(
    package: Package,
    baseline_name: str,
) -> dict[str, np.ndarray]:
    """Load the accepted MD arrays that an Otter-only run must not replace."""
    if not package.preserve_md:
        return {}
    path = package.baseline_dir / baseline_name
    forbidden = FORBIDDEN_ARCHIVE_FIELDS.get(package.name, set())
    with np.load(path, allow_pickle=False) as archive:
        return {
            key: np.asarray(archive[key])
            for key in archive.files
            if key.startswith("md_") and key not in forbidden
        }


def _with_preserved_md(
    package: Package,
    baseline_name: str,
    candidate: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Merge immutable accepted MD measurements into a new Otter result."""
    payload = dict(candidate)
    md_arrays = _accepted_md_arrays(package, baseline_name)
    if md_arrays:
        # Old MD cannot certify a newly calculated electronic potential. Keep
        # the accepted paired curves together until matching-potential evidence
        # (or a reviewed new MD run) is available; numerical closeness is not
        # evidence that the underlying potentials are identical.
        with np.load(package.baseline_dir / baseline_name, allow_pickle=False) as old:
            for key in ("gii_r", "sii_k", "gij_r", "sij_k", "hnc_gij_r", "hnc_sij_k",
                        "vmhnc_gii_r", "vmhnc_sii_k"):
                if key in payload and key in old and not np.array_equal(payload[key], old[key]):
                    raise ValueError(
                        f"{package.name}/{baseline_name}: cannot pair changed {key} "
                        "with archived MD without verified matching-potential provenance. "
                        "Keep the complete archived comparison or review new MD separately."
                    )
    for key, accepted in md_arrays.items():
        if key in payload and not np.array_equal(payload[key], accepted):
            raise ValueError(
                f"{package.name}/{baseline_name}: Otter-only candidate tried "
                f"to modify preserved MD field {key!r}."
            )
        payload[key] = accepted
    return payload


def _build_ch2_hnc_candidate(
    package: Package,
    baseline_name: str,
) -> Path:
    """Combine fresh HNC cases with the untouched accepted CH2 MD statistics."""
    summary_path = package.candidate_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing CH2 HNC summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = {
        (float(item["te_ev"]), float(item["ti_ev"])): dict(item)
        for item in summary.get("cases", ())
    }
    accepted_path = package.baseline_dir / baseline_name
    with np.load(accepted_path, allow_pickle=False) as archive:
        accepted = {key: np.asarray(archive[key]) for key in archive.files}

    te_values = np.asarray(accepted["te_ev"], dtype=float)
    ti_values = np.asarray(accepted["ti_ev"], dtype=float)
    if te_values.shape != (9,) or ti_values.shape != (9,):
        raise ValueError("The accepted CH2 state ordering is not the expected 9 cases.")

    r_target = np.linspace(0.0, 10.0, 2001)
    k_target = np.linspace(0.0, 4.5, 1801)
    pair_indices = ((0, 0), (0, 1), (1, 1))
    hnc_g = np.empty((9, 3, r_target.size), dtype=float)
    hnc_s = np.empty((9, 3, k_target.size), dtype=float)
    hnc_elapsed = np.empty(9, dtype=float)
    hnc_residual = np.empty(9, dtype=float)
    hnc_closure = np.empty(9, dtype=float)
    signatures: list[str] = []

    for index, (te_ev, ti_ev) in enumerate(
        zip(te_values, ti_values, strict=True)
    ):
        key = (float(te_ev), float(ti_ev))
        record = records.get(key)
        if record is None:
            raise FileNotFoundError(f"Missing CH2 HNC record for Te/Ti={key}.")
        if record.get("status") != "success" or record.get("hnc_status") != "success":
            raise RuntimeError(f"CH2 HNC record did not succeed for Te/Ti={key}.")
        case_dir = package.candidate_dir / (
            f"Te{te_ev:07.3f}_Ti{ti_ev:07.3f}"
        )
        result_path = case_dir / "hnc_results.npz"
        with np.load(result_path, allow_pickle=False) as archive:
            r_source = np.asarray(archive["r_bohr"], dtype=float)
            k_source = np.asarray(archive["k_bohr_inv"], dtype=float)
            gij = np.asarray(archive["gij_r"], dtype=float)
            sij = np.asarray(archive["sij_k"], dtype=float)
        if gij.shape[:2] != (2, 2) or sij.shape[:2] != (2, 2):
            raise ValueError(f"Unexpected CH2 pair layout in {result_path}.")
        for pair, (left, right) in enumerate(pair_indices):
            hnc_g[index, pair] = np.interp(
                r_target, r_source, gij[left, right]
            )
            hnc_s[index, pair] = np.interp(
                k_target, k_source, sij[left, right]
            )
        hnc_elapsed[index] = float(record["hnc_elapsed_s"])
        hnc_residual[index] = float(record["hnc_output_residual"])
        hnc_closure[index] = float(record["hnc_closure_mismatch"])
        signatures.append(str(record["signature"]))

    payload = {
        key: value
        for key, value in accepted.items()
        if key != "metadata_json"
        and not key.startswith("hnc_")
        and key not in {
            "g_rmse",
            "s_rmse",
            "qoz_potential_max_abs_delta_within_te",
            "source_case_signature",
        }
    }
    payload.update(
        {
            "schema_version": np.asarray("otter_ch2_hnc_md_v2"),
            "storage_profile": np.asarray("hnc_with_preserved_md"),
            "hnc_r_bohr": r_target,
            "hnc_gij_r": hnc_g,
            "hnc_k_bohr_inv": k_target,
            "hnc_sij_k": hnc_s,
            "hnc_elapsed_s": hnc_elapsed,
            "hnc_output_residual": hnc_residual,
            "hnc_closure_mismatch": hnc_closure,
            "qoz_potential_max_abs_delta_within_te": np.zeros(9),
            "source_case_signature": np.asarray(signatures),
        }
    )

    md_r = np.asarray(payload["md_r_bohr"], dtype=float)
    md_k = np.asarray(payload["md_k_bohr_inv"], dtype=float)
    reliable = (
        (md_k <= 4.0)
        & (np.asarray(payload["md_vectors_per_k_bin"], dtype=int) >= 12)
    )
    g_rmse = np.empty((9, 3), dtype=float)
    s_rmse = np.empty((9, 3), dtype=float)
    for state_index in range(9):
        for pair_index in range(3):
            g_reference = np.interp(
                md_r, r_target, hnc_g[state_index, pair_index]
            )
            s_reference = np.interp(
                md_k, k_target, hnc_s[state_index, pair_index]
            )
            g_delta = (
                np.asarray(payload["md_gij_r"])[state_index, pair_index]
                - g_reference
            )
            s_delta = (
                np.asarray(payload["md_sij_k"])[state_index, pair_index]
                - s_reference
            )
            g_rmse[state_index, pair_index] = float(
                np.sqrt(np.mean(g_delta**2))
            )
            s_rmse[state_index, pair_index] = float(
                np.sqrt(np.mean(s_delta[reliable] ** 2))
            )
    payload["g_rmse"] = g_rmse
    payload["s_rmse"] = s_rmse

    for key, value in payload.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"CH2 candidate contains object field {key!r}.")
        if array.dtype.kind in "fiu" and not np.all(np.isfinite(array)):
            raise ValueError(f"CH2 candidate contains non-finite field {key!r}.")
    candidate_path = package.candidate_dir / baseline_name
    save_npz_atomic(candidate_path, payload)
    return candidate_path


def _refresh_ch2_manifest(
    manifest: dict[str, Any],
    promoted: dict[str, tuple[dict[str, np.ndarray], str]],
) -> None:
    """Refresh HNC-derived CH2 diagnostics while retaining MD provenance."""
    arrays = next(iter(promoted.values()))[0]
    producer = manifest.setdefault("producer", {})
    driver = ROOT / "applications" / "ch2_xrts_dataset" / "compare_hnc_md.py"
    producer["project_version"] = otter_version
    producer["source_driver_sha256_at_packaging"] = sha256_file(driver)
    producer["otter_hnc_recomputed_with_md_preserved"] = True
    producer["md_arrays_reused_without_modification"] = True
    acceptance = manifest.setdefault("acceptance", {})
    g_rmse = np.asarray(arrays["g_rmse"], dtype=float)
    s_rmse = np.asarray(arrays["s_rmse"], dtype=float)
    hnc_elapsed = np.asarray(arrays["hnc_elapsed_s"], dtype=float)
    acceptance.update(
        {
            "successful_states": int(hnc_elapsed.size),
            "maximum_hnc_output_residual": float(
                np.max(arrays["hnc_output_residual"])
            ),
            "pairwise_g_rmse_range": [float(np.min(g_rmse)), float(np.max(g_rmse))],
            "pairwise_s_rmse_range": [float(np.min(s_rmse)), float(np.max(s_rmse))],
            "hnc_elapsed_s_range": [
                float(np.min(hnc_elapsed)),
                float(np.max(hnc_elapsed)),
            ],
            "hnc_elapsed_s_sum": float(np.sum(hnc_elapsed)),
        }
    )


def _refresh_producer_metadata(manifest: dict[str, Any]) -> None:
    producer = manifest.get("producer")
    if not isinstance(producer, dict):
        return
    relative = producer.get("script_relative_path")
    if not relative:
        return
    script = ROOT / str(relative)
    if not script.is_file():
        raise FileNotFoundError(f"Producer script is missing: {relative}")
    digest = sha256_file(script)
    producer["script_sha256_current"] = digest
    producer.setdefault(
        "script_sha256_at_generation",
        producer.get("script_sha256", digest),
    )
    if "script_sha256" in producer:
        producer["script_sha256"] = digest
    if (
        "current_controller_sha256" in producer
        or str(relative).startswith("benchmarks/examples/")
    ):
        producer["current_controller_sha256"] = digest


def _merge_manifest(
    package: Package,
    *,
    candidate_manifest: dict[str, Any] | None,
    promoted: dict[str, tuple[dict[str, np.ndarray], str]],
) -> dict[str, Any]:
    manifest_path = package.baseline_dir / "manifest.json"
    accepted = json.loads(manifest_path.read_text(encoding="utf-8"))
    accepted_configuration = deepcopy(accepted.get("configuration", {}))

    if candidate_manifest is not None:
        for key in (
            "producer",
            "configuration",
            "scientific_audit",
            "shell_charge_diagnostic",
        ):
            if key in candidate_manifest:
                accepted[key] = deepcopy(candidate_manifest[key])
    if "status" in accepted:
        accepted["status"] = "accepted"
    if any(
        key.startswith("md_")
        for arrays, _ in promoted.values()
        for key in arrays
    ):
        configuration = accepted.setdefault("configuration", {})
        for key in ("same_potential_md", "wunsch_same_potential_md"):
            if key in accepted_configuration:
                configuration[key] = deepcopy(accepted_configuration[key])
    producer_commits = {
        str(commit)
        for arrays, _ in promoted.values()
        if (commit := _npz_scalar(arrays, "otter_git_commit")) is not None
    }
    if len(producer_commits) == 1:
        accepted.setdefault("producer", {})["git_commit"] = producer_commits.pop()
    _refresh_producer_metadata(accepted)

    if isinstance(accepted.get("states"), list):
        updated = []
        for baseline_record in accepted["states"]:
            baseline_name = _record_filename(baseline_record)
            if baseline_name is None or baseline_name not in promoted:
                raise ValueError(
                    f"{package.name}: manifest record lacks promoted data: "
                    f"{baseline_record}"
                )
            arrays, digest = promoted[baseline_name]
            candidate_name = _candidate_name(package, baseline_name)
            updated.append(
                _update_record(
                    dict(baseline_record),
                    _candidate_record(
                        candidate_manifest,
                        candidate_name=candidate_name,
                        baseline_record=dict(baseline_record),
                    ),
                    arrays,
                    baseline_name=baseline_name,
                    digest=digest,
                )
            )
        accepted["states"] = updated
    elif isinstance(accepted.get("state"), dict):
        baseline_record = dict(accepted["state"])
        baseline_name = _record_filename(baseline_record)
        if baseline_name is None or baseline_name not in promoted:
            raise ValueError(f"{package.name}: singleton state was not promoted.")
        arrays, digest = promoted[baseline_name]
        accepted["state"] = _update_record(
            baseline_record,
            _candidate_record(
                candidate_manifest,
                candidate_name=_candidate_name(package, baseline_name),
                baseline_record=baseline_record,
            ),
            arrays,
            baseline_name=baseline_name,
            digest=digest,
        )
    elif isinstance(accepted.get("baseline"), dict):
        baseline_record = dict(accepted["baseline"])
        baseline_name = _record_filename(baseline_record)
        if baseline_name is None or baseline_name not in promoted:
            raise ValueError(f"{package.name}: baseline record was not promoted.")
        arrays, digest = promoted[baseline_name]
        accepted["baseline"] = _update_record(
            baseline_record,
            None,
            arrays,
            baseline_name=baseline_name,
            digest=digest,
        )
    if package.ch2_hnc:
        _refresh_ch2_manifest(accepted, promoted)
    if package.name == "starrett_single_species_2013_2014":
        _refresh_starrett_single_species_rmse(accepted, promoted)
    if package.name == "starrett_saumon_2013_electronic":
        _refresh_starrett_saumon_electronic_audit(accepted, promoted)
    return accepted


def _validate_package(
    package: Package,
) -> tuple[
    dict[str, tuple[Path, dict[str, np.ndarray]]],
    dict[str, Any] | None,
]:
    baseline_names = {
        path.name for path in package.baseline_dir.glob("*.npz")
    }
    if not baseline_names:
        raise FileNotFoundError(f"{package.name}: no baseline NPZ files.")
    candidate_files: dict[str, tuple[Path, dict[str, np.ndarray]]] = {}
    for baseline_name in sorted(baseline_names):
        candidate_name = _candidate_name(package, baseline_name)
        candidate_path = package.candidate_dir / candidate_name
        if package.ch2_hnc and not candidate_path.is_file():
            candidate_path = _build_ch2_hnc_candidate(package, baseline_name)
        if not candidate_path.is_file():
            raise FileNotFoundError(
                f"{package.name}: missing candidate {candidate_name}."
            )
        arrays = _with_preserved_md(
            package,
            baseline_name,
            _load_candidate(candidate_path),
        )
        _validate_storage_contract(package, candidate_path, arrays)
        candidate_files[baseline_name] = (candidate_path, arrays)

    extra = {
        path.name
        for path in package.candidate_dir.glob("*.npz")
        if path.name
        not in {
            _candidate_name(package, baseline_name)
            for baseline_name in baseline_names
        }
    }
    if extra:
        raise ValueError(f"{package.name}: unexpected candidate files {sorted(extra)}.")
    return candidate_files, _candidate_manifest(package)


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


METRIC_RUNNERS = {
    "al_qm_tf": (
        "benchmarks/runners/plot_al_qm_tf.py",
        "benchmarks/outputs/al_qm_tf/al_qm_tf_offline_metrics.csv",
        "metrics.csv",
    ),
    "carbon_lfc_sensitivity": (
        "benchmarks/runners/plot_carbon_lfc_sensitivity.py",
        "benchmarks/outputs/carbon_lfc_sensitivity/carbon_lfc_sensitivity_metrics.csv",
        "metrics.csv",
    ),
    "starrett_et_al_2014_mixtures_fig3": (
        "benchmarks/runners/plot_starrett_et_al_2014_mixtures_fig3.py",
        "benchmarks/outputs/starrett_et_al_2014_mixtures_fig3/fig3_ch1p36_offline_metrics.csv",
        "metrics.csv",
    ),
}


def _refresh_metrics(package: Package) -> None:
    if package.name not in METRIC_RUNNERS:
        return
    runner, generated_relative, baseline_name = METRIC_RUNNERS[package.name]
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("MPLCONFIGDIR", "/tmp/otter-matplotlib")
    subprocess.run(
        [sys.executable, str(ROOT / runner)],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    generated = ROOT / generated_relative
    destination = package.baseline_dir / baseline_name
    _atomic_copy(generated, destination)
    manifest_path = package.baseline_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if package.name == "starrett_et_al_2014_mixtures_fig3":
        manifest["baseline"]["metrics_file"] = baseline_name
        manifest["baseline"]["metrics_sha256"] = sha256_file(destination)
    else:
        manifest["metrics_file"] = baseline_name
        manifest["metrics_sha256"] = sha256_file(destination)
    _write_manifest(manifest_path, manifest)


def _refresh_existing_metadata(package: Package) -> None:
    promoted: dict[str, tuple[dict[str, np.ndarray], str]] = {}
    for path in sorted(package.baseline_dir.glob("*.npz")):
        arrays = _load_candidate(path)
        arrays.pop("metadata_json", None)
        manifest = json.loads(
            (package.baseline_dir / "manifest.json").read_text(encoding="utf-8")
        )
        payload = _with_compact_metadata(
            package,
            manifest=manifest,
            baseline_name=path.name,
            arrays=arrays,
        )
        save_npz_atomic(path, payload)
        promoted[path.name] = (payload, sha256_file(path))
    manifest = _merge_manifest(
        package,
        candidate_manifest=None,
        promoted=promoted,
    )
    _write_manifest(package.baseline_dir / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="replace baselines after all candidates pass validation",
    )
    parser.add_argument(
        "--keep-candidates",
        action="store_true",
        help="retain candidate directories after a successful promotion",
    )
    parser.add_argument(
        "--refresh-metadata-only",
        action="store_true",
        help="refresh embedded metadata and checksums in accepted baselines",
    )
    args = parser.parse_args()

    if args.refresh_metadata_only:
        for package in PACKAGES:
            _refresh_existing_metadata(package)
            print(f"[metadata] {package.name}")
        return

    validated = {
        package.name: _validate_package(package) for package in PACKAGES
    }
    total = sum(len(files) for files, _ in validated.values())
    print(f"Validated {total} candidate NPZ files in {len(PACKAGES)} packages.")
    if not args.apply:
        print("Dry run only; use --apply to replace project-generated baselines.")
        return

    for package in PACKAGES:
        candidate_files, candidate_manifest = validated[package.name]
        provisional = {
            baseline_name: (arrays, sha256_file(source))
            for baseline_name, (source, arrays) in candidate_files.items()
        }
        metadata_manifest = _merge_manifest(
            package,
            candidate_manifest=candidate_manifest,
            promoted=provisional,
        )
        promoted: dict[str, tuple[dict[str, np.ndarray], str]] = {}
        for baseline_name, (source, arrays) in candidate_files.items():
            destination = package.baseline_dir / baseline_name
            payload = _with_compact_metadata(
                package,
                manifest=metadata_manifest,
                baseline_name=baseline_name,
                arrays=arrays,
            )
            save_npz_atomic(destination, payload)
            promoted[baseline_name] = (payload, sha256_file(destination))
        manifest = _merge_manifest(
            package,
            candidate_manifest=candidate_manifest,
            promoted=promoted,
        )
        _write_manifest(package.baseline_dir / "manifest.json", manifest)
        _refresh_metrics(package)
        print(f"[promoted] {package.name}: {len(promoted)} state file(s)")

    if not args.keep_candidates:
        for package in PACKAGES:
            if package.name == "carbon_ionization_levels":
                for path in (
                    package.candidate_dir / "C_Te100eV_density_scan.npz",
                    package.candidate_dir
                    / "C_Te100eV_density_scan.manifest.json",
                    package.candidate_dir / "point_cache",
                    package.candidate_dir / "point_failures",
                ):
                    if path.is_dir():
                        shutil.rmtree(path)
                    elif path.exists():
                        path.unlink()
            elif package.candidate_dir.exists():
                shutil.rmtree(package.candidate_dir)
    print("Promotion complete; " + (
        "candidate data retained." if args.keep_candidates
        else "no recomputed NPZ candidates remain."
    ))


if __name__ == "__main__":
    main()
