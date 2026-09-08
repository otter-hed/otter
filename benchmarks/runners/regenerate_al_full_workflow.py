"""Regenerate the complete Al 8.1 g/cc, 1 eV Otter gallery state.

The accepted documentation reference result is read-only.  Running this program
writes a candidate archive to
``benchmarks/outputs/al_full_workflow_1ev/recomputed`` for review.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time
from types import ModuleType
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "benchmarks" / "outputs" / "al_full_workflow_1ev" / "recomputed"
OUTPUT_PATH = OUTPUT_DIR / "Al_rho8p1gcc_Te1eV_Ti1eV.npz"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
STATE = {
    "state_id": "al_full_workflow_rho8p1_te1_ti1",
    "element": "Al",
    "rho_g_cc": 8.1,
    "te_ev": 1.0,
    "ti_ev": 1.0,
}
HNC_TOL = 1.0e-6
HNC_CLOSURE_TOL = 1.0e-3
HNC_MAX_ITER = 1000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_status_porcelain() -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return "git metadata unavailable\n"


def _load_library_regenerator() -> ModuleType:
    path = ROOT / "benchmarks" / "runners" / "regenerate_ion_structure_library.py"
    spec = importlib.util.spec_from_file_location(
        "otter_ion_structure_regenerator",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configuration(producer: ModuleType):
    """Return this gallery state with a strict HNC stopping contract.

    The reusable ion-library scan uses a looser nonlinear tolerance across
    several states.  At this cold, strongly coupled Al point that stopping
    point can satisfy the map residual while retaining a larger finite-DST
    ``g(r) <-> S(k)`` mismatch.  The complete workflow example continues the
    same equations to a tighter fixed point; no physical model is changed.
    """
    cfg = producer._configuration(STATE)
    cfg.hnc_tol = float(HNC_TOL)
    cfg.hnc_closure_transform_tol = float(HNC_CLOSURE_TOL)
    cfg.hnc_max_iter = int(HNC_MAX_ITER)
    return cfg


def _augment_v2_payload(
    payload: dict[str, np.ndarray],
    portable: dict[str, np.ndarray],
    workflow: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Add explicit ion and raw/charge-closed screening form factors."""
    output = dict(payload)
    payload_k = np.asarray(output["k_bohr_inv"], dtype=float)
    portable_k = np.asarray(portable["k_bohr_inv"], dtype=float)
    n_ion_k = np.asarray(portable["n_ion_k"], dtype=float)
    if n_ion_k.shape != (1, portable_k.size):
        raise ValueError(
            "The single-species portable n_ion(k) array must have shape " "(1, n_k)."
        )
    if portable_k.shape != payload_k.shape or not np.allclose(
        portable_k,
        payload_k,
        rtol=1.0e-12,
        atol=1.0e-13,
    ):
        raise ValueError("Portable f(k) grid does not match the workflow grid.")
    output["schema_version"] = np.asarray("otter_al_full_workflow_v2")
    output["benchmark_id"] = np.asarray("al_full_workflow_1ev")
    output["n_ion_k_electrons"] = n_ion_k[0]
    if workflow is not None:
        charge_fix = dict(workflow["ion"]["charge_fix"])
        q_scale = float(charge_fix["scale_factor"])
        q_used = np.asarray(portable["n_scr_k"], dtype=float)[0]
        output["q_scr_grid_raw"] = np.asarray(float(charge_fix["q_scr_raw"]))
        output["q_scr_used"] = np.asarray(float(charge_fix["q_scr_used"]))
        output["q_scr_scale_factor"] = np.asarray(q_scale)
        output["n_scr_k_electrons"] = q_used
        output["n_scr_k_raw_electrons"] = q_used / q_scale
    return output


def _pack_gallery_result(
    workflow: dict[str, Any],
    *,
    elapsed_s: float,
) -> dict[str, np.ndarray]:
    """Keep exactly the arrays consumed by the complete-workflow gallery."""
    from otter.io.state import StateExportOptions, build_state_arrays

    electronic = dict(workflow["electronic"]["result"])
    ion = dict(workflow["ion"])
    exported = build_state_arrays(
        workflow,
        options=StateExportOptions(
            profile="ion_structure",
            include_groups=(
                "electronic_profiles",
                "electronic_potentials",
                "pair_potential",
            ),
        ),
    )
    charge_fix = dict(ion["charge_fix"])
    q_scale = float(charge_fix["scale_factor"])
    q_used = np.asarray(exported["q_k"], dtype=float)[0]
    prefix = "species_0_"

    def species(field: str) -> np.ndarray:
        return np.asarray(exported[prefix + field])

    v_full_key = (
        prefix + "v_full_r_ha"
        if prefix + "v_full_r_ha" in exported
        else prefix + "v_scf_r_ha"
    )
    payload = {
        "schema_version": np.asarray("otter_al_full_workflow_v2"),
        "benchmark_id": np.asarray("al_full_workflow_1ev"),
        "storage_profile": np.asarray("gallery_analysis"),
        "state_id": np.asarray(STATE["state_id"]),
        "element": np.asarray(STATE["element"]),
        "rho_g_cc": np.asarray(STATE["rho_g_cc"]),
        "te_ev": np.asarray(STATE["te_ev"]),
        "ti_ev": np.asarray(STATE["ti_ev"]),
        "producer_elapsed_s": np.asarray(float(elapsed_s)),
        "r_e_bohr": species("r_bohr"),
        "n_full_bohr3": species("n_full_r"),
        "n_bound_bohr3": species("n_bound_r"),
        "n_ext_bohr3": species("n_ext_r"),
        "n_pa_bohr3": species("n_pa_r"),
        "n_scr_bohr3": species("n_scr_r_native"),
        "v_full_ha": np.asarray(exported[v_full_key]),
        "v_ext_ha": species("v_ext_r_ha"),
        "v_hartree_ha": species("v_hartree_r_ha"),
        "v_xc_ha": species("v_xc_r_ha"),
        "n0_bohr3": np.asarray(float(electronic["n0"])),
        "r_ws_bohr": np.asarray(float(electronic["r_ws"])),
        "mu_ha": np.asarray(float(electronic["mu"])),
        "zbar_aa": np.asarray(float(electronic["zbar"])),
        "zbar_partition": np.asarray(float(ion["zbar_partition"])),
        "zbar_qoz": np.asarray(float(ion["zbar_qoz"])),
        "q_scr_raw": np.asarray(float(ion["zbar_screening_integral_raw"])),
        "q_scr_grid_raw": np.asarray(float(charge_fix["q_scr_raw"])),
        "q_scr_used": np.asarray(float(charge_fix["q_scr_used"])),
        "q_scr_scale_factor": np.asarray(q_scale),
        "threshold_state_status": np.asarray(
            str(electronic.get("threshold_state_status", "none"))
        ),
        "threshold_state_representation": np.asarray(
            str(electronic.get("threshold_state_representation", "none"))
        ),
        "r_bohr": np.asarray(exported["r_bohr"]),
        "k_bohr_inv": np.asarray(exported["k_bohr_inv"]),
        "gii_r": np.asarray(exported["gij_r"])[0, 0],
        "sii_k": np.asarray(exported["sij_k"])[0, 0],
        "vii_r_ha": np.asarray(exported["vij_r"])[0, 0],
        "vii_k_ha_bohr3": np.asarray(exported["vij_k"])[0, 0],
        "n_scr_k_electrons": q_used,
        "n_ion_k_electrons": np.asarray(exported["f_k"])[0],
        "hnc_best_residual": np.asarray(float(ion["hnc_best_residual"])),
        "hnc_closure_mismatch": np.asarray(
            float(ion["closure_transform_max_abs"])
        ),
        "hnc_closure_tolerance": np.asarray(float(ion["closure_transform_tol"])),
        "hnc_iters": np.asarray(int(ion["hnc_iters"])),
        "bound_l": species("bound_l"),
        "bound_n_index": species("bound_n_index"),
        "bound_energy_ha": species("bound_energy_ha"),
        "bound_fd": species("bound_fd"),
        "bound_occ_deg_fd": species("bound_occ_deg_fd"),
    }
    for key, value in payload.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"Object dtype is forbidden for {key!r}.")
        if array.dtype.kind in "fiu" and not np.all(np.isfinite(array)):
            raise ValueError(f"Non-finite values in {key!r}.")
    return payload


def _candidate_manifest(
    output_path: Path,
    *,
    worktree_status: str,
    payload: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Describe a review-only v2 result without claiming it is accepted."""
    script = Path(__file__).resolve()
    shared = ROOT / "benchmarks" / "runners" / "regenerate_ion_structure_library.py"
    manifest: dict[str, Any] = {
        "schema_version": "otter_benchmark_manifest_v1",
        "benchmark_id": "al_full_workflow_1ev",
        "status": "candidate_not_accepted",
        "title": ("Complete Al 8.1 g/cc, Te=Ti=1 eV " "electronic-to-ionic workflow"),
        "data_rights": {
            "origin_type": "project_generated_numerical_output",
            "third_party_reference_data_included": False,
            "method_citations_only": True,
        },
        "method_references": [
            {
                "citation_key": "StarrettSaumon2014",
                "doi": "10.1016/j.hedp.2013.12.001",
                "scope": "average atom, pseudoatom, QOZ, and HNC workflow",
            },
            {
                "citation_key": "Chabrier1990",
                "doi": "10.1051/jphys:0199000510150160700",
                "scope": "finite-temperature jellium local-field correction",
            },
        ],
        "units": {
            "radius": "Bohr",
            "wavenumber": "Bohr^-1",
            "electronic_density": "Bohr^-3",
            "screening_cloud_k": "electron number",
            "ion_electron_form_factor_k": "electron number",
            "effective_pair_potential_r": "Hartree",
            "effective_pair_potential_k": "Hartree Bohr^3",
            "energy": "Hartree",
            "gii_and_sii": "dimensionless",
        },
        "producer": {
            "project": "Otter",
            "git_commit": _git_commit(),
            "script_relative_path": str(script.relative_to(ROOT)),
            "script_sha256": _sha256(script),
            "shared_generator_relative_path": str(shared.relative_to(ROOT)),
            "shared_generator_sha256": _sha256(shared),
            "worktree_clean_at_generation": not bool(worktree_status),
            "git_status_porcelain_sha256": hashlib.sha256(
                worktree_status.encode("utf-8")
            ).hexdigest(),
            "note": (
                "git_commit is a reproducible producer snapshot only when "
                "worktree_clean_at_generation is true."
            ),
        },
        "configuration": {
            "element": "Al",
            "rho_g_cc": 8.1,
            "te_ev": 1.0,
            "ti_ev": 1.0,
            "electronic_model": "qm",
            "structure_model": "IS",
            "bound_occ_mode": "fd",
            "bound_rmax_mult": None,
            "bound_zero_tail_refine": False,
            "b3_tail_model": "full",
            "chi0_model": "lindhard_fd",
            "lfc_model": "chabrier1990",
            "qoz_n_points_before_padding": 4096,
            "qoz_zbar_mode": "pseudoatom_partition",
            "qoz_renormalize_nscr_to_zbar": True,
            "hnc_tolerance": HNC_TOL,
            "hnc_transform_closure_tolerance": HNC_CLOSURE_TOL,
            "hnc_max_iterations": HNC_MAX_ITER,
            "r_retained_max_bohr": 20.0,
            "k_retained_max_bohr_inv": 20.0,
        },
        "state": {
            "state_id": STATE["state_id"],
            "data_file": output_path.name,
            "data_sha256": _sha256(output_path),
        },
    }
    if payload is not None:
        manifest["scientific_audit"] = {
            "zbar_aa_ws": float(payload["zbar_aa"]),
            "zbar_partition": float(payload["zbar_partition"]),
            "zbar_qoz": float(payload["zbar_qoz"]),
            "q_scr_native_raw": float(payload["q_scr_raw"]),
            "q_scr_grid_raw": float(payload["q_scr_grid_raw"]),
            "q_scr_used": float(payload["q_scr_used"]),
            "q_scr_scale_factor": float(payload["q_scr_scale_factor"]),
            "hnc_best_residual": float(payload["hnc_best_residual"]),
            "hnc_closure_mismatch": float(payload["hnc_closure_mismatch"]),
        }
    return manifest


def regenerate(*, output_path: Path = OUTPUT_PATH) -> Path:
    """Compute and write the candidate gallery archive."""
    producer = _load_library_regenerator()
    worktree_status = _git_status_porcelain()
    start = time.perf_counter()
    result = producer.solve_plasma_workflow(_configuration(producer))
    payload = _pack_gallery_result(
        result,
        elapsed_s=time.perf_counter() - start,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    manifest_path = output_path.parent / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            _candidate_manifest(
                output_path,
                worktree_status=worktree_status,
                payload=payload,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[saved] {output_path.relative_to(ROOT)} "
        f"({float(payload['producer_elapsed_s']):.1f} s)"
    )
    print(f"[saved] {manifest_path.relative_to(ROOT)}")
    print("Accepted documentation reference result was not modified.")
    return output_path


if __name__ == "__main__":
    regenerate()
