"""Regenerate the Otter side of the ion-structure library benchmark.

This program is intentionally separate from the offline documentation runner.
It performs expensive average-atom and QOZ/HNC calculations and writes only
to ``benchmarks/outputs/ion_structure_library/recomputed``.  It never
overwrites an accepted reference result.

Independent thermodynamic state groups run concurrently. Each average-atom
calculation inherits Otter's single-worker default.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
from scipy.constants import physical_constants


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from otter.electronic.full_external import FullExternalConfig
from otter import (  # noqa: E402
    PlasmaWorkflowConfig,
    continue_plasma_workflow_from_electronic_result,
    solve_plasma_workflow,
)
from otter_lammps_md import (  # noqa: E402
    MDConfig,
    MDSpecies,
    PairPotential,
    run_otter_lammps_md,
)


# Parallelize independent states, not the energies within one AA.
MAX_STATE_WORKERS = 3
R_RETAIN_MAX_BOHR = 20.0
K_RETAIN_MAX_BOHR_INV = 20.0
VMHNC_ETA_TOL = 1.0e-6

# The Wünsch Be state can additionally compare HNC and VMHNC with classical MD
# using the identical IS-QOZ pair potential.  Full Otter-only recomputations
# disable this optional external calculation through the environment.
RUN_WUNSCH_SAME_POTENTIAL_MD = (
    os.environ.get("OTTER_RUN_WUNSCH_SAME_POTENTIAL_MD", "1") == "1"
)
REUSE_COMPLETED_GROUPS = (
    os.environ.get("OTTER_REUSE_ION_STRUCTURE_GROUPS", "0") == "1"
)
LAMMPS_EXECUTABLE = "lmp"
MPI_LAUNCHER = "mpirun"
MPI_PROCESSES_PER_MD_CASE = 10
MD_ATOMS = 2048
MD_TIMESTEP_OMEGA_P_INV = 0.005
MD_EQUILIBRATION_OMEGA_P_INV = 50.0
MD_PRODUCTION_OMEGA_P_INV = 500.0
WUNSCH_MD_K_MAX_ANGSTROM_INV = 10.2

# ``None`` reproduces the complete library.  A tuple such as
# ``("be_wunsch",)`` recomputes only the named state group.
STATE_GROUPS_TO_RUN: tuple[str, ...] | None = None

OUTPUT_DIR = ROOT / "benchmarks" / "outputs" / "ion_structure_library" / "recomputed"
SCHEMA = "otter_ion_structure_library_state_v2"


STATE_GROUPS: dict[str, tuple[dict[str, Any], ...]] = {
    "al_gill": (
        {
            "state_id": "al_gill_rho2p7_te5_ti5",
            "element": "Al",
            "rho_g_cc": 2.7,
            "te_ev": 5.0,
            "ti_ev": 5.0,
        },
    ),
    "al_clerouin": (
        {
            "state_id": "al_clerouin_rho8p1_te10_ti10",
            "element": "Al",
            "rho_g_cc": 8.1,
            "te_ev": 10.0,
            "ti_ev": 10.0,
        },
        {
            "state_id": "al_clerouin_rho8p1_te10_ti2",
            "element": "Al",
            "rho_g_cc": 8.1,
            "te_ev": 10.0,
            "ti_ev": 2.0,
        },
    ),
    "al_clerouin_tf": (
        {
            "state_id": "al_clerouin_rho8p1_te10_ti10_tf",
            "element": "Al",
            "rho_g_cc": 8.1,
            "te_ev": 10.0,
            "ti_ev": 10.0,
            "electronic_model": "tf",
        },
        {
            "state_id": "al_clerouin_rho8p1_te10_ti2_tf",
            "element": "Al",
            "rho_g_cc": 8.1,
            "te_ev": 10.0,
            "ti_ev": 2.0,
            "electronic_model": "tf",
        },
    ),
    "be_wunsch": (
        {
            "state_id": "be_wunsch_rho5p544_te13_ti13",
            "element": "Be",
            "rho_g_cc": 5.544,
            "te_ev": 13.0,
            "ti_ev": 13.0,
        },
    ),
    "c_starrett_hot": (
        {
            "state_id": "c_starrett_rho20_te50_ti50",
            "element": "C",
            "rho_g_cc": 20.0,
            "te_ev": 50.0,
            "ti_ev": 50.0,
        },
    ),
}


def _git_commit() -> str:
    """Return the source revision without storing an absolute checkout path."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configuration(
    state: dict[str, Any],
    *,
    bridge_model: str = "none",
) -> PlasmaWorkflowConfig:
    """Build the documented production configuration for one state."""
    model = str(state.get("electronic_model", "qm"))
    model_override = {} if model == "qm" else {"electronic_model": model}
    return PlasmaWorkflowConfig(
        elements=[str(state["element"])],
        temperature_ev=float(state["te_ev"]),
        ion_temperature_ev=float(state["ti_ev"]),
        rho_g_cc=float(state["rho_g_cc"]),
        **model_override,
        aa_overrides={
            "bound_zero_tail_refine": True,
            "bound_zero_tail_max_binding_ha": 1.0e-2,
            "bound_zero_tail_scan_points": 64,
            "bound_zero_tail_edge_rel_tol": 0.1,
        },
        # Keep the nonlinear root strict while independently allowing the
        # measured ~2e-3 finite-DST g<->S mismatch of cold, strongly coupled
        # Al.  This does not relax positivity or fixed-point checks.
        hnc_closure_transform_tol=2.5e-3,
        hnc_max_iter=1000,
        hnc_bridge_model=bridge_model,
        vmhnc_eta_tol=VMHNC_ETA_TOL,
    )


def _pack_result(
    result: dict[str, Any],
    state: dict[str, Any],
    *,
    elapsed_s: float,
) -> dict[str, np.ndarray]:
    """Convert a workflow payload to a compact pickle-free state archive."""
    electronic = dict(result["electronic"]["result"])
    ion = dict(result["ion"])
    r_i = np.asarray(ion["r"], dtype=float)
    k = np.asarray(ion["k"], dtype=float)
    ion_mask = r_i <= R_RETAIN_MAX_BOHR
    k_mask = k <= K_RETAIN_MAX_BOHR_INV

    signature = {
        "state": state,
        "structure_model": "IS",
        # Store the complete resolved workflow configuration.  The producer
        # code above only spells out non-default overrides, while this record
        # remains reproducible if a future Otter release changes its defaults.
        "resolved_configuration": result.get(
            "configuration", asdict(_configuration(state))
        ),
    }
    if str(state["state_id"]).startswith("be_wunsch"):
        signature["ionic_comparison"] = {
            "closures": ["HNC", "Rosenfeld--Ashcroft VMHNC"],
            "same_potential_md": bool(RUN_WUNSCH_SAME_POTENTIAL_MD),
            "vmhnc_eta_tol": VMHNC_ETA_TOL,
        }
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA),
        "benchmark_id": np.asarray("ion_structure_library"),
        "storage_profile": np.asarray("benchmark_analysis"),
        "state_id": np.asarray(str(state["state_id"])),
        "electronic_model": np.asarray(str(state.get("electronic_model", "qm"))),
        "element": np.asarray(str(state["element"])),
        "rho_g_cc": np.asarray(float(state["rho_g_cc"])),
        "te_ev": np.asarray(float(state["te_ev"])),
        "ti_ev": np.asarray(float(state["ti_ev"])),
        "otter_git_commit": np.asarray(_git_commit()),
        "producer_signature_json": np.asarray(
            json.dumps(signature, sort_keys=True, separators=(",", ":"))
        ),
        "producer_elapsed_s": np.asarray(float(elapsed_s)),
        "n0_bohr3": np.asarray(float(electronic["n0"])),
        "r_ws_bohr": np.asarray(float(electronic["r_ws"])),
        "mu_ha": np.asarray(float(electronic["mu"])),
        "zbar_aa": np.asarray(float(electronic["zbar"])),
        "zbar_partition": np.asarray(float(ion["zbar_partition"])),
        "zbar_qoz": np.asarray(float(ion["zbar_qoz"])),
        "ion_density_bohr3": np.asarray(float(ion["n_i"])),
        "threshold_state_status": np.asarray(
            str(electronic.get("threshold_state_status", "none"))
        ),
        "threshold_state_representation": np.asarray(
            str(electronic.get("threshold_state_representation", "none"))
        ),
        "q_scr_raw": np.asarray(float(ion["zbar_screening_integral_raw"])),
        "r_bohr": r_i[ion_mask],
        "k_bohr_inv": k[k_mask],
        "gii_r": np.asarray(ion["gii_r"], dtype=float)[ion_mask],
        "sii_k": np.asarray(ion["sii_k"], dtype=float)[k_mask],
        "hnc_best_residual": np.asarray(float(ion["hnc_best_residual"])),
        "hnc_closure_mismatch": np.asarray(float(ion["closure_transform_max_abs"])),
        "hnc_closure_tolerance": np.asarray(float(ion["closure_transform_tol"])),
        "hnc_iters": np.asarray(int(ion["hnc_iters"])),
    }
    for key, value in payload.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"Object dtype is forbidden for {key!r}.")
        if array.dtype.kind in "fiu" and not np.all(np.isfinite(array)):
            # Missing optional potential components are not acceptable in a
            # canonical output; fail loudly instead of publishing a partial
            # gallery state.
            raise ValueError(f"Non-finite values in {key!r}.")
    return payload


def _add_wunsch_vmhnc(
    payload: dict[str, np.ndarray],
    hnc_result: dict[str, Any],
    vmhnc_result: dict[str, Any],
) -> None:
    """Append VMHNC fields after proving that only the closure changed."""
    hnc = dict(hnc_result["ion"])
    vmhnc = dict(vmhnc_result["ion"])
    if str(hnc["hnc_bridge_model"]) != "none":
        raise RuntimeError("The Wünsch ordinary-HNC result used a bridge.")
    if str(vmhnc["hnc_bridge_model"]) != "rosenfeld_ashcroft":
        raise RuntimeError("The Wünsch VMHNC result used the wrong bridge.")
    if not np.allclose(hnc["vii_r"], vmhnc["vii_r"], rtol=0.0, atol=0.0):
        raise RuntimeError("HNC and VMHNC did not reuse the same QOZ potential.")
    if vmhnc.get("hnc_converged") is not True:
        raise RuntimeError("Wünsch VMHNC did not converge.")
    if float(vmhnc["closure_transform_max_abs"]) > 2.5e-3:
        raise RuntimeError("Wünsch VMHNC failed the transform-closure audit.")

    r = np.asarray(vmhnc["r"], dtype=float)
    k = np.asarray(vmhnc["k"], dtype=float)
    r_mask = r <= R_RETAIN_MAX_BOHR
    k_mask = k <= K_RETAIN_MAX_BOHR_INV
    payload.update(
        {
            "vmhnc_gii_r": np.asarray(vmhnc["gii_r"], dtype=float)[r_mask],
            "vmhnc_sii_k": np.asarray(vmhnc["sii_k"], dtype=float)[k_mask],
            "vmhnc_best_residual": np.asarray(float(vmhnc["hnc_output_residual"])),
            "vmhnc_closure_mismatch": np.asarray(
                float(vmhnc["closure_transform_max_abs"])
            ),
            "vmhnc_eta": np.asarray(float(vmhnc["vmhnc_eta"])),
            "vmhnc_sigma_bohr": np.asarray(float(vmhnc["vmhnc_sigma_bohr"])),
            "vmhnc_variational_residual": np.asarray(
                float(vmhnc["vmhnc_variational_residual"])
            ),
        }
    )


def _wunsch_md_config(
    state: dict[str, Any],
    ion: dict[str, Any],
    output_dir: Path,
) -> MDConfig:
    """Use a dimensionless plasma-frequency protocol for the Be MD run."""
    mass_u = 9.0121831
    mass_ratio = (
        physical_constants["atomic mass constant"][0]
        / physical_constants["electron mass"][0]
    )
    atomic_time_ps = physical_constants["atomic unit of time"][0] * 1.0e12
    n_i = float(ion["n_i"])
    zbar = float(ion["zbar_partition"])
    omega_p = np.sqrt(4.0 * np.pi * n_i * zbar**2 / (mass_u * mass_ratio))
    return MDConfig(
        output_dir=output_dir,
        species=(MDSpecies("Be", mass_u, MD_ATOMS),),
        ion_density_bohr3=n_i,
        ion_temperature_ev=float(state["ti_ev"]),
        timestep_ps=MD_TIMESTEP_OMEGA_P_INV / omega_p * atomic_time_ps,
        thermostat_damp_ps=0.5 / omega_p * atomic_time_ps,
        equilibration_steps=round(
            MD_EQUILIBRATION_OMEGA_P_INV / MD_TIMESTEP_OMEGA_P_INV
        ),
        production_steps=round(MD_PRODUCTION_OMEGA_P_INV / MD_TIMESTEP_OMEGA_P_INV),
        rdf_bins=500,
        rdf_every=100,
        rdf_repeat=50,
        trajectory_every=5_000,
        table_points=8_192,
        k_max_angstrom_inv=WUNSCH_MD_K_MAX_ANGSTROM_INV,
        random_seed=20_260_825,
        lammps_executable=LAMMPS_EXECUTABLE,
        mpi_launcher=MPI_LAUNCHER,
        mpi_processes=MPI_PROCESSES_PER_MD_CASE,
        structure_factor_workers=MPI_PROCESSES_PER_MD_CASE,
    )


def _add_wunsch_md(
    payload: dict[str, np.ndarray],
    result: dict[str, Any],
    state: dict[str, Any],
    output_dir: Path,
) -> None:
    """Run Be MD with the HNC/VMHNC pair potential and append its statistics."""
    ion = dict(result["ion"])
    md = run_otter_lammps_md(
        _wunsch_md_config(state, ion, output_dir),
        [
            PairPotential(
                "Be",
                "Be",
                np.asarray(ion["r"], dtype=float),
                np.asarray(ion["vii_r"], dtype=float),
            )
        ],
    )
    payload.update(md)
    # Friendly one-component aliases keep gallery access explicit.
    payload["md_gii_r"] = np.asarray(md["md_gij_r"])[:, 0]
    payload["md_gii_block_sem"] = np.asarray(md["md_gij_block_sem"])[:, 0]
    payload["md_sii_k"] = np.asarray(md["md_snn_k"])
    payload["md_sii_frame_sem"] = np.asarray(md["md_snn_frame_sem"])
    payload["md_sii_vectors_per_bin"] = np.asarray(md["md_vectors_per_k_bin"])
    for key in (
        "md_gij_r",
        "md_gij_block_sem",
        "md_snn_k",
        "md_snn_frame_sem",
        "md_sij_k",
        "md_sij_frame_sem",
        "md_vectors_per_k_bin",
    ):
        payload.pop(key, None)


def _attach_metadata(
    payload: dict[str, np.ndarray],
    state: dict[str, Any],
) -> None:
    """Embed portable provenance after every optional result is attached."""
    state_id = str(state["state_id"])
    citation_keys = ["StarrettSaumon2013", "StarrettSaumon2014", "Chabrier1990"]
    if state_id.startswith("be_wunsch"):
        citation_keys += [
            "WunschEtAl2009",
            "RosenfeldAshcroft1979",
            "Faussurier2004",
            "ThompsonEtAl2022",
        ]
    metadata = {
        "schema_version": "otter_compact_archive_metadata_v1",
        "archive_role": "project_generated_example_or_benchmark_baseline",
        "archive_schema_version": SCHEMA,
        "package_id": "ion_structure_library",
        "configuration": json.loads(str(payload["producer_signature_json"].item())),
        "state": {
            "state_id": state_id,
            "element": str(state["element"]),
            "rho_g_cc": float(state["rho_g_cc"]),
            "te_ev": float(state["te_ev"]),
            "ti_ev": float(state["ti_ev"]),
            "electronic_model": str(state.get("electronic_model", "qm")),
        },
        "producer": {
            "project": "Otter",
            "git_commit": _git_commit(),
            "script_relative_path": str(Path(__file__).resolve().relative_to(ROOT)),
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "citation_keys": citation_keys,
        "convergence": {
            "aa_stage2_converged": True,
            "aa_ext_converged": True,
            "threshold_state_status": str(payload["threshold_state_status"].item()),
            "hnc_best_residual": float(payload["hnc_best_residual"]),
            "hnc_closure_mismatch": float(payload["hnc_closure_mismatch"]),
            **(
                {
                    "vmhnc_best_residual": float(payload["vmhnc_best_residual"]),
                    "vmhnc_closure_mismatch": float(payload["vmhnc_closure_mismatch"]),
                    "vmhnc_variational_residual": float(
                        payload["vmhnc_variational_residual"]
                    ),
                }
                if "vmhnc_best_residual" in payload
                else {}
            ),
            **(
                {
                    "md_nve_relative_energy_drift": float(
                        payload["md_nve_relative_energy_drift"]
                    )
                }
                if "md_nve_relative_energy_drift" in payload
                else {}
            ),
        },
        "fields": sorted(payload),
    }
    payload["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )


def _solve_group(
    group_name: str,
    states: tuple[dict[str, Any], ...],
    output_dir: Path,
) -> dict[str, dict[str, np.ndarray]]:
    """Solve one electronic state and all requested ion temperatures."""
    first = states[0]
    start = time.perf_counter()
    first_result = solve_plasma_workflow(_configuration(first))
    first_id = str(first["state_id"])
    packed = {
        first_id: _pack_result(
            first_result, first, elapsed_s=time.perf_counter() - start
        )
    }
    electronic_kind = str(first_result["electronic"]["kind"])
    electronic_result = dict(first_result["electronic"]["result"])
    if group_name == "be_wunsch":
        vmhnc_started = time.perf_counter()
        vmhnc_result = continue_plasma_workflow_from_electronic_result(
            _configuration(first, bridge_model="rosenfeld_ashcroft"),
            electronic_kind=electronic_kind,
            electronic_result=electronic_result,
        )
        _add_wunsch_vmhnc(packed[first_id], first_result, vmhnc_result)
        packed[first_id]["vmhnc_elapsed_s"] = np.asarray(
            time.perf_counter() - vmhnc_started
        )
        if RUN_WUNSCH_SAME_POTENTIAL_MD:
            _add_wunsch_md(
                packed[first_id],
                first_result,
                first,
                output_dir / "md_work" / first_id,
            )
    _attach_metadata(packed[first_id], first)
    if len(states) == 1:
        return packed

    for state in states[1:]:
        ion_start = time.perf_counter()
        result = continue_plasma_workflow_from_electronic_result(
            _configuration(state),
            electronic_kind=electronic_kind,
            electronic_result=electronic_result,
        )
        payload = _pack_result(
            result,
            state,
            elapsed_s=time.perf_counter() - ion_start,
        )
        _attach_metadata(payload, state)
        packed[str(state["state_id"])] = payload
    return packed


def regenerate(
    *,
    output_dir: Path = OUTPUT_DIR,
    max_state_workers: int = MAX_STATE_WORKERS,
) -> list[Path]:
    """Run all groups and return the written portable NPZ paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    failures: list[tuple[str, str]] = []
    selected = (
        STATE_GROUPS
        if STATE_GROUPS_TO_RUN is None
        else {name: STATE_GROUPS[name] for name in STATE_GROUPS_TO_RUN}
    )
    if REUSE_COMPLETED_GROUPS:
        pending: dict[str, tuple[dict[str, Any], ...]] = {}
        for group, states in selected.items():
            paths = [output_dir / f"{state['state_id']}.npz" for state in states]
            complete = True
            for state, path in zip(states, paths, strict=True):
                try:
                    with np.load(path, allow_pickle=False) as archive:
                        complete = (
                            str(archive["schema_version"].item()) == SCHEMA
                            and str(archive["state_id"].item())
                            == str(state["state_id"])
                        )
                except (OSError, KeyError, ValueError):
                    complete = False
                if not complete:
                    break
            if complete:
                written.extend(paths)
                print(f"[reused] group={group}")
            else:
                pending[group] = states
        selected = pending
    if not selected:
        return written

    workers = max(1, min(int(max_state_workers), len(selected)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_solve_group, name, states, output_dir): name
            for name, states in selected.items()
        }
        for future in as_completed(futures):
            group = futures[future]
            try:
                results = future.result()
            except Exception as exc:
                failures.append((group, f"{type(exc).__name__}: {exc}"))
                print(f"[failed] group={group}: {type(exc).__name__}: {exc}")
                continue
            for state_id, payload in results.items():
                path = output_dir / f"{state_id}.npz"
                np.savez_compressed(path, **payload)
                written.append(path)
                print(
                    f"[saved] {state_id}: {path.relative_to(ROOT)} "
                    f"({float(payload['producer_elapsed_s']):.1f} s)"
                )
            print(f"[complete] group={group}")
    if failures:
        details = "; ".join(f"{group}: {reason}" for group, reason in failures)
        raise RuntimeError(
            "One or more core benchmark groups failed after successful "
            f"states were preserved: {details}"
        )
    return sorted(written)


def main() -> None:
    print(
        "Otter recomputation: "
        f"{MAX_STATE_WORKERS} state workers x "
        f"{FullExternalConfig.cont_n_jobs} continuum workers"
    )
    paths = regenerate()
    print(f"Wrote {len(paths)} staged states under {OUTPUT_DIR.relative_to(ROOT)}")
    print("Accepted reference results were not modified.")


if __name__ == "__main__":
    # Guard required by multiprocessing on spawn-based platforms.
    main()
