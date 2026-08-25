r"""
Two-temperature aluminium: Johnson et al. (2025)
=================================================

This benchmark compares Otter IS-QOZ :math:`g_{ii}(r)` obtained with ordinary
HNC and Rosenfeld--Ashcroft VMHNC closures with the three reference curve
families in Fig. 2(a)--2(d) of :cite:t:`JohnsonEtAl2025`.
The states are
aluminium at :math:`\rho=2.7` g cm\ :sup:`-3`,
:math:`T_i=1` eV, and :math:`T_e=1,3,10,30` eV.  The paper labels the
reference methods as 2TTCP HNC+bridge, DFT-MD, and YOCP HNC+bridge.  In every
panel, the two Otter integral-equation curves reuse exactly the same ion-
sphere (IS) electronic result and effective ion--ion potential; only the
ionic closure changes.  Classical LAMMPS MD :cite:p:`ThompsonEtAl2022` with
that same potential is also shown, so closure error can be separated from
pseudoatom pair-potential error.

The VMHNC bridge is the Percus--Yevick hard-sphere bridge with its packing
fraction fixed by the variational condition of :cite:t:`Faussurier2004`, not
by fitting these DFT-MD curves.  The comparison therefore tests a predictive
closure rather than a calibrated overlay.

At low electron temperature, a spherical pseudoatom/average-atom
construction cannot represent directional chemical bonding or transient
molecular structure.  A bridge function improves the classical ionic
closure, but it cannot restore electronic or chemical structure absent from
the pseudoatom pair potential.  Low-temperature disagreement is consequently
not, by itself, evidence that the bridge implementation is wrong.

Several similarly named models must not be conflated.  OCP denotes bare
Coulomb ions in a uniform neutralizing background; YOCP denotes a screened
Yukawa interaction.  IEMHNC maps the simulation-derived OCP bridge of
:cite:t:`IyetomiOgataIchimaru1992` to a YOCP state along an isomorph
:cite:p:`ToliasLuccoCastello2019`.  The earlier Otter panel-(d) study first
fitted the non-Yukawa QOZ potential at long wavelength to an effective YOCP;
it is therefore an IEMHNC-inspired diagnostic mapping, not a general IEMHNC
closure for arbitrary QOZ potentials.  Johnson's plotted YOCP curve instead
cites the earlier empirical Yukawa bridge of
:cite:t:`DaughtonMurilloThode2000`.  Neither is Otter's hard-sphere VMHNC.

The reference abscissa is :math:`r` in atomic units (Bohr), as shown on the
published Fig. 2 axis.  No coordinate conversion is applied.

Edit only the input block below.  ``USE_PRECOMPUTED_DATA = True`` verifies
the dedicated accepted-baseline manifest and every NPZ checksum.  With
``False``, this same file calls the public Otter workflow, writes candidate
results under ``benchmarks/outputs``, and plots them.
``USE_RECOMPUTED_CANDIDATES = True`` loads those still-unreviewed candidates
with their separate manifest and checksums for local review; it never promotes
or overwrites an accepted baseline.

When recomputing, ``RUN_SAME_POTENTIAL_MD = True`` follows Johnson's stated
dimensionless protocol: 2048 ions, :math:`\Delta t=0.005\omega_p^{-1}`;
:math:`50\omega_p^{-1}` NVT equilibration followed by
:math:`500\omega_p^{-1}` NVE production.  It requires ``lmp`` and ``mpirun``.

The project maintainer digitized the publication curves from Fig. 2(a)--2(d).
They are distributed with article/panel attribution and license status
``NOASSERTION``.
See :doc:`the provenance and reuse notice
</benchmarks/johnson_et_al_2025_two_temperature_al>`.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import physical_constants

from otter import (
    PlasmaWorkflowConfig,
    continue_plasma_workflow_from_electronic_result,
    solve_plasma_workflow,
)
from otter.plotting import grid_figsize, save_figure, set_style


# =============================================================================
# User input
# =============================================================================
USE_PRECOMPUTED_DATA = True
if os.environ.get("OTTER_RECOMPUTE_JOHNSON_AL", "0") == "1":
    USE_PRECOMPUTED_DATA = False
USE_RECOMPUTED_CANDIDATES = False
if os.environ.get("OTTER_USE_JOHNSON_CANDIDATES", "0") == "1":
    USE_RECOMPUTED_CANDIDATES = True

# A fresh benchmark recomputation also performs classical MD with the same
# Otter IS-QOZ pair potential.  Set this to False only when LAMMPS is absent.
RUN_SAME_POTENTIAL_MD = True
LAMMPS_EXECUTABLE = "lmp"
MPI_LAUNCHER = "mpirun"
MPI_PROCESSES_PER_MD_STATE = 10
MAX_PARALLEL_MD_STATES = 2
MD_CELLS_PER_AXIS = 8  # FCC: 4 * 8^3 = 2048 Al ions.
MD_TIMESTEP_OMEGA_P_INV = 5.0e-3
MD_EQUILIBRATION_OMEGA_P_INV = 50.0
MD_PRODUCTION_OMEGA_P_INV = 500.0
MD_RDF_BINS = 500

# Two states run concurrently; each AA solve parallelizes its continuum channels.
MAX_STATE_WORKERS = 2
CONTINUUM_WORKERS_PER_STATE = 6
QOZ_N_POINTS = 4096

LFC_MODEL = "chabrier1990"
HNC_TOL = 1.0e-4
HNC_CLOSURE_TOL = 2.5e-3
VMHNC_ETA_TOL = 1.0e-6
R_RETAIN_MAX_BOHR = 20.0
K_RETAIN_MAX_BOHR_INV = 20.0
# =============================================================================


BENCHMARK_ID = "johnson_et_al_2025_two_temperature_al"
STATES: tuple[dict[str, Any], ...] = (
    {
        "state_id": "al_rho2p7_te1_ti1",
        "te_ev": 1.0,
        "panel": "Fig. 2(a)",
    },
    {
        "state_id": "al_rho2p7_te3_ti1",
        "te_ev": 3.0,
        "panel": "Fig. 2(b)",
    },
    {
        "state_id": "al_rho2p7_te10_ti1",
        "te_ev": 10.0,
        "panel": "Fig. 2(c)",
    },
    {
        "state_id": "al_rho2p7_te30_ti1",
        "te_ev": 30.0,
        "panel": "Fig. 2(d)",
    },
)
REFERENCE_METHODS: tuple[tuple[str, str, str], ...] = (
    ("AA", "2TTCP HNC+bridge", "o"),
    ("DFTMD", "DFT-MD", "x"),
    ("YOCP", "YOCP HNC+bridge", "s"),
)


def repository_root() -> Path:
    """Locate the checkout when run directly or through Sphinx-Gallery."""
    candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    source_file = globals().get("__file__")
    if source_file is not None:
        source = Path(str(source_file)).resolve()
        candidates.extend([source.parent, *source.parents])
    for candidate in candidates:
        manifest = (
            candidate
            / "benchmarks"
            / "baselines"
            / BENCHMARK_ID
            / "manifest.json"
        )
        if manifest.is_file():
            return candidate
    raise FileNotFoundError("Cannot locate the Otter checkout.")


ROOT = repository_root()
BASELINE_DIR = ROOT / "benchmarks" / "baselines" / BENCHMARK_ID
REFERENCE_DIR = ROOT / "benchmarks" / "reference_data" / BENCHMARK_ID
OUTPUT_DIR = ROOT / "benchmarks" / "outputs" / BENCHMARK_ID
CANDIDATE_DIR = OUTPUT_DIR / "gallery_recomputed"
FIGURE_DIR = OUTPUT_DIR / "figures"
SCRIPT_PATH = ROOT / "benchmarks" / "examples" / Path(
    "plot_johnson_et_al_2025_two_temperature_al.py"
)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load only portable numeric/string arrays."""
    with np.load(path, allow_pickle=False) as archive:
        result = {key: np.asarray(archive[key]) for key in archive.files}
    if any(value.dtype.hasobject for value in result.values()):
        raise TypeError(f"Object arrays are forbidden in {path}.")
    return result


def load_precomputed_states() -> dict[str, dict[str, np.ndarray]]:
    """Load all and only accepted, checksummed dedicated baselines."""
    manifest = json.loads(
        (BASELINE_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("Unexpected baseline manifest.")
    expected_ids = {str(state["state_id"]) for state in STATES}
    records = {
        str(record["state_id"]): record for record in manifest["states"]
    }
    if not records or not set(records).issubset(expected_ids):
        raise RuntimeError("Baseline manifest state coverage is invalid.")
    terminal_statuses = {"accepted", "reference_only_strict_hnc_rejected"}
    unresolved = {
        state_id: str(record.get("status"))
        for state_id, record in records.items()
        if record.get("status") not in terminal_statuses
    }
    if unresolved:
        detail = ", ".join(
            f"{state_id}={status}"
            for state_id, status in sorted(unresolved.items())
        )
        raise RuntimeError(
            "The dedicated benchmark manifest still contains unreviewed "
            f"states ({detail}). Set USE_PRECOMPUTED_DATA = False to "
            "calculate candidates directly; review them before promotion."
        )

    loaded: dict[str, dict[str, np.ndarray]] = {}
    for state_id, record in records.items():
        if record["status"] == "reference_only_strict_hnc_rejected":
            if record.get("baseline_file") is not None:
                raise RuntimeError(
                    f"Rejected state {state_id} must not name a baseline."
                )
            continue
        filename = record.get("baseline_file")
        expected_hash = record.get("baseline_sha256")
        if not filename or not expected_hash:
            raise RuntimeError(f"Accepted record {state_id} is incomplete.")
        path = BASELINE_DIR / str(filename)
        if sha256_file(path) != str(expected_hash):
            raise RuntimeError(f"Checksum mismatch for {path}.")
        payload = load_npz(path)
        if str(payload["state_id"].item()) != state_id:
            raise ValueError(f"State identifier mismatch in {path}.")
        loaded[state_id] = payload
    missing = expected_ids - set(records)
    if missing:
        print(
            "Accepted baseline does not yet contain: "
            + ", ".join(sorted(missing))
        )
    return loaded


def load_recomputed_candidates() -> dict[str, dict[str, np.ndarray]]:
    """Load a complete candidate set without treating it as accepted data."""
    manifest = json.loads(
        (CANDIDATE_DIR / "candidate_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("Unexpected candidate manifest.")
    records = {
        str(record["state_id"]): record for record in manifest["states"]
    }
    expected_ids = {str(state["state_id"]) for state in STATES}
    if set(records) != expected_ids:
        raise RuntimeError("Candidate manifest does not cover all four states.")

    loaded: dict[str, dict[str, np.ndarray]] = {}
    for state_id, record in records.items():
        if record.get("status") != "candidate_unreviewed":
            raise RuntimeError(
                f"Candidate {state_id} is not reviewable: {record.get('status')}."
            )
        path = CANDIDATE_DIR / str(record["baseline_file"])
        if sha256_file(path) != str(record["baseline_sha256"]):
            raise RuntimeError(f"Candidate checksum mismatch for {path}.")
        loaded[state_id] = load_npz(path)
    return loaded


def workflow_config(
    state: dict[str, Any],
    *,
    bridge_model: str = "none",
) -> PlasmaWorkflowConfig:
    """Build one IS workflow; only ``bridge_model`` changes between curves."""
    aa_overrides: dict[str, Any] = {
        "cont_n_jobs": int(CONTINUUM_WORKERS_PER_STATE),
        "cont_shards": int(2 * CONTINUUM_WORKERS_PER_STATE),
    }
    return PlasmaWorkflowConfig(
        elements=["Al"],
        temperature_ev=float(state["te_ev"]),
        ion_temperature_ev=1.0,
        rho_g_cc=2.7,
        aa_overrides=aa_overrides,
        hnc_tol=float(HNC_TOL),
        hnc_closure_transform_tol=float(HNC_CLOSURE_TOL),
        hnc_max_iter=500,
        hnc_bridge_model=str(bridge_model),
        vmhnc_eta_tol=float(VMHNC_ETA_TOL),
        show_progress=False,
        verbose=False,
    )


def _pack_structure_result(
    workflow: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Validate and retain one IS electronic/QOZ structure."""
    electronic = dict(workflow["electronic"]["result"])
    ion = dict(workflow["ion"])
    if electronic.get("stage2_converged") is not True:
        raise RuntimeError("Full average-atom stage 2 did not converge.")
    if dict(electronic.get("ext_status", {})).get("converged") is not True:
        raise RuntimeError("External fixed-mu average atom did not converge.")
    if str(electronic.get("threshold_state_status", "")).lower() == "unresolved":
        raise RuntimeError("The threshold-state representation is unresolved.")
    if ion.get("hnc_converged") is not True:
        raise RuntimeError("HNC did not reach a physical fixed point.")
    if float(ion["hnc_output_residual"]) > HNC_TOL:
        raise RuntimeError("HNC residual exceeds the configured tolerance.")
    if float(ion["closure_transform_max_abs"]) > HNC_CLOSURE_TOL:
        raise RuntimeError("The g/S transform-closure audit failed.")

    r = np.asarray(ion["r"], dtype=float)
    k = np.asarray(ion["k"], dtype=float)
    r_electronic = np.asarray(electronic["r"], dtype=float)
    r_mask = r <= R_RETAIN_MAX_BOHR
    k_mask = k <= K_RETAIN_MAX_BOHR_INV
    return {
        "r_bohr": r[r_mask],
        "gii_r": np.asarray(ion["gii_r"], dtype=float)[r_mask],
        "vii_r_ha": np.asarray(ion["vii_r"], dtype=float)[r_mask],
        "k_bohr_inv": k[k_mask],
        "sii_k": np.asarray(ion["sii_k"], dtype=float)[k_mask],
        "ion_density_bohr3": np.asarray(float(ion["n_i"])),
        "mu_ha": np.asarray(float(electronic["mu"])),
        "r_ws_bohr": np.asarray(float(electronic["r_ws"])),
        "zbar_partition": np.asarray(float(ion["zbar_partition"])),
        "bound_charge": np.asarray(
            4.0
            * np.pi
            * np.trapezoid(
                np.asarray(electronic["n_bound"], dtype=float)
                * r_electronic**2,
                r_electronic,
            )
        ),
        "hnc_solver_path": np.asarray(str(ion["hnc_solver_path"])),
        "hnc_fallback_used": np.asarray(bool(ion["hnc_fallback_used"])),
        "hnc_primary_best_residual": np.asarray(
            float(ion["hnc_primary_best_residual"])
        ),
        "hnc_best_residual": np.asarray(float(ion["hnc_output_residual"])),
        "hnc_s_min": np.asarray(float(ion["hnc_s_min"])),
        "hnc_closure_mismatch": np.asarray(
            float(ion["closure_transform_max_abs"])
        ),
        "hnc_bridge_model": np.asarray(str(ion["hnc_bridge_model"])),
    }


def pack_result(
    hnc_workflow: dict[str, Any],
    vmhnc_workflow: dict[str, Any],
    state: dict[str, Any],
    hnc_elapsed_s: float,
    vmhnc_elapsed_s: float,
) -> dict[str, np.ndarray]:
    """Store paired HNC/VMHNC closures from one IS electronic result."""
    hnc = _pack_structure_result(hnc_workflow)
    vmhnc = _pack_structure_result(vmhnc_workflow)
    vmhnc_ion = dict(vmhnc_workflow["ion"])
    if str(hnc["hnc_bridge_model"].item()) != "none":
        raise RuntimeError("The ordinary-HNC curve unexpectedly used a bridge.")
    if str(vmhnc["hnc_bridge_model"].item()) != "rosenfeld_ashcroft":
        raise RuntimeError("The VMHNC curve did not use Rosenfeld--Ashcroft.")
    if not np.isclose(
        float(hnc["mu_ha"]),
        float(vmhnc["mu_ha"]),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("HNC and VMHNC did not reuse the same IS electronic state.")

    payload = {
        "schema_version": np.asarray("otter_johnson_2025_al_v3"),
        "state_id": np.asarray(str(state["state_id"])),
        "rho_g_cc": np.asarray(2.7),
        "te_ev": np.asarray(float(state["te_ev"])),
        "ti_ev": np.asarray(1.0),
        "paper_panel": np.asarray(str(state["panel"])),
        "structure_model": np.asarray("IS"),
        "producer_elapsed_s": np.asarray(
            float(hnc_elapsed_s + vmhnc_elapsed_s)
        ),
        "hnc_elapsed_s": np.asarray(float(hnc_elapsed_s)),
        "vmhnc_elapsed_s": np.asarray(float(vmhnc_elapsed_s)),
        **hnc,
        **{
            f"vmhnc_{key}": value
            for key, value in vmhnc.items()
            if key
            in {
                "r_bohr",
                "gii_r",
                "k_bohr_inv",
                "sii_k",
                "hnc_solver_path",
                "hnc_fallback_used",
                "hnc_primary_best_residual",
                "hnc_best_residual",
                "hnc_s_min",
                "hnc_closure_mismatch",
                "hnc_bridge_model",
            }
        },
        "vmhnc_eta": np.asarray(float(vmhnc_ion["vmhnc_eta"])),
        "vmhnc_sigma_bohr": np.asarray(
            float(vmhnc_ion["vmhnc_sigma_bohr"])
        ),
        "vmhnc_variational_residual": np.asarray(
            float(vmhnc_ion["vmhnc_variational_residual"])
        ),
    }
    metadata = {
        "archive_role": "project_generated_example_or_benchmark_baseline",
        "archive_schema_version": "otter_johnson_2025_al_v3",
        "citation_keys": list(
            workflow_config(state, bridge_model="vmhnc").citation_keys
        )
        + (["ThompsonEtAl2022"] if RUN_SAME_POTENTIAL_MD else []),
        "closures": ["HNC", "Rosenfeld--Ashcroft VMHNC"],
        "electronic_state_reused_between_closures": True,
        "same_potential_md": {
            "enabled": bool(RUN_SAME_POTENTIAL_MD),
            "engine_citation_key": "ThompsonEtAl2022",
            "ensemble_sequence": "NVT->NVE",
        },
        "producer": {
            "script_relative_path": str(SCRIPT_PATH.relative_to(ROOT)),
            "script_sha256": sha256_file(SCRIPT_PATH),
            **git_revision(),
        },
    }
    payload["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    return payload


def _write_lammps_pair_table(
    path: Path,
    r_bohr: np.ndarray,
    potential_ha: np.ndarray,
) -> tuple[int, float]:
    """Write a shifted-force Al pair table in LAMMPS metal units."""
    bohr_to_angstrom = 0.529177210903
    selected = (r_bohr >= 0.1) & (r_bohr <= R_RETAIN_MAX_BOHR)
    radius = np.asarray(r_bohr[selected], dtype=float) * bohr_to_angstrom
    energy = np.asarray(potential_ha[selected], dtype=float) * 27.211386245988
    force = -np.gradient(energy, radius, edge_order=2)

    # Make both V and F continuous where the tabulated pair list ends.
    force_cutoff = force[-1]
    energy = energy - energy[-1] + (radius - radius[-1]) * force_cutoff
    force = force - force_cutoff
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# Otter IS-QOZ Al-Al pair potential\n\nAL_AL\n")
        stream.write(f"N {radius.size}\n\n")
        for index, (r_value, v_value, f_value) in enumerate(
            zip(radius, energy, force),
            start=1,
        ):
            stream.write(
                f"{index} {r_value:.12e} {v_value:.12e} {f_value:.12e}\n"
            )
    return int(radius.size), float(radius[-1])


def _read_lammps_rdf(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return radius, block-mean RDF, and block standard error."""
    rows = [
        line.split()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    blocks: list[np.ndarray] = []
    cursor = 0
    while cursor < len(rows):
        block_size = int(rows[cursor][1])
        blocks.append(
            np.asarray(rows[cursor + 1 : cursor + 1 + block_size], dtype=float)
        )
        cursor += block_size + 1
    if len(blocks) < 2:
        raise RuntimeError("LAMMPS produced fewer than two RDF blocks.")
    values = np.stack(blocks)
    g_blocks = values[:, :, 2]
    return (
        values[0, :, 1] / 0.529177210903,
        np.mean(g_blocks, axis=0),
        np.std(g_blocks, axis=0, ddof=1) / np.sqrt(g_blocks.shape[0]),
    )


def _read_lammps_nve_audit(path: Path) -> tuple[float, float, str]:
    """Return relative NVE energy drift, mean temperature, and version."""
    lines = path.read_text(encoding="utf-8").splitlines()
    headers = [
        index
        for index, line in enumerate(lines)
        if line.split()[:5] == ["Step", "Temp", "PotEng", "KinEng", "TotEng"]
    ]
    if len(headers) < 2:
        raise RuntimeError("LAMMPS log does not contain the NVE thermo block.")
    thermo_rows: list[list[float]] = []
    for line in lines[headers[-1] + 1 :]:
        if line.startswith("Loop time"):
            break
        values = line.split()
        if len(values) < 7:
            continue
        try:
            thermo_rows.append([float(value) for value in values[:7]])
        except ValueError:
            continue
    if len(thermo_rows) < 2:
        raise RuntimeError("LAMMPS NVE thermo block is incomplete.")
    initial_energy = thermo_rows[0][4]
    final_energy = thermo_rows[-1][4]
    relative_drift = (final_energy - initial_energy) / abs(initial_energy)
    mean_temperature = float(np.mean([row[1] for row in thermo_rows]))
    version = lines[0].removeprefix("LAMMPS ").strip()
    return float(relative_drift), mean_temperature, version


def run_same_potential_md(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Run Johnson's NVT-to-NVE protocol with the stored Otter pair potential."""
    lammps = shutil.which(LAMMPS_EXECUTABLE)
    launcher = shutil.which(MPI_LAUNCHER)
    if lammps is None or launcher is None:
        raise FileNotFoundError("LAMMPS and an MPI launcher are required for MD.")

    state_id = str(payload["state_id"].item())
    workdir = CANDIDATE_DIR / "md_work" / state_id
    workdir.mkdir(parents=True, exist_ok=True)
    table_points, cutoff_angstrom = _write_lammps_pair_table(
        workdir / "al_al.table",
        np.asarray(payload["r_bohr"], dtype=float),
        np.asarray(payload["vii_r_ha"], dtype=float),
    )

    n_i = float(payload["ion_density_bohr3"])
    zbar = float(payload["zbar_partition"])
    aluminium_mass_u = 26.9815385
    mass_ratio = (
        physical_constants["atomic mass constant"][0]
        / physical_constants["electron mass"][0]
    )
    atomic_time_ps = physical_constants["atomic unit of time"][0] * 1.0e12
    omega_p_au = np.sqrt(
        4.0 * np.pi * n_i * zbar**2 / (aluminium_mass_u * mass_ratio)
    )
    timestep_ps = (
        MD_TIMESTEP_OMEGA_P_INV / omega_p_au * atomic_time_ps
    )
    thermostat_damp_ps = 0.5 / omega_p_au * atomic_time_ps
    equilibration_steps = int(
        round(MD_EQUILIBRATION_OMEGA_P_INV / MD_TIMESTEP_OMEGA_P_INV)
    )
    production_steps = int(
        round(MD_PRODUCTION_OMEGA_P_INV / MD_TIMESTEP_OMEGA_P_INV)
    )
    rdf_every = 100
    rdf_repeat = 50
    rdf_block_steps = rdf_every * rdf_repeat
    lattice_angstrom = (4.0 / (n_i / 0.529177210903**3)) ** (1.0 / 3.0)
    seed = 20260825 + int(round(float(payload["te_ev"]) * 10.0))
    cells = MD_CELLS_PER_AXIS
    temperature_k = 11604.51812155008
    rdf_fix = (
        f"fix             rdf all ave/time {rdf_every} {rdf_repeat} "
        f"{rdf_block_steps} c_gr[*] mode vector ave one file rdf_blocks.dat"
    )
    thermostat_fix = (
        f"fix             thermostat all nvt temp {temperature_k:.8f} "
        f"{temperature_k:.8f} {thermostat_damp_ps:.12e}"
    )
    input_text = f"""# Same-potential MD for {state_id}
units           metal
dimension       3
boundary        p p p
atom_style      atomic

lattice         fcc {lattice_angstrom:.12f}
region          box block 0 {cells} 0 {cells} 0 {cells} units lattice
create_box      1 box
create_atoms    1 box
mass            1 {aluminium_mass_u:.10f}

pair_style      table linear {table_points}
pair_coeff      1 1 al_al.table AL_AL {cutoff_angstrom:.12f}
neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes

velocity        all create {temperature_k:.8f} {seed} mom yes rot no dist gaussian
timestep        {timestep_ps:.12e}
thermo          1000
thermo_style    custom step temp pe ke etotal press density
thermo_modify   flush yes

{thermostat_fix}
run             {equilibration_steps}
unfix           thermostat

reset_timestep  0
fix             integrator all nve
compute         gr all rdf {MD_RDF_BINS} 1 1
{rdf_fix}
run             {production_steps}
"""
    input_path = workdir / "in.al_md"
    input_path.write_text(input_text, encoding="utf-8")
    command = [
        launcher,
        "-np",
        str(MPI_PROCESSES_PER_MD_STATE),
        lammps,
        "-in",
        input_path.name,
        "-log",
        "log.lammps",
    ]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    started = time.perf_counter()
    with (workdir / "screen.log").open("w", encoding="utf-8") as screen:
        subprocess.run(
            command,
            cwd=workdir,
            env=environment,
            stdout=screen,
            stderr=subprocess.STDOUT,
            check=True,
        )
    radius, g_r, sem = _read_lammps_rdf(workdir / "rdf_blocks.dat")
    energy_drift, mean_temperature, lammps_version = _read_lammps_nve_audit(
        workdir / "log.lammps"
    )
    return {
        "md_r_bohr": radius,
        "md_gii_r": g_r,
        "md_gii_block_sem": sem,
        "md_atoms": np.asarray(4 * MD_CELLS_PER_AXIS**3),
        "md_rdf_blocks": np.asarray(production_steps // rdf_block_steps),
        "md_timestep_omega_p_inv": np.asarray(MD_TIMESTEP_OMEGA_P_INV),
        "md_equilibration_omega_p_inv": np.asarray(
            MD_EQUILIBRATION_OMEGA_P_INV
        ),
        "md_production_omega_p_inv": np.asarray(
            MD_PRODUCTION_OMEGA_P_INV
        ),
        "md_ensemble_sequence": np.asarray("NVT->NVE"),
        "md_nve_relative_energy_drift": np.asarray(energy_drift),
        "md_nve_mean_temperature_k": np.asarray(mean_temperature),
        "md_lammps_version": np.asarray(lammps_version),
        "md_elapsed_s": np.asarray(time.perf_counter() - started),
    }


def add_same_potential_md(
    states: dict[str, dict[str, np.ndarray]],
) -> None:
    """Add independent same-potential MD results with bounded MPI concurrency."""
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_MD_STATES) as pool:
        futures = {
            pool.submit(run_same_potential_md, payload): state_id
            for state_id, payload in states.items()
        }
        for future in as_completed(futures):
            state_id = futures[future]
            states[state_id].update(future.result())
            print(f"[computed MD] {state_id}")


def solve_state(state: dict[str, Any]) -> dict[str, np.ndarray]:
    """Calculate one IS state, then solve both ionic closures on it."""
    config = workflow_config(state, bridge_model="none")
    started = time.perf_counter()
    hnc_workflow = solve_plasma_workflow(config)
    hnc_elapsed_s = time.perf_counter() - started

    electronic = dict(hnc_workflow["electronic"])
    started = time.perf_counter()
    vmhnc_workflow = continue_plasma_workflow_from_electronic_result(
        workflow_config(state, bridge_model="rosenfeld_ashcroft"),
        electronic_kind=str(electronic["kind"]),
        electronic_result=dict(electronic["result"]),
    )
    return pack_result(
        hnc_workflow,
        vmhnc_workflow,
        state,
        hnc_elapsed_s,
        time.perf_counter() - started,
    )


def git_revision() -> dict[str, Any]:
    """Record the checkout revision without pretending dirty output is committed."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "worktree_dirty": None}
    return {"git_commit": head, "worktree_dirty": dirty}


def refresh_archive_metadata(payload: dict[str, np.ndarray]) -> None:
    """Bring a completed closure/MD payload onto the compact archive schema."""
    metadata = json.loads(str(payload["metadata_json"].item()))
    metadata["schema_version"] = "otter_compact_archive_metadata_v1"
    metadata["configuration"] = {
        "structure_model": "IS",
        "lfc_model": LFC_MODEL,
        "ionic_closures": ["HNC", "Rosenfeld--Ashcroft VMHNC"],
        "same_potential_md": bool("md_gii_r" in payload),
        "hnc_tolerance": HNC_TOL,
        "vmhnc_eta_tolerance": VMHNC_ETA_TOL,
    }
    metadata["state"] = {
        "state_id": str(payload["state_id"].item()),
        "element": "Al",
        "rho_g_cc": float(payload["rho_g_cc"]),
        "te_ev": float(payload["te_ev"]),
        "ti_ev": float(payload["ti_ev"]),
    }
    metadata["producer"]["project"] = "Otter"
    metadata["convergence"] = {
        "hnc_best_residual": float(payload["hnc_best_residual"]),
        "hnc_closure_mismatch": float(payload["hnc_closure_mismatch"]),
        "vmhnc_best_residual": float(payload["vmhnc_hnc_best_residual"]),
        "vmhnc_closure_mismatch": float(
            payload["vmhnc_hnc_closure_mismatch"]
        ),
        "vmhnc_variational_residual": float(
            payload["vmhnc_variational_residual"]
        ),
        "md_nve_relative_energy_drift": (
            float(payload["md_nve_relative_energy_drift"])
            if "md_nve_relative_energy_drift" in payload
            else None
        ),
    }
    metadata["fields"] = sorted(
        key for key in payload if key != "metadata_json"
    )
    payload["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )


def save_candidates(
    states: dict[str, dict[str, np.ndarray]],
    failures: dict[str, str],
) -> None:
    """Write candidates and a manifest with real file hashes."""
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for definition in STATES:
        state_id = str(definition["state_id"])
        if state_id in failures:
            records.append(
                {
                    "state_id": state_id,
                    "status": "strict_calculation_rejected",
                    "baseline_file": None,
                    "baseline_sha256": None,
                    "reason": failures[state_id],
                }
            )
            continue
        filename = f"{state_id}.npz"
        path = CANDIDATE_DIR / filename
        refresh_archive_metadata(states[state_id])
        np.savez_compressed(path, **states[state_id])
        record = {
            "state_id": state_id,
            "status": "candidate_unreviewed",
            "baseline_file": filename,
            "baseline_sha256": sha256_file(path),
            "structure_model": "IS",
            "ionic_closures": ["HNC", "Rosenfeld--Ashcroft VMHNC"],
            "zbar_partition": float(states[state_id]["zbar_partition"]),
            "vmhnc_eta": float(states[state_id]["vmhnc_eta"]),
            "vmhnc_variational_residual": float(
                states[state_id]["vmhnc_variational_residual"]
            ),
            "same_potential_md": "md_gii_r" in states[state_id],
        }
        records.append(record)
        print(f"[candidate] {path}")

    candidate_manifest = {
        "schema_version": "otter_candidate_manifest_v1",
        "benchmark_id": BENCHMARK_ID,
        "producer": {
            "script_relative_path": (
                "benchmarks/examples/"
                "plot_johnson_et_al_2025_two_temperature_al.py"
            ),
            # Sphinx-Gallery executes generated source without defining
            # ``__file__``; ROOT was already located in both execution modes.
            "script_sha256": sha256_file(SCRIPT_PATH),
            **git_revision(),
        },
        "configuration": {
            "aa_n_points": 4096,
            "qoz_n_points_before_padding": QOZ_N_POINTS,
            "continuum_workers_per_state": CONTINUUM_WORKERS_PER_STATE,
            "lfc_model": LFC_MODEL,
            "hnc_tolerance": HNC_TOL,
            "hnc_transform_closure_tolerance": HNC_CLOSURE_TOL,
            "hnc_primary_solver": "anderson",
            "hnc_fallback_solver": "newton_krylov",
            "hnc_potential_scales": [
                float(value)
                for value in workflow_config(STATES[0]).hnc_potential_scales
            ],
            "hnc_adaptive_continuation": bool(
                workflow_config(STATES[0]).hnc_adaptive_continuation
            ),
            "ionic_closures": ["HNC", "Rosenfeld--Ashcroft VMHNC"],
            "electronic_state_reused_between_closures": True,
            "vmhnc_points_per_diameter": int(
                workflow_config(STATES[0]).vmhnc_points_per_diameter
            ),
            "vmhnc_eta_bounds": [
                float(value)
                for value in workflow_config(STATES[0]).vmhnc_eta_bounds
            ],
            "vmhnc_eta_tolerance": float(
                workflow_config(STATES[0]).vmhnc_eta_tol
            ),
            "same_potential_md": {
                "enabled": bool(RUN_SAME_POTENTIAL_MD),
                "atoms": 4 * MD_CELLS_PER_AXIS**3,
                "ensemble_sequence": "NVT->NVE",
                "timestep_omega_p_inv": MD_TIMESTEP_OMEGA_P_INV,
                "equilibration_omega_p_inv": MD_EQUILIBRATION_OMEGA_P_INV,
                "production_omega_p_inv": MD_PRODUCTION_OMEGA_P_INV,
                "rdf_bins": MD_RDF_BINS,
            },
        },
        "states": records,
    }
    (CANDIDATE_DIR / "candidate_manifest.json").write_text(
        json.dumps(candidate_manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def solve_all_states() -> dict[str, dict[str, np.ndarray]]:
    """Calculate the four independent states with bounded parallelism."""
    solved: dict[str, dict[str, np.ndarray]] = {}
    failures: dict[str, str] = {}
    if MAX_STATE_WORKERS <= 1:
        for state in STATES:
            state_id = str(state["state_id"])
            try:
                solved[state_id] = solve_state(state)
            except Exception as exc:
                failures[state_id] = f"{type(exc).__name__}: {exc}"
                print(f"[strictly rejected] {state_id}: {exc}")
            else:
                print(f"[computed] {state_id}")
    else:
        with ProcessPoolExecutor(max_workers=MAX_STATE_WORKERS) as pool:
            futures = {pool.submit(solve_state, state): state for state in STATES}
            for future in as_completed(futures):
                state = futures[future]
                state_id = str(state["state_id"])
                try:
                    solved[state_id] = future.result()
                except Exception as exc:
                    failures[state_id] = f"{type(exc).__name__}: {exc}"
                    print(f"[strictly rejected] {state_id}: {exc}")
                else:
                    print(f"[computed] {state_id}")
    if not solved:
        raise RuntimeError("Every Otter state failed its strict checks.")
    if RUN_SAME_POTENTIAL_MD:
        add_same_potential_md(solved)
    save_candidates(solved, failures)
    return solved


def reference_filename(te_ev: float, token: str) -> str:
    return (
        f"Zak_2025_Al_rho2.7_Te{te_ev:.1f}_Ti1.0_{token}.csv"
    )


def load_references() -> dict[tuple[float, str], tuple[np.ndarray, np.ndarray]]:
    """Verify the attributed reference manifest and load unchanged Bohr values."""
    manifest = json.loads(
        (REFERENCE_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("reference_id") != BENCHMARK_ID:
        raise ValueError("Unexpected reference-data manifest.")
    if manifest.get("units", {}).get("column_1") != "Bohr":
        raise ValueError("Johnson Fig. 2 reference radius must be Bohr.")
    by_path = {str(item["path"]): item for item in manifest["files"]}
    references: dict[tuple[float, str], tuple[np.ndarray, np.ndarray]] = {}
    for state in STATES:
        te_ev = float(state["te_ev"])
        for token, _label, _marker in REFERENCE_METHODS:
            filename = reference_filename(te_ev, token)
            record = by_path.get(filename)
            if record is None:
                raise FileNotFoundError(f"Missing manifest record {filename}.")
            path = REFERENCE_DIR / filename
            if sha256_file(path) != str(record["sha256"]):
                raise RuntimeError(f"Checksum mismatch for {path}.")
            values = np.asarray(np.genfromtxt(path, delimiter=","), dtype=float)
            if values.ndim != 2 or values.shape[1] < 2:
                raise ValueError(f"Expected two columns in {path}.")
            mask = np.isfinite(values[:, 0]) & np.isfinite(values[:, 1])
            references[(te_ev, token)] = (
                values[mask, 0],
                values[mask, 1],
            )
    return references


if not USE_PRECOMPUTED_DATA:
    states = solve_all_states()
elif USE_RECOMPUTED_CANDIDATES:
    states = load_recomputed_candidates()
else:
    states = load_precomputed_states()
references = load_references()
print(
    "Using "
    + (
        "new candidates calculated directly by this gallery script."
        if not USE_PRECOMPUTED_DATA
        else (
            "checksummed, unreviewed recomputation candidates."
            if USE_RECOMPUTED_CANDIDATES
            else "reviewed, checksummed Otter baselines."
        )
    )
)


def print_metrics() -> None:
    """Report interpolation errors without treating unlike methods as exact."""
    print(
        f"{'state':24s} {'Otter closure':16s} {'reference':20s} "
        f"{'RMSE':>10s} {'MAE':>10s} {'max':>10s}"
    )
    for definition in STATES:
        state_id = str(definition["state_id"])
        if state_id not in states:
            print(f"{state_id:24s} {'no accepted Otter state':20s}")
            continue
        te_ev = float(definition["te_ev"])
        payload = states[state_id]
        closures = (
            ("HNC", "r_bohr", "gii_r"),
            ("VMHNC", "vmhnc_r_bohr", "vmhnc_gii_r"),
            ("same-potential MD", "md_r_bohr", "md_gii_r"),
        )
        for closure, r_key, g_key in closures:
            if r_key not in payload:
                continue
            r_otter = np.asarray(payload[r_key], dtype=float)
            g_otter = np.asarray(payload[g_key], dtype=float)
            for token, label, _marker in REFERENCE_METHODS:
                r_ref, g_ref = references[(te_ev, token)]
                mask = (r_ref >= r_otter[0]) & (r_ref <= r_otter[-1])
                delta = np.interp(r_ref[mask], r_otter, g_otter) - g_ref[mask]
                print(
                    f"{state_id:24s} {closure:16s} {label:20s} "
                    f"{np.sqrt(np.mean(delta**2)):10.4e} "
                    f"{np.mean(np.abs(delta)):10.4e} "
                    f"{np.max(np.abs(delta)):10.4e}"
                )
        print(
            f"{state_id}: shared IS Zbar={float(payload['zbar_partition']):.8f}, "
            f"VMHNC eta={float(payload['vmhnc_eta']):.8f}, "
            "variational residual="
            f"{float(payload['vmhnc_variational_residual']):.3e}"
        )


print_metrics()


# %%
# Pair-distribution comparison
# ----------------------------
#
# Both Otter curves use the same pseudoatom IS-QOZ potential.  Their difference
# isolates the ionic closure; neither is relabelled as the paper's 2TTCP model.
# At low Te, neither closure can restore chemical bonding absent from the
# spherical pseudoatom pair potential.

set_style("thesis", palette="bing")
fig, axes = plt.subplots(
    2,
    2,
    figsize=grid_figsize(2, 2),
    sharex=True,
    sharey=True,
)
axes = np.asarray(axes).ravel()
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for panel_index, (axis, definition) in enumerate(zip(axes, STATES)):
    state_id = str(definition["state_id"])
    te_ev = float(definition["te_ev"])
    if state_id in states:
        payload = states[state_id]
        axis.plot(
            np.asarray(payload["r_bohr"]),
            np.asarray(payload["gii_r"]),
            color="black",
            lw=2.3,
            label="Otter IS-QOZ/HNC",
            zorder=2,
        )
        axis.plot(
            np.asarray(payload["vmhnc_r_bohr"]),
            np.asarray(payload["vmhnc_gii_r"]),
            color=colors[3 % len(colors)],
            lw=2.1,
            ls="--",
            alpha=0.9,
            label="Otter IS-QOZ/VMHNC",
            zorder=2,
        )
        if "md_gii_r" in payload:
            r_md = np.asarray(payload["md_r_bohr"], dtype=float)
            g_md = np.asarray(payload["md_gii_r"], dtype=float)
            sem_md = np.asarray(payload["md_gii_block_sem"], dtype=float)
            axis.plot(
                r_md,
                g_md,
                color=colors[4 % len(colors)],
                lw=1.9,
                ls="-.",
                label="Otter IS-potential MD",
                zorder=2,
            )
            axis.fill_between(
                r_md,
                g_md - 2.0 * sem_md,
                g_md + 2.0 * sem_md,
                color=colors[4 % len(colors)],
                alpha=0.12,
                linewidth=0.0,
                zorder=1,
            )
    else:
        axis.text(
            0.97,
            0.05,
            "No accepted Otter baseline",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            color="0.25",
            fontsize=9,
        )
    for method_index, (token, label, marker) in enumerate(REFERENCE_METHODS):
        r_ref, g_ref = references[(te_ev, token)]
        options: dict[str, Any] = {
            "s": 40,
            "marker": marker,
            "linewidths": 1.3,
            "label": label if panel_index == 0 else "_nolegend_",
            "zorder": 3,
        }
        color = colors[method_index % len(colors)]
        if marker == "x":
            options["color"] = color
        else:
            options["facecolors"] = "none"
            options["edgecolors"] = color
        axis.scatter(r_ref, g_ref, **options)

    axis.set_title(
        rf"{definition['panel']}: $T_e={te_ev:g}$ eV, $T_i=1$ eV"
    )
    axis.set_xlabel(r"$r$ [Bohr]")
    axis.set_xlim(2.5, 8.2)
    axis.set_ylim(-0.05, 2.30)
    axis.axhline(1.0, color="0.55", lw=0.8, ls=":")
    if panel_index == 0:
        axis.set_ylabel(r"$g_{ii}(r)$")
    if panel_index == 0:
        axis.legend(
            fontsize="small",
            loc="best",
        )

fig.suptitle(
    r"Al, $\rho=2.7$ g cm$^{-3}$: "
    "Otter versus Johnson et al. (2025)",
    y=0.985,
)
fig.text(
    0.5,
    0.006,
    "Reference data: Johnson et al. (2025), "
    "doi:10.1103/5c29-kdx1.",
    ha="center",
    va="bottom",
    fontsize=7.5,
)
fig.tight_layout(rect=(0.0, 0.025, 1.0, 0.95), pad=0.55)
saved_paths = save_figure(
    fig,
    FIGURE_DIR / "johnson_et_al_2025_two_temperature_al_gii",
)
print(
    "[figure] "
    + ", ".join(
        f"{kind}={path.relative_to(ROOT)}"
        for kind, path in saved_paths.items()
    )
)
if "agg" not in plt.get_backend().lower():
    plt.show()
