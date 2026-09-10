"""Reproduce the CH2 AA -> QOZ/HNC -> LAMMPS comparison from physical inputs.

Run ``python benchmarks/examples/plot_ch2_hnc_md.py`` for all six figures.
The three electronic states and nine MD cases are calculated afresh. This
requires LAMMPS/MPI and can take hours; progress and per-case timings are printed.
Numerical results stay under benchmarks/outputs/ch2_hnc_md/recomputed.
The input block records the full MD protocol, including the hot-state core
regularization (not a multicomponent VMHNC bridge).
"""

from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Keep NumPy/SciPy from multiplying the 16 explicit LAMMPS MPI ranks by hidden
# BLAS threads.  Change these values here only if the hardware allocation changes.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np
from scipy.constants import physical_constants


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(ROOT / "benchmarks/outputs/ch2_hnc_md/.matplotlib"),
)

import matplotlib.pyplot as plt  # noqa: E402

from otter import (  # noqa: E402
    PlasmaWorkflowConfig,
    solve_plasma_workflow,
    continue_plasma_workflow_from_electronic_result,
    prepare_multicomponent_ion_structure_from_electronic_result,
)
from otter.plotting import (  # noqa: E402
    PAIR_COLORS,
    grid_figsize,
    save_figure,
    set_style,
)


# ============================== Inputs =====================================
TARGET_TE_EV = (9.0, 29.0, 99.0)
TI_OVER_TE = (0.2, 0.5, 1.0)
OUTPUT_DIR = Path(os.environ.get("OTTER_CH2_OUTPUT_DIR", ROOT / "benchmarks/outputs/ch2_hnc_md/recomputed"))
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = ROOT / OUTPUT_DIR

# 1024 C + 2048 H gives exact CH2 stoichiometry.  Cases run sequentially and
# LAMMPS uses only 16 MPI ranks so the workstation remains responsive.
CH2_FORMULA_UNITS = 1024
LAMMPS_MPI_PROCESSES = 16
STRUCTURE_FACTOR_WORKERS = 8

# These reproduce the benchmark NVT -> NVE protocol in units of the fastest
# species plasma period.  The 0.005 step gives 10,000 + 100,000 MD steps.
TIMESTEP_OMEGA_P_INV = 5.0e-3
EQUILIBRATION_OMEGA_P_INV = 50.0
PRODUCTION_OMEGA_P_INV = 500.0
# The plasma-frequency step alone does not shrink as Ti raises the hydrogen
# thermal speed. Limit the one-step H RMS displacement at hot states; the
# number of steps is increased below so the sampled physical time is unchanged.
MAX_THERMAL_STEP_BOHR = 3.0e-3
THERMAL_GUARD_MIN_TI_EV = 40.0
DEFAULT_MD_R_MIN_BOHR = 0.1
HOT_MD_R_MIN_BOHR = 1.0e-5
MAX_NVE_RELATIVE_ENERGY_DRIFT = 2.0e-3
# A finite-k DST representation rings around the exact Zbar_i Zbar_j/r core.
# Hot MD states reach that otherwise irrelevant region, so replace only the
# core and blend back to the unchanged QOZ potential before the first shell.
COULOMB_CORE_BOHR = 0.3
COULOMB_BLEND_END_BOHR = 0.5
MD_TABLE_POINTS = 8192
MD_RDF_BINS = 500
MD_STATISTICAL_BLOCKS = 20
MD_RDF_REPEAT = 50
MD_K_MAX_ANGSTROM_INV = 8.3
MD_K_BIN_WIDTH_ANGSTROM_INV = 0.1
MIN_VECTORS_PER_K_BIN = 12

PROGRESS_INTERVAL_S = 30.0
REUSE_COMPLETED_CASES = False
RUN_MD = os.environ.get("OTTER_CH2_RUN_MD", "1") == "1"
RECOMPUTE_ELECTRONIC = True
REQUIRE_ALL_CASES = True
# ============================================================================


RHO_G_CC = 0.946
ELEMENTS = ("C", "H")
COUNTS = (1.0, 2.0)
BRIDGE_STATUS = "not_implemented_for_multicomponent_ch2"
PAIR_INDICES = ((0, 0), (0, 1), (1, 1))
PAIR_LABELS = ("CC", "CH", "HH")


def _load_md_module():
    path = ROOT / "tools/otter_lammps_md.py"
    spec = importlib.util.spec_from_file_location("otter_lammps_md", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


md = _load_md_module()


def workflow_config(te_ev: float, ti_ev: float | None) -> PlasmaWorkflowConfig:
    """Use production defaults; Ti=None selects the electronic-only workflow."""
    return PlasmaWorkflowConfig(
        elements=list(ELEMENTS),
        counts=list(COUNTS),
        temperature_ev=float(te_ev),
        ion_temperature_ev=None if ti_ev is None else float(ti_ev),
        rho_g_cc=RHO_G_CC,
        show_progress=False,
    )


def calculate_electronic(te_ev: float) -> tuple[str, dict[str, Any]]:
    """Run a cold common-mu/full+external solve; never read an old cache."""
    print(f"\nTe={te_ev:g} eV: cold common-mu + full/external AA", flush=True)
    cfg = workflow_config(te_ev, None)
    cfg.show_progress = True
    workflow = solve_plasma_workflow(cfg)
    kind = workflow["electronic"]["kind"]
    result = workflow["electronic"]["result"]
    if not result["meta"]["final_electronic_eligible"]:
        raise RuntimeError("Fresh electronic state is not eligible for QOZ")
    _atomic_json(OUTPUT_DIR / f"electronic_Te{te_ev:g}.json", {
        "configuration": asdict(cfg), "runtime_s": workflow["runtime_s"],
        "final_electronic_eligible": True,
    })
    return kind, result


def plasma_clock(prepared: Any) -> tuple[float, float, float]:
    """Return omega_max [a.u.], MD timestep [ps], and thermostat damping [ps]."""
    mass_u = np.asarray((12.011, 1.008), dtype=float)
    mass_ratio = (
        physical_constants["atomic mass constant"][0]
        / physical_constants["electron mass"][0]
    )
    omega = np.sqrt(
        4.0
        * np.pi
        * np.asarray(prepared.n_i, dtype=float)
        * np.asarray(prepared.zbar, dtype=float) ** 2
        / (mass_u * mass_ratio)
    )
    omega_max = float(np.max(omega))
    atomic_time_ps = physical_constants["atomic unit of time"][0] * 1.0e12
    return (
        omega_max,
        TIMESTEP_OMEGA_P_INV / omega_max * atomic_time_ps,
        0.5 / omega_max * atomic_time_ps,
    )


def md_controls(prepared: Any, ti_ev: float) -> dict[str, float | bool]:
    """Add a thermal-motion guard to the plasma-frequency MD clock."""
    omega_max, base_timestep_ps, thermostat_damp_ps = plasma_clock(prepared)
    mass_ratio = (
        physical_constants["atomic mass constant"][0]
        / physical_constants["electron mass"][0]
    )
    temperature_ha = ti_ev / physical_constants["Hartree energy in eV"][0]
    mass_u = np.asarray((12.011, 1.008), dtype=float)
    fastest_rms_speed_au = float(
        np.max(np.sqrt(3.0 * temperature_ha / (mass_u * mass_ratio)))
    )
    base_timestep_au = TIMESTEP_OMEGA_P_INV / omega_max
    base_thermal_step_bohr = fastest_rms_speed_au * base_timestep_au
    scale = (
        min(1.0, MAX_THERMAL_STEP_BOHR / base_thermal_step_bohr)
        if ti_ev >= THERMAL_GUARD_MIN_TI_EV
        else 1.0
    )
    timestep_omega_p_inv = TIMESTEP_OMEGA_P_INV * scale
    thermal_limited = scale < 1.0 - 1.0e-12
    r_min_bohr = HOT_MD_R_MIN_BOHR if thermal_limited else DEFAULT_MD_R_MIN_BOHR
    return {
        "omega_max_au": omega_max,
        "timestep_ps": base_timestep_ps * scale,
        "thermostat_damp_ps": thermostat_damp_ps,
        "timestep_omega_p_inv": timestep_omega_p_inv,
        "thermal_step_bohr": base_thermal_step_bohr * scale,
        "thermal_limited": thermal_limited,
        "r_min_bohr": r_min_bohr,
    }


def regularize_coulomb_core(
    r_bohr: np.ndarray,
    potential_matrix_ha: np.ndarray,
    zbar: np.ndarray,
) -> np.ndarray:
    """Restore the exact Coulomb core without changing resolved separations."""
    r = np.asarray(r_bohr, dtype=float)
    matrix = np.asarray(potential_matrix_ha, dtype=float)
    charges = np.asarray(zbar, dtype=float)
    if matrix.shape != (charges.size, charges.size, r.size):
        raise ValueError("Potential matrix, zbar, and radius shapes do not match.")
    fit = (r >= COULOMB_CORE_BOHR) & (r <= COULOMB_BLEND_END_BOHR)
    if np.count_nonzero(fit) < 4:
        raise ValueError("The QOZ grid does not resolve the Coulomb blend window.")

    result = matrix.copy()
    blend_x = np.clip(
        (r - COULOMB_CORE_BOHR) / (COULOMB_BLEND_END_BOHR - COULOMB_CORE_BOHR),
        0.0,
        1.0,
    )
    blend = blend_x**3 * (10.0 + blend_x * (-15.0 + 6.0 * blend_x))
    for left in range(charges.size):
        for right in range(left, charges.size):
            coefficient = charges[left] * charges[right]
            raw = 0.5 * (matrix[left, right] + matrix[right, left])
            regular_remainder = raw - coefficient / r
            offset = float(np.median(regular_remainder[fit]))
            analytic = coefficient / r + offset
            repaired = (1.0 - blend) * analytic + blend * raw
            result[left, right] = repaired
            result[right, left] = repaired
    return result


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _case_signature(
    te_ev: float, ti_ev: float, controls: dict[str, float | bool]
) -> str:
    payload = {
        "schema": "ch2_hnc_md_comparison_v1",
        "run_md": RUN_MD,
        "te_ev": te_ev,
        "ti_ev": ti_ev,
        "formula_units": CH2_FORMULA_UNITS,
        "mpi_processes": LAMMPS_MPI_PROCESSES,
        "dt_omega_p_inv": TIMESTEP_OMEGA_P_INV,
        "equilibration_omega_p_inv": EQUILIBRATION_OMEGA_P_INV,
        "production_omega_p_inv": PRODUCTION_OMEGA_P_INV,
        "table_points": MD_TABLE_POINTS,
        "rdf_bins": MD_RDF_BINS,
        "k_max_angstrom_inv": MD_K_MAX_ANGSTROM_INV,
        "k_bin_width_angstrom_inv": MD_K_BIN_WIDTH_ANGSTROM_INV,
    }
    # Preserve the signatures of accepted cooler runs. Only states requiring
    # the high-T guard are invalidated and recomputed.
    if bool(controls["thermal_limited"]):
        payload["high_temperature_md_guard"] = {
            "max_thermal_step_bohr": MAX_THERMAL_STEP_BOHR,
            "minimum_ti_ev": THERMAL_GUARD_MIN_TI_EV,
            "timestep_omega_p_inv": controls["timestep_omega_p_inv"],
            "r_min_bohr": controls["r_min_bohr"],
            "coulomb_core_bohr": COULOMB_CORE_BOHR,
            "coulomb_blend_end_bohr": COULOMB_BLEND_END_BOHR,
            "analytic_coulomb_overlay": True,
            "statistical_blocks": MD_STATISTICAL_BLOCKS,
            "rdf_repeat": MD_RDF_REPEAT,
        }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _completed_case(case_dir: Path, signature: str) -> dict[str, Any] | None:
    path = case_dir / "case_summary.json"
    if not REUSE_COMPLETED_CASES or not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    required = [case_dir / "hnc_results.npz"]
    if RUN_MD:
        required.append(case_dir / "md/md_results.npz")
    reusable = record.get("status") in {"success", "partial"}
    return (
        record
        if reusable
        and record.get("signature") == signature
        and all(path.is_file() for path in required)
        else None
    )


def _lammps_progress(
    screen: Path, equilibration_steps: int, production_steps: int
) -> tuple[str, float]:
    if not screen.is_file():
        return "waiting for LAMMPS output", 0.0
    text = screen.read_text(encoding="utf-8", errors="replace")
    if "Total wall time:" in text:
        return "LAMMPS complete; analysing RDF and S(k)", 1.0
    phase = -1
    steps = [0.0, 0.0]
    for line in text.splitlines():
        tokens = line.split()
        if tokens[:5] == ["Step", "Temp", "PotEng", "KinEng", "TotEng"]:
            phase += 1
        elif phase in (0, 1) and tokens:
            try:
                steps[phase] = max(steps[phase], float(tokens[0]))
            except ValueError:
                pass
    completed = min(steps[0], equilibration_steps) + min(steps[1], production_steps)
    total = equilibration_steps + production_steps
    label = "NVT equilibration" if phase <= 0 else "NVE production"
    return f"{label}, step={steps[max(phase, 0)]:.0f}", completed / total


def _monitor_md(
    case_dir: Path, config: Any, stop: threading.Event, started: float
) -> None:
    while not stop.wait(PROGRESS_INTERVAL_S):
        label, fraction = _lammps_progress(
            case_dir / "md/screen.log",
            int(config.equilibration_steps),
            int(config.production_steps),
        )
        elapsed = time.perf_counter() - started
        eta = "unknown"
        if 0.0 < fraction < 1.0:
            eta = f"{elapsed * (1.0 - fraction) / fraction / 60.0:.1f} min"
        print(
            f"      MD progress: {100.0 * fraction:5.1f}% | {label} | "
            f"elapsed={elapsed / 60.0:.1f} min | case ETA={eta}",
            flush=True,
        )


def save_hnc(case_dir: Path, te_ev: float, ti_ev: float, ion: dict[str, Any]) -> None:
    """Save only the HNC quantities used by the comparison and XRTS analysis."""
    np.savez_compressed(
        case_dir / "hnc_results.npz",
        schema_version=np.asarray("ch2_hnc_result_v2"),
        storage_profile=np.asarray("ion_structure"),
        te_ev=np.asarray(te_ev),
        ti_ev=np.asarray(ti_ev),
        bridge_status=np.asarray(BRIDGE_STATUS),
        species=np.asarray(ELEMENTS),
        r_bohr=np.asarray(ion["r"], dtype=float),
        k_bohr_inv=np.asarray(ion["k"], dtype=float),
        gij_r=np.asarray(ion["gij_r"], dtype=float),
        sij_k=np.asarray(ion["sij_k"], dtype=float),
        f_k=np.asarray(ion["f_k"], dtype=float),
        q_k=np.asarray(ion["q_k"], dtype=float),
        n_i_bohr3=np.asarray(ion["n_i"], dtype=float),
        zbar=np.asarray(ion["zbar"], dtype=float),
        hnc_output_residual=np.asarray(ion["hnc_output_residual"]),
        closure_transform_max_abs=np.asarray(ion["closure_transform_max_abs"]),
        hnc_solve_s=np.asarray(ion["hnc_solve_s"]),
    )


def plot_comparison(
    case_dir: Path,
    hnc: dict[str, Any],
    md_result: dict[str, np.ndarray],
    te_ev: float,
    ti_ev: float,
) -> None:
    set_style("thesis", palette="bing")
    fig, axes = plt.subplots(2, 3, figsize=grid_figsize(2, 3), sharex="row")
    r_hnc = np.asarray(hnc["r"], dtype=float)
    k_hnc = np.asarray(hnc["k"], dtype=float)
    r_md = np.asarray(md_result["md_r_bohr"], dtype=float)
    k_md = np.asarray(md_result["md_k_bohr_inv"], dtype=float)
    reliable_k = np.asarray(md_result["md_vectors_per_k_bin"]) >= MIN_VECTORS_PER_K_BIN

    for pair_index, ((left, right), label) in enumerate(
        zip(PAIR_INDICES, PAIR_LABELS, strict=True)
    ):
        color = PAIR_COLORS[label]
        axes[0, pair_index].plot(
            r_hnc, hnc["gij_r"][left, right], color=color, label="HNC"
        )
        g_md = np.asarray(md_result["md_gij_r"])[:, pair_index]
        g_sem = np.asarray(md_result["md_gij_block_sem"])[:, pair_index]
        axes[0, pair_index].plot(
            r_md, g_md, color=color, ls="--", alpha=0.85, label="MD"
        )
        axes[0, pair_index].fill_between(
            r_md,
            g_md - g_sem,
            g_md + g_sem,
            color=color,
            alpha=0.16,
            linewidth=0,
        )

        axes[1, pair_index].plot(
            k_hnc, hnc["sij_k"][left, right], color=color, label="HNC"
        )
        s_md = np.asarray(md_result["md_sij_k"])[:, pair_index]
        s_sem = np.asarray(md_result["md_sij_frame_sem"])[:, pair_index]
        axes[1, pair_index].errorbar(
            k_md[reliable_k],
            s_md[reliable_k],
            yerr=s_sem[reliable_k],
            color=color,
            ls="none",
            marker="o",
            ms=3.0,
            alpha=0.72,
            label="MD",
        )
        axes[0, pair_index].set_title(label)
        axes[0, pair_index].set_xlabel(r"$r$ [Bohr]")
        axes[1, pair_index].set_xlabel(r"$k$ [Bohr$^{-1}$]")
    axes[0, 0].set_ylabel(r"$g_{ab}(r)$")
    axes[1, 0].set_ylabel(r"$S_{ab}(k)$")
    axes[0, 0].legend(loc="best")
    fig.suptitle(
        rf"CH$_2$: $T_e={te_ev:g}$ eV, $T_i={ti_ev:g}$ eV; "
        "binary bridge not implemented",
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, case_dir / "hnc_md_comparison", close=True)


def _write_summary(records: list[dict[str, Any]], selected: list[float]) -> None:
    payload = {
        "schema_version": "ch2_hnc_md_scan_v1",
        "md_enabled": RUN_MD,
        "electronic_recomputed": RECOMPUTE_ELECTRONIC,
        "selected_te_ev": selected,
        "ti_over_te": list(TI_OVER_TE),
        "bridge_status": BRIDGE_STATUS,
        "cases": records,
    }
    _atomic_json(OUTPUT_DIR / "summary.json", payload)
    if records:
        fields = sorted({key for record in records for key in record})
        temporary = OUTPUT_DIR / "summary.csv.tmp"
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        temporary.replace(OUTPUT_DIR / "summary.csv")


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = list(TARGET_TE_EV)
    print(f"Fresh electronic temperatures: {selected} eV")
    if RUN_MD:
        print(
            f"Cases: {len(selected) * len(TI_OVER_TE)} sequential MD runs; "
            f"atoms={3 * CH2_FORMULA_UNITS}, MPI ranks={LAMMPS_MPI_PROCESSES}"
        )
    else:
        print(f"Cases: {len(selected) * len(TI_OVER_TE)} Otter HNC runs; MD disabled")
    print(f"Bridge: {BRIDGE_STATUS}")

    records: list[dict[str, Any]] = []
    total_cases = len(selected) * len(TI_OVER_TE)
    completed = 0
    scan_started = time.perf_counter()

    for te_ev in selected:
        try:
            electronic_kind, electronic_result = calculate_electronic(te_ev)
        except Exception as exc:
            print(f"    Electronic FAILED: {type(exc).__name__}: {exc}", flush=True)
            for alpha in TI_OVER_TE:
                records.append(dict(te_ev=te_ev, ti_ev=alpha * te_ev,
                                    status="failed", electronic_error=str(exc),
                                    hnc_status="not_run", md_status="not_run"))
            completed += len(TI_OVER_TE)
            _write_summary(records, selected)
            continue
        prep_started = time.perf_counter()
        prepared = prepare_multicomponent_ion_structure_from_electronic_result(
            workflow_config(te_ev, None),
            electronic_kind=electronic_kind,
            electronic_result=electronic_result,
        )
        preparation_s = time.perf_counter() - prep_started
        print(f"\nTe={te_ev:g} eV: shared QOZ prepared in {preparation_s:.2f} s")

        for alpha in TI_OVER_TE:
            ti_ev = float(alpha * te_ev)
            controls = md_controls(prepared, ti_ev)
            completed += 1
            case_dir = OUTPUT_DIR / f"Te{te_ev:07.3f}_Ti{ti_ev:07.3f}"
            case_dir.mkdir(parents=True, exist_ok=True)
            signature = _case_signature(te_ev, ti_ev, controls)
            cached_record = _completed_case(case_dir, signature)
            if cached_record is not None:
                records.append(cached_record)
                print(
                    f"[{completed}/{total_cases}] Te={te_ev:g}, Ti={ti_ev:g}: "
                    "complete, skipped"
                )
                continue

            print(
                f"[{completed}/{total_cases}] Te={te_ev:g} eV, "
                f"Ti={ti_ev:g} eV (alpha={alpha:g})"
            )
            if RUN_MD and bool(controls["thermal_limited"]):
                print(
                    "    high-T MD guard: "
                    f"dt*omega_p={controls['timestep_omega_p_inv']:.3e}, "
                    f"thermal step={controls['thermal_step_bohr']:.3e} Bohr, "
                    f"table r_min={controls['r_min_bohr']:.3e} Bohr"
                )
            case_started = time.perf_counter()
            record: dict[str, Any] = {
                "signature": signature,
                "te_ev": te_ev,
                "ti_ev": ti_ev,
                "alpha": alpha,
                "bridge_status": BRIDGE_STATUS,
                "qoz_preparation_s": preparation_s,
                "status": "running",
            }
            ion: dict[str, Any] | None = None
            try:
                hnc_started = time.perf_counter()
                workflow = continue_plasma_workflow_from_electronic_result(
                    workflow_config(te_ev, ti_ev),
                    electronic_kind=electronic_kind,
                    electronic_result=electronic_result,
                    multicomponent_preparation=prepared,
                )
                hnc_elapsed = time.perf_counter() - hnc_started
                ion = dict(workflow["ion"])
                save_hnc(case_dir, te_ev, ti_ev, ion)
                record.update(
                    hnc_status="success",
                    hnc_elapsed_s=hnc_elapsed,
                    hnc_output_residual=float(ion["hnc_output_residual"]),
                    hnc_closure_mismatch=float(ion["closure_transform_max_abs"]),
                )
                print(f"    HNC success in {hnc_elapsed:.2f} s")
            except Exception as exc:
                record.update(
                    hnc_status="failed",
                    hnc_error_type=type(exc).__name__,
                    hnc_error=" ".join(str(exc).split()),
                )
                print(f"    HNC FAILED: {type(exc).__name__}: {exc}")

            if not RUN_MD:
                record.update(
                    md_status="not_requested",
                    status="success" if ion is not None else "failed",
                    case_elapsed_s=time.perf_counter() - case_started,
                )
                records.append(record)
                _atomic_json(case_dir / "case_summary.json", record)
                _write_summary(records, selected)
                elapsed = time.perf_counter() - scan_started
                eta = elapsed / completed * (total_cases - completed)
                print(
                    f"    Overall: {completed}/{total_cases}, "
                    f"elapsed={elapsed / 60.0:.2f} min, "
                    f"ETA={eta / 60.0:.2f} min"
                )
                continue

            md_result: dict[str, np.ndarray] | None = None
            try:
                equilibration_steps = int(
                    round(
                        EQUILIBRATION_OMEGA_P_INV
                        / float(controls["timestep_omega_p_inv"])
                    )
                )
                production_steps = int(
                    round(
                        PRODUCTION_OMEGA_P_INV / float(controls["timestep_omega_p_inv"])
                    )
                )
                rdf_every = max(
                    1,
                    production_steps // (MD_STATISTICAL_BLOCKS * MD_RDF_REPEAT),
                )
                trajectory_every = max(1, production_steps // MD_STATISTICAL_BLOCKS)
                md_config = md.MDConfig(
                    output_dir=case_dir / "md",
                    species=(
                        md.MDSpecies(
                            "C",
                            12.011,
                            CH2_FORMULA_UNITS,
                            float(prepared.zbar[0]),
                        ),
                        md.MDSpecies(
                            "H",
                            1.008,
                            2 * CH2_FORMULA_UNITS,
                            float(prepared.zbar[1]),
                        ),
                    ),
                    ion_density_bohr3=float(np.sum(prepared.n_i)),
                    ion_temperature_ev=ti_ev,
                    timestep_ps=float(controls["timestep_ps"]),
                    thermostat_damp_ps=float(controls["thermostat_damp_ps"]),
                    equilibration_steps=equilibration_steps,
                    production_steps=production_steps,
                    rdf_bins=MD_RDF_BINS,
                    rdf_every=rdf_every,
                    rdf_repeat=MD_RDF_REPEAT,
                    trajectory_every=trajectory_every,
                    table_points=MD_TABLE_POINTS,
                    r_min_bohr=float(controls["r_min_bohr"]),
                    k_max_angstrom_inv=MD_K_MAX_ANGSTROM_INV,
                    k_bin_width_angstrom_inv=MD_K_BIN_WIDTH_ANGSTROM_INV,
                    structure_factor_workers=STRUCTURE_FACTOR_WORKERS,
                    random_seed=20_260_825 + int(round(100.0 * te_ev + 1000.0 * alpha)),
                    mpi_processes=LAMMPS_MPI_PROCESSES,
                    reuse_completed=False,
                )
                md_potential_matrix = (
                    regularize_coulomb_core(
                        prepared.r,
                        prepared.qoz.vij_r,
                        prepared.zbar,
                    )
                    if bool(controls["thermal_limited"])
                    else np.asarray(prepared.qoz.vij_r, dtype=float)
                )
                potentials = md.pair_potentials_from_otter(
                    prepared,
                    potential_matrix_ha=md_potential_matrix,
                )
                md_started = time.perf_counter()
                stop = threading.Event()
                monitor = threading.Thread(
                    target=_monitor_md,
                    args=(case_dir, md_config, stop, md_started),
                    daemon=True,
                )
                monitor.start()
                try:
                    md_result = md.run_otter_lammps_md(md_config, potentials)
                finally:
                    stop.set()
                    monitor.join()
                md_elapsed = time.perf_counter() - md_started
                record.update(
                    md_status="success",
                    md_elapsed_s=md_elapsed,
                    md_lammps_elapsed_s=float(md_result["md_lammps_elapsed_s"]),
                    md_analysis_elapsed_s=float(md_result["md_analysis_elapsed_s"]),
                    md_nve_relative_energy_drift=float(
                        md_result["md_nve_relative_energy_drift"]
                    ),
                    md_nve_elapsed_ps=float(md_result["md_nve_elapsed_ps"]),
                    md_sampled_min_timestep_ps=float(
                        md_result["md_thermo_sampled_min_timestep_ps"]
                    ),
                    md_sampled_max_timestep_ps=float(
                        md_result["md_thermo_sampled_max_timestep_ps"]
                    ),
                    md_reused_existing_run=bool(md_result["md_reused_existing_run"]),
                    omega_max_au=float(controls["omega_max_au"]),
                    timestep_ps=float(controls["timestep_ps"]),
                    timestep_omega_p_inv=float(controls["timestep_omega_p_inv"]),
                    thermal_step_bohr=float(controls["thermal_step_bohr"]),
                    thermal_limited=bool(controls["thermal_limited"]),
                    md_r_min_bohr=float(controls["r_min_bohr"]),
                    md_rdf_every=rdf_every,
                    md_trajectory_every=trajectory_every,
                    md_coulomb_core_regularized=bool(controls["thermal_limited"]),
                )
                drift = abs(float(record["md_nve_relative_energy_drift"]))
                record["md_energy_quality"] = (
                    "passed" if drift <= MAX_NVE_RELATIVE_ENERGY_DRIFT else "warning"
                )
                record["status"] = "success" if ion is not None and drift <= MAX_NVE_RELATIVE_ENERGY_DRIFT else "failed"
                print(
                    f"    MD + analysis success in {md_elapsed / 60.0:.1f} min; "
                    f"NVE drift={record['md_nve_relative_energy_drift']:.3e} "
                    f"({record['md_energy_quality']})"
                )
            except Exception as exc:
                record.update(
                    status="failed",
                    md_status="failed",
                    md_error_type=type(exc).__name__,
                    md_error=" ".join(str(exc).split()),
                )
                print(f"    MD FAILED: {type(exc).__name__}: {exc}")
            if md_result is not None and ion is not None:
                try:
                    plot_comparison(case_dir, ion, md_result, te_ev, ti_ev)
                    record["plot_status"] = "success"
                except Exception as exc:
                    record.update(
                        status="partial",
                        plot_status="failed",
                        plot_error_type=type(exc).__name__,
                        plot_error=" ".join(str(exc).split()),
                    )
                    print(f"    PLOT FAILED: {type(exc).__name__}: {exc}")
            record["case_elapsed_s"] = time.perf_counter() - case_started
            records.append(record)
            _atomic_json(case_dir / "case_summary.json", record)
            _write_summary(records, selected)

            elapsed = time.perf_counter() - scan_started
            eta = elapsed / completed * (total_cases - completed)
            print(
                f"    Overall: {completed}/{total_cases}, "
                f"elapsed={elapsed / 3600.0:.2f} h, "
                f"ETA={eta / 3600.0:.2f} h"
            )

    _write_summary(records, selected)
    failures = sum(record.get("status") != "success" for record in records)
    successes = len(records) - failures
    print(f"\nFinished: {successes}/{len(records)} successful")
    print(f"Results: {OUTPUT_DIR}")
    if REQUIRE_ALL_CASES and successes != total_cases:
        raise RuntimeError(
            "Full CH2 benchmark recomputation requires every case to succeed; "
            f"got {successes}/{total_cases}. Inspect summary.json and the "
            "per-case md/screen.log files."
        )


def rdf_structure_factors(
    case_dir: Path,
    md: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform every RDF block, then average the resulting partial S(k)."""
    from otter.numerics.transforms import precompute_dst_lattice_transform_like, radial_forward
    md_reader = globals()["md"]
    rdf_path = case_dir / "md/rdf_blocks.dat"
    r_rdf, g_blocks = md_reader._read_rdf(rdf_path, len(PAIR_INDICES))

    # LAMMPS reports RDF values at bin centres.  Interpolate h=g-1 onto the
    # strict r_i=i*dr DST-I lattice used by Otter and close the unresolved tail
    # to h=0 at the next lattice point.
    dr = float(np.mean(np.diff(r_rdf)))
    r_lattice = dr * np.arange(1, r_rdf.size + 1, dtype=float)
    transform = precompute_dst_lattice_transform_like(
        r_lattice,
        n_grid=r_rdf.size + 1,
    )
    h_lattice = np.empty(
        (g_blocks.shape[0], len(PAIR_INDICES), transform.r.size),
        dtype=float,
    )
    for block in range(g_blocks.shape[0]):
        for pair in range(len(PAIR_INDICES)):
            h_rdf = g_blocks[block, :, pair] - 1.0
            h_lattice[block, pair] = np.interp(
                transform.r,
                r_rdf,
                h_rdf,
                left=float(h_rdf[0]),
                right=0.0,
            )

    h_k = radial_forward(h_lattice, transform)
    counts = np.asarray(md["md_particle_counts"], dtype=float)
    box_length = float(md["md_box_length_bohr"])
    partial_density = counts / box_length**3
    pair_scale = np.asarray(
        [
            np.sqrt(partial_density[left] * partial_density[right])
            for left, right in PAIR_INDICES
        ]
    )
    diagonal = np.asarray([1.0 if left == right else 0.0
                           for left, right in PAIR_INDICES])
    s_blocks = (
        diagonal[None, :, None]
        + pair_scale[None, :, None] * h_k
    )
    s_mean = np.mean(s_blocks, axis=0).T
    s_sem = np.std(s_blocks, axis=0, ddof=1).T / np.sqrt(s_blocks.shape[0])
    return np.asarray(transform.k), s_mean, s_sem


def collect_results() -> dict[str, np.ndarray]:
    """Assemble only the outputs of the just-completed public calculation.

    No accepted archive or private application directory is used. All nine
    cases must pass their convergence and MD energy checks before plotting.
    """
    summary = json.loads((OUTPUT_DIR / "summary.json").read_text())
    records = summary["cases"]
    if len(records) != len(TARGET_TE_EV) * len(TI_OVER_TE) or any(
        item.get("status") != "success" or item.get("md_energy_quality") != "passed"
        for item in records
    ):
        raise RuntimeError("Incomplete or failed CH2 run; inspect summary.json")
    r_hnc = np.linspace(0.0, 10.0, 2001)
    k_hnc = np.linspace(0.0, 4.5, 1801)
    data = {"hnc_r_bohr": r_hnc, "hnc_k_bohr_inv": k_hnc}
    arrays = {key: [] for key in (
        "te_ev", "ti_ev", "alpha", "hnc_gij_r", "hnc_sij_k",
        "md_gij_r", "md_gij_sem", "md_sij_k", "md_sij_sem",
        "md_sij_from_rdf", "md_sij_from_rdf_sem", "hnc_elapsed_s", "md_elapsed_s",
        "g_rmse", "s_rmse", "md_sij_estimator_rmse", "md_sij_estimator_signed_mean",
        "md_sij_estimator_max_abs",
    )}
    for record in sorted(records, key=lambda row: (row["te_ev"], row["ti_ev"])):
        te, ti = record["te_ev"], record["ti_ev"]
        directory = OUTPUT_DIR / f"Te{te:07.3f}_Ti{ti:07.3f}"
        with np.load(directory / "hnc_results.npz", allow_pickle=False) as archive:
            hnc = dict(archive)
        with np.load(directory / "md/md_results.npz", allow_pickle=False) as archive:
            result = dict(archive)
        k_rdf, s_rdf, sem_rdf = rdf_structure_factors(directory, result)
        grids = {
            "md_r_bohr": result["md_r_bohr"], "md_k_bohr_inv": result["md_k_bohr_inv"],
            "md_vectors_per_k_bin": result["md_vectors_per_k_bin"], "md_rdf_k_bohr_inv": k_rdf,
        }
        for key, value in grids.items():
            if key in data and (data[key].shape != value.shape or not np.allclose(data[key], value)):
                raise ValueError(f"CH2 comparison requires matching grids: {key}")
            data[key] = value
        g = np.asarray([np.interp(r_hnc, hnc["r_bohr"], hnc["gij_r"][a,b]) for a,b in PAIR_INDICES])
        s = np.asarray([np.interp(k_hnc, hnc["k_bohr_inv"], hnc["sij_k"][a,b]) for a,b in PAIR_INDICES])
        values = dict(te_ev=te, ti_ev=ti, alpha=ti/te, hnc_gij_r=g, hnc_sij_k=s,
            md_gij_r=result["md_gij_r"].T, md_gij_sem=result["md_gij_block_sem"].T,
            md_sij_k=result["md_sij_k"].T, md_sij_sem=result["md_sij_frame_sem"].T,
            md_sij_from_rdf=s_rdf.T, md_sij_from_rdf_sem=sem_rdf.T,
            hnc_elapsed_s=record["hnc_elapsed_s"], md_elapsed_s=record["md_elapsed_s"])
        k = data["md_k_bohr_inv"]
        reliable = (data["md_vectors_per_k_bin"] >= MIN_VECTORS_PER_K_BIN) & (k <= 4.0)
        overlap = reliable & (k >= k_rdf[0]) & (k <= k_rdf[-1])
        if not np.any(overlap):
            raise ValueError("No reliable common structure-factor bins")
        g_delta = result["md_gij_r"].T - np.asarray([np.interp(data["md_r_bohr"], r_hnc, y) for y in g])
        s_delta = result["md_sij_k"].T - np.asarray([np.interp(k, k_hnc, y) for y in s])
        estimator_delta = np.asarray([np.interp(k[overlap], k_rdf, y) for y in s_rdf.T]) - result["md_sij_k"][overlap].T
        values.update(g_rmse=np.sqrt(np.mean(g_delta**2, axis=1)),
            s_rmse=np.sqrt(np.mean(s_delta[:,reliable]**2, axis=1)),
            md_sij_estimator_rmse=np.sqrt(np.mean(estimator_delta**2, axis=1)),
            md_sij_estimator_signed_mean=np.mean(estimator_delta, axis=1),
            md_sij_estimator_max_abs=np.max(np.abs(estimator_delta), axis=1))
        for key in arrays:
            arrays[key].append(values[key])
        print(f"Te={te:g}, Ti={ti:g}: HNC={record['hnc_elapsed_s']:.3f} s, "
              f"MD={record['md_elapsed_s']/60:.2f} min; "
              f"dt={record['timestep_ps']:.6g} ps, MPI={LAMMPS_MPI_PROCESSES}, "
              f"L={float(result['md_box_length_bohr']):.6g} Bohr", flush=True)
    data.update({key: np.asarray(value) for key, value in arrays.items()})
    return data


if __name__ == "__main__":
    main()
