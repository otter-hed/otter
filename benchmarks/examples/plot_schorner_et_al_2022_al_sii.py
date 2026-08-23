r"""
Equilibrium aluminium structure factors: Schörner et al. (2022)
================================================================

This benchmark compares Otter PA-HNC :math:`S_{ii}(k)` with two curves
digitized from Figure 2 of :cite:t:`SchornerEtAl2022`. Each state is calculated
with LDA-PW92 and PBE and labelled ``Otter-LDA`` and ``Otter-PBE``. Both states
have :math:`T_e=T_i`: 1 eV at :math:`\rho=4.712` g cm\ :sup:`-3` and 5 eV at
:math:`\rho=8.1` g cm\ :sup:`-3`.

The raw 1 eV ordinate is displaced downward because only the 5 eV curve was
used to calibrate the digitization. The loader therefore applies the audited
correction :math:`S_{ii}^{\mathrm{corrected}}=S_{ii}^{\mathrm{stored}}+1.5`
to the 1 eV curve only. The source CSV remains unchanged; the transformation
is recorded in its manifest and tested explicitly.

Set ``USE_PRECOMPUTED_DATA = False`` to calculate all four state/XC cases with
the public Otter workflow and write candidates below ``benchmarks/outputs``.
This requires the optional Libxc bindings. The default loads checksum-verified,
reviewed Otter baselines so documentation builds stay fast and deterministic.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from otter import PlasmaWorkflowConfig, solve_plasma_workflow
from otter.literature import citation_keys_for_xc_model
from otter.plotting import MODEL_STYLES, grid_figsize, save_figure, set_style


# =============================================================================
# User input
# =============================================================================
USE_PRECOMPUTED_DATA = True
if os.environ.get("OTTER_RECOMPUTE_SCHORNER_AL", "0") == "1":
    USE_PRECOMPUTED_DATA = False

# Four independent state/XC cases x four continuum workers use at most about
# sixteen CPU workers.
MAX_CASE_WORKERS = 4
CONTINUUM_WORKERS_PER_CASE = 4
HNC_TOL = 1.0e-4
HNC_CLOSURE_TOL = 2.5e-3
HNC_MAX_ITER = 1000
K_RETAIN_MAX_BOHR_INV = 20.0
# =============================================================================


BENCHMARK_ID = "schorner_et_al_2022_al_sii"
SCHEMA = "otter_schorner_2022_al_sii_v2"
BOHR_TO_ANGSTROM = 0.529177210903
REFERENCE_SII_OFFSET = {
    "al_rho4p712_te1_ti1": 1.5,
    "al_rho8p1_te5_ti5": 0.0,
}
XC_MODELS: tuple[dict[str, str], ...] = (
    {"key": "lda", "xc_model": "lda_pw", "label": "Otter-LDA"},
    {"key": "pbe", "xc_model": "pbe", "label": "Otter-PBE"},
)
STATES: tuple[dict[str, Any], ...] = (
    {
        "state_id": "al_rho4p712_te1_ti1",
        "rho_g_cc": 4.712,
        "te_ev": 1.0,
        "ti_ev": 1.0,
    },
    {
        "state_id": "al_rho8p1_te5_ti5",
        "rho_g_cc": 8.1,
        "te_ev": 5.0,
        "ti_ev": 5.0,
    },
)


def case_id(state: dict[str, Any], xc: dict[str, str]) -> str:
    """Return the stable identifier for one thermodynamic state/XC pair."""
    return f"{state['state_id']}_{xc['key']}"


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
            / "reference_data"
            / BENCHMARK_ID
            / "manifest.json"
        )
        if manifest.is_file():
            return candidate
    raise FileNotFoundError("Cannot locate the Otter checkout.")


ROOT = repository_root()
REFERENCE_DIR = ROOT / "benchmarks" / "reference_data" / BENCHMARK_ID
BASELINE_DIR = ROOT / "benchmarks" / "baselines" / BENCHMARK_ID
OUTPUT_DIR = ROOT / "benchmarks" / "outputs" / BENCHMARK_ID
CANDIDATE_DIR = OUTPUT_DIR / "gallery_recomputed"
FIGURE_DIR = OUTPUT_DIR / "figures"


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference_curves() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load the raw CSV and apply only the manifest-declared y corrections."""
    manifest = json.loads(
        (REFERENCE_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("reference_id") != BENCHMARK_ID:
        raise ValueError("Unexpected Schörner reference manifest.")
    file_record = manifest["files"][0]
    path = REFERENCE_DIR / str(file_record["path"])
    if sha256_file(path) != str(file_record["sha256"]):
        raise RuntimeError(f"Checksum mismatch for {path}.")
    values = np.genfromtxt(path, delimiter=",", skip_header=2)
    finite_four_columns = (
        values.ndim == 2
        and values.shape[1] == 4
        and np.all(np.isfinite(values))
    )
    if not finite_four_columns:
        raise ValueError("Expected four finite numeric reference columns.")

    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for state in manifest["states"]:
        state_id = str(state["state_id"])
        correction = float(state["sii_additive_correction"])
        if correction != REFERENCE_SII_OFFSET[state_id]:
            raise ValueError(f"Unexpected ordinate correction for {state_id}.")
        k = np.asarray(values[:, int(state["k_column"])], dtype=float)
        sii = (
            np.asarray(values[:, int(state["sii_column"])], dtype=float)
            + correction
        )
        if np.any(np.diff(k) <= 0.0) or np.any(sii < 0.0):
            raise ValueError(f"Invalid corrected reference curve for {state_id}.")
        curves[state_id] = (k, sii)
    if set(curves) != {str(state["state_id"]) for state in STATES}:
        raise ValueError("Reference state coverage is incomplete.")
    return curves


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load one portable, pickle-free Otter archive."""
    with np.load(path, allow_pickle=False) as archive:
        result = {key: np.asarray(archive[key]) for key in archive.files}
    if any(value.dtype.hasobject for value in result.values()):
        raise TypeError(f"Object arrays are forbidden in {path}.")
    return result


def load_precomputed_states() -> dict[str, dict[str, np.ndarray]]:
    """Verify and load the four accepted Otter state/XC baselines."""
    manifest = json.loads(
        (BASELINE_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("Unexpected Schörner benchmark manifest.")
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for record in manifest["states"]:
        if record.get("status") != "accepted":
            raise RuntimeError(f"Unaccepted state: {record['state_id']}.")
        path = BASELINE_DIR / str(record["baseline_file"])
        if sha256_file(path) != str(record["baseline_sha256"]):
            raise RuntimeError(f"Checksum mismatch for {path}.")
        payload = load_npz(path)
        state_id = str(payload["state_id"].item())
        if state_id != str(record["state_id"]):
            raise ValueError(f"State identifier mismatch in {path}.")
        loaded[state_id] = payload
    expected = {case_id(state, xc) for state in STATES for xc in XC_MODELS}
    if set(loaded) != expected:
        raise RuntimeError("Accepted baseline state coverage is incomplete.")
    return loaded


def workflow_config(
    state: dict[str, Any], xc: dict[str, str]
) -> PlasmaWorkflowConfig:
    """Build one equilibrium aluminium PA-HNC calculation."""
    return PlasmaWorkflowConfig(
        elements=["Al"],
        temperature_ev=float(state["te_ev"]),
        ion_temperature_ev=float(state["ti_ev"]),
        rho_g_cc=float(state["rho_g_cc"]),
        xc_model=str(xc["xc_model"]),
        aa_overrides={
            "cont_n_jobs": int(CONTINUUM_WORKERS_PER_CASE),
            "cont_shards": int(2 * CONTINUUM_WORKERS_PER_CASE),
        },
        hnc_tol=float(HNC_TOL),
        hnc_closure_transform_tol=float(HNC_CLOSURE_TOL),
        hnc_max_iter=int(HNC_MAX_ITER),
        show_progress=False,
    )


def pack_result(
    workflow: dict[str, Any],
    state: dict[str, Any],
    xc: dict[str, str],
    elapsed_s: float,
) -> dict[str, np.ndarray]:
    """Reject unreliable results and retain the structure-factor audit data."""
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
    xc_model = str(xc["xc_model"])
    xc_provenance = dict(electronic["xc_provenance"])
    if str(xc_provenance.get("model")) != xc_model:
        raise RuntimeError("The electronic result reports the wrong XC model.")

    k = np.asarray(ion["k"], dtype=float)
    mask = k <= K_RETAIN_MAX_BOHR_INV
    payload = {
        "schema_version": np.asarray(SCHEMA),
        "state_id": np.asarray(case_id(state, xc)),
        "thermodynamic_state_id": np.asarray(str(state["state_id"])),
        "element": np.asarray("Al"),
        "rho_g_cc": np.asarray(float(state["rho_g_cc"])),
        "te_ev": np.asarray(float(state["te_ev"])),
        "ti_ev": np.asarray(float(state["ti_ev"])),
        "xc_model": np.asarray(xc_model),
        "xc_label": np.asarray(str(xc["label"])),
        "xc_provenance_json": np.asarray(
            json.dumps(xc_provenance, sort_keys=True, separators=(",", ":"))
        ),
        "producer_elapsed_s": np.asarray(float(elapsed_s)),
        "k_bohr_inv": k[mask],
        "sii_k": np.asarray(ion["sii_k"], dtype=float)[mask],
        "zbar_partition": np.asarray(float(ion["zbar_partition"])),
        "threshold_state_status": np.asarray(
            str(electronic.get("threshold_state_status", "none"))
        ),
        "hnc_solver_path": np.asarray(str(ion["hnc_solver_path"])),
        "hnc_output_residual": np.asarray(float(ion["hnc_output_residual"])),
        "closure_transform_max_abs": np.asarray(
            float(ion["closure_transform_max_abs"])
        ),
    }
    metadata = {
        "schema_version": "otter_compact_archive_metadata_v1",
        "archive_role": "project_generated_example_or_benchmark_baseline",
        "archive_schema_version": SCHEMA,
        "package_id": BENCHMARK_ID,
        "configuration": {
            "elements": ["Al"],
            "temperature_ev": float(state["te_ev"]),
            "ion_temperature_ev": float(state["ti_ev"]),
            "rho_g_cc": float(state["rho_g_cc"]),
            "xc_model": xc_model,
            "continuum_workers_per_case": CONTINUUM_WORKERS_PER_CASE,
            "hnc_tolerance": HNC_TOL,
            "hnc_transform_closure_tolerance": HNC_CLOSURE_TOL,
            "hnc_max_iterations": HNC_MAX_ITER,
        },
        "state": {
            "state_id": case_id(state, xc),
            "thermodynamic_state_id": str(state["state_id"]),
            "element": "Al",
            "rho_g_cc": float(state["rho_g_cc"]),
            "te_ev": float(state["te_ev"]),
            "ti_ev": float(state["ti_ev"]),
            "xc_model": xc_model,
            "xc_label": str(xc["label"]),
        },
        "producer": {
            "project": "Otter",
            "version": "0.2.2",
            "script_relative_path": (
                "benchmarks/examples/plot_schorner_et_al_2022_al_sii.py"
            ),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "citation_keys": [
            "SchornerEtAl2022",
            *citation_keys_for_xc_model(xc_model),
        ],
        "convergence": {
            "aa_stage2_converged": True,
            "aa_ext_converged": True,
            "threshold_state_status": str(
                electronic.get("threshold_state_status", "none")
            ),
            "hnc_converged": True,
            "hnc_output_residual": float(ion["hnc_output_residual"]),
            "closure_transform_max_abs": float(
                ion["closure_transform_max_abs"]
            ),
        },
        "fields": sorted(payload),
    }
    payload["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    return payload


def solve_case(
    state: dict[str, Any], xc: dict[str, str]
) -> tuple[str, dict[str, np.ndarray]]:
    """Calculate and validate one state/XC case in a worker process."""
    started = time.perf_counter()
    workflow = solve_plasma_workflow(workflow_config(state, xc))
    payload = pack_result(workflow, state, xc, time.perf_counter() - started)
    return case_id(state, xc), payload


def solve_all_states() -> dict[str, dict[str, np.ndarray]]:
    """Calculate all independent state/XC cases and save review candidates."""
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    solved: dict[str, dict[str, np.ndarray]] = {}
    cases = [(state, xc) for state in STATES for xc in XC_MODELS]
    workers = max(1, min(int(MAX_CASE_WORKERS), len(cases)))
    if workers == 1:
        for state, xc in cases:
            identifier, payload = solve_case(state, xc)
            solved[identifier] = payload
            print(f"[computed] {identifier}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(solve_case, state, xc): (state, xc)
                for state, xc in cases
            }
            for future in as_completed(futures):
                identifier, payload = future.result()
                solved[identifier] = payload
                print(f"[computed] {identifier}")
    for identifier, payload in solved.items():
        np.savez_compressed(CANDIDATE_DIR / f"{identifier}.npz", **payload)
    return solved


def inverse_bohr_sii(
    state: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return one Otter curve with wavenumber in inverse ångström."""
    k_angstrom_inv = (
        np.asarray(state["k_bohr_inv"], dtype=float) / BOHR_TO_ANGSTROM
    )
    return k_angstrom_inv, np.asarray(state["sii_k"], dtype=float)


reference_curves = load_reference_curves()
otter_states = (
    load_precomputed_states()
    if USE_PRECOMPUTED_DATA
    else solve_all_states()
)

print(f"{'case':>29s} {'RMSE':>12s} {'MAE':>12s} {'max|delta|':>12s}")
for state in STATES:
    reference_id = str(state["state_id"])
    k_ref, sii_ref = reference_curves[reference_id]
    for xc in XC_MODELS:
        identifier = case_id(state, xc)
        k_otter, sii_otter = inverse_bohr_sii(otter_states[identifier])
        mask = (k_ref >= k_otter[0]) & (k_ref <= k_otter[-1])
        delta = np.interp(k_ref[mask], k_otter, sii_otter) - sii_ref[mask]
        print(
            f"{identifier:>29s} {np.sqrt(np.mean(delta**2)):12.4e} "
            f"{np.mean(np.abs(delta)):12.4e} "
            f"{np.max(np.abs(delta)):12.4e}"
        )


# %%
# Static ion structure factors
# ----------------------------
#
# Lines are current Otter PA-HNC LDA/PBE results. Markers are the corrected,
# digitized Figure 2 values; the raw CSV is never rewritten.

set_style("thesis", palette="bing")
fig, axes = plt.subplots(1, 2, figsize=grid_figsize(1, 2))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
xc_line_styles = (
    {**MODEL_STYLES["otter"], "alpha": 0.82},
    {
        **MODEL_STYLES["otter"],
        "color": "#D55E00",
        "linestyle": "--",
        "alpha": 0.90,
    },
)
for axis, state in zip(axes, STATES, strict=True):
    reference_id = str(state["state_id"])
    k_ref, sii_ref = reference_curves[reference_id]
    for xc, line_style in zip(XC_MODELS, xc_line_styles, strict=True):
        k_otter, sii_otter = inverse_bohr_sii(
            otter_states[case_id(state, xc)]
        )
        axis.plot(
            k_otter,
            sii_otter,
            label=str(xc["label"]),
            **line_style,
        )
    axis.scatter(
        k_ref,
        sii_ref,
        s=25,
        marker="o",
        linewidths=1.2,
        facecolors="none",
        edgecolors=colors[0],
        label="Schörner et al. (2022), Fig. 2",
        zorder=3,
    )
    axis.axhline(1.0, color="0.55", ls=":", lw=0.8)
    axis.set(
        xlim=(0.0, 8.3),
        ylim=(0.0, 2.25 if float(state["te_ev"]) == 1.0 else 1.55),
        xlabel=r"$k\ (\mathrm{\AA}^{-1})$",
        ylabel=r"$S_{ii}(k)$",
        title=(
            rf"Al: $T_e=T_i={state['te_ev']:g}$ eV, "
            rf"$\rho={state['rho_g_cc']:g}$ g cc$^{{-1}}$"
        ),
    )
    axis.legend(fontsize=8.0)

fig.tight_layout()
saved_paths = save_figure(
    fig,
    FIGURE_DIR / "schorner_et_al_2022_al_sii",
    formats=("png", "pdf"),
)
print(
    "Saved figures: "
    + ", ".join(str(path.relative_to(ROOT)) for path in saved_paths.values())
)
if "agg" not in plt.get_backend().lower():
    plt.show()
