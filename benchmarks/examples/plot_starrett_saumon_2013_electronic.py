r"""
Starrett--Saumon electronic levels and ionization
==================================================

This benchmark compares current Otter ion-sphere (IS) and experimental
self-consistent (SC) diagnostics with Tables I--III of
:cite:t:`StarrettSaumon2013` (doi:
`10.1103/PhysRevE.87.013104
<https://doi.org/10.1103/PhysRevE.87.013104>`__).  It was added specifically
to make pressure ionization auditable before interpreting the
:math:`\bar Z` difference in Johnson *et al.* (2025).

The article's reported results are **not** isolated ion-sphere calculations.
Its average-atom and two-component-plasma (QTCP or TFTCP) equations are
iterated self-consistently; Appendix B identifies the ion-sphere model as the
initial guess.  Table I is discussed as part of that coupled model and its
15-eV 3s entry is :math:`E=-0.0125` Hartree (about -0.340 eV), not -0.0125 eV.

For a bound level of energy :math:`E<0`, Otter uses Eq. (81),

.. math::

   M(E)=\operatorname{erf}\!\left[-2\sqrt{\ln 2}\,E/\gamma\right],

where :math:`\gamma` is obtained from the continuum phase shifts through the
transport-cross-section, Ziman-conductivity, and Drude relaxation-time path.
These sparse diagnostics are clearer as tables than as plots, so this gallery
uses direct numerical comparisons throughout.  Otter values are rounded to
three significant digits; published values retain the precision of the
article.  Every :math:`E` and :math:`\gamma` entry uses Hartree, while
:math:`M(E)` and both ionization definitions are dimensionless.

.. raw:: html

   <table class="docutils align-default">
   <caption>Al electronic diagnostics at 2.7 g cm<sup>-3</sup> and
   \(T=2\,\mathrm{eV}\)</caption>
   <thead>
   <tr><th rowspan="2">State</th><th colspan="3">\(E\) [Ha]</th>
   <th colspan="3">\(M(E)\)</th></tr>
   <tr><th>Paper SC</th><th>Otter-IS</th><th>Otter-SC</th>
   <th>Paper SC</th><th>Otter-IS</th><th>Otter-SC</th></tr>
   </thead><tbody>
   <tr><td>1s</td><td>-54.6</td><td>-54.6</td><td>-54.6</td>
   <td>1.00</td><td>1.00</td><td>1.00</td></tr>
   <tr><td>2s</td><td>-3.41</td><td>-3.39</td><td>-3.40</td>
   <td>1.00</td><td>1.00</td><td>1.00</td></tr>
   <tr><td>2p</td><td>-2.04</td><td>-2.02</td><td>-2.02</td>
   <td>1.00</td><td>1.00</td><td>1.00</td></tr>
   <tr><td>3s</td><td>unbound</td><td>unbound</td><td>unbound</td>
   <td>&mdash;</td><td>&mdash;</td><td>&mdash;</td></tr>
   </tbody></table>

   <table class="docutils align-default">
   <caption>Al electronic diagnostics at 2.7 g cm<sup>-3</sup> and
   \(T=15\,\mathrm{eV}\)</caption>
   <thead>
   <tr><th rowspan="2">State</th><th colspan="3">\(E\) [Ha]</th>
   <th colspan="3">\(M(E)\)</th></tr>
   <tr><th>Paper SC</th><th>Otter-IS</th><th>Otter-SC</th>
   <th>Paper SC</th><th>Otter-IS</th><th>Otter-SC</th></tr>
   </thead><tbody>
   <tr><td>1s</td><td>-54.9</td><td>-54.8</td><td>-54.8</td>
   <td>1.00</td><td>1.00</td><td>1.00</td></tr>
   <tr><td>2s</td><td>-3.60</td><td>-3.50</td><td>-3.57</td>
   <td>1.00</td><td>1.00</td><td>1.00</td></tr>
   <tr><td>2p</td><td>-2.23</td><td>-2.14</td><td>-2.21</td>
   <td>1.00</td><td>1.00</td><td>1.00</td></tr>
   <tr><td>3s</td><td>-0.0125</td><td>unbound</td><td>-0.00763</td>
   <td>0.134</td><td>&mdash;</td><td>0.0842</td></tr>
   </tbody></table>

   <table class="docutils align-default">
   <caption>Scattering width \(\gamma\) [Ha]</caption>
   <thead><tr><th>\(T\) [eV]</th><th>Paper SC</th>
   <th>Otter-IS</th><th>Otter-SC</th></tr></thead><tbody>
   <tr><td>2</td><td>0.0698</td><td>0.0367</td><td>0.0617</td></tr>
   <tr><td>15</td><td>0.174</td><td>0.109</td><td>0.170</td></tr>
   </tbody></table>

As a formula-only audit, evaluating Otter's Eq. (81) with the *published*
15-eV 3s inputs :math:`E=-0.0125` Ha and :math:`\gamma=0.174` Ha gives
:math:`M=0.1343`, consistent with the published 0.134.  This is not a fourth
physical model, so it is reported as an audit rather than an extra table model.
The accepted NPZ retains full precision; the executable summary follows the
same three-significant-digit presentation.  Tables II and III use two distinct
ionization definitions:

.. math::

   Z^*=n_e^0/n_I^0, \qquad
   \bar Z=Z-\int n_e^{\rm ion}(r)\,d^3r.

The following tables compare both definitions directly.  The
:math:`\Gamma_{\rm OCP}` and :math:`\Gamma_{\rm TCP}` columns are transcribed
paper parameters, not Otter observables; this benchmark does not manufacture
nominally corresponding Otter values.

.. raw:: html

   <table class="docutils align-default">
   <caption>Al at 2.7 g cm<sup>-3</sup>
   (\(n_I^0=8.93\times10^{-3}\,\mathrm{Bohr}^{-3}\)):
   Table II versus Otter</caption>
   <thead>
   <tr><th rowspan="2">\(T\) [eV]</th>
   <th colspan="3">\(Z^*\)</th>
   <th colspan="3">\(\bar Z\)</th>
   <th colspan="2">Paper coupling</th></tr>
   <tr><th>Paper</th><th>Otter-IS</th><th>Otter-SC</th>
   <th>Paper</th><th>Otter-IS</th><th>Otter-SC</th>
   <th>\(\Gamma_{\rm OCP}\)</th><th>\(\Gamma_{\rm TCP}\)</th></tr>
   </thead><tbody>
   <tr><td>2</td><td>1.98</td><td>2.09</td><td>2.09</td>
   <td>3.00</td><td>3.00</td><td>3.00</td><td>41.0</td><td>5.05</td></tr>
   <tr><td>6</td><td>2.11</td><td>2.19</td><td>2.19</td>
   <td>3.00</td><td>3.00</td><td>3.00</td><td>13.6</td><td>2.04</td></tr>
   <tr><td>10</td><td>2.24</td><td>2.33</td><td>2.33</td>
   <td>3.00</td><td>3.03</td><td>3.02</td><td>8.14</td><td>1.60</td></tr>
   <tr><td>15</td><td>2.51</td><td>2.56</td><td>2.56</td>
   <td>3.18</td><td>3.24</td><td>3.20</td><td>6.12</td><td>1.34</td></tr>
   </tbody></table>

   <table class="docutils align-default">
   <caption>Fe principal Hugoniot: Table III versus Otter</caption>
   <thead>
   <tr><th rowspan="2">\(\rho\) [g cm\(^{-3}\)]</th>
   <th rowspan="2">\(n_I^0\) [Bohr\(^{-3}\)]</th>
   <th rowspan="2">\(T\) [eV]</th>
   <th colspan="3">\(Z^*\)</th>
   <th colspan="3">\(\bar Z\)</th>
   <th colspan="2">Paper coupling</th></tr>
   <tr><th>Paper</th><th>Otter-IS</th><th>Otter-SC</th>
   <th>Paper</th><th>Otter-IS</th><th>Otter-SC</th>
   <th>\(\Gamma_{\rm OCP}\)</th><th>\(\Gamma_{\rm TCP}\)</th></tr>
   </thead><tbody>
   <tr><td>22.5</td><td>0.0360</td><td>10</td>
   <td>5.85</td><td>5.85</td><td>5.85</td>
   <td>8.78</td><td>8.81</td><td>8.74</td><td>112</td><td>14.3</td></tr>
   <tr><td>34.5</td><td>0.0551</td><td>100</td>
   <td>9.54</td><td>9.54</td><td>9.54</td>
   <td>11.6</td><td>11.9</td><td>11.5</td><td>22.4</td><td>4.85</td></tr>
   <tr><td>39.65</td><td>0.0634</td><td>1000</td>
   <td>20.4</td><td>20.4</td><td>20.4</td>
   <td>21.7</td><td>21.9</td><td>21.6</td><td>8.22</td><td>3.32</td></tr>
   <tr><td>34.37</td><td>0.0549</td><td>5000</td>
   <td>25.1</td><td>25.1</td><td>25.1</td>
   <td>25.5</td><td>25.6</td><td>25.6</td><td>2.18</td><td>1.35</td></tr>
   </tbody></table>

Otter-IS is the converged production ion-sphere workflow.  Otter-SC is the
experimental feedback path based on Sec. 2.4 of
:cite:t:`StarrettSaumon2014`: it feeds QOZ/HNC ion structure back into the
average atom while holding the converged IS chemical potential fixed.  This
is a useful controlled comparison, but it is not claimed to reproduce every
detail of the simultaneous 2013 QTCP/TFTCP solver.  Al uses quantum orbitals;
Fe uses Thomas--Fermi electrons, matching the model named for Table III.

Set ``USE_PRECOMPUTED_DATA = False`` below to recompute all eight independent
IS/SC pairs.  New calculations are staged under ``benchmarks/outputs`` and
never overwrite the accepted, checksummed baseline.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

from otter import PlasmaWorkflowConfig, solve_plasma_workflow
from otter.data.helpers import ion_level_weight
from otter.experimental import SCFeedbackConfig, solve_sc_feedback_workflow


# =============================================================================
# User input
# =============================================================================
USE_PRECOMPUTED_DATA = True
MAX_STATE_WORKERS = 2
CONTINUUM_WORKERS_PER_STATE = 4
# =============================================================================


BENCHMARK_ID = "starrett_saumon_2013_electronic"
SCHEMA = "otter_starrett_saumon_2013_electronic_v2"
STRUCTURE_LABELS = ("is", "sc")
LEVEL_LABELS = ("1s", "2s", "2p", "3s")
AL_TEMPERATURES_EV = (2.0, 6.0, 10.0, 15.0)
FE_HUGONIOT = (
    (22.5, 10.0),
    (34.5, 100.0),
    (39.65, 1000.0),
    (34.37, 5000.0),
)
SC_CONTROLS = SCFeedbackConfig(
    max_outer=16,
    g_tol=5.0e-4,
    v_corr_tol=5.0e-4,
    v_corr_mix=0.5,
    require_converged=True,
)


def repository_root() -> Path:
    """Locate the checkout when run directly or through Sphinx-Gallery."""
    candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    source = Path(str(globals().get("__file__", Path.cwd()))).resolve()
    candidates.extend([source.parent, *source.parents])
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "otter"
        ).is_dir():
            return candidate
    raise FileNotFoundError("Cannot locate the Otter checkout.")


ROOT = repository_root()
REFERENCE_DIR = ROOT / "benchmarks" / "reference_data" / BENCHMARK_ID
BASELINE_DIR = ROOT / "benchmarks" / "baselines" / BENCHMARK_ID
OUTPUT_DIR = ROOT / "benchmarks" / "outputs" / BENCHMARK_ID
BASELINE_PATH = BASELINE_DIR / "electronic_states.npz"
BASELINE_MANIFEST = BASELINE_DIR / "manifest.json"
CANDIDATE_PATH = OUTPUT_DIR / "electronic_states.npz"
CANDIDATE_MANIFEST = OUTPUT_DIR / "manifest.json"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_definitions() -> tuple[dict[str, Any], ...]:
    """Return the four Al-QM and four Fe-TF benchmark states."""
    aluminium = tuple(
        {
            "state_id": f"al_rho2p7_te{temperature:g}_qm",
            "element": "Al",
            "electronic_model": "qm",
            "rho_g_cc": 2.7,
            "temperature_ev": temperature,
        }
        for temperature in AL_TEMPERATURES_EV
    )
    iron = tuple(
        {
            "state_id": f"fe_rho{rho:g}_te{temperature:g}_tf",
            "element": "Fe",
            "electronic_model": "tf",
            "rho_g_cc": rho,
            "temperature_ev": temperature,
        }
        for rho, temperature in FE_HUGONIOT
    )
    return aluminium + iron


STATES = state_definitions()


def load_reference_tables() -> dict[str, Any]:
    """Verify and load the three published numerical tables."""
    manifest = json.loads(
        (REFERENCE_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("schema_version") != "otter_reference_manifest_v1"
        or manifest.get("reference_id") != BENCHMARK_ID
        or manifest["publication"].get("doi")
        != "10.1103/PhysRevE.87.013104"
    ):
        raise ValueError("Unexpected Starrett--Saumon reference manifest.")
    for record in manifest["files"]:
        path = REFERENCE_DIR / str(record["path"])
        if sha256_file(path) != str(record["sha256"]):
            raise RuntimeError(f"Reference checksum mismatch: {path.name}.")

    with (REFERENCE_DIR / "table_i_al_levels.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        level_rows = tuple(csv.DictReader(stream))
    return {
        "levels": level_rows,
        "al_ionization": np.genfromtxt(
            REFERENCE_DIR / "table_ii_al_ionization.csv",
            delimiter=",",
            names=True,
        ),
        "fe_ionization": np.genfromtxt(
            REFERENCE_DIR / "table_iii_fe_hugoniot.csv",
            delimiter=",",
            names=True,
        ),
    }


def workflow_config(state: dict[str, Any]) -> PlasmaWorkflowConfig:
    """Return the complete IS workflow used to initialise SC feedback."""
    is_tf = str(state["electronic_model"]) == "tf"
    return PlasmaWorkflowConfig(
        elements=[str(state["element"])],
        temperature_ev=float(state["temperature_ev"]),
        ion_temperature_ev=float(state["temperature_ev"]),
        rho_g_cc=float(state["rho_g_cc"]),
        electronic_model=str(state["electronic_model"]),
        aa_overrides={
            # The nondegenerate 5000-eV Fe chemical potential lies below the
            # ordinary warm-dense bracket.  Only the scalar search interval
            # is enlarged; the converged TF equations are unchanged.
            "mu_bounds": (-2000.0, 200.0) if is_tf else (-200.0, 200.0),
            "cont_n_jobs": int(CONTINUUM_WORKERS_PER_STATE),
            "cont_shards": int(2 * CONTINUUM_WORKERS_PER_STATE),
            # SC feedback can move an s state through the E=0 threshold.
            # Match that candidate to its all-space tail before allowing its
            # finite-box sign to define the bound/free partition.
            "bound_zero_tail_refine": not is_tf,
            # The paper's 15-eV 3s level is -0.0125 Ha.  Include that entire
            # physical comparison window rather than the library's narrower
            # near-zero diagnostic default (1e-3 Ha).
            "bound_zero_tail_max_binding_ha": 3.0e-2,
        },
        hnc_closure_transform_tol=2.5e-3,
        hnc_max_iter=500,
        show_progress=False,
    )


def level_label(l_value: int, radial_index: int) -> str:
    """Convert stored angular/radial indices to a spectroscopic label."""
    letters = "spdfgh"
    return f"{radial_index + l_value}{letters[l_value]}"


def reduce_electronic(
    result: dict[str, Any], *, electronic_model: str
) -> dict[str, Any]:
    """Apply convergence gates and retain compact electronic diagnostics."""
    if result.get("stage2_converged") is not True:
        raise RuntimeError("full average-atom stage 2 did not converge")
    threshold = str(result.get("threshold_state_status", "none")).lower()
    if threshold == "unresolved":
        raise RuntimeError("threshold-state representation is unresolved")

    n_i = float(dict(result["meta"])["n_i_bohr3"])
    zstar = float(result["n0"]) / n_i
    zbar = float(result["zbar_partition"])
    if not np.all(np.isfinite((n_i, zstar, zbar))):
        raise RuntimeError("full AA returned a non-finite ionization diagnostic")

    energies = np.full(len(LEVEL_LABELS), np.nan)
    weights = np.full(len(LEVEL_LABELS), np.nan)
    gamma = float(
        result.get(
            "ion_gamma",
            dict(result.get("meta", {})).get("ion_gamma_final", np.nan),
        )
    )
    if str(electronic_model) != "tf":
        energy_table = np.asarray(result["bound_energy_ha"], dtype=float)
        weight_table = np.asarray(result["bound_m"], dtype=float)
        l_values = np.asarray(result["bound_l_list"], dtype=int)
        n_indices = np.asarray(result["bound_n_index"], dtype=int)
        for l_row, l_value in enumerate(l_values):
            for column, radial_index in enumerate(n_indices):
                label = level_label(int(l_value), int(radial_index))
                if label not in LEVEL_LABELS:
                    continue
                energy = float(energy_table[l_row, column])
                if not np.isfinite(energy) or energy >= 0.0:
                    continue
                index = LEVEL_LABELS.index(label)
                energies[index] = energy
                weights[index] = float(weight_table[l_row, column])
                expected = ion_level_weight(energy, gamma)
                if not np.isclose(weights[index], expected, atol=2.0e-13):
                    raise RuntimeError(f"stored M(E) is inconsistent for {label}")

    history = list(result.get("history", ()))
    final_error = float(history[-1].get("err", np.nan)) if history else np.nan
    return {
        "n_i_bohr3": n_i,
        "r_ws_bohr": float(result["r_ws"]),
        "zstar": zstar,
        "zbar_partition": zbar,
        "ion_gamma_ha": gamma,
        "level_energy_ha": energies,
        "level_m": weights,
        "stage2_error": final_error,
        "stage2_iters": int(result.get("stage2_iters", len(history))),
        "threshold_state_status": threshold,
    }


def electronic_result(workflow: dict[str, Any]) -> dict[str, Any]:
    """Return the single-species full+external result from one workflow."""
    if str(workflow["electronic"]["kind"]) != "single_species":
        raise ValueError("This benchmark expects one species per workflow.")
    return dict(workflow["electronic"]["result"])


def solve_state(state: dict[str, Any]) -> dict[str, Any]:
    """Solve one IS workflow and continue it to experimental SC feedback."""
    config = workflow_config(state)
    started = time.perf_counter()
    is_workflow = solve_plasma_workflow(config)
    is_elapsed_s = time.perf_counter() - started

    started = time.perf_counter()
    sc_workflow = solve_sc_feedback_workflow(
        config,
        is_workflow,
        feedback_cfg=SC_CONTROLS,
    )
    sc_extension_elapsed_s = time.perf_counter() - started
    feedback = dict(sc_workflow["sc_feedback"])
    return {
        **state,
        "is": reduce_electronic(
            electronic_result(is_workflow),
            electronic_model=str(state["electronic_model"]),
        ),
        "sc": reduce_electronic(
            electronic_result(sc_workflow),
            electronic_model=str(state["electronic_model"]),
        ),
        "is_elapsed_s": float(is_elapsed_s),
        "sc_extension_elapsed_s": float(sc_extension_elapsed_s),
        "sc_converged": bool(feedback["converged"]),
        "sc_iterations": int(feedback["iterations"]),
        "fixed_is_mu_ha": float(feedback["fixed_is_mu_ha"]),
    }


def pack_rows(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Pack ordered IS/SC rows into one pickle-free portable archive."""
    by_id = {str(row["state_id"]): row for row in rows}
    ordered = [by_id[str(state["state_id"])] for state in STATES]
    archive: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA),
        "structure_labels": np.asarray(STRUCTURE_LABELS),
        "level_labels": np.asarray(LEVEL_LABELS),
    }
    for field in (
        "state_id",
        "element",
        "electronic_model",
        "rho_g_cc",
        "temperature_ev",
        "is_elapsed_s",
        "sc_extension_elapsed_s",
        "sc_converged",
        "sc_iterations",
        "fixed_is_mu_ha",
    ):
        archive[field] = np.asarray([row[field] for row in ordered])
    for field in (
        "n_i_bohr3",
        "r_ws_bohr",
        "zstar",
        "zbar_partition",
        "ion_gamma_ha",
        "stage2_error",
        "stage2_iters",
        "threshold_state_status",
    ):
        archive[field] = np.asarray(
            [[row[structure][field] for structure in STRUCTURE_LABELS]
             for row in ordered]
        )
    for field in ("level_energy_ha", "level_m"):
        archive[field] = np.asarray(
            [[row[structure][field] for structure in STRUCTURE_LABELS]
             for row in ordered],
            dtype=float,
        )
    return add_archive_metadata(archive)


def add_archive_metadata(
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Attach the compact provenance block required of accepted baselines."""
    archive = {
        key: np.asarray(value)
        for key, value in arrays.items()
        if key != "metadata_json"
    }
    state_count = int(np.asarray(archive["state_id"]).size)
    metadata = {
        "schema_version": "otter_compact_archive_metadata_v1",
        "configuration": {
            "scope": "is_and_experimental_sc_feedback",
            "aa_n_points": 4096,
            "bound_energy_cut_mode": "zero",
            "bound_zero_tail_refine": True,
            "bound_zero_tail_max_binding_ha": 3.0e-2,
            "ion_gamma_mode": "scattering",
            "al_electronic_model": "qm",
            "fe_electronic_model": "tf",
            "fe_mu_bounds_ha": [-2000.0, 200.0],
            "sc_max_outer": int(SC_CONTROLS.max_outer),
            "sc_g_tol": float(SC_CONTROLS.g_tol),
            "sc_v_corr_tol_ha": float(SC_CONTROLS.v_corr_tol),
            "sc_v_corr_mix": float(SC_CONTROLS.v_corr_mix),
            "sc_fixed_is_mu": True,
        },
        "state": {"benchmark_id": BENCHMARK_ID, "count": state_count},
        "producer": {
            "project": "Otter",
            "script_relative_path": (
                "benchmarks/examples/"
                "plot_starrett_saumon_2013_electronic.py"
            ),
        },
        "citation_keys": ["StarrettSaumon2013", "StarrettSaumon2014"],
        "convergence": {
            "accepted_state_pairs": state_count,
            "failed_state_pairs": 0,
            "is_stage2_converged": True,
            "sc_feedback_converged": True,
        },
        "fields": sorted(archive),
        "units": {
            "rho_g_cc": "g cm^-3",
            "temperature_ev": "eV",
            "n_i_bohr3": "Bohr^-3",
            "r_ws_bohr": "Bohr",
            "ion_gamma_ha": "Hartree",
            "level_energy_ha": "Hartree",
        },
    }
    archive["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    return archive


def git_head() -> str:
    """Return the producing revision when Git metadata is available."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def compute_candidate() -> dict[str, np.ndarray]:
    """Compute all IS/SC pairs in bounded parallelism and stage a candidate."""
    rows: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=MAX_STATE_WORKERS) as pool:
        futures = {pool.submit(solve_state, state): state for state in STATES}
        for future in as_completed(futures):
            state = futures[future]
            state_id = str(state["state_id"])
            try:
                row = future.result()
            except Exception as error:
                failures[state_id] = f"{type(error).__name__}: {error}"
                print(f"[rejected] {state_id}: {failures[state_id]}")
            else:
                rows.append(row)
                total = row["is_elapsed_s"] + row["sc_extension_elapsed_s"]
                print(
                    f"[accepted] {state_id}: {total:.1f} s, "
                    f"SC outer={row['sc_iterations']}"
                )
    if failures:
        raise RuntimeError(f"IS/SC benchmark failures: {failures}")

    archive = pack_rows(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CANDIDATE_PATH, **archive)
    manifest = {
        "schema_version": "otter_benchmark_manifest_v1",
        "benchmark_id": BENCHMARK_ID,
        "status": "candidate_not_accepted",
        "configuration": {
            "scope": "is_and_experimental_sc_feedback",
            "bound_energy_cut_mode": "zero",
            "bound_zero_tail_refine": True,
            "bound_zero_tail_max_binding_ha": 3.0e-2,
            "al_electronic_model": "qm",
            "fe_electronic_model": "tf",
            "fe_mu_bounds_ha": [-2000.0, 200.0],
            "aa_n_points": 4096,
            "sc_controls": {
                "max_outer": int(SC_CONTROLS.max_outer),
                "g_tol": float(SC_CONTROLS.g_tol),
                "v_corr_tol_ha": float(SC_CONTROLS.v_corr_tol),
                "v_corr_mix": float(SC_CONTROLS.v_corr_mix),
                "fixed_is_mu": True,
            },
        },
        "producer": {
            "script_relative_path": str(Path(__file__).resolve().relative_to(ROOT)),
            "script_sha256_current": sha256_file(Path(__file__).resolve()),
            "git_commit": git_head(),
        },
        "state": {
            "data_file": CANDIDATE_PATH.name,
            "data_sha256": sha256_file(CANDIDATE_PATH),
            "count": len(STATES),
        },
    }
    CANDIDATE_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return archive


def validate_archive(state: dict[str, np.ndarray]) -> None:
    """Validate state order, IS/SC dimensions, and accepted diagnostics."""
    if str(state["schema_version"].item()) != SCHEMA:
        raise ValueError("Unsupported electronic benchmark schema.")
    expected_ids = np.asarray([str(item["state_id"]) for item in STATES])
    if not np.array_equal(np.asarray(state["state_id"], dtype=str), expected_ids):
        raise ValueError("Electronic benchmark state order changed.")
    if not np.array_equal(
        np.asarray(state["structure_labels"], dtype=str), STRUCTURE_LABELS
    ):
        raise ValueError("Electronic benchmark structure order changed.")
    if np.asarray(state["level_energy_ha"]).shape != (len(STATES), 2, 4):
        raise ValueError("Malformed bound-level table.")
    for field in ("n_i_bohr3", "r_ws_bohr", "zstar", "zbar_partition"):
        values = np.asarray(state[field], dtype=float)
        if values.shape != (len(STATES), 2) or not np.all(np.isfinite(values)):
            raise ValueError(f"Malformed or non-finite accepted field: {field}.")
    if not np.all(np.asarray(state["sc_converged"], dtype=bool)):
        raise ValueError("The accepted archive contains unconverged SC feedback.")


def load_precomputed() -> dict[str, np.ndarray]:
    """Verify and load the accepted Otter IS/SC baseline."""
    manifest = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "otter_benchmark_manifest_v1"
        or manifest.get("benchmark_id") != BENCHMARK_ID
        or manifest.get("status") != "accepted"
    ):
        raise ValueError("Unexpected electronic benchmark manifest.")
    record = dict(manifest["state"])
    if record.get("data_file") != BASELINE_PATH.name:
        raise ValueError("Electronic benchmark manifest names the wrong file.")
    if sha256_file(BASELINE_PATH) != str(record.get("data_sha256")):
        raise RuntimeError("Electronic benchmark baseline checksum mismatch.")
    with np.load(BASELINE_PATH, allow_pickle=False) as archive:
        state = {key: np.asarray(archive[key]) for key in archive.files}
    validate_archive(state)
    return state


def state_index(state: dict[str, np.ndarray], state_id: str) -> int:
    """Return the row index of one stable state ID."""
    matches = np.flatnonzero(np.asarray(state["state_id"], dtype=str) == state_id)
    if matches.size != 1:
        raise KeyError(state_id)
    return int(matches[0])


def print_comparison(state: dict[str, np.ndarray], reference: dict[str, Any]) -> None:
    """Print the table values using the gallery's three-digit convention."""
    def cell(
        value: float,
        width: int,
        *,
        missing: str = "--",
    ) -> str:
        """Format three significant digits and label absent bound states."""
        if not np.isfinite(value):
            return f"{missing:>{width}}"
        if value == 0.0:
            rendered = "0.00"
        else:
            exponent = int(np.floor(np.log10(abs(value))))
            decimals = max(0, 2 - exponent)
            rendered = f"{value:.{decimals}f}"
        return f"{rendered:>{width}}"

    print("\nAl bound levels: all energies E are in Hartree")
    print(
        f"{'T[eV]':>6} {'shell':>5} {'paper E[Ha]':>12} "
        f"{'IS E[Ha]':>11} {'SC E[Ha]':>11} "
        f"{'M paper':>9} {'M IS':>9} {'M SC':>9}"
    )
    for row in reference["levels"]:
        temperature = float(row["temperature_ev"])
        shell = str(row["shell"])
        index = state_index(state, f"al_rho2p7_te{temperature:g}_qm")
        column = LEVEL_LABELS.index(shell)
        paper_energy = float(row["energy_ha"])
        values = (
            temperature,
            shell,
            paper_energy,
            float(state["level_energy_ha"][index, 0, column]),
            float(state["level_energy_ha"][index, 1, column]),
            float(row["m_weight"]),
            float(state["level_m"][index, 0, column]),
            float(state["level_m"][index, 1, column]),
        )
        print(
            f"{values[0]:6g} {values[1]:>5} "
            f"{cell(values[2], 12, missing='unbound')} "
            f"{cell(values[3], 11, missing='unbound')} "
            f"{cell(values[4], 11, missing='unbound')} "
            f"{cell(values[5], 9)} {cell(values[6], 9)} "
            f"{cell(values[7], 9)}"
        )

    paper_3s_m = ion_level_weight(-0.0125, 0.174)
    print(
        "Eq. (81) formula audit at 15 eV 3s: "
        f"paper inputs give M={paper_3s_m:.3f}; published M=0.134."
    )

    print("\nScattering width gamma: all values are in Hartree")
    print(
        f"{'T[eV]':>6} {'paper gamma[Ha]':>16} "
        f"{'IS gamma[Ha]':>13} {'SC gamma[Ha]':>13}"
    )
    for temperature in (2.0, 15.0):
        index = state_index(state, f"al_rho2p7_te{temperature:g}_qm")
        paper_gamma = float(
            next(
                row["gamma_ha"]
                for row in reference["levels"]
                if float(row["temperature_ev"]) == temperature
            )
        )
        print(
            f"{temperature:6g} {cell(paper_gamma, 16)} "
            f"{cell(float(state['ion_gamma_ha'][index, 0]), 13)} "
            f"{cell(float(state['ion_gamma_ha'][index, 1]), 13)}"
        )

    print("\nMean ionization: all values are dimensionless")
    print(
        f"{'state':>24} {'Z* paper':>9} {'Z* IS':>9} {'Z* SC':>9} "
        f"{'Zbar paper':>11} {'Zbar IS':>10} {'Zbar SC':>10}"
    )
    for element, table in (
        ("Al", reference["al_ionization"]),
        ("Fe", reference["fe_ionization"]),
    ):
        for row in np.atleast_1d(table):
            temperature = float(row["temperature_ev"])
            state_id = (
                f"al_rho2p7_te{temperature:g}_qm"
                if element == "Al"
                else (
                    f"fe_rho{float(row['rho_g_cc']):g}_"
                    f"te{temperature:g}_tf"
                )
            )
            index = state_index(state, state_id)
            print(
                f"{state_id:>24} {cell(float(row['zstar']), 9)} "
                f"{cell(float(state['zstar'][index, 0]), 9)} "
                f"{cell(float(state['zstar'][index, 1]), 9)} "
                f"{cell(float(row['zbar']), 11)} "
                f"{cell(float(state['zbar_partition'][index, 0]), 10)} "
                f"{cell(float(state['zbar_partition'][index, 1]), 10)}"
            )


# Load the same validated state table once for all gallery cells below.
if __name__ == "__main__":
    reference = load_reference_tables()
    state = compute_candidate() if not USE_PRECOMPUTED_DATA else load_precomputed()
    validate_archive(state)
    print(
        "Using "
        + (
            "checksummed, accepted Otter IS/SC states."
            if USE_PRECOMPUTED_DATA
            else "new candidate Otter IS/SC states."
        )
    )
    print_comparison(state, reference)
