r"""
Al: complete electronic-to-ionic workflow
=========================================

This example follows one aluminium state through the complete Otter pipeline:

.. math::

   \mathrm{KS\!-\!DFT\ AA}
   \rightarrow \{n_{\rm ion},n_{\rm scr}\}
   \rightarrow V_{ii}
   \rightarrow \mathrm{OZ/HNC}
   \rightarrow \{g_{ii},S_{ii}\}.

The full/external pseudoatom partition, continuum density, asymptotic density
tail, and QOZ construction follow :cite:t:`StarrettSaumon2014`. The default
finite-temperature jellium local-field correction follows
:cite:t:`Chabrier1990`.

The first figure shows unweighted bound-state wavefunctions, ionic electron
densities, and form factors by level. The following figures show the electronic
structure, screening cloud and pair interactions, and ionic correlations.
The Rayleigh weight per ion is
:math:`W_R(k)=|f(k)+q(k)|^2 S_{ii}(k)`.
Only the element, density and temperatures are set; numerical controls use
Otter's defaults.
Set ``EXPORT_SLIDE_FIGURES = True`` to also save individual density,
:math:`g_{ii}(r)`, and :math:`S_{ii}(k)` figures for slides.

Reproduction
------------

From the root of the complete Otter checkout, using Poetry, run::

    poetry run python docs/examples/plot_al_full_workflow.py

Downloads are optional: ``.ipynb`` launches this repository script; ``.zip``
contains both formats. See :doc:`/user_guide/reproducing_galleries` for setup.

The script calculates the states from their input parameters and then plots
the results. No bundled Otter NPZ is required. Numerical outputs are written
locally; literature reference tables remain inputs to the comparison.

Recorded results
----------------

The figures and output below are from the recorded validation run; running
the source recalculates them with the installed Otter version.

.. include:: /_static/gallery_results/plot_al_full_workflow/results.rst

Orbital access and NPZ export
------------------------------------------------------------

The completed workflow provides the raw orbital arrays without another solve:

.. code-block:: python

   from otter import bound_wavefunctions, ion_orbital_form_factors

   aa = result["electronic"]["result"]
   ion = result["ion"]
   Zbar = aa["zbar_partition"]
   Zstar = aa["zstar"]
   R_nl = bound_wavefunctions(aa)             # raw R_nl on aa["r_bound"]
   n_nl = aa["ion_orbital_density_r"]         # on aa["r"]
   f_nl = ion_orbital_form_factors(aa, r=ion["r"], k=ion["k"])

This script also saves ``benchmarks/outputs/al_full_workflow_1ev/Al_orbitals_state.npz``
with :func:`otter.save_plasma_state`. It contains unweighted wavefunctions,
ionic density components, per-level form factors, level labels, and basic
state metadata. Access the saved arrays with:

.. code-block:: python

   from otter import load_plasma_state

   state = load_plasma_state(
       "benchmarks/outputs/al_full_workflow_1ev/Al_orbitals_state.npz"
   )
   R_nl = state["species_0_bound_wavefunction_r"]
   n_nl = state["species_0_ion_orbital_density_r"]
   f_nl = state["species_0_ion_orbital_density_k"]

See :ref:`orbital-npz-access` for selective saving, radial grids, level
indexing, and mixture access. The :math:`4\pi r^2` display factors and the
optional FD wavefunction weighting do not modify the saved arrays.

"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from otter import PlasmaWorkflowConfig, solve_plasma_workflow, save_plasma_state, load_plasma_state
from otter.electronic import FullExternalConfig
from otter.io.state import StateExportOptions, build_state_arrays
from otter.numerics.constants import HA_TO_EV
from otter.plotting import grid_figsize, save_figure, style_context


# =============================================================================
# User input
# =============================================================================
RECOMPUTE_WITH_OTTER = True
EXPORT_SLIDE_FIGURES = False
if os.environ.get("OTTER_RECOMPUTE_AL_FULL", "0") == "1":
    RECOMPUTE_WITH_OTTER = True

ELEMENT = "Al"
RHO_G_CC = 8.1
TE_EV = 1.0
TI_EV = 1.0

# =============================================================================


SCHEMA = "otter_al_full_workflow_v2"


def repository_root() -> Path:
    """Locate source and reference inputs, independently of numerical outputs."""
    candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    source_file = globals().get("__file__")
    if source_file is not None:
        candidates.extend(Path(source_file).resolve().parents)
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/otter").is_dir():
            return candidate
    raise FileNotFoundError("Run from an Otter source checkout with its dependencies installed.")


ROOT = repository_root()
BASELINE_DIR = ROOT / "benchmarks" / "baselines" / "al_full_workflow_1ev"
OUTPUT_DIR = ROOT / "benchmarks" / "outputs" / "al_full_workflow_1ev"
FIGURE_DIR = OUTPUT_DIR / "figures"
RECOMPUTED_PATH = OUTPUT_DIR / "recomputed" / "Al_rho8p1gcc_Te1eV_Ti1eV.npz"
ORBITAL_STATE_PATH = OUTPUT_DIR / "Al_orbitals_state.npz"


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 checksum."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reviewed_state() -> dict[str, np.ndarray]:
    """Verify and load the reviewed current-Otter result."""
    manifest = json.loads((BASELINE_DIR / "manifest.json").read_text(encoding="utf-8"))
    record = manifest["state"]
    path = BASELINE_DIR / str(record["data_file"])
    if sha256_file(path) != str(record["data_sha256"]):
        raise RuntimeError(f"Checksum mismatch for {path}.")
    with np.load(path, allow_pickle=False) as archive:
        state = {key: np.asarray(archive[key]) for key in archive.files}
    if str(state["schema_version"].item()) != SCHEMA:
        raise ValueError("Unsupported Al full-workflow archive.")
    if any(value.dtype.hasobject for value in state.values()):
        raise TypeError("Object arrays are forbidden in gallery archives.")
    return state


def workflow_config() -> PlasmaWorkflowConfig:
    """Return the complete public Otter configuration used on this page."""
    return PlasmaWorkflowConfig(
        elements=[ELEMENT],
        temperature_ev=TE_EV,
        ion_temperature_ev=TI_EV,
        rho_g_cc=RHO_G_CC,
    )


def finite_bound_levels(electronic: dict[str, Any]) -> dict[str, np.ndarray]:
    """Flatten finite negative-energy levels and ordinary FD occupations."""
    energies = np.asarray(electronic["bound_energy_ha"], dtype=float)
    l_values = np.asarray(electronic["bound_l_list"], dtype=int)
    fd = np.asarray(electronic["bound_fd"], dtype=float)
    occupation = np.asarray(electronic["bound_occ_deg_fd"], dtype=float)
    records: list[tuple[int, int, float, float, float]] = []
    for l_index, l_value in enumerate(l_values):
        for radial_index in range(energies.shape[1]):
            energy = float(energies[l_index, radial_index])
            if np.isfinite(energy) and energy < 0.0:
                records.append(
                    (
                        int(l_value),
                        int(radial_index + 1),
                        energy,
                        float(fd[l_index, radial_index]),
                        float(occupation[l_index, radial_index]),
                    )
                )
    values = np.asarray(records, dtype=float)
    return {
        "bound_l": values[:, 0].astype(int),
        "bound_n_index": values[:, 1].astype(int),
        "bound_energy_ha": values[:, 2],
        "bound_fd": values[:, 3],
        "bound_occ_deg_fd": values[:, 4],
    }


def pack_workflow(
    workflow: dict[str, Any],
    *,
    elapsed_s: float,
) -> dict[str, np.ndarray]:
    """Pack the plotted fields without relying on an external producer."""
    electronic = dict(workflow["electronic"]["result"])
    ion = dict(workflow["ion"])
    portable = build_state_arrays(
        workflow,
        options=StateExportOptions(
            profile="ion_structure",
        ),
    )
    r_e = np.asarray(electronic["r"], dtype=float)
    r = np.asarray(ion["r"], dtype=float)
    k = np.asarray(ion["k"], dtype=float)
    e_mask = r_e <= 20.0
    r_mask = r <= 20.0
    k_mask = k <= 20.0
    threshold_status = (
        str(electronic.get("threshold_state_status", "none")).strip().lower()
    )
    charge_fix = dict(ion["charge_fix"])
    q_scale = float(charge_fix["scale_factor"])
    if not np.isfinite(q_scale) or q_scale <= 0.0:
        raise RuntimeError("The QOZ screening-charge scale is not physical.")
    q_used = np.asarray(ion["n_scr_k"], dtype=float)
    if threshold_status == "unresolved":
        raise RuntimeError("The final average atom has an unresolved threshold state.")
    if not bool(ion["hnc_converged"]):
        raise RuntimeError("The final HNC state did not converge.")
    if float(ion["closure_transform_max_abs"]) > float(ion["closure_transform_tol"]):
        raise RuntimeError(
            "The final finite-transform g(r)/S(k) mismatch exceeds tolerance."
        )

    def profile(name: str, fallback: str | None = None) -> np.ndarray:
        key = name if name in electronic else fallback
        if key is None or key not in electronic:
            raise KeyError(f"Electronic result has no {name!r} profile.")
        return np.asarray(electronic[key], dtype=float)[e_mask]

    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA),
        "benchmark_id": np.asarray("al_full_workflow_1ev"),
        "storage_profile": np.asarray("gallery_analysis"),
        "state_id": np.asarray("al_full_workflow_rho8p1_te1_ti1"),
        "element": np.asarray(ELEMENT),
        "rho_g_cc": np.asarray(RHO_G_CC),
        "te_ev": np.asarray(TE_EV),
        "ti_ev": np.asarray(TI_EV),
        "producer_signature_json": np.asarray(
            json.dumps(
                {
                    "resolved_configuration": workflow["configuration"],
                    "aa_final_settings": {
                        key: value for key, value in electronic["meta"].items()
                        if (key.startswith("bound_zero_tail_") and key in FullExternalConfig.__dataclass_fields__)
                        or key in ("n_points", "cont_rmax_mult", "b3_tail_target")
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "producer_elapsed_s": np.asarray(elapsed_s),
        "r_e_bohr": r_e[e_mask],
        "n_full_bohr3": profile("n_full"),
        "n_bound_bohr3": profile("n_bound"),
        "n_ext_bohr3": profile("n_ext"),
        "n_pa_bohr3": profile("n_pa"),
        "n_scr_bohr3": profile("n_scr"),
        "v_full_ha": profile("v_full", fallback="v_scf"),
        "v_ext_ha": profile("v_ext"),
        "v_hartree_ha": profile("v_H"),
        "v_xc_ha": profile("v_xc"),
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
        "r_bohr": r[r_mask],
        "k_bohr_inv": k[k_mask],
        "gii_r": np.asarray(ion["gii_r"], dtype=float)[r_mask],
        "sii_k": np.asarray(ion["sii_k"], dtype=float)[k_mask],
        "vii_r_ha": np.asarray(ion["vii_r"], dtype=float)[r_mask],
        "vii_k_ha_bohr3": np.asarray(ion["vii_k"], dtype=float)[k_mask],
        "n_scr_k_electrons": q_used[k_mask],
        "n_ion_k_electrons": np.asarray(portable["n_ion_k"], dtype=float)[0],
        "hnc_best_residual": np.asarray(float(ion["hnc_best_residual"])),
        "hnc_closure_mismatch": np.asarray(float(ion["closure_transform_max_abs"])),
        "hnc_closure_tolerance": np.asarray(float(ion["closure_transform_tol"])),
        "hnc_iters": np.asarray(int(ion["hnc_iters"])),
    }
    payload.update(finite_bound_levels(electronic))
    return payload


def calculate_state() -> dict[str, np.ndarray]:
    """Run AA -> pseudoatom -> QOZ/HNC and stage a portable result."""
    started = time.perf_counter()
    workflow = solve_plasma_workflow(workflow_config())
    state = pack_workflow(workflow, elapsed_s=time.perf_counter() - started)
    save_plasma_state(
        ORBITAL_STATE_PATH,
        workflow,
        options=StateExportOptions(
            profile="electronic_summary", include_groups=("orbital_densities",),
        ),
    )
    RECOMPUTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RECOMPUTED_PATH, **state)
    print(f"Saved newly calculated state: {RECOMPUTED_PATH}")
    return state


def plot_bound_orbitals(state, *, title, r_max=5.0, k_max=10.0):
    """Plot the orbital arrays from this example's standard NPZ export."""
    r_wave = state["species_0_r_bound_bohr"]
    r = state["species_0_r_bohr"]
    k = state["species_0_orbital_k_bohr_inv"]
    wave = state["species_0_bound_wavefunction_r"]
    density = state["species_0_ion_orbital_density_r"]
    factors = state["species_0_ion_orbital_density_k"]
    angular = state["species_0_bound_l"]
    principal = state["species_0_bound_principal_n"]
    r_ws = float(state["species_0_r_ws_bohr"])
    total_density = density.sum(axis=0)
    wave_mask, r_mask, k_mask = r_wave <= r_max, r <= r_max, k <= k_max
    wave_shell, density_shell = 4 * np.pi * r_wave**2, 4 * np.pi * r**2
    shell_letters = "spdfghiklm"
    with style_context("thesis", palette="bing"):
        fig, axes = plt.subplots(1, 3, figsize=grid_figsize(1, 3), layout="constrained")
        for i, (l, n) in enumerate(zip(angular, principal)):
            l, n = int(l), int(n)
            label = f"{n}{shell_letters[l]}" if l < len(shell_letters) else f"n={n}, l={l}"
            axes[0].plot(r_wave[wave_mask], (wave_shell * wave[i])[wave_mask], label=label)
            axes[1].plot(r[r_mask], (density_shell * density[i])[r_mask], label=label)
            axes[2].plot(k[k_mask], factors[i, k_mask], label=label)
        axes[1].plot(r[r_mask], (density_shell * total_density)[r_mask], color="black", ls="--", label="Total")
        axes[2].plot(k[k_mask], factors.sum(axis=0)[k_mask], color="black", ls="--", label="Total")
        axes[0].set(xlabel=r"$r$ [$a_B$]", xlim=(-0.25, r_max),
                    ylabel=r"$4\pi r^2 R_{nl}(r)$ [$a_B^{1/2}$]",
                    title="Unweighted wavefunctions")
        axes[1].set(xlabel=r"$r$ [$a_B$]", xlim=(-0.25, r_max),
                    ylabel=r"$4\pi r^2 n^{\rm ion}_{nl}(r)$ [$a_B^{-1}$]", title="Ion-orbital densities")
        axes[2].set(xlabel=r"$k$ [$a_B^{-1}$]", xlim=(0, k_max),
                    ylabel=r"$f_{nl}(k)=n^{\rm ion}_{nl}(k)$", title="Ion-orbital form factors")
        for ax in axes[:2]:
            ax.axvline(r_ws, color="0.6", ls="--", lw=1.0, label=r"$R_{\rm WS}$")
        for ax in axes:
            if ax.lines:
                ax.legend()
        fig.suptitle(title)
    return fig


def main() -> None:
    state = calculate_state() if RECOMPUTE_WITH_OTTER else load_reviewed_state()
    print(
        "Using "
        + (
            "a newly calculated Otter state."
            if RECOMPUTE_WITH_OTTER
            else "the checksummed current-Otter state."
        )
    )


    def level_rows() -> list[tuple[str, float, float, float, float]]:
        """Return spectroscopic labels and energies for terminal output."""
        symbols = ("s", "p", "d", "f", "g", "h")
        rows = []
        for l_value, radial_index, energy, fd, occupation in zip(
            state["bound_l"],
            state["bound_n_index"],
            state["bound_energy_ha"],
            state["bound_fd"],
            state["bound_occ_deg_fd"],
            strict=True,
        ):
            l_int = int(l_value)
            principal_n = int(radial_index) + l_int
            label = (
                f"{principal_n}{symbols[l_int]}"
                if l_int < len(symbols)
                else f"n={principal_n},l={l_int}"
            )
            rows.append(
                (
                    label,
                    float(energy),
                    HA_TO_EV * float(energy),
                    float(fd),
                    float(occupation),
                )
            )
        return rows


    print(
        f"Al: rho={float(state['rho_g_cc']):g} g/cc, " f"Te=Ti={float(state['te_ev']):g} eV"
    )
    print(
        f"mu={float(state['mu_ha']):.8f} Ha, "
        f"HNC residual={float(state['hnc_best_residual']):.3e}"
    )
    print(
        "Zbar(AA-WS/partition/QOZ)="
        f"{float(state['zbar_aa']):.8f}/"
        f"{float(state['zbar_partition']):.8f}/"
        f"{float(state['zbar_qoz']):.8f}"
    )
    print(
        "Qscr(native/grid/used)="
        f"{float(state['q_scr_raw']):.8f}/"
        f"{float(state['q_scr_grid_raw']):.8f}/"
        f"{float(state['q_scr_used']):.8f}; "
        f"scale={float(state['q_scr_scale_factor']):.8f}"
    )
    print(
        f"{'level':>7s} {'E [Ha]':>13s} {'E [eV]':>13s} " f"{'FD':>10s} {'occupation':>12s}"
    )
    for row in level_rows():
        print(
            f"{row[0]:>7s} {row[1]:13.6f} {row[2]:13.6f} " f"{row[3]:10.6f} {row[4]:12.6f}"
        )


    electronic_title = (
        rf"Al, $\rho={float(state['rho_g_cc']):g}$ g cm$^{{-3}}$, "
        rf"$T_e={float(state['te_ev']):g}$ eV, "
        rf"$\mu={float(state['mu_ha']):.5f}$ Ha"
    )
    orbital_state = load_plasma_state(ORBITAL_STATE_PATH)
    fig_orbitals = plot_bound_orbitals(orbital_state, title=electronic_title)
    save_figure(fig_orbitals, FIGURE_DIR / "al_full_workflow_orbitals", close=False)

    # %%
    # Electronic structure
    # --------------------
    #
    # Potentials use the solver's outer-potential gauge.

    with style_context("thesis", palette="bing"):
        r_e = np.asarray(state["r_e_bohr"], dtype=float)
        shell = 4.0 * np.pi * r_e**2
        r_ws = float(state["r_ws_bohr"])
        mask_e = r_e <= 8.0

        fig_electronic, (
            ax_density,
            ax_potential,
            ax_decomposition,
        ) = plt.subplots(
            1,
            3,
            figsize=grid_figsize(1, 3),
            layout="constrained",
        )
        density_curves = (
            (state["n_full_bohr3"], r"$n^{\rm full}$"),
            (np.interp(r_e, orbital_state["species_0_r_bohr"],
                       orbital_state["species_0_ion_orbital_density_r"].sum(axis=0)),
             r"$n^{\rm ion}$"),
            (state["n_ext_bohr3"], r"$n^{\rm ext}$"),
            (state["n_pa_bohr3"], r"$n^{\rm PA}$"),
            (state["n_scr_bohr3"], r"$n^{\rm scr}$"),
            (state["n0_bohr3"], r"$n_0$"),
        )
        for density, label in density_curves:
            ax_density.plot(
                r_e[mask_e],
                (shell * density)[mask_e],
                label=label,
            )
        ax_density.axvline(r_ws, color="0.25", ls=":", lw=1.1, label=r"$R_{\rm WS}$")
        ax_density.set(
            xlabel=r"$r$ [$a_B$]",
            ylabel=r"$4\pi r^2n(r)$ [$a_B^{-1}$]",
            xlim=(-0.5, 8.0),
            ylim=(-1.0, 15.0),
            title="Electronic densities",
        )
        ax_density.legend(ncol=2)

        safe_r = np.maximum(r_e, np.finfo(float).tiny)
        effective_potentials = (
            ("v_full_ha", r"$V_{\rm eff}^{\rm full}$"),
            ("v_ext_ha", r"$V_{\rm eff}^{\rm ext}$"),
        )
        for key, label in effective_potentials:
            ax_potential.plot(
                r_e[mask_e],
                np.asarray(state[key], dtype=float)[mask_e],
                ls="-",
                lw=2.0,
                label=label,
            )
        ax_potential.axhline(0.0, color="0.5", ls=":", lw=0.9)
        ax_potential.axvline(r_ws, color="0.25", ls=":", lw=1.1, label=r"$R_{\rm WS}$")
        ax_potential.set(
            xlabel=r"$r$ [$a_B$]",
            ylabel=r"$V(r)$ [Ha]",
            xlim=(-0.5, 5.0),
            ylim=(-1.0, 1.0),
            title="Effective potentials",
        )
        ax_potential.legend()

        ax_decomposition.plot(
            r_e[mask_e],
            np.asarray(state["v_full_ha"], dtype=float)[mask_e],
            ls="-",
            lw=2.0,
            label=r"$V_{\rm eff}^{\rm full}$",
        )
        ax_decomposition.plot(
            r_e[mask_e],
            np.asarray(state["v_hartree_ha"], dtype=float)[mask_e],
            ls="--",
            lw=1.5,
            label=r"$V_{\rm H}$",
        )
        ax_decomposition.plot(
            r_e[mask_e],
            np.asarray(state["v_xc_ha"], dtype=float)[mask_e],
            ls="--",
            lw=1.5,
            label=r"$V_{\rm xc}$",
        )
        ax_decomposition.plot(
            r_e[mask_e],
            (-13.0 / safe_r)[mask_e],
            ls="--",
            lw=1.5,
            label=r"$V_{\rm nuc}$",
        )
        ax_decomposition.axhline(0.0, color="0.5", ls=":", lw=0.9)
        ax_decomposition.axvline(r_ws, color="0.25", ls=":", lw=1.1, label=r"$R_{\rm WS}$")
        ax_decomposition.set(
            xlabel=r"$r$ [$a_B$]",
            ylabel=r"$V(r)$ [Ha]",
            xlim=(-0.5, 8.0),
            ylim=(-8.0, 8.0),
            title="Full-AA potential components",
        )
        ax_decomposition.legend(ncol=2)

        fig_electronic.suptitle(electronic_title)
        save_figure(
            fig_electronic,
            FIGURE_DIR / "al_full_workflow_electronic",
            close=False,
        )


    # %%
    # Pseudoatom to ion structure
    # ---------------------------
    #
    # f(k)=n_ion(k) and q(k)=n_scr(k) use electron-number normalization.

    with style_context("thesis", palette="bing"):
        k = np.asarray(state["k_bohr_inv"], dtype=float)
        r = np.asarray(state["r_bohr"], dtype=float)
        k_mask = k <= 8.0
        r_mask = r <= 12.0
        fig_pipeline, (ax_q, ax_vk, ax_vr) = plt.subplots(
            1, 3, figsize=grid_figsize(1, 3), layout="constrained",
        )
        fig_ionic, (ax_g, ax_s, ax_w) = plt.subplots(
            1, 3, figsize=grid_figsize(1, 3), layout="constrained",
        )

        ax_q.plot(
            k[k_mask],
            np.asarray(state["n_scr_k_electrons"])[k_mask],
        )
        ax_q.set(
            title=r"$q(k)=n_{\rm scr}(k)$",
            xlabel=r"$k$ [$a_B^{-1}$]",
            ylabel="electrons",
        )

        ax_vk.plot(k[k_mask], np.asarray(state["vii_k_ha_bohr3"])[k_mask])
        ax_vk.set(
            title=r"$V_{ii}(k)$",
            xlabel=r"$k$ [$a_B^{-1}$]",
            ylabel=r"Ha $a_B^3$",
        )

        ax_vr.plot(r[r_mask], np.asarray(state["vii_r_ha"])[r_mask])
        ax_vr.set(
            title=r"$V_{ii}(r)$",
            xlabel=r"$r$ [$a_B$]",
            ylabel="Ha",
            xlim=(-0.5, 12.0),
        )

        ax_g.plot(r[r_mask], np.asarray(state["gii_r"])[r_mask])
        ax_g.axhline(1.0, color="0.5", lw=0.8, ls=":")
        ax_g.set(
            title=r"$g_{ii}(r)$",
            xlabel=r"$r$ [$a_B$]",
            ylabel=r"$g_{ii}(r)$",
            xlim=(-0.5, 12.0),
        )

        ax_s.plot(k[k_mask], np.asarray(state["sii_k"])[k_mask])
        ax_s.axhline(1.0, color="0.5", lw=0.8, ls=":")
        ax_s.set(
            title=r"$S_{ii}(k)$",
            xlabel=r"$k$ [$a_B^{-1}$]",
            ylabel=r"$S_{ii}(k)$",
        )

        weight = np.abs(state["n_ion_k_electrons"] + state["n_scr_k_electrons"])**2 * state["sii_k"]
        ax_w.plot(k[k_mask], weight[k_mask])
        ax_w.set(
            title="Rayleigh weight",
            xlabel=r"$k$ [$a_B^{-1}$]",
            ylabel=r"$W_R(k)$",
        )

        ionic_title = (
            rf"Al pseudoatom/QOZ/HNC, "
            rf"$\rho={float(state['rho_g_cc']):g}$ g cm$^{{-3}}$, "
            rf"$T_e={float(state['te_ev']):g}$ eV, "
            rf"$T_i={float(state['ti_ev']):g}$ eV"
        )
        fig_pipeline.suptitle(ionic_title)
        fig_ionic.suptitle(ionic_title)
        save_figure(
            fig_pipeline,
            FIGURE_DIR / "al_full_workflow_ionic_pipeline",
            close=False,
        )

        save_figure(
            fig_ionic,
            FIGURE_DIR / "al_full_workflow_ionic_structure",
            close=False,
        )

        if EXPORT_SLIDE_FIGURES:
            slide_title = (
                rf"Al, $\rho={float(state['rho_g_cc']):g}$ g cm$^{{-3}}$, "
                rf"$T_e=T_i={float(state['te_ev']):g}$ eV"
            )

            fig_density, ax_density_slide = plt.subplots(figsize=grid_figsize(1, 1))
            for density, label in density_curves:
                ax_density_slide.plot(
                    r_e[mask_e],
                    (shell * density)[mask_e],
                    label=label,
                )
            ax_density_slide.axvline(r_ws, color="0.25", ls=":", lw=1.1, label=r"$R_{\rm WS}$")
            ax_density_slide.set(
                xlabel=r"$r$ [$a_B$]",
                ylabel=r"$4\pi r^2 n(r)$ [$a_B^{-1}$]",
                xlim=(-0.5, 8.0),
                ylim=(-1.0, 15.0),
                title=electronic_title,
            )
            ax_density_slide.legend(ncol=2)
            fig_density.tight_layout()
            save_figure(
                fig_density,
                FIGURE_DIR / "al_full_workflow_electronic_densities",
                close=True,
            )

            fig_gii, ax_gii_slide = plt.subplots(figsize=grid_figsize(1, 1))
            ax_gii_slide.plot(r[r_mask], np.asarray(state["gii_r"])[r_mask])
            ax_gii_slide.axhline(1.0, color="0.5", lw=0.9, ls=":")
            ax_gii_slide.set(
                xlabel=r"$r$ [$a_B$]",
                ylabel=r"$g_{ii}(r)$",
                xlim=(-0.5, 12.0),
                title=slide_title + r", $g_{ii}(r)$",
            )
            fig_gii.tight_layout()
            save_figure(fig_gii, FIGURE_DIR / "al_full_workflow_gii", close=True)

            fig_sii, ax_sii_slide = plt.subplots(figsize=grid_figsize(1, 1))
            ax_sii_slide.plot(k[k_mask], np.asarray(state["sii_k"])[k_mask])
            ax_sii_slide.axhline(1.0, color="0.5", lw=0.9, ls=":")
            ax_sii_slide.set(
                xlabel=r"$k$ [$a_B^{-1}$]",
                ylabel=r"$S_{ii}(k)$",
                xlim=(0.0, 8.0),
                title=slide_title + r", $S_{ii}(k)$",
            )
            fig_sii.tight_layout()
            save_figure(fig_sii, FIGURE_DIR / "al_full_workflow_sii", close=True)

    plt.show()



if __name__ == "__main__":
    main()

# sphinx_gallery_thumbnail_path = "_static/gallery_results/plot_al_full_workflow/thumbnail.png"
