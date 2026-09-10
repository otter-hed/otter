r"""
Beryllium ionization: Döppner et al. (2023)
===========================================

Beryllium ionization at :math:`T_e=50,100,150` eV and densities
1--70 g cm\ :sup:`-3` is compared with Fig. 3(a) of
:cite:t:`DoppnerEtAl2023`. Otter uses the QM ion-sphere model with Dirac
exchange. The two ionization definitions are

.. math::

   \bar Z_{\mathrm{AA,WS}}=4-4\pi\int_0^{R_{\rm WS}}n_{\rm ion}(r)r^2dr,
   \qquad Z^*=n_e^0/n_i.

where :math:`n_{\rm ion}` includes the pressure-ionization weight and radial
cutoff of :cite:t:`StarrettSaumon2013,StarrettSaumon2014`.
Ionization definitions depend on the electron partition used by each model.
The Stewart--Pyatt and OPAL reference curves in the 150 eV panel are at
160 eV, as labelled in the paper.

Reproduction
------------

From the root of the complete Otter checkout, using Poetry, run::

    poetry run python benchmarks/examples/plot_doppner_2023_be_ionization.py

Downloads are optional: ``.ipynb`` launches this repository script; ``.zip``
contains both formats. See :doc:`/user_guide/reproducing_galleries` for setup.

The script calculates the states from their input parameters and then plots
the results. No bundled Otter NPZ is required. Numerical outputs are written
locally; literature reference tables remain inputs to the comparison.

Recorded results
----------------

The figures and output below are from the recorded validation run; running
the source recalculates them with the installed Otter version.

.. include:: /_static/gallery_results/plot_doppner_2023_be_ionization/results.rst

"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np

from otter.plotting import PALETTES, grid_figsize, save_figure, set_style

RECOMPUTE_WITH_OTTER = True
USE_CANDIDATES = True
if os.environ.get("OTTER_USE_CANDIDATE_BE_IONIZATION", "0") == "1":
    USE_CANDIDATES = True
PACKAGE = "doppner_2023_be_ionization"


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


def load_reference() -> list[dict]:
    """Verify and parse ragged two-column series, never treating blanks as zero."""
    directory = ROOT / "benchmarks/reference_data/doppner_2023_Be_ionization_fig3a"
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / manifest["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Be reference checksum mismatch")
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    curves = []
    for series in manifest["series"]:
        i, j = series["columns"]
        if rows[0][i] != series["header"] or [rows[1][i], rows[1][j]] != ["rho", "Zbar"]:
            raise ValueError("Be reference column mapping changed")
        values = []
        for row in rows[2:]:
            if bool(row[i]) != bool(row[j]):
                raise ValueError("Incomplete reference coordinate pair")
            if row[i]:
                values.append((float(row[i]), float(row[j])))
        array = np.asarray(values)
        if (len(array) != series["points"] or not np.all(np.isfinite(array))
                or not np.all(np.diff(array[:, 0]) > 0)):
            raise ValueError("Invalid Be reference series")
        curves.append({**series, "rho": array[:, 0], "z": array[:, 1]})
    return curves


def load_scan() -> dict[str, np.ndarray]:
    directory = ROOT / "benchmarks" / ("outputs" if USE_CANDIDATES else "baselines") / PACKAGE
    if USE_CANDIDATES:
        directory /= "recomputed"
    manifest = json.loads((directory / "manifest.json").read_text())
    record = manifest["states"][0]
    path = directory / record["baseline_file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["baseline_sha256"]:
        raise ValueError("Be Otter checksum mismatch")
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    audit = manifest["scientific_audit"]
    if (data["zbar"].size != audit["full_scf_converged_states"]
            or not np.all(data["stage2_converged"])):
        raise ValueError("Be scan disagrees with its convergence audit")
    return data


def plot_comparison(data: dict[str, np.ndarray], references: list[dict]):
    """Use the Bethkenhagen gallery's Otter/reference visual convention."""
    set_style("thesis", palette="bing")
    colors = PALETTES["bing"]
    reference_styles = {
        "DFT-MD": (colors[2], ":", "o"),
        "Stewart-Pyatt": (colors[3], "-.", None),
        "OPAL": (colors[4], "--", None),
    }
    fig, axes = plt.subplots(1, 3, figsize=grid_figsize(1, 3), sharey=True)
    for ax, temperature in zip(axes, (50, 100, 150), strict=True):
        mask = data["te_ev"] == temperature
        rho = np.asarray([1, 3, 6, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
        indices = np.searchsorted(rho, data["rho_g_cc"][mask])
        resolved = data["threshold_state_status"][mask] == "resolved"
        for field, label, color in (
            ("zbar", r"Otter $\bar Z_{\rm AA,WS}$", "#131313"),
            ("zstar", r"Otter $Z^*$", colors[1]),
        ):
            y = data[field][mask]
            plotted = np.full(rho.size, np.nan)
            plotted[indices] = np.where(resolved, y, np.nan)
            ax.plot(rho, plotted, color=color, ls="-", lw=2.3,
                    alpha=0.45, marker="o", ms=4.0, label=label)
            if np.any(~resolved):
                ax.scatter(rho[indices[~resolved]], y[~resolved], color=color, marker="x", zorder=5)
        for reference in references:
            t_ref = reference["temperature_ev"]
            if t_ref != temperature and not (temperature == 150 and t_ref == 160):
                continue
            method = reference["method"]
            label = method + (" (160 eV)" if t_ref == 160 else "")
            color, line_style, marker = reference_styles[method]
            ax.plot(reference["rho"], reference["z"],
                    color=color, ls=line_style, lw=1.75,
                    marker=marker, ms=5.0 if marker else None,
                    markerfacecolor="white" if marker else None,
                    markeredgewidth=1.1 if marker else None,
                    alpha=0.75, label=label)
        ax.set(title=rf"Be: $T_e={temperature}$ eV", xlabel=r"$\rho$ (g cm$^{-3}$)",
               xlim=(0, 72))
        ax.legend(ncol=1 if temperature == 150 else 2, fontsize="small",
                  loc="best", columnspacing=0.8, handlelength=1.8)
    axes[0].set_ylabel("Ionization (electrons per Be)")
    fig.tight_layout()
    return fig


def main() -> None:
    if RECOMPUTE_WITH_OTTER:
        subprocess.run([sys.executable, str(ROOT / "benchmarks/runners/regenerate_doppner_2023_be_ionization.py")], check=True)
    data = load_scan()
    references = load_reference()
    fig = plot_comparison(data, references)
    save_figure(fig, ROOT / "benchmarks/outputs" / PACKAGE / "figures/be_ionization")
    print("QM ion-sphere, Dirac exchange.")
    audit = json.loads(data["metadata_json"].item())["convergence"]
    counts = ", ".join(f"{key}={value}" for key, value in sorted(audit["threshold_counts"].items()))
    print(f"Full SCF: {audit['full_scf_converged_states']}/{audit['requested_states']}; "
          f"threshold: {counts}; failed={audit['stage2_nonconverged_states']}")
    if np.any(data["threshold_state_status"] != "resolved"):
        print("Crosses mark non-resolved threshold states; failed points are omitted.")
    print(f"{'T [eV]':>8} {'rho [g/cc]':>12} {'Zbar AA,WS':>12} {'Zstar':>10} {'mu [Ha]':>12} {'threshold':>12} {'time [s]':>10}")
    for i in range(data["zbar"].size):
        print(f"{data['te_ev'][i]:8g} {data['rho_g_cc'][i]:12g} {data['zbar'][i]:12.6f} "
              f"{data['zstar'][i]:10.6f} {data['mu_ha'][i]:12.6f} "
              f"{data['threshold_state_status'][i]:>12} {data['elapsed_s'][i]:10.2f}")
    for failure in audit.get("failures", []):
        print(f"FAILED: T={failure['te_ev']} eV, rho={failure['rho_g_cc']} g/cc: {failure['error']}")


if __name__ == "__main__":
    main()

# %%
# Reproduction
# ------------
# The runner called above computes all 48 states, one AA at a time. Results are saved in
# ``benchmarks/outputs/doppner_2023_be_ionization/recomputed``.

# sphinx_gallery_thumbnail_path = "_static/gallery_results/plot_doppner_2023_be_ionization/thumbnail.png"
