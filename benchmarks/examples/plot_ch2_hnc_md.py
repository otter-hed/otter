r"""
CH2: HNC versus same-potential MD
=================================

This benchmark tests whether ordinary multicomponent HNC is an adequate fast
ion-structure approximation for CH2 at :math:`\rho=0.946` g cm\ :sup:`-3`.
It compares Otter QOZ/HNC with classical LAMMPS MD
:cite:p:`ThompsonEtAl2022` at
:math:`T_e=9,29,99` eV and :math:`T_i/T_e=0.2,0.5,1.0`.  Six of the nine
states are explicitly two-temperature states; the three
:math:`T_i/T_e=1` cases are equilibrium controls.

At each electron temperature, one converged IS electronic result is reused
for all three ion temperatures.  For each state, HNC and MD use the same
QOZ-derived :math:`V_{ab}^{\mathrm{eff}}(r)`, so their difference comes from
the ionic statistical treatment rather than a different electronic
structure.  For the two hottest MD runs only, the analytically known Coulomb
core is restored below 0.3 Bohr and blended back to the unchanged QOZ
potential by 0.5 Bohr; the resolved first-shell potential is not refitted.

Across the tested window, the pairwise HNC--MD RMSE is
:math:`0.00281\text{--}0.0183` for :math:`g_{ab}(r)` and
:math:`0.00632\text{--}0.0169` for :math:`S_{ab}(k)`.  HNC takes
:math:`0.38\text{--}1.16` s per state, while the same-potential MD takes
:math:`8.6\text{--}224` min.  Thus HNC is a good fast approximation for the
specific density and temperature range tested here; this is not a universal
claim for chemically bonded or more strongly coupled mixtures.

The MD :math:`S_{ab}(k)` is evaluated directly from periodic density modes,
not from a truncated transform of the RDF.  Shaded regions show
:math:`\pm2` SEM.  Only reciprocal shells containing at least 12 vectors are
plotted.  The accepted arrays are compact, checksummed outputs of
``applications/ch2_xrts_dataset/compare_hnc_md.py``; documentation builds do
not rerun the 6.7-hour MD scan.

Results
-------

.. image:: /_static/benchmarks/ch2_hnc_md/ch2_hnc_md_gab.png
   :alt: CH2 HNC and same-potential MD partial pair distributions
   :width: 100%

.. image:: /_static/benchmarks/ch2_hnc_md/ch2_hnc_md_sab.png
   :alt: CH2 HNC and same-potential MD partial structure factors
   :width: 100%

.. image:: /_static/benchmarks/ch2_hnc_md/ch2_hnc_md_sab_residual.png
   :alt: CH2 MD minus HNC partial structure-factor residuals
   :width: 100%

Representative OVITO rendering of the 3072-ion CH2 trajectory
----------------------------------------------------------------

The animation is illustrative and is not used for the numerical comparison.

.. raw:: html

   <video controls loop muted playsinline preload="metadata"
          poster="../../_static/benchmarks/ch2_hnc_md/ch2_md.png"
          style="display:block; width:min(100%,760px); margin:0 auto;">
     <source src="../../_static/benchmarks/ch2_hnc_md/ch2_md.mp4"
             type="video/mp4">
     Your browser does not support embedded MP4 video.
   </video>

.. only:: not html

   .. image:: /_static/benchmarks/ch2_hnc_md/ch2_md.png
      :alt: OVITO rendering of the CH2 same-potential MD trajectory
"""

# sphinx_gallery_thumbnail_path = '_static/benchmarks/ch2_hnc_md/ch2_hnc_md_gab.png'

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from otter.plotting import PALETTES, grid_figsize, save_figure, set_style


USE_PRECOMPUTED_DATA = True

BENCHMARK_ID = "ch2_hnc_md"
PAIR_LABELS = ("CC", "CH", "HH")
ALPHA_VALUES = (0.2, 0.5, 1.0)
R_RANGE_BOHR = (0.0, 10.0)
K_RANGE_BOHR_INV = (0.0, 4.0)
MIN_VECTORS_PER_K_BIN = 12
UNCERTAINTY_SIGMA = 2.0
UNCERTAINTY_ALPHA = 0.22


def repository_root() -> Path:
    """Locate the checkout in direct and Sphinx-Gallery execution modes."""
    candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    source = Path(str(globals().get("__file__", Path.cwd()))).resolve()
    candidates.extend([source.parent, *source.parents])
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "benchmarks" / "baselines" / BENCHMARK_ID
        ).is_dir():
            return candidate
    raise FileNotFoundError("Cannot locate the Otter checkout.")


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


ROOT = repository_root()
BASELINE_DIR = ROOT / "benchmarks" / "baselines" / BENCHMARK_ID
OUTPUT_DIR = ROOT / "benchmarks" / "outputs" / BENCHMARK_ID
FIGURE_DIR = OUTPUT_DIR / "figures"


def load_baseline() -> dict[str, np.ndarray]:
    """Verify the accepted numerical archive and bundled media checksums."""
    if not USE_PRECOMPUTED_DATA:
        raise RuntimeError(
            "Run applications/ch2_xrts_dataset/compare_hnc_md.py to create "
            "fresh long-running candidates; this gallery never overwrites "
            "the accepted baseline."
        )
    manifest = json.loads(
        (BASELINE_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("Unexpected CH2 HNC--MD benchmark manifest.")
    record = manifest["baseline"]
    path = BASELINE_DIR / str(record["path"])
    if sha256_file(path) != str(record["sha256"]):
        raise RuntimeError(f"Checksum mismatch for {path}.")
    for media in manifest["media"]:
        media_path = ROOT / str(media["path"])
        if sha256_file(media_path) != str(media["sha256"]):
            raise RuntimeError(f"Checksum mismatch for {media_path}.")
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    if str(payload["schema_version"].item()) != "otter_ch2_hnc_md_v1":
        raise ValueError("Obsolete CH2 HNC--MD baseline schema.")
    if payload["te_ev"].shape != (9,) or np.count_nonzero(
        ~np.isclose(payload["alpha"], 1.0)
    ) != 6:
        raise ValueError("Expected nine states, including six non-equilibrium states.")
    return payload


data = load_baseline()
print("Loaded nine checksummed HNC--MD states (six two-temperature states).")


def alpha_handles() -> list[object]:
    """Return shared temperature-ratio legend handles."""
    return [
        Line2D(
            [], [], color=ALPHA_COLORS[alpha], lw=2.2,
            label=rf"$T_i/T_e={alpha:g}$",
        )
        for alpha in ALPHA_VALUES
    ]


def comparison_handles() -> list[object]:
    """Return the common method and uncertainty legend."""
    return alpha_handles() + [
        Line2D([], [], color="0.15", lw=2.0, ls="-", label="HNC"),
        Line2D([], [], color="0.15", lw=1.6, ls="--", label="MD"),
        Patch(
            facecolor="0.35", alpha=UNCERTAINTY_ALPHA, edgecolor="none",
            label=r"MD $\pm2$ SEM",
        ),
    ]


def finish_grid(fig: object, title: str, handles: list[object]) -> None:
    """Apply the shared compact title and top legend layout."""
    fig.suptitle(title, y=0.99)
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.925),
        ncol=len(handles), frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), h_pad=0.8, w_pad=1.2)


TE_VALUES = tuple(float(value) for value in np.unique(data["te_ev"]))
ALPHA_COLORS = dict(zip(ALPHA_VALUES, PALETTES["bing"][1:4], strict=True))
set_style("thesis", palette="bing")


# %%
# Pair distributions
# ------------------

fig_g, axes = plt.subplots(
    3, 3, figsize=grid_figsize(3, 3), sharex=True, sharey=True
)
for column, te_ev in enumerate(TE_VALUES):
    axes[0, column].set_title(rf"$T_e={te_ev:g}$ eV")
    for state in np.flatnonzero(np.isclose(data["te_ev"], te_ev)):
        alpha = float(data["alpha"][state])
        color = ALPHA_COLORS[alpha]
        for pair, label in enumerate(PAIR_LABELS):
            axes[pair, column].plot(
                data["hnc_r_bohr"][state], data["hnc_gij_r"][state, pair],
                color=color, lw=2.0, alpha=0.88,
            )
            g_md = data["md_gij_r"][state, pair]
            sem = data["md_gij_sem"][state, pair]
            axes[pair, column].plot(
                data["md_r_bohr"], g_md, color=color, lw=1.6,
                ls="--", alpha=0.76,
            )
            axes[pair, column].fill_between(
                data["md_r_bohr"],
                g_md - UNCERTAINTY_SIGMA * sem,
                g_md + UNCERTAINTY_SIGMA * sem,
                color=color, alpha=UNCERTAINTY_ALPHA, linewidth=0.0,
            )
for row, label in enumerate(PAIR_LABELS):
    axes[row, 0].set_ylabel(rf"$g_{{{label}}}(r)$")
for axis in axes[-1]:
    axis.set_xlabel(r"$r$ [Bohr]")
for axis in axes.flat:
    axis.set(xlim=R_RANGE_BOHR, ylim=(-0.04, 1.25))
finish_grid(fig_g, r"CH$_2$: HNC vs MD $g_{ab}(r)$", comparison_handles())
_ = save_figure(fig_g, FIGURE_DIR / "ch2_hnc_md_gab", close=True)


# %%
# Partial structure factors
# -------------------------

fig_s, axes = plt.subplots(
    3, 3, figsize=grid_figsize(3, 3), sharex=True, sharey="row"
)
reliable = data["md_vectors_per_k_bin"] >= MIN_VECTORS_PER_K_BIN
for column, te_ev in enumerate(TE_VALUES):
    axes[0, column].set_title(rf"$T_e={te_ev:g}$ eV")
    for state in np.flatnonzero(np.isclose(data["te_ev"], te_ev)):
        alpha = float(data["alpha"][state])
        color = ALPHA_COLORS[alpha]
        for pair, label in enumerate(PAIR_LABELS):
            axes[pair, column].plot(
                data["hnc_k_bohr_inv"][state],
                data["hnc_sij_k"][state, pair],
                color=color, lw=2.0, alpha=0.88,
            )
            s_md = data["md_sij_k"][state, pair, reliable]
            sem = data["md_sij_sem"][state, pair, reliable]
            k_md = data["md_k_bohr_inv"][reliable]
            axes[pair, column].plot(
                k_md, s_md, color=color, lw=1.4, ls="--", marker="o",
                ms=2.3, markevery=3, alpha=0.72,
            )
            axes[pair, column].fill_between(
                k_md,
                s_md - UNCERTAINTY_SIGMA * sem,
                s_md + UNCERTAINTY_SIGMA * sem,
                color=color, alpha=UNCERTAINTY_ALPHA, linewidth=0.0,
            )
for row, label in enumerate(PAIR_LABELS):
    axes[row, 0].set_ylabel(rf"$S_{{{label}}}(k)$")
for axis in axes[-1]:
    axis.set_xlabel(r"$k$ [Bohr$^{-1}$]")
for axis in axes.flat:
    axis.set_xlim(*K_RANGE_BOHR_INV)
    axis.margins(y=0.08)
finish_grid(fig_s, r"CH$_2$: HNC vs MD $S_{ab}(k)$", comparison_handles())
_ = save_figure(fig_s, FIGURE_DIR / "ch2_hnc_md_sab", close=True)


# %%
# Structure-factor residuals
# --------------------------

fig_d, axes = plt.subplots(
    3, 3, figsize=grid_figsize(3, 3), sharex=True, sharey=True
)
residual_extent: list[np.ndarray] = []
visible = reliable & (data["md_k_bohr_inv"] <= K_RANGE_BOHR_INV[1])
for column, te_ev in enumerate(TE_VALUES):
    axes[0, column].set_title(rf"$T_e={te_ev:g}$ eV")
    for state in np.flatnonzero(np.isclose(data["te_ev"], te_ev)):
        alpha = float(data["alpha"][state])
        color = ALPHA_COLORS[alpha]
        for pair, label in enumerate(PAIR_LABELS):
            hnc = np.interp(
                data["md_k_bohr_inv"], data["hnc_k_bohr_inv"][state],
                data["hnc_sij_k"][state, pair],
            )
            delta = data["md_sij_k"][state, pair] - hnc
            sem = data["md_sij_sem"][state, pair]
            residual_extent.append(
                np.abs(delta[visible]) + UNCERTAINTY_SIGMA * sem[visible]
            )
            axes[pair, column].plot(
                data["md_k_bohr_inv"][reliable], delta[reliable],
                color=color, lw=1.5, alpha=0.78,
            )
            axes[pair, column].fill_between(
                data["md_k_bohr_inv"][reliable],
                delta[reliable] - UNCERTAINTY_SIGMA * sem[reliable],
                delta[reliable] + UNCERTAINTY_SIGMA * sem[reliable],
                color=color, alpha=UNCERTAINTY_ALPHA, linewidth=0.0,
            )
limit = 1.08 * max(float(np.max(values)) for values in residual_extent)
for row, label in enumerate(PAIR_LABELS):
    axes[row, 0].set_ylabel(
        r"$S_{%s}^{\mathrm{MD}}-S_{%s}^{\mathrm{HNC}}$" % (label, label)
    )
for axis in axes[-1]:
    axis.set_xlabel(r"$k$ [Bohr$^{-1}$]")
for axis in axes.flat:
    axis.axhline(0.0, color="0.35", lw=0.8, alpha=0.65)
    axis.set(xlim=K_RANGE_BOHR_INV, ylim=(-limit, limit))
finish_grid(
    fig_d,
    r"CH$_2$: MD $-$ HNC $\Delta S_{ab}(k)$",
    alpha_handles() + [
        Line2D([], [], color="0.15", lw=1.6, label="MD $-$ HNC"),
        Patch(
            facecolor="0.35", alpha=UNCERTAINTY_ALPHA, edgecolor="none",
            label=r"MD $\pm2$ SEM",
        ),
    ],
)
_ = save_figure(fig_d, FIGURE_DIR / "ch2_hnc_md_sab_residual", close=True)


print(
    "HNC seconds: "
    f"{np.min(data['hnc_elapsed_s']):.3f}--{np.max(data['hnc_elapsed_s']):.3f}; "
    "MD minutes: "
    f"{np.min(data['md_elapsed_s']) / 60.0:.2f}--"
    f"{np.max(data['md_elapsed_s']) / 60.0:.2f}."
)
print(
    "Pairwise RMSE ranges: "
    f"g={np.min(data['g_rmse']):.5f}--{np.max(data['g_rmse']):.5f}; "
    f"S={np.min(data['s_rmse']):.5f}--{np.max(data['s_rmse']):.5f}."
)

if "agg" not in plt.get_backend().lower():
    plt.show()
