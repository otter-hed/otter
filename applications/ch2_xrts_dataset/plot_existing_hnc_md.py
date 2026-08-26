"""Plot the completed CH2 HNC--MD comparisons without running any solver.

Edit the short input block below if the displayed r/k ranges should change,
then run this file directly.  Only cases marked successful in ``summary.json``
are included.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(ROOT / "applications/ch2_xrts_dataset/outputs/.matplotlib"),
)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from otter.plotting import PALETTES, grid_figsize, save_figure, set_style  # noqa: E402


# ============================== Inputs =====================================
RESULTS_DIR = Path(__file__).resolve().parent / "outputs/ch2_hnc_md_comparison"
R_RANGE_BOHR = (0.0, 10.0)
K_RANGE_BOHR_INV = (0.0, 4.0)
MIN_VECTORS_PER_K_BIN = 12
UNCERTAINTY_SIGMA = 2.0
UNCERTAINTY_ALPHA = 0.16
# ============================================================================


PAIR_INDICES = ((0, 0), (0, 1), (1, 1))
PAIR_LABELS = ("CC", "CH", "HH")
ALPHA_VALUES = (0.2, 0.5, 1.0)
ALPHA_COLORS = dict(zip(ALPHA_VALUES, PALETTES["bing"][1:4], strict=True))


def _case_dir(case: dict[str, object]) -> Path:
    return RESULTS_DIR / (
        f"Te{float(case['te_ev']):07.3f}_Ti{float(case['ti_ev']):07.3f}"
    )


def load_completed_cases() -> tuple[list[float], list[dict[str, object]]]:
    summary = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
    completed: list[dict[str, object]] = []
    for case in summary["cases"]:
        directory = _case_dir(case)
        if (
            case.get("hnc_status") == "success"
            and case.get("md_status") == "success"
            and (directory / "hnc_results.npz").is_file()
            and (directory / "md/md_results.npz").is_file()
        ):
            completed.append(case)
    completed.sort(key=lambda item: (float(item["te_ev"]), float(item["alpha"])))
    return [float(value) for value in summary["selected_te_ev"]], completed


def load_arrays(
    case: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    directory = _case_dir(case)
    with np.load(directory / "hnc_results.npz") as archive:
        hnc = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(directory / "md/md_results.npz") as archive:
        md = {name: np.asarray(archive[name]) for name in archive.files}
    return hnc, md


def _column_title(te_ev: float) -> str:
    return rf"$T_e={te_ev:g}$ eV"


def _alpha_handles() -> list[object]:
    return [
        Line2D([], [], color=ALPHA_COLORS[alpha], lw=2.2, label=rf"$T_i/T_e={alpha:g}$")
        for alpha in ALPHA_VALUES
    ]


def _comparison_legend_handles() -> list[object]:
    handles = _alpha_handles()
    handles.extend(
        (
            Line2D([], [], color="0.15", lw=2.0, ls="-", label="HNC"),
            Line2D([], [], color="0.15", lw=1.7, ls="--", label="MD"),
            Patch(
                facecolor="0.35",
                alpha=UNCERTAINTY_ALPHA,
                edgecolor="none",
                label=r"MD $\pm2$ SEM",
            ),
        )
    )
    return handles


def _finish_grid(fig: object, title: str, handles: list[object]) -> None:
    fig.suptitle(title, y=0.99)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=len(handles),
        frameon=False,
    )
    # Keep the shared legend close to the column titles without overlapping them.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), h_pad=0.8, w_pad=1.2)


def plot_gij(te_values: list[float], cases: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(
        3, 3, figsize=grid_figsize(3, 3), sharex=True, sharey=True
    )
    for column, te_ev in enumerate(te_values):
        axes[0, column].set_title(_column_title(te_ev))
        for case in cases:
            if not np.isclose(float(case["te_ev"]), te_ev):
                continue
            hnc, md = load_arrays(case)
            alpha = float(case["alpha"])
            color = ALPHA_COLORS[alpha]
            for row, (left, right) in enumerate(PAIR_INDICES):
                axes[row, column].plot(
                    hnc["r_bohr"], hnc["gij_r"][left, right],
                    color=color, lw=2.0, alpha=0.88,
                )
                g_md = md["md_gij_r"][:, row]
                g_sem = md["md_gij_block_sem"][:, row]
                axes[row, column].plot(
                    md["md_r_bohr"], g_md,
                    color=color, lw=1.6, ls="--", alpha=0.76,
                )
                axes[row, column].fill_between(
                    md["md_r_bohr"],
                    g_md - UNCERTAINTY_SIGMA * g_sem,
                    g_md + UNCERTAINTY_SIGMA * g_sem,
                    color=color, alpha=UNCERTAINTY_ALPHA, linewidth=0,
                )

    for row, label in enumerate(PAIR_LABELS):
        axes[row, 0].set_ylabel(rf"$g_{{{label}}}(r)$")
    for axis in axes[-1]:
        axis.set_xlabel(r"$r$ [Bohr]")
    for axis in axes.flat:
        axis.set_xlim(*R_RANGE_BOHR)
        axis.set_ylim(-0.04, 1.25)
    _finish_grid(
        fig,
        r"CH$_2$: HNC vs MD $g_{ab}(r)$",
        _comparison_legend_handles(),
    )
    save_figure(fig, RESULTS_DIR / "existing_hnc_md_gij", close=True)


def plot_sij(te_values: list[float], cases: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(
        3, 3, figsize=grid_figsize(3, 3), sharex=True, sharey="row"
    )
    for column, te_ev in enumerate(te_values):
        axes[0, column].set_title(_column_title(te_ev))
        for case in cases:
            if not np.isclose(float(case["te_ev"]), te_ev):
                continue
            hnc, md = load_arrays(case)
            alpha = float(case["alpha"])
            color = ALPHA_COLORS[alpha]
            reliable = md["md_vectors_per_k_bin"] >= MIN_VECTORS_PER_K_BIN
            for row, (left, right) in enumerate(PAIR_INDICES):
                axes[row, column].plot(
                    hnc["k_bohr_inv"], hnc["sij_k"][left, right],
                    color=color, lw=2.0, alpha=0.88,
                )
                k_md = md["md_k_bohr_inv"][reliable]
                s_md = md["md_sij_k"][reliable, row]
                s_sem = md["md_sij_frame_sem"][reliable, row]
                axes[row, column].plot(
                    k_md, s_md, color=color, lw=1.4, ls="--",
                    marker="o", ms=2.3, markevery=3, alpha=0.72,
                )
                axes[row, column].fill_between(
                    k_md,
                    s_md - UNCERTAINTY_SIGMA * s_sem,
                    s_md + UNCERTAINTY_SIGMA * s_sem,
                    color=color, alpha=UNCERTAINTY_ALPHA, linewidth=0,
                )

    for row, label in enumerate(PAIR_LABELS):
        axes[row, 0].set_ylabel(rf"$S_{{{label}}}(k)$")
    for axis in axes[-1]:
        axis.set_xlabel(r"$k$ [Bohr$^{{-1}}$]")
    for axis in axes.flat:
        axis.set_xlim(*K_RANGE_BOHR_INV)
        axis.margins(y=0.08)
    _finish_grid(
        fig,
        r"CH$_2$: HNC vs MD $S_{ab}(k)$",
        _comparison_legend_handles(),
    )
    save_figure(fig, RESULTS_DIR / "existing_hnc_md_sij", close=True)


def plot_gij_difference(te_values: list[float], cases: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(
        3, 3, figsize=grid_figsize(3, 3), sharex=True, sharey=True
    )
    plotted: list[np.ndarray] = []
    for column, te_ev in enumerate(te_values):
        axes[0, column].set_title(_column_title(te_ev))
        for case in cases:
            if not np.isclose(float(case["te_ev"]), te_ev):
                continue
            hnc, md = load_arrays(case)
            alpha = float(case["alpha"])
            color = ALPHA_COLORS[alpha]
            r_md = md["md_r_bohr"]
            visible = (r_md >= R_RANGE_BOHR[0]) & (r_md <= R_RANGE_BOHR[1])
            for row, (left, right) in enumerate(PAIR_INDICES):
                g_hnc = np.interp(r_md, hnc["r_bohr"], hnc["gij_r"][left, right])
                difference = md["md_gij_r"][:, row] - g_hnc
                sem = md["md_gij_block_sem"][:, row]
                plotted.append(np.abs(difference[visible]) + UNCERTAINTY_SIGMA * sem[visible])
                axes[row, column].plot(
                    r_md, difference, color=color, lw=1.6, alpha=0.80,
                )
                axes[row, column].fill_between(
                    r_md,
                    difference - UNCERTAINTY_SIGMA * sem,
                    difference + UNCERTAINTY_SIGMA * sem,
                    color=color, alpha=UNCERTAINTY_ALPHA, linewidth=0,
                )

    limit = 1.08 * max(float(np.max(values)) for values in plotted)
    for row, label in enumerate(PAIR_LABELS):
        axes[row, 0].set_ylabel(rf"$g^{{\rm MD}}_{{{label}}}-g^{{\rm HNC}}_{{{label}}}$")
    for axis in axes[-1]:
        axis.set_xlabel(r"$r$ [Bohr]")
    for axis in axes.flat:
        axis.axhline(0.0, color="0.35", lw=0.8, alpha=0.65)
        axis.set_xlim(*R_RANGE_BOHR)
        axis.set_ylim(-limit, limit)
    residual_handles = _alpha_handles()
    residual_handles.extend(
        (
            Line2D([], [], color="0.15", lw=1.7, label="MD $-$ HNC"),
            Patch(
                facecolor="0.35",
                alpha=UNCERTAINTY_ALPHA,
                edgecolor="none",
                label=r"MD $\pm2$ SEM",
            ),
        )
    )
    _finish_grid(
        fig,
        r"CH$_2$: MD $-$ HNC $\Delta g_{ab}(r)$",
        residual_handles,
    )
    save_figure(fig, RESULTS_DIR / "existing_hnc_md_gij_difference", close=True)


def plot_sij_difference(te_values: list[float], cases: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(
        3, 3, figsize=grid_figsize(3, 3), sharex=True, sharey="row"
    )
    plotted: list[list[np.ndarray]] = [[], [], []]
    for column, te_ev in enumerate(te_values):
        axes[0, column].set_title(_column_title(te_ev))
        for case in cases:
            if not np.isclose(float(case["te_ev"]), te_ev):
                continue
            hnc, md = load_arrays(case)
            alpha = float(case["alpha"])
            color = ALPHA_COLORS[alpha]
            k_md = md["md_k_bohr_inv"]
            visible = (
                (k_md >= K_RANGE_BOHR_INV[0])
                & (k_md <= K_RANGE_BOHR_INV[1])
                & (md["md_vectors_per_k_bin"] >= MIN_VECTORS_PER_K_BIN)
            )
            k_visible = k_md[visible]
            for row, (left, right) in enumerate(PAIR_INDICES):
                s_hnc = np.interp(
                    k_visible,
                    hnc["k_bohr_inv"],
                    hnc["sij_k"][left, right],
                )
                difference = md["md_sij_k"][visible, row] - s_hnc
                sem = md["md_sij_frame_sem"][visible, row]
                plotted[row].append(
                    np.abs(difference) + UNCERTAINTY_SIGMA * sem
                )
                axes[row, column].plot(
                    k_visible,
                    difference,
                    color=color,
                    lw=1.4,
                    marker="o",
                    ms=2.3,
                    markevery=3,
                    alpha=0.76,
                )
                axes[row, column].fill_between(
                    k_visible,
                    difference - UNCERTAINTY_SIGMA * sem,
                    difference + UNCERTAINTY_SIGMA * sem,
                    color=color,
                    alpha=UNCERTAINTY_ALPHA,
                    linewidth=0,
                )

    for row, label in enumerate(PAIR_LABELS):
        axes[row, 0].set_ylabel(
            rf"$S^{{\rm MD}}_{{{label}}}-S^{{\rm HNC}}_{{{label}}}$"
        )
        limit = 1.08 * max(float(np.max(values)) for values in plotted[row])
        for axis in axes[row]:
            axis.set_ylim(-limit, limit)
    for axis in axes[-1]:
        axis.set_xlabel(r"$k$ [Bohr$^{{-1}}$]")
    for axis in axes.flat:
        axis.axhline(0.0, color="0.35", lw=0.8, alpha=0.65)
        axis.set_xlim(*K_RANGE_BOHR_INV)
    residual_handles = _alpha_handles()
    residual_handles.extend(
        (
            Line2D([], [], color="0.15", lw=1.7, label="MD $-$ HNC"),
            Patch(
                facecolor="0.35",
                alpha=UNCERTAINTY_ALPHA,
                edgecolor="none",
                label=r"MD $\pm2$ SEM",
            ),
        )
    )
    _finish_grid(
        fig,
        r"CH$_2$: MD $-$ HNC $\Delta S_{ab}(k)$",
        residual_handles,
    )
    save_figure(fig, RESULTS_DIR / "existing_hnc_md_sij_difference", close=True)


def write_metrics(cases: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        hnc, md = load_arrays(case)
        r_md = md["md_r_bohr"]
        r_mask = (r_md >= R_RANGE_BOHR[0]) & (r_md <= R_RANGE_BOHR[1])
        k_md = md["md_k_bohr_inv"]
        k_mask = (
            (k_md >= K_RANGE_BOHR_INV[0])
            & (k_md <= K_RANGE_BOHR_INV[1])
            & (md["md_vectors_per_k_bin"] >= MIN_VECTORS_PER_K_BIN)
        )
        for pair, (left, right) in zip(PAIR_LABELS, PAIR_INDICES, strict=True):
            pair_index = PAIR_LABELS.index(pair)
            g_hnc = np.interp(r_md, hnc["r_bohr"], hnc["gij_r"][left, right])
            s_hnc = np.interp(k_md, hnc["k_bohr_inv"], hnc["sij_k"][left, right])
            g_difference = md["md_gij_r"][:, pair_index] - g_hnc
            s_difference = md["md_sij_k"][:, pair_index] - s_hnc
            rows.append(
                {
                    "te_ev": float(case["te_ev"]),
                    "ti_ev": float(case["ti_ev"]),
                    "ti_over_te": float(case["alpha"]),
                    "pair": pair,
                    "g_rmse": float(np.sqrt(np.mean(g_difference[r_mask] ** 2))),
                    "g_mae": float(np.mean(np.abs(g_difference[r_mask]))),
                    "g_max_abs": float(np.max(np.abs(g_difference[r_mask]))),
                    "s_rmse": float(np.sqrt(np.mean(s_difference[k_mask] ** 2))),
                    "s_mae": float(np.mean(np.abs(s_difference[k_mask]))),
                    "s_max_abs": float(np.max(np.abs(s_difference[k_mask]))),
                }
            )

    path = RESULTS_DIR / "existing_hnc_md_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.8g}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )
    return rows


def main() -> None:
    set_style("thesis", palette="bing")
    te_values, cases = load_completed_cases()
    if not cases:
        raise RuntimeError(f"No completed HNC--MD cases found in {RESULTS_DIR}")
    metrics = write_metrics(cases)
    plot_gij(te_values, cases)
    plot_sij(te_values, cases)
    plot_gij_difference(te_values, cases)
    plot_sij_difference(te_values, cases)
    g_rmse = np.asarray([float(row["g_rmse"]) for row in metrics])
    s_rmse = np.asarray([float(row["s_rmse"]) for row in metrics])
    print(f"Plotted {len(cases)}/9 completed HNC--MD cases")
    print(f"g(r) RMSE range: {g_rmse.min():.3g}--{g_rmse.max():.3g}")
    print(f"S(k) RMSE range: {s_rmse.min():.3g}--{s_rmse.max():.3g}")
    print(f"Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
