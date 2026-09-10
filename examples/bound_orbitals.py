"""Plot final KS wavefunctions, ion-orbital densities, and their form factors.

From a source checkout: poetry run python examples/bound_orbitals.py
"""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from otter import (
    PlasmaWorkflowConfig, bound_wavefunctions, ion_orbital_form_factors,
    solve_plasma_workflow,
)
from otter.plotting import grid_figsize, save_figure, style_context

# User inputs. Numerical solver controls retain their defaults.
ELEMENT = "Al"
RHO_G_CC = 8.1
TE_EV = 1.0
TI_EV = TE_EV
MULTIPLY_WAVEFUNCTION_BY_FD = False
SAVE_FIGURES = False
FIGURE_STEM = Path(__file__).resolve().parents[1] / "outputs/bound_orbitals"


def plot_orbitals(aa, ion, *, multiply_fd=False, r_max=5.0, k_max=10.0):
    """Display raw wavefunctions and ionic density/form-factor components."""
    energies = np.asarray(aa["bound_energy_ha"])
    valid = np.isfinite(energies) & (energies < aa["bound_energy_cut_ha"])
    li, ni = np.nonzero(valid)
    angular = np.asarray(aa["bound_l_list"], dtype=int)[li]
    principal = ni + angular + 1
    r_wave, r, k = np.asarray(aa["r_bound"]), np.asarray(aa["r"]), np.asarray(ion["k"])
    wave = bound_wavefunctions(aa, multiply_fd=multiply_fd)[valid]
    density = np.asarray(aa["ion_orbital_density_r"])[valid]
    factors = ion_orbital_form_factors(aa, r=ion["r"], k=ion["k"])[valid]
    total_density, r_ws = np.asarray(aa["n_ion"]), float(aa["r_ws"])
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
                    ylabel=(r"$4\pi r^2 f_{\rm FD}R_{nl}(r)$" if multiply_fd else r"$4\pi r^2 R_{nl}(r)$") + r" [$a_B^{1/2}$]",
                    title="FD-weighted wavefunctions" if multiply_fd else "Unweighted wavefunctions")
        axes[1].set(xlabel=r"$r$ [$a_B$]", xlim=(-0.25, r_max),
                    ylabel=r"$4\pi r^2 n^{\rm ion}_{nl}(r)$ [$a_B^{-1}$]", title="Ion-orbital densities")
        axes[2].set(xlabel=r"$k$ [$a_B^{-1}$]", xlim=(0, k_max),
                    ylabel=r"$f_{nl}(k)=n^{\rm ion}_{nl}(k)$", title="Ion-orbital form factors")
        for ax in axes[:2]:
            ax.axvline(r_ws, color="0.6", ls="--", lw=1.0, label=r"$R_{\rm WS}$")
        for ax in axes:
            if ax.lines:
                ax.legend()
    return fig


def main():
    config = PlasmaWorkflowConfig(
        elements=[ELEMENT], rho_g_cc=RHO_G_CC,
        temperature_ev=TE_EV, ion_temperature_ev=TI_EV,
    )
    start = perf_counter()
    result = solve_plasma_workflow(config)
    print(f"Calculation wall time: {perf_counter() - start:.2f} s")
    aa, ion = result["electronic"]["result"], result["ion"]
    factors = ion_orbital_form_factors(aa, r=ion["r"], k=ion["k"])
    print("max |sum(n_ion_nl) - n_ion| =", np.max(np.abs(
        aa["ion_orbital_density_r"].sum(axis=(0, 1)) - aa["n_ion"])))
    print("max |sum(f_nl) - f| =", np.max(np.abs(factors.sum(axis=(0, 1)) - ion["f_k"])))
    fig = plot_orbitals(aa, ion, multiply_fd=MULTIPLY_WAVEFUNCTION_BY_FD)
    if SAVE_FIGURES:
        print(save_figure(fig, FIGURE_STEM))
    plt.show()


if __name__ == "__main__":
    main()
