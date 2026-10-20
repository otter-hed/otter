"""Check free angular completion against the uniform spectral density.

The reference k/pi^2 follows Starrett and Saumon (2014), Eq. A3, and
NIST DLMF 10.60.12. Fixed small angular cutoffs deliberately expose the
free-density deficit; they are model controls, not average-atom settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from otter.electronic.continuum import scattering as sc
from otter.numerics.grids import create_sqrt_grid
from otter.plotting import grid_figsize, save_figure, style_context


def calculate(output: Path) -> dict:
    """Calculate a zero-potential control on a declared square-root grid."""
    output.mkdir(parents=True, exist_ok=True)
    grid = create_sqrt_grid(30.0, 2400, rmin=1e-5)
    rows, profiles = [], {"r_bohr": grid.r}
    for energy in (0.001, 0.02, 0.2, 2.0):
        for cap in (2, 8, 32):
            params = dict(
                v_eff=np.zeros_like(grid.r), r=grid.r, mu=0.0, temperature=1.0,
                energy=energy, l_max=cap, grid_kind="sqrt", grid_step=grid.dxi,
                l_pad=2, match_fraction=0.15, match_slice=None,
                match_r_cut=25.5, match_fraction_mode="r", match_width=4.5,
                match_kr_min=3.0, match_v_tol=0.1, match_min_points=16,
                match_asymptotic="free", match_coulomb_tol=0.1,
                match_allow_shift=False, match_fallback="free",
                prop_rescale_limit=1e6, apply_occ=False, l_cap_strategy="none",
            )
            raw, phases = sc._scattering_density_and_phase(**params)
            completed, completed_phases = sc._scattering_density_and_phase(
                **params, free_completion=True
            )
            reference = np.sqrt(2 * energy) / np.pi**2
            mask = grid.r >= 1.0
            error = float(np.max(abs(completed[mask] / reference - 1)))
            row = dict(
                energy_ha=energy, angular_cutoff=cap,
                raw_error_over_free=float(np.max(abs(raw[mask] / reference - 1))),
                completed_error_over_free=error,
                phases_unchanged=bool(np.array_equal(phases, completed_phases)),
                passed=error <= 2e-7,
            )
            rows.append(row)
            if energy == 0.2 and cap == 2:
                profiles.update(raw=raw, completed=completed,
                                free_spectrum=np.full_like(grid.r, reference))
    np.savez_compressed(output / "profiles.npz", **profiles)
    result = dict(
        reference="k/pi^2; NIST DLMF 10.60.12; StarrettSaumon2014 Eq. A3",
        grid=dict(points=2400, rmax_bohr=30.0, rmin_bohr=1e-5),
        comparison_min_radius_bohr=1.0, tolerance_over_free=2e-7,
        rows=rows, passed=all(r["passed"] and r["phases_unchanged"] for r in rows),
        solver_sha256=hashlib.sha256(Path(sc.__file__).read_bytes()).hexdigest(),
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (output / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def plot(output: Path) -> None:
    """Export the analytic density and residual control using Otter style."""
    import matplotlib.pyplot as plt

    with np.load(output / "profiles.npz", allow_pickle=False) as data:
        r = data["r_bohr"]
        raw = data["raw"] / data["free_spectrum"]
        completed = data["completed"] / data["free_spectrum"]
    with style_context("thesis", palette="bing"):
        fig, axes = plt.subplots(
            2, 1, figsize=grid_figsize(2, 1), sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.0]},
        )
        axes[0].plot(r, raw, "--", label="Truncated")
        axes[0].plot(r, completed, label="Free-channel completion")
        axes[0].axhline(1, color="0.3", ls=":", label=r"$k/\pi^2$")
        axes[0].set_ylabel(r"$n_E/(k/\pi^2)$")
        axes[0].legend()
        mask = r >= 1.0
        axes[1].semilogy(r[mask], np.maximum(abs(raw[mask] - 1), 1e-16), "--")
        axes[1].semilogy(r[mask], np.maximum(abs(completed[mask] - 1), 1e-16))
        axes[1].set_ylabel("Absolute residual")
        axes[1].set_xlabel(r"$r\ [a_0]$")
        axes[0].set_xlim(0, 30)
        fig.tight_layout()
        save_figure(fig, output / "free_spectrum", close=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/outputs/free_channel_completion"))
    args = parser.parse_args()
    result = calculate(args.output)
    plot(args.output)
    print(json.dumps({"passed": result["passed"], "controls": len(result["rows"])}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
