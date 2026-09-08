"""Run reproducible classical MD from Otter ion--ion pair potentials.

Use :func:`pair_potentials_from_otter` to extract every unique ``V_ab(r)``
from a prepared QOZ object, a workflow result, or an exported Otter state.
Then :func:`run_otter_lammps_md` writes every LAMMPS input artifact, performs
NVT equilibration followed by NVE sampling, and saves the sampled ``g_ab(r)``
and directly estimated ``S_ab(k)`` to ``md_results.npz``.  Both functions
support one or many ion species.

A complete call is intentionally only two lines after constructing
``MDConfig``::

    potentials = pair_potentials_from_otter(prepared_qoz)
    result = run_otter_lammps_md(config, potentials)

The pair columns in ``result["md_gij_r"]`` and ``result["md_sij_k"]`` are
identified by ``result["md_pair_labels"]``.  Pair potentials are shifted in
both energy and force at the finite MD cutoff.

When species carry nonzero ``charge_e``, LAMMPS evaluates the exact
``q_i q_j/r`` core and the table contains only the finite screened remainder.
This avoids interpolating the Coulomb singularity while preserving the full
Otter potential and its shifted-force cutoff.

This module deliberately has no command-line parser: benchmark and user
scripts keep their physical settings in readable Python code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Iterable

import numpy as np
from scipy.interpolate import PchipInterpolator

from otter.numerics.constants import (
    BOHR_TO_ANGSTROM,
    EV_TO_KELVIN,
    HA_TO_EV,
)


@dataclass(frozen=True)
class MDSpecies:
    """One LAMMPS atom type with an exact particle count."""

    symbol: str
    mass_u: float
    count: int
    charge_e: float = 0.0


@dataclass(frozen=True)
class PairPotential:
    """One unordered Otter pair potential in atomic units."""

    left: str
    right: str
    r_bohr: np.ndarray
    potential_ha: np.ndarray


@dataclass(frozen=True)
class MDConfig:
    """LAMMPS and statistical controls shared by pure and mixed systems."""

    output_dir: Path
    species: tuple[MDSpecies, ...]
    ion_density_bohr3: float
    ion_temperature_ev: float
    timestep_ps: float
    thermostat_damp_ps: float
    equilibration_steps: int = 10_000
    production_steps: int = 100_000
    rdf_bins: int = 500
    rdf_every: int = 100
    rdf_repeat: int = 50
    trajectory_every: int = 5_000
    table_points: int = 8_192
    r_min_bohr: float = 0.1
    cutoff_box_fraction: float = 0.48
    k_bin_width_angstrom_inv: float = 0.1
    k_max_angstrom_inv: float = 8.3
    reciprocal_batch_size: int = 512
    structure_factor_workers: int = 1
    random_seed: int = 20_260_825
    lammps_executable: str = "lmp"
    mpi_launcher: str = "mpirun"
    mpi_processes: int = 1
    reuse_completed: bool = True


def pair_potentials_from_matrix(
    symbols: tuple[str, ...],
    r_bohr: np.ndarray,
    potential_matrix_ha: np.ndarray,
) -> list[PairPotential]:
    """Convert an Otter ``(N, N, Nr)`` potential matrix to pair records."""
    matrix = np.asarray(potential_matrix_ha, dtype=float)
    radius = np.asarray(r_bohr, dtype=float)
    expected = (len(symbols), len(symbols), radius.size)
    if matrix.shape != expected:
        raise ValueError(
            f"Expected potential matrix shape {expected}, got {matrix.shape}."
        )
    if not np.allclose(matrix, np.swapaxes(matrix, 0, 1), rtol=0.0, atol=1.0e-12):
        raise ValueError("The Otter pair-potential matrix is not symmetric.")
    return [
        PairPotential(symbols[left], symbols[right], radius, matrix[left, right])
        for left in range(len(symbols))
        for right in range(left, len(symbols))
    ]


def pair_potentials_from_otter(
    source: object,
    *,
    potential_matrix_ha: np.ndarray | None = None,
) -> list[PairPotential]:
    """Extract ``V_ab(r)`` from a native or saved Otter ion-structure state.

    ``source`` may be a ``PreparedMulticomponentIonStructure``, a complete
    workflow result, its ``result["ion"]`` mapping, or an ``NpzFile`` produced
    by Otter's state exporter.  ``potential_matrix_ha`` is an explicit
    override for studies that intentionally modify the native QOZ potential.
    """
    qoz = getattr(source, "qoz", None)
    if qoz is not None:
        symbols = tuple(str(value) for value in getattr(source, "species"))
        radius = np.asarray(getattr(source, "r"), dtype=float)
        native_matrix = np.asarray(getattr(qoz, "vij_r"), dtype=float)
    else:
        record: Any = source
        if hasattr(record, "files"):
            keys = set(record.files)  # NpzFile
        elif hasattr(record, "keys"):
            keys = set(record.keys())
        else:
            raise TypeError(
                "Expected a prepared QOZ object, workflow/ion result, or "
                "exported Otter state containing species, r, and vij_r."
            )
        if "ion" in keys:
            record = record["ion"]
            keys = set(record.keys())
        if {"species_symbols", "r_bohr", "vij_r"} <= keys:
            symbols = tuple(str(value) for value in record["species_symbols"])
            radius = np.asarray(record["r_bohr"], dtype=float)
        elif {"species", "r", "vij_r"} <= keys:
            symbols = tuple(str(value) for value in record["species"])
            radius = np.asarray(record["r"], dtype=float)
        else:
            raise TypeError(
                "Expected a prepared QOZ object, workflow/ion result, or "
                "exported Otter state containing species, r, and vij_r."
            )
        native_matrix = np.asarray(record["vij_r"], dtype=float)

    matrix = (
        native_matrix
        if potential_matrix_ha is None
        else np.asarray(potential_matrix_ha, dtype=float)
    )
    if len(symbols) == 1 and matrix.ndim == 1:
        matrix = matrix[None, None, :]
    return pair_potentials_from_matrix(symbols, radius, matrix)


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((str(left), str(right))))  # type: ignore[return-value]


def _section(left: str, right: str) -> str:
    return "V_" + "_".join(value.upper() for value in _pair_key(left, right))


def _validate(
    config: MDConfig,
    potentials: Iterable[PairPotential],
) -> dict[tuple[str, str], PairPotential]:
    if not config.species or any(item.count <= 0 for item in config.species):
        raise ValueError("Every MD species must have a positive particle count.")
    if any(not np.isfinite(item.charge_e) for item in config.species):
        raise ValueError("Every MD species charge must be finite.")
    symbols = [item.symbol for item in config.species]
    if len(set(symbols)) != len(symbols):
        raise ValueError("MD species symbols must be unique.")
    if config.ion_density_bohr3 <= 0.0 or config.ion_temperature_ev <= 0.0:
        raise ValueError("Ion density and temperature must be positive.")
    if config.table_points < 4 or not 0.0 < config.cutoff_box_fraction < 0.5:
        raise ValueError("Invalid table size or cutoff/box fraction.")
    if not np.isfinite(config.r_min_bohr) or config.r_min_bohr <= 0.0:
        raise ValueError("The table inner radius must be finite and positive.")
    if config.structure_factor_workers < 1:
        raise ValueError("structure_factor_workers must be positive.")

    by_pair: dict[tuple[str, str], PairPotential] = {}
    for potential in potentials:
        key = _pair_key(potential.left, potential.right)
        r = np.asarray(potential.r_bohr, dtype=float)
        v = np.asarray(potential.potential_ha, dtype=float)
        if key in by_pair:
            raise ValueError(f"Duplicate pair potential {key}.")
        if r.ndim != 1 or v.shape != r.shape or r.size < 4:
            raise ValueError(f"Pair potential {key} must contain matching 1D arrays.")
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(v)):
            raise ValueError(f"Pair potential {key} contains non-finite values.")
        if not np.all(np.diff(r) > 0.0):
            raise ValueError(f"Pair-potential radius for {key} is not increasing.")
        by_pair[key] = potential

    expected = {
        _pair_key(left, right)
        for index, left in enumerate(symbols)
        for right in symbols[index:]
    }
    if set(by_pair) != expected:
        raise ValueError(
            "Pair-potential coverage mismatch: "
            f"missing={sorted(expected - set(by_pair))}, "
            f"extra={sorted(set(by_pair) - expected)}"
        )
    return by_pair


def _fcc_positions(config: MDConfig) -> tuple[np.ndarray, np.ndarray, float]:
    """Return reproducible positions, one-based types, and cubic box length."""
    count = sum(item.count for item in config.species)
    density_angstrom3 = config.ion_density_bohr3 / BOHR_TO_ANGSTROM**3
    box_length = (count / density_angstrom3) ** (1.0 / 3.0)
    cells = int(np.ceil((count / 4.0) ** (1.0 / 3.0)))
    basis = np.asarray(
        ((0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)),
        dtype=float,
    )
    lattice = np.asarray(
        [
            (np.asarray((i, j, k), dtype=float) + offset) / cells
            for i in range(cells)
            for j in range(cells)
            for k in range(cells)
            for offset in basis
        ]
    )
    rng = np.random.default_rng(config.random_seed)
    selected = np.sort(rng.choice(lattice.shape[0], count, replace=False))
    positions = lattice[selected] * box_length
    types = np.concatenate(
        [np.full(item.count, index + 1, dtype=int)
         for index, item in enumerate(config.species)]
    )
    rng.shuffle(types)
    return positions, types, float(box_length)


def _write_data(
    path: Path,
    config: MDConfig,
    positions: np.ndarray,
    types: np.ndarray,
    box_length: float,
) -> None:
    charged = any(item.charge_e != 0.0 for item in config.species)
    lines = [
        "LAMMPS data generated from Otter pair potentials",
        "",
        f"{positions.shape[0]} atoms",
        f"{len(config.species)} atom types",
        "",
        f"0.0 {box_length:.12e} xlo xhi",
        f"0.0 {box_length:.12e} ylo yhi",
        f"0.0 {box_length:.12e} zlo zhi",
        "",
        "Atoms # charge" if charged else "Atoms # atomic",
        "",
    ]
    if charged:
        lines.extend(
            f"{index} {atom_type} "
            f"{config.species[atom_type - 1].charge_e:.12e} "
            f"{x:.12e} {y:.12e} {z:.12e}"
            for index, (atom_type, (x, y, z)) in enumerate(
                zip(types, positions), start=1
            )
        )
    else:
        lines.extend(
            f"{index} {atom_type} {x:.12e} {y:.12e} {z:.12e}"
            for index, (atom_type, (x, y, z)) in enumerate(
                zip(types, positions), start=1
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_table(
    path: Path,
    config: MDConfig,
    potentials: dict[tuple[str, str], PairPotential],
    box_length: float,
) -> tuple[float, dict[str, float]]:
    charge_by_symbol = {item.symbol: item.charge_e for item in config.species}
    charged = any(value != 0.0 for value in charge_by_symbol.values())
    r_min = (
        float(config.r_min_bohr)
        if charged
        else max(
            config.r_min_bohr,
            max(float(np.asarray(item.r_bohr)[0]) for item in potentials.values()),
        )
    )
    r_cut = min(
        config.cutoff_box_fraction * box_length / BOHR_TO_ANGSTROM,
        min(float(np.asarray(item.r_bohr)[-1]) for item in potentials.values()),
    )
    if r_cut <= r_min:
        raise ValueError(
            "The common pair-potential range does not reach the MD cutoff."
        )
    radius_bohr = np.linspace(r_min, r_cut, config.table_points)
    radius_angstrom = radius_bohr * BOHR_TO_ANGSTROM
    lines = ["# Shifted-force Otter ion-ion pair potentials", ""]
    endpoint: dict[str, float] = {}
    for key in sorted(potentials):
        item = potentials[key]
        source_r = np.asarray(item.r_bohr, dtype=float)
        source_v = np.asarray(item.potential_ha, dtype=float)
        coefficient = charge_by_symbol[key[0]] * charge_by_symbol[key[1]]
        if charged:
            source_v = source_v - coefficient / source_r
        interpolator = PchipInterpolator(source_r, source_v, extrapolate=False)
        interpolation_r = np.maximum(radius_bohr, source_r[0])
        energy = (
            np.asarray(interpolator(interpolation_r), dtype=float)
            * HA_TO_EV
        )
        force = (
            -np.asarray(interpolator.derivative()(interpolation_r), dtype=float)
            * HA_TO_EV
            / BOHR_TO_ANGSTROM
        )
        force[radius_bohr < source_r[0]] = 0.0
        if charged:
            energy = energy + coefficient / radius_bohr * HA_TO_EV
            force = force + (
                coefficient
                / radius_bohr**2
                * HA_TO_EV
                / BOHR_TO_ANGSTROM
            )
        cutoff_force = float(force[-1])
        energy = (
            energy
            - energy[-1]
            + (radius_angstrom - radius_angstrom[-1]) * cutoff_force
        )
        force = force - cutoff_force
        coulomb_energy = np.zeros_like(energy)
        coulomb_force = np.zeros_like(force)
        if charged:
            coulomb_energy = coefficient / radius_bohr * HA_TO_EV
            coulomb_force = (
                coefficient
                / radius_bohr**2
                * HA_TO_EV
                / BOHR_TO_ANGSTROM
            )
            energy = energy - coulomb_energy
            force = force - coulomb_force
        name = _section(*key)
        endpoint[f"{name}_energy_ev"] = float(
            energy[-1] + coulomb_energy[-1]
        )
        endpoint[f"{name}_force_ev_per_angstrom"] = float(
            force[-1] + coulomb_force[-1]
        )
        lines.extend((name, f"N {config.table_points}", ""))
        lines.extend(
            f"{index} {r:.12e} {v:.12e} {f:.12e}"
            for index, (r, v, f) in enumerate(
                zip(radius_angstrom, energy, force), start=1
            )
        )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return float(radius_angstrom[-1] - 1.0e-9), endpoint


def _write_input(path: Path, config: MDConfig, cutoff_angstrom: float) -> None:
    pairs = [
        (left + 1, right + 1)
        for left in range(len(config.species))
        for right in range(left, len(config.species))
    ]
    charged = any(item.charge_e != 0.0 for item in config.species)
    pair_coeff_lines: list[str] = []
    if charged:
        pair_coeff_lines.append("pair_coeff      * * coul/cut")
    for left, right in pairs:
        section = _section(
            config.species[left - 1].symbol,
            config.species[right - 1].symbol,
        )
        table_style = "table " if charged else ""
        pair_coeff_lines.append(
            f"pair_coeff      {left} {right} {table_style}pair_potentials.table "
            f"{section} {cutoff_angstrom:.12e}"
        )
    pair_coeff = "\n".join(pair_coeff_lines)
    masses = "\n".join(
        f"mass            {index} {item.mass_u:.10f}"
        for index, item in enumerate(config.species, start=1)
    )
    rdf_pairs = " ".join(f"{left} {right}" for left, right in pairs)
    temperature_k = config.ion_temperature_ev * EV_TO_KELVIN
    rdf_block_steps = config.rdf_every * config.rdf_repeat
    atom_style = "charge" if charged else "atomic"
    pair_style = (
        f"hybrid/overlay coul/cut {cutoff_angstrom:.12e} "
        f"table linear {config.table_points}"
        if charged
        else f"table linear {config.table_points}"
    )
    text = f"""# NVT -> NVE same-potential MD generated by tools/otter_lammps_md.py
units           metal
dimension       3
boundary        p p p
atom_style      {atom_style}
read_data       atoms.data
{masses}
pair_style      {pair_style}
{pair_coeff}
neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes
velocity        all create {temperature_k:.8f} {config.random_seed} mom yes rot no dist gaussian
timestep        {config.timestep_ps:.12e}
thermo          1000
thermo_style    custom step temp pe ke etotal press density time dt
thermo_modify   flush yes
fix             thermostat all nvt temp {temperature_k:.8f} {temperature_k:.8f} {config.thermostat_damp_ps:.12e}
run             {config.equilibration_steps}
unfix           thermostat
reset_timestep  0
fix             integrator all nve
compute         rdf_all all rdf {config.rdf_bins} {rdf_pairs}
fix             rdf_blocks all ave/time {config.rdf_every} {config.rdf_repeat} {rdf_block_steps} c_rdf_all[*] mode vector ave one file rdf_blocks.dat
dump            snapshots all custom {config.trajectory_every} trajectory.lammpstrj id type x y z
dump_modify     snapshots sort id
run             {config.production_steps}
"""
    path.write_text(text, encoding="utf-8")


def prepare_lammps_run(
    config: MDConfig,
    potentials: Iterable[PairPotential],
) -> dict[str, object]:
    """Validate inputs and write a complete, runnable LAMMPS directory."""
    by_pair = _validate(config, potentials)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    positions, types, box_length = _fcc_positions(config)
    _write_data(output / "atoms.data", config, positions, types, box_length)
    cutoff, endpoint = _write_table(
        output / "pair_potentials.table", config, by_pair, box_length
    )
    _write_input(output / "in.otter_md", config, cutoff)
    return {
        "atom_count": int(types.size),
        "type_counts": [int(np.count_nonzero(types == index + 1))
                        for index in range(len(config.species))],
        "box_length_angstrom": box_length,
        "cutoff_angstrom": cutoff,
        "table_endpoint": endpoint,
    }


def _read_rdf(path: Path, pair_count: int) -> tuple[np.ndarray, np.ndarray]:
    rows = [
        line.split() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    blocks: list[np.ndarray] = []
    cursor = 0
    while cursor < len(rows):
        size = int(rows[cursor][1])
        blocks.append(np.asarray(rows[cursor + 1:cursor + 1 + size], dtype=float))
        cursor += size + 1
    if len(blocks) < 2:
        raise RuntimeError("LAMMPS produced fewer than two RDF blocks.")
    values = np.stack(blocks)
    g_columns = 2 + 2 * np.arange(pair_count)
    return values[0, :, 1] / BOHR_TO_ANGSTROM, values[:, :, g_columns]


def _read_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    lines = path.read_text(encoding="utf-8").splitlines()
    frames: list[np.ndarray] = []
    frame_types: np.ndarray | None = None
    box_length: float | None = None
    cursor = 0
    while cursor < len(lines):
        if lines[cursor] != "ITEM: TIMESTEP":
            raise RuntimeError("Unexpected LAMMPS trajectory block.")
        atoms = int(lines[cursor + 3])
        bounds = np.asarray(
            [[float(x) for x in lines[cursor + offset].split()[:2]]
             for offset in (5, 6, 7)]
        )
        lengths = bounds[:, 1] - bounds[:, 0]
        if not np.allclose(lengths, lengths[0], rtol=0.0, atol=1.0e-10):
            raise RuntimeError("Direct S(k) requires a cubic MD cell.")
        header = lines[cursor + 8]
        if header != "ITEM: ATOMS id type x y z":
            raise RuntimeError("Unexpected trajectory atom columns.")
        values = np.asarray(
            [[float(x) for x in line.split()]
             for line in lines[cursor + 9:cursor + 9 + atoms]], dtype=float
        )
        values = values[np.argsort(values[:, 0])]
        current_types = values[:, 1].astype(int)
        if frame_types is None:
            frame_types = current_types
        elif not np.array_equal(frame_types, current_types):
            raise RuntimeError("Atom types changed between trajectory frames.")
        frames.append(values[:, 2:5] / BOHR_TO_ANGSTROM)
        box_length = float(lengths[0] / BOHR_TO_ANGSTROM)
        cursor += 9 + atoms
    if len(frames) < 2 or frame_types is None or box_length is None:
        raise RuntimeError("LAMMPS produced fewer than two trajectory frames.")
    return np.stack(frames), frame_types, box_length


def _direct_structure_factors(
    positions: np.ndarray,
    types: np.ndarray,
    box_length_bohr: float,
    config: MDConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average all periodic half-space density modes by shell and frame."""
    fundamental = 2.0 * np.pi / box_length_bohr
    k_max = config.k_max_angstrom_inv * BOHR_TO_ANGSTROM
    n_max = int(np.floor(k_max / fundamental))
    integers = np.arange(-n_max, n_max + 1, dtype=int)
    nx, ny, nz = np.meshgrid(integers, integers, integers, indexing="ij")
    triplets = np.column_stack((nx.ravel(), ny.ravel(), nz.ravel()))
    half = ((triplets[:, 0] > 0)
            | ((triplets[:, 0] == 0) & (triplets[:, 1] > 0))
            | ((triplets[:, 0] == 0) & (triplets[:, 1] == 0)
               & (triplets[:, 2] > 0)))
    triplets = triplets[half]
    magnitudes = fundamental * np.sqrt(np.sum(triplets**2, axis=1))
    keep = magnitudes <= k_max
    triplets, magnitudes = triplets[keep], magnitudes[keep]
    radial = np.floor(
        magnitudes / (config.k_bin_width_angstrom_inv * BOHR_TO_ANGSTROM)
    ).astype(int)
    bins = np.unique(radial)
    vector_bin = np.searchsorted(bins, radial)
    vectors_per_bin = np.bincount(vector_bin, minlength=bins.size)
    k_bin = np.bincount(vector_bin, weights=magnitudes) / vectors_per_bin
    vectors = fundamental * triplets.astype(float)
    pairs = [(a, b) for a in range(1, len(config.species) + 1)
             for b in range(a, len(config.species) + 1)]
    counts = {kind: int(np.count_nonzero(types == kind))
              for kind in range(1, len(config.species) + 1)}
    def analyse_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        total_modes = np.empty(vectors.shape[0])
        partial_modes = np.empty((vectors.shape[0], len(pairs)))
        for start in range(0, vectors.shape[0], config.reciprocal_batch_size):
            stop = min(start + config.reciprocal_batch_size, vectors.shape[0])
            amplitudes = {
                kind: np.exp(
                    1j * (frame[types == kind] @ vectors[start:stop].T)
                ).sum(axis=0)
                for kind in counts
            }
            total = sum(amplitudes.values())
            total_modes[start:stop] = np.abs(total) ** 2 / frame.shape[0]
            for pair_index, (left, right) in enumerate(pairs):
                product = amplitudes[left] * np.conjugate(amplitudes[right])
                partial_modes[start:stop, pair_index] = (
                    np.real(product) / np.sqrt(counts[left] * counts[right])
                )
        total_by_bin = np.bincount(
            vector_bin, weights=total_modes
        ) / vectors_per_bin
        partial_by_bin = np.empty((bins.size, len(pairs)))
        for pair_index in range(len(pairs)):
            partial_by_bin[:, pair_index] = np.bincount(
                vector_bin, weights=partial_modes[:, pair_index]
            ) / vectors_per_bin
        return total_by_bin, partial_by_bin

    workers = min(config.structure_factor_workers, positions.shape[0])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        analysed = list(pool.map(analyse_frame, positions))
    frame_total = np.stack([item[0] for item in analysed])
    frame_partial = np.stack([item[1] for item in analysed])
    sem_scale = np.sqrt(positions.shape[0])
    return (
        k_bin,
        np.mean(frame_total, axis=0),
        np.std(frame_total, axis=0, ddof=1) / sem_scale,
        np.mean(frame_partial, axis=0),
        np.std(frame_partial, axis=0, ddof=1) / sem_scale,
        vectors_per_bin,
    )


def _nve_audit(path: Path) -> tuple[float, float, str, float, float, float]:
    lines = path.read_text(encoding="utf-8").splitlines()
    headers = [index for index, line in enumerate(lines)
               if line.split()[:5] == ["Step", "Temp", "PotEng", "KinEng", "TotEng"]]
    if len(headers) < 2:
        raise RuntimeError("LAMMPS log does not contain the NVE thermo block.")
    header = lines[headers[-1]].split()
    rows: list[list[float]] = []
    for line in lines[headers[-1] + 1:]:
        if line.startswith("Loop time"):
            break
        try:
            values = [float(value) for value in line.split()[: len(header)]]
        except ValueError:
            continue
        if len(values) == len(header):
            rows.append(values)
    if len(rows) < 2:
        raise RuntimeError("LAMMPS NVE thermo block is incomplete.")
    drift = (rows[-1][4] - rows[0][4]) / abs(rows[0][4])
    version = next((line for line in lines if line.startswith("LAMMPS ")), "unknown")
    index = {name: position for position, name in enumerate(header)}
    elapsed_ps = (
        rows[-1][index["Time"]] - rows[0][index["Time"]]
        if "Time" in index
        else np.nan
    )
    sampled_dt = np.asarray(
        [row[index["Dt"]] for row in rows], dtype=float
    ) if "Dt" in index else None
    return (
        float(drift),
        float(np.mean([row[1] for row in rows])),
        version,
        float(elapsed_ps),
        float(np.min(sampled_dt)) if sampled_dt is not None else np.nan,
        float(np.max(sampled_dt)) if sampled_dt is not None else np.nan,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _completed_lammps_run(output: Path) -> bool:
    """Accept only a normal exit produced by the current input artifacts."""
    required = (
        output / "rdf_blocks.dat",
        output / "trajectory.lammpstrj",
        output / "log.lammps",
        output / "run_metadata.json",
    )
    if not all(path.is_file() for path in required):
        return False
    log_text = (output / "log.lammps").read_text(encoding="utf-8")
    if "Total wall time:" not in log_text:
        return False
    try:
        metadata = json.loads(
            (output / "run_metadata.json").read_text(encoding="utf-8")
        )
        recorded = metadata["artifacts"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return False
    return all(
        recorded.get(name) == _sha256(output / name)
        for name in ("atoms.data", "pair_potentials.table", "in.otter_md")
    )


def run_otter_lammps_md(
    config: MDConfig,
    potentials: Iterable[PairPotential],
) -> dict[str, np.ndarray]:
    """Prepare, run, audit, and analyse one single- or multi-species MD case."""
    prepared = prepare_lammps_run(config, potentials)
    output = Path(config.output_dir)
    rdf_path = output / "rdf_blocks.dat"
    trajectory_path = output / "trajectory.lammpstrj"
    log_path = output / "log.lammps"
    complete = _completed_lammps_run(output)
    lammps_elapsed = 0.0
    if not (config.reuse_completed and complete):
        lammps = shutil.which(config.lammps_executable)
        launcher = (
            shutil.which(config.mpi_launcher)
            if config.mpi_processes > 1
            else None
        )
        if lammps is None or (config.mpi_processes > 1 and launcher is None):
            raise FileNotFoundError(
                "LAMMPS or the requested MPI launcher was not found."
            )
        command = ([launcher, "-np", str(config.mpi_processes), lammps]
                   if launcher is not None else [lammps])
        command += ["-in", "in.otter_md", "-log", "log.lammps"]
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = "1"
        started = time.perf_counter()
        with (output / "screen.log").open("w", encoding="utf-8") as screen:
            subprocess.run(
                command,
                cwd=output,
                env=environment,
                stdout=screen,
                stderr=subprocess.STDOUT,
                check=True,
            )
        lammps_elapsed = time.perf_counter() - started

    analysis_started = time.perf_counter()
    pairs = len(config.species) * (len(config.species) + 1) // 2
    r, rdf_blocks = _read_rdf(rdf_path, pairs)
    positions, types, box_length = _read_trajectory(trajectory_path)
    k, sk, sk_sem, partial_sk, partial_sk_sem, vectors_per_bin = (
        _direct_structure_factors(positions, types, box_length, config)
    )
    (
        drift,
        mean_temperature,
        version,
        nve_elapsed_ps,
        sampled_min_dt_ps,
        sampled_max_dt_ps,
    ) = _nve_audit(log_path)
    analysis_elapsed = time.perf_counter() - analysis_started
    result = {
        "md_r_bohr": r,
        "md_gij_r": np.mean(rdf_blocks, axis=0),
        "md_gij_block_sem": np.std(rdf_blocks, axis=0, ddof=1)
        / np.sqrt(rdf_blocks.shape[0]),
        "md_k_bohr_inv": k,
        "md_snn_k": sk,
        "md_snn_frame_sem": sk_sem,
        "md_sij_k": partial_sk,
        "md_sij_frame_sem": partial_sk_sem,
        "md_vectors_per_k_bin": vectors_per_bin,
        "md_type_pairs": np.asarray(
            [(left, right) for left in range(1, len(config.species) + 1)
             for right in range(left, len(config.species) + 1)], dtype=int
        ),
        "md_pair_labels": np.asarray(
            [
                f"{config.species[left].symbol}-{config.species[right].symbol}"
                for left in range(len(config.species))
                for right in range(left, len(config.species))
            ]
        ),
        "md_particle_counts": np.asarray(
            [item.count for item in config.species], dtype=int
        ),
        "md_box_length_bohr": np.asarray(box_length),
        "md_nve_relative_energy_drift": np.asarray(drift),
        "md_nve_elapsed_ps": np.asarray(nve_elapsed_ps),
        "md_thermo_sampled_min_timestep_ps": np.asarray(sampled_min_dt_ps),
        "md_thermo_sampled_max_timestep_ps": np.asarray(sampled_max_dt_ps),
        "md_nve_mean_temperature_k": np.asarray(mean_temperature),
        "md_atoms": np.asarray(types.size),
        "md_rdf_blocks": np.asarray(rdf_blocks.shape[0]),
        "md_trajectory_frames": np.asarray(positions.shape[0]),
        "md_ensemble_sequence": np.asarray("NVT->NVE"),
        "md_structure_factor_estimator": np.asarray(
            "complete periodic reciprocal-shell and trajectory-frame average"
        ),
        "md_gij_uncertainty_definition": np.asarray(
            "SEM across independent RDF sampling blocks"
        ),
        "md_sij_uncertainty_definition": np.asarray(
            "SEM across shell-averaged saved production frames"
        ),
        "md_lammps_version": np.asarray(version),
        "md_reused_existing_run": np.asarray(config.reuse_completed and complete),
        "md_lammps_elapsed_s": np.asarray(lammps_elapsed),
        "md_analysis_elapsed_s": np.asarray(analysis_elapsed),
        "md_elapsed_s": np.asarray(lammps_elapsed + analysis_elapsed),
    }
    np.savez_compressed(output / "md_results.npz", **result)
    metadata = {
        "schema_version": "otter_lammps_md_run_v1",
        "driver": {
            "relative_path": "tools/otter_lammps_md.py",
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "config": {
            **asdict(config),
            "output_dir": ".",
            "species": [asdict(item) for item in config.species],
        },
        "prepared": prepared,
        "lammps_version": version,
        "lammps_command": (
            [
                config.mpi_launcher,
                "-np",
                str(config.mpi_processes),
                config.lammps_executable,
                "-in",
                "in.otter_md",
                "-log",
                "log.lammps",
            ]
            if config.mpi_processes > 1
            else [
                config.lammps_executable,
                "-in",
                "in.otter_md",
                "-log",
                "log.lammps",
            ]
        ),
        "artifacts": {
            name: _sha256(output / name)
            for name in (
                "atoms.data",
                "pair_potentials.table",
                "in.otter_md",
                "log.lammps",
                "rdf_blocks.dat",
                "trajectory.lammpstrj",
                "md_results.npz",
            )
        },
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
