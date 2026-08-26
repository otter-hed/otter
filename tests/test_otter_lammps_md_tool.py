"""Fast, LAMMPS-free checks for the reusable Otter-to-MD driver."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "otter_lammps_md.py"
SPEC = importlib.util.spec_from_file_location("otter_lammps_md", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
md = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = md
SPEC.loader.exec_module(md)


def _potential(left: str, right: str, scale: float = 1.0):
    radius = np.linspace(0.05, 20.0, 256)
    return md.PairPotential(left, right, radius, scale * np.exp(-radius) / radius)


def _config(tmp_path: Path, species: tuple) -> object:
    return md.MDConfig(
        output_dir=tmp_path,
        species=species,
        ion_density_bohr3=0.02,
        ion_temperature_ev=5.0,
        timestep_ps=1.0e-5,
        thermostat_damp_ps=1.0e-3,
        table_points=64,
    )


def test_prepare_single_component_run(tmp_path: Path) -> None:
    config = _config(tmp_path, (md.MDSpecies("Be", 9.0121831, 32),))
    audit = md.prepare_lammps_run(config, [_potential("Be", "Be")])

    assert audit["atom_count"] == 32
    assert audit["type_counts"] == [32]
    assert max(abs(value) for value in audit["table_endpoint"].values()) < 1.0e-14
    input_text = (tmp_path / "in.otter_md").read_text(encoding="utf-8")
    assert "pair_coeff      1 1 pair_potentials.table V_BE_BE" in input_text
    assert "compute         rdf_all all rdf 500 1 1" in input_text
    assert "density time dt" in input_text
    assert "unfix           thermostat" in input_text
    assert "fix             integrator all nve" in input_text


def test_prepare_mixture_has_exact_counts_and_every_pair(tmp_path: Path) -> None:
    species = (
        md.MDSpecies("C", 12.011, 10),
        md.MDSpecies("H", 1.008, 20),
        md.MDSpecies("O", 15.999, 5),
    )
    potentials = [
        _potential("C", "C", 6.0),
        _potential("C", "H", 3.0),
        _potential("C", "O", 4.0),
        _potential("H", "H", 1.0),
        _potential("H", "O", 2.0),
        _potential("O", "O", 5.0),
    ]
    audit = md.prepare_lammps_run(_config(tmp_path, species), potentials)

    assert audit["type_counts"] == [10, 20, 5]
    atom_rows = (
        (tmp_path / "atoms.data")
        .read_text(encoding="utf-8")
        .split("Atoms # atomic\n\n", 1)[1]
        .splitlines()
    )
    assert [
        sum(row.split()[1] == str(kind) for row in atom_rows)
        for kind in (1, 2, 3)
    ] == [10, 20, 5]
    input_text = (tmp_path / "in.otter_md").read_text(encoding="utf-8")
    for pair in ("1 1", "1 2", "1 3", "2 2", "2 3", "3 3"):
        assert f"pair_coeff      {pair}" in input_text
    assert "compute         rdf_all all rdf 500 1 1 1 2 1 3 2 2 2 3 3 3" in input_text


def test_missing_mixture_cross_potential_is_rejected(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        (md.MDSpecies("C", 12.011, 10), md.MDSpecies("H", 1.008, 20)),
    )
    with pytest.raises(ValueError, match="coverage mismatch"):
        md.prepare_lammps_run(
            config,
            [_potential("C", "C"), _potential("H", "H")],
        )


def test_charged_species_use_analytic_coulomb_plus_tabulated_remainder(
    tmp_path: Path,
) -> None:
    species = (
        md.MDSpecies("C", 12.011, 10, 4.0),
        md.MDSpecies("H", 1.008, 20, 1.0),
    )
    config = _config(tmp_path, species)
    radius = np.linspace(0.05, 20.0, 512)
    potentials = [
        md.PairPotential("C", "C", radius, 16.0 / radius - 0.4),
        md.PairPotential("C", "H", radius, 4.0 / radius - 0.2),
        md.PairPotential("H", "H", radius, 1.0 / radius - 0.1),
    ]
    audit = md.prepare_lammps_run(config, potentials)

    data = (tmp_path / "atoms.data").read_text(encoding="utf-8")
    script = (tmp_path / "in.otter_md").read_text(encoding="utf-8")
    assert "Atoms # charge" in data
    assert "atom_style      charge" in script
    assert "pair_style      hybrid/overlay coul/cut" in script
    assert "pair_coeff      * * coul/cut" in script
    assert "pair_coeff      1 2 table pair_potentials.table V_C_H" in script
    assert max(abs(value) for value in audit["table_endpoint"].values()) < 1.0e-12


def test_otter_potential_matrix_conversion() -> None:
    radius = np.linspace(0.1, 2.0, 8)
    matrix = np.empty((2, 2, radius.size))
    matrix[0, 0] = 3.0 / radius
    matrix[0, 1] = matrix[1, 0] = 2.0 / radius
    matrix[1, 1] = 1.0 / radius
    pairs = md.pair_potentials_from_matrix(("C", "H"), radius, matrix)
    assert [(item.left, item.right) for item in pairs] == [
        ("C", "C"),
        ("C", "H"),
        ("H", "H"),
    ]


def test_parallel_structure_factor_matches_serial(tmp_path: Path) -> None:
    species = (md.MDSpecies("C", 12.011, 4), md.MDSpecies("H", 1.008, 4))
    rng = np.random.default_rng(7)
    positions = rng.random((3, 8, 3)) * 20.0
    types = np.asarray([1, 1, 1, 1, 2, 2, 2, 2])
    serial = md.MDConfig(
        **{
            **md.asdict(_config(tmp_path, species)),
            "output_dir": tmp_path,
            "species": species,
            "k_max_angstrom_inv": 2.0,
            "structure_factor_workers": 1,
        }
    )
    parallel = md.MDConfig(
        **{**md.asdict(serial), "output_dir": tmp_path, "species": species,
           "structure_factor_workers": 2}
    )
    serial_result = md._direct_structure_factors(positions, types, 20.0, serial)
    parallel_result = md._direct_structure_factors(positions, types, 20.0, parallel)
    for left, right in zip(serial_result, parallel_result):
        np.testing.assert_allclose(left, right, rtol=0.0, atol=0.0)


def test_completion_requires_the_end_of_the_nve_run(tmp_path: Path) -> None:
    for name in (
        "atoms.data",
        "pair_potentials.table",
        "in.otter_md",
        "rdf_blocks.dat",
        "trajectory.lammpstrj",
    ):
        (tmp_path / name).write_text("partial output\n", encoding="utf-8")
    artifacts = {
        name: md._sha256(tmp_path / name)
        for name in ("atoms.data", "pair_potentials.table", "in.otter_md")
    }
    (tmp_path / "run_metadata.json").write_text(
        md.json.dumps({"artifacts": artifacts}), encoding="utf-8"
    )
    log = tmp_path / "log.lammps"
    log.write_text("Loop time of 1.0 for the NVT segment\n", encoding="utf-8")
    assert not md._completed_lammps_run(tmp_path)
    log.write_text(
        "Loop time of 1.0 for the NVT segment\nTotal wall time: 0:00:02\n",
        encoding="utf-8",
    )
    assert md._completed_lammps_run(tmp_path)
    (tmp_path / "in.otter_md").write_text("changed input\n", encoding="utf-8")
    assert not md._completed_lammps_run(tmp_path)
