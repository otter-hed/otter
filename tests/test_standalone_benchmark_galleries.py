"""Contracts for the downloadable, independently executable benchmarks."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GALLERIES = (
    ROOT / "benchmarks" / "examples" / "plot_ion_structure_library.py",
    ROOT
    / "benchmarks"
    / "examples"
    / "plot_johnson_et_al_2025_two_temperature_al.py",
    ROOT
    / "benchmarks"
    / "examples"
    / "plot_argha_roy_carbon_sii.py",
    ROOT
    / "benchmarks"
    / "examples"
    / "plot_schorner_et_al_2022_al_sii.py",
    ROOT
    / "benchmarks"
    / "examples"
    / "plot_starrett_et_al_2014_mixtures_fig3.py",
    ROOT
    / "benchmarks"
    / "examples"
    / "plot_starrett_single_species_2013_2014.py",
    ROOT
    / "benchmarks"
    / "examples"
    / "plot_starrett_saumon_2013_electronic.py",
)


@pytest.mark.parametrize("path", GALLERIES, ids=lambda path: path.stem)
def test_benchmark_gallery_is_one_complete_otter_script(path: Path) -> None:
    """A live gallery must not delegate its scientific work to another file."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    assert "USE_PRECOMPUTED_DATA = True" in source
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module is not None
        and (node.module == "otter" or node.module.startswith("otter."))
        for node in ast.walk(tree)
    )
    plasma_workflow = (
        "PlasmaWorkflowConfig" in source
        and "solve_plasma_workflow(" in source
    )
    full_average_atom = (
        "FullExternalConfig" in source and "solve_full_only(" in source
    )
    assert plasma_workflow or full_average_atom
    if path.name == "plot_starrett_saumon_2013_electronic.py":
        # Sparse level and ionization diagnostics are intentionally presented
        # as direct article-style tables rather than redundant figures.
        assert "<table class=\"docutils align-default\">" in source
        assert "from otter.plotting import" not in source
        assert "save_figure(" not in source
    else:
        assert "from otter.plotting import" in source
        assert "save_figure(" in source

    assert "importlib" not in source
    assert "benchmarks/runners" not in source
    assert "benchmarks\" / \"runners" not in source
    assert "regenerate_" not in source


def test_bethkenhagen_benchmark_reuses_the_carbon_ionization_scan() -> None:
    """The literature overlay must not repeat the 100 eV carbon scan."""
    path = (
        ROOT
        / "benchmarks"
        / "examples"
        / "plot_bethkenhagen_et_al_2020_carbon_ionization.py"
    )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    assert "carbon_ionization_levels" in source
    assert "C_Te100eV_density_scan.npz" in source
    assert "solve_full_only(" not in source
    assert "solve_plasma_workflow(" not in source
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "otter.plotting"
        for node in ast.walk(tree)
    )


def test_ion_structure_library_adds_wunsch_vmhnc_and_reproducible_md() -> None:
    path = ROOT / "benchmarks" / "examples" / "plot_ion_structure_library.py"
    source = path.read_text(encoding="utf-8")
    assert "Otter-HNC" in source
    assert "Otter-VMHNC" in source
    assert "Otter-MD" in source
    assert 'bridge_model="rosenfeld_ashcroft"' in source
    assert "from tools.otter_lammps_md import" in source
    assert "RUN_WUNSCH_SAME_POTENTIAL_MD = True" in source
    assert "MD_MIN_HALF_SPACE_MODES_PER_BIN = 4" in source
    assert 'states[result_id]["md_sii_vectors_per_bin"]' in source


def test_johnson_gallery_compares_hnc_and_vmhnc_on_the_same_is_state() -> None:
    """All four Johnson panels must isolate the ionic closure, not IS/SC."""
    path = (
        ROOT
        / "benchmarks"
        / "examples"
        / "plot_johnson_et_al_2025_two_temperature_al.py"
    )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    otter_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "otter"
        for alias in node.names
    }
    assert {
        "continue_plasma_workflow_from_electronic_result",
        "solve_plasma_workflow",
    }.issubset(otter_imports)
    assert 'bridge_model="none"' in source
    assert 'bridge_model="rosenfeld_ashcroft"' in source
    assert "Otter IS-QOZ/HNC" in source
    assert "Otter IS-QOZ/VMHNC" in source
    assert "Otter IS-potential MD" in source
    assert "RUN_SAME_POTENTIAL_MD = True" in source
    assert '"NVT->NVE"' in source
    assert "solve_sc_feedback_workflow" not in source
    assert "otter.experimental" not in source
    assert "for definition in STATES" in source


def test_schorner_gallery_compares_xc_closures_and_same_potential_md() -> None:
    """Schörner panels must vary XC and closure without changing the IS model."""
    path = (
        ROOT
        / "benchmarks"
        / "examples"
        / "plot_schorner_et_al_2022_al_sii.py"
    )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    otter_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "otter"
        for alias in node.names
    }
    assert {
        "continue_plasma_workflow_from_electronic_result",
        "solve_plasma_workflow",
    } <= otter_imports
    assert 'bridge_model="none"' in source
    assert 'bridge_model="rosenfeld_ashcroft"' in source
    assert '"xc_model": "lda_pw"' in source
    assert '"xc_model": "pbe"' in source
    assert "RUN_SAME_POTENTIAL_MD = True" in source
    assert "complete periodic reciprocal-shell" in source
    assert "solve_sc_feedback_workflow" not in source
