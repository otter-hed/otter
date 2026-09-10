"""Contracts for the downloadable, independently executable benchmarks."""

from __future__ import annotations

import ast
from pathlib import Path
import runpy

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


def test_recomputation_queue_does_not_require_private_ch2_caches() -> None:
    tasks = runpy.run_path(str(ROOT / "tools/recompute_all_data.py"))["TASKS"]
    settings = tasks["ch2_hnc_md"].environment
    assert settings["OTTER_CH2_RECOMPUTE_ELECTRONIC"] == "1"
    assert settings["OTTER_CH2_RUN_MD"] == "0"
    assert settings["OTTER_REQUIRE_ALL_CH2_HNC_MD"] == "1"


def test_examples_and_benchmarks_inherit_the_single_aa_worker_default() -> None:
    """Parallelize states, without silently changing per-AA worker counts."""
    from otter.electronic.full_external import FullExternalConfig

    assert FullExternalConfig.cont_n_jobs == 1
    for directory in ("examples", "docs/examples", "benchmarks/examples", "benchmarks/runners"):
        for path in (ROOT / directory).glob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.keyword):
                    assert node.arg not in {"cont_n_jobs", "cont_shards"}, path
                elif isinstance(node, ast.Dict):
                    assert not any(
                        isinstance(key, ast.Constant)
                        and key.value in {"cont_n_jobs", "cont_shards"}
                        for key in node.keys
                    ), path


@pytest.mark.parametrize("path", GALLERIES, ids=lambda path: path.stem)
def test_benchmark_gallery_is_one_complete_otter_script(path: Path) -> None:
    """A live gallery must not delegate its scientific work to another file."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    assert "USE_PRECOMPUTED_DATA = False" in source
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


def test_bethkenhagen_benchmark_reuses_the_carbon_calculation_routine() -> None:
    """Share the implementation, not a required precomputed scan."""
    path = (
        ROOT
        / "benchmarks"
        / "examples"
        / "plot_bethkenhagen_et_al_2020_carbon_ionization.py"
    )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    assert "carbon_ionization_levels" in source
    assert "state = module._compute_and_stage()" in source
    assert "RECOMPUTE_WITH_OTTER = True" in source
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


def test_argha_roy_archive_retains_only_the_displayed_structure_factor() -> None:
    path = ROOT / "benchmarks" / "examples" / "plot_argha_roy_carbon_sii.py"
    source = path.read_text(encoding="utf-8")
    assert '"sii_k"' in source
    assert '"n_scr_k_electrons"' not in source
    assert '"vii_k_ha_bohr3"' not in source
    assert '"f_k"' not in source
    assert '"q_k"' not in source
    assert "QOZ_N_POINTS = 4096" not in source
    assert "HNC_TOL = 1.0e-4" not in source


def test_rayleigh_gallery_reconstructs_derived_weight() -> None:
    path = ROOT / "docs" / "examples" / "plot_al_rayleigh_weight.py"
    source = path.read_text(encoding="utf-8")
    assert "def rayleigh_weight(" in source
    assert '"q_k": q' in source
    assert '"f_k": f' in source
    assert '"sii_k": sii' in source
    assert '"rayleigh_weight": weight' not in source


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
