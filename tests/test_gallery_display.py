"""Gallery display must reach Matplotlib, including interactive Agg backends."""
from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXED_PAGES = (
    "docs/examples/plot_al_full_workflow.py",
    "docs/examples/plot_al_is_sc_comparison.py",
    "docs/examples/plot_al_qm_tf.py",
    "docs/examples/plot_carbon_ionization_levels.py",
    "docs/examples/plot_carbon_lfc_sensitivity.py",
    "docs/examples/plot_ch136_mixture_workflow.py",
    "benchmarks/examples/plot_argha_roy_carbon_sii.py",
    "benchmarks/examples/plot_bethkenhagen_et_al_2020_carbon_ionization.py",
    "benchmarks/examples/plot_ch2_hnc_md.py",
    "benchmarks/examples/plot_ion_structure_library.py",
    "benchmarks/examples/plot_johnson_et_al_2025_two_temperature_al.py",
    "benchmarks/examples/plot_schorner_et_al_2022_al_sii.py",
    "benchmarks/examples/plot_starrett_et_al_2014_mixtures_fig3.py",
)


def is_pyplot_call(node, name):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "plt"
        and node.func.attr == name
    )


@pytest.mark.parametrize("path", FIXED_PAGES, ids=lambda path: Path(path).stem)
@pytest.mark.parametrize("backend", [
    "TkAgg", "QtAgg", "Qt5Agg", "GTK3Agg", "WebAgg", "MacOSX",
    "module://matplotlib_inline.backend_inline", "Agg",
])
def test_gallery_delegates_display_for_each_backend(path, backend):
    # Execute the actual show statement and any enclosing display guard,
    # without launching AA/MD or requiring GUI libraries in the CI runner.
    tree = ast.parse((ROOT / path).read_text())
    parents = {child: parent for parent in ast.walk(tree)
               for child in ast.iter_child_nodes(parent)}
    calls = [node for node in ast.walk(tree) if is_pyplot_call(node, "show")]
    assert len(calls) == 1
    statement = parents[calls[0]]
    assert isinstance(statement, ast.Expr)
    while isinstance(parents.get(statement), ast.If):
        statement = parents[statement]
    display = ast.Module(body=[statement], type_ignores=[])
    pyplot = Mock()
    pyplot.get_backend.return_value = backend
    exec(compile(display, path, "exec"), {"plt": pyplot})
    pyplot.show.assert_called_once_with()


def test_examples_do_not_filter_or_force_matplotlib_backends():
    for directory in ("examples", "docs/examples", "benchmarks/examples"):
        for path in (ROOT / directory).glob("*.py"):
            tree = ast.parse(path.read_text())
            assert not any(is_pyplot_call(node, name)
                           for node in ast.walk(tree)
                           for name in ("get_backend", "switch_backend")), path


@pytest.mark.parametrize("backend", ["TkAgg", "QtAgg", "Agg"])
@pytest.mark.parametrize("show_figures", [False, True])
def test_partition_diagnostic_respects_only_the_display_switch(backend, show_figures):
    path = ROOT / "tools/diagnostics/bound_energy_partition_sensitivity.py"
    tree = ast.parse(path.read_text())
    guards = [node for node in ast.walk(tree) if isinstance(node, ast.If)
              and isinstance(node.test, ast.Name) and node.test.id == "SHOW_FIGURES"]
    assert len(guards) == 1
    pyplot = Mock()
    pyplot.get_backend.return_value = backend
    exec(compile(ast.Module(body=guards, type_ignores=[]), str(path), "exec"),
         {"plt": pyplot, "SHOW_FIGURES": show_figures})
    if show_figures:
        pyplot.show.assert_called_once_with()
        pyplot.close.assert_not_called()
    else:
        pyplot.show.assert_not_called()
        pyplot.close.assert_called_once_with("all")
