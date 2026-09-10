"""Standalone Colab uses only installed Otter and its default numerical controls."""
from __future__ import annotations

import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pytest

import otter
from otter import PlasmaWorkflowConfig
from otter.plotting import PALETTES

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/00-otter_intro.ipynb"


def cells():
    return {cell["id"]: cell for cell in json.loads(NOTEBOOK.read_text())["cells"]}


def execute(name, namespace):
    source = "".join(cells()[name]["source"])
    exec(compile(source, f"{NOTEBOOK}:{name}", "exec"), namespace)


def workflow_result():
    """Synthetic workflow-shaped data, not archived or calculated test fixtures."""
    r = np.linspace(0.05, 20.0, 90)
    k = np.linspace(0.05, 20.0, 80)
    electronic = {"r": r, "r_ws": 2.0, "n0": 0.02, "mu": 1.0, "zbar": 3.0}
    for index, key in enumerate(("n_full", "n_ion", "n_ext", "n_pa", "n_scr"), 1):
        electronic[key] = index * np.exp(-r)
    # Ensure plotting n_ion never silently substitutes the bound-state density.
    electronic["n_bound"] = 99.0 * np.exp(-r)
    for index, key in enumerate(("v_full", "v_ext", "v_H", "v_xc", "v_nuc"), 1):
        electronic[key] = -index * np.exp(-r)
    ion = {
        "r": r, "k": k, "zbar": 3.0, "hnc_best_residual": 1e-5,
        "f_k": 10.0 * np.exp(-k), "q_k": 3.0 * np.exp(-k),
        "vii_k": 1.0 / (1.0 + k**2), "vii_r": np.exp(-r) / r,
        "gii_r": 1.0 - np.exp(-r), "sii_k": 1.0 - np.exp(-k),
    }
    return {"electronic": {"result": electronic}, "ion": ion}


@pytest.mark.parametrize("temperatures", [(1.0, 1.0), (5.0, 10.0)])
def test_colab_runs_outside_checkout_with_defaults_and_redraws(
    tmp_path, monkeypatch, temperatures, capsys,
):
    namespace, calls = {}, []
    result = workflow_result()
    monkeypatch.chdir(tmp_path)

    def solve(config):
        calls.append(config)
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError("Notebook must not read archives or recalculate when plotting")

    monkeypatch.setattr(otter, "solve_plasma_workflow", solve)
    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(plt, "show", lambda: None)
    execute("imports", namespace)
    execute("input", namespace)
    assert namespace["TE_EV"] == namespace["TI_EV"] == 1.0
    namespace["TE_EV"], namespace["TI_EV"] = temperatures
    namespace["perf_counter"] = iter((10.0, 22.5)).__next__
    execute("calculate", namespace)
    assert len(calls) == 1
    assert namespace["calculation_elapsed"] == 12.5
    assert "Calculation wall time: 12.50 s (0.21 min)" in capsys.readouterr().out
    expected = PlasmaWorkflowConfig(
        elements=["Al"], rho_g_cc=8.1,
        temperature_ev=temperatures[0], ion_temperature_ev=temperatures[1],
    )
    assert asdict(calls[0]) == asdict(expected)
    assert calls[0].show_progress is True
    assert namespace["electronic"] is result["electronic"]["result"]
    assert namespace["ion"] is result["ion"]
    namespace["solve_plasma_workflow"] = forbidden
    try:
        execute("plot-electronic", namespace)
        execute("plot-ionic", namespace)
        electronic, ionic = namespace["fig_electronic"], namespace["fig_ionic"]
        assert [len(fig.axes) for fig in (electronic, ionic)] == [3, 6]
        assert ionic.axes[1].get_legend() is None
        assert [line.get_color() for line in electronic.axes[0].lines[:6]] == list(
            PALETTES["bing"][:6]
        )
        assert [ax.get_title() for ax in electronic.axes] == [
            "Electronic densities", "Effective potentials", "Full-AA potential components",
        ]
        assert [ax.get_title() for ax in ionic.axes] == [
            r"$f(k)=n_{\rm ion}(k)$", r"$q(k)=n_{\rm scr}(k)$",
            r"$V_{ii}(k)$", r"$V_{ii}(r)$", r"$g_{ii}(r)$", r"$S_{ii}(k)$",
        ]
        r_e = result["electronic"]["result"]["r"]
        mask_e = r_e <= 8.0
        np.testing.assert_array_equal(
            electronic.axes[0].lines[1].get_ydata(),
            (4 * np.pi * r_e**2 * result["electronic"]["result"]["n_ion"])[mask_e],
        )
        np.testing.assert_array_equal(
            electronic.axes[2].lines[3].get_ydata(),
            result["electronic"]["result"]["v_nuc"][mask_e],
        )
        for ax, key in zip(ionic.axes, ("f_k", "q_k", "vii_k", "vii_r", "gii_r", "sii_k")):
            mask = result["ion"]["r"] <= 12.0 if key in ("vii_r", "gii_r") else result["ion"]["k"] <= 8.0
            np.testing.assert_array_equal(ax.lines[0].get_ydata(), result["ion"][key][mask])
            assert ax.lines[0].get_color() == PALETTES["bing"][0]
        if temperatures[0] != temperatures[1]:
            assert "T_e=T_i" not in electronic._suptitle.get_text()
            assert "T_i=10" in ionic._suptitle.get_text()

        def pixels(fig):
            fig.canvas.draw()
            return hashlib.sha256(fig.canvas.buffer_rgba()).hexdigest()

        before = [pixels(fig) for fig in (electronic, ionic)]
        plt.close("all")
        execute("plot-electronic", namespace)
        execute("plot-ionic", namespace)
        assert [pixels(namespace[key]) for key in ("fig_electronic", "fig_ionic")] == before
        assert len(calls) == 1
        assert not list(tmp_path.iterdir())
    finally:
        plt.close("all")


def test_colab_has_no_repository_dependency_or_numerical_overrides():
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    assert "".join(cells()["install"]["source"]) == "%pip install -q otter-hed"
    sources = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None and cell["outputs"] == []
        if cell["id"] == "install":
            continue
        sources.append("".join(cell["source"]))
    combined = "\n".join(sources)
    tree = ast.parse(combined)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                       else [node.module])
            assert all(module.split(".")[0] in {"otter", "numpy", "matplotlib", "time"} for module in modules)
    configs = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Name) and node.func.id == "PlasmaWorkflowConfig"]
    assert len(configs) == 1
    assert {kw.arg for kw in configs[0].keywords} == {
        "elements", "rho_g_cc", "temperature_ev", "ion_temperature_ev",
    }
    for forbidden in ("importlib", "subprocess", "gallery.", "np.load", "runpy", "REPOSITORY",
                      "redirect_stdout", "capture_output", "%%capture"):
        assert forbidden not in combined


def test_html_copyright_includes_chinese_name():
    tree = ast.parse((ROOT / "docs/source/conf.py").read_text())
    values = {node.targets[0].id: ast.literal_eval(node.value)
              for node in tree.body if isinstance(node, ast.Assign)
              and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id in {"author", "copyright"}}
    assert values["copyright"] == "2026, Chongbing Qu (瞿崇兵)"
    assert values["author"] == "Chongbing Qu (瞿崇兵) and Dominik Kraus"
