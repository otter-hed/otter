"""The single-species example plots the workflow's unscaled pair potentials."""
from dataclasses import asdict
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from otter import PlasmaWorkflowConfig


def test_single_species_potential_panels_keep_coordinates_and_defaults(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples/single_species_workflow.py"
    spec = importlib.util.spec_from_file_location("single_species_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    r, k = np.linspace(.01, 25., 80), np.linspace(.03, 30., 90)
    electronic = {"r": r, "r_ws": 2., "n_full": np.exp(-r), "v_full": -np.exp(-r)}
    ion = {"r": r, "k": k, "gij_r": np.ones((1, 1, r.size)),
           "sij_k": np.ones((1, 1, k.size)),
           "vij_r": np.exp(-r)[None, None, :], "vij_k": np.exp(-k)[None, None, :],
           "v_ei_r": -2 * np.exp(-r)[None, :], "v_ei_k": -3 * np.exp(-k)[None, :]}
    calls = []

    def solve(config):
        calls.append(config)
        return {"electronic": {"species": [{"result": electronic}]}, "ion": ion}

    monkeypatch.setattr(example, "solve_plasma_workflow", solve)
    monkeypatch.setattr(example, "SAVE_NPZ", False)
    monkeypatch.setattr(example, "SAVE_FIGURES", False)
    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        example.main()
        assert len(calls) == 1
        assert asdict(calls[0]) == asdict(PlasmaWorkflowConfig(
            elements=["Al"], rho_g_cc=8.1, temperature_ev=15., ion_temperature_ev=15.,
            save_state_path=example.OUTPUT_PATH))
        fig = plt.gcf()
        assert len(fig.axes) == 6
        for ax, space, x, limit in ((fig.axes[2], "r", r, 12.), (fig.axes[5], "k", k, 8.)):
            mask = x <= limit
            for line, values in zip(ax.lines, (ion[f"vij_{space}"][0, 0], ion[f"v_ei_{space}"][0])):
                np.testing.assert_array_equal(line.get_xdata(), x[mask])
                np.testing.assert_array_equal(line.get_ydata(), values[mask])
            assert ax.get_yscale() == "symlog"
            assert [text.get_text() for text in ax.get_legend().get_texts()] == [r"$V_{ii}$", r"$V_{ei}$"]
        fig.canvas.draw()
    finally:
        plt.close("all")
