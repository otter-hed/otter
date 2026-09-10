"""Check the Be reference mapping, full-only diagnostics and plotting source."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _gallery():
    path = ROOT / "benchmarks/examples/plot_doppner_2023_be_ionization.py"
    spec = importlib.util.spec_from_file_location("be_gallery", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # These tests exercise recorded private data/plotting, never a new AA scan.
    module.RECOMPUTE_WITH_OTTER = False
    module.USE_CANDIDATES = False
    return module


def test_be_reference_has_independent_grids_and_correct_temperature_labels():
    curves = _gallery().load_reference()
    assert len(curves) == 9
    assert {c["temperature_ev"] for c in curves if c["method"] == "DFT-MD"} == {50, 100, 150}
    assert {c["temperature_ev"] for c in curves if c["method"] == "OPAL"} == {50, 100, 160}
    assert [c["points"] for c in curves] == [7, 40, 40, 9, 40, 40, 9, 40, 40]


@pytest.mark.private_baseline
def test_be_scan_preserves_full_only_definitions_and_diagnostics():
    data = _gallery().load_scan()
    assert data["rho_g_cc"].shape == (48,)
    for t in (50, 100, 150):
        expected = [1, 3, 6, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
        np.testing.assert_array_equal(data["rho_g_cc"][data["te_ev"] == t],
                                      expected)
    np.testing.assert_allclose(data["zstar"], data["n0_bohr3"] / data["n_i_bohr3"])
    assert np.all(data["stage2_converged"])
    assert np.all(data["continuum_matching_window_full_valid"])
    assert set(data["threshold_state_status"]) <= {"resolved", "marginal", "unresolved"}
    assert np.all(np.isfinite(data["zbar"]))
    assert not {"n_full", "n_ext", "v_eff", "r"}.intersection(data)
    metadata = json.loads(data["metadata_json"].item())
    assert metadata["configuration"]["run_mode"] == "full"
    assert metadata["configuration"]["cont_n_jobs"] == 1
    assert metadata["convergence"]["stage2_nonconverged_states"] == 0
    assert metadata["convergence"]["threshold_counts"] == {"resolved": 48}


@pytest.mark.private_baseline
def test_be_plot_uses_distinct_bethkenhagen_palette_and_preserves_data():
    import matplotlib.pyplot as plt

    module = _gallery()
    data = module.load_scan()
    original = {key: value.copy() for key, value in data.items()}
    references = module.load_reference()
    figure = module.plot_comparison(data, references)
    try:
        palette = module.PALETTES["bing"]
        expected_colors = ["#131313", palette[1], palette[2], palette[3], palette[4]]
        for ax in figure.axes:
            assert [line.get_color() for line in ax.lines] == expected_colors
            assert len(set(expected_colors)) == 5
            assert [line.get_linestyle() for line in ax.lines] == ["-", "-", ":", "-.", "--"]
            assert ax.lines[2].get_markerfacecolor() == "white"
        y = figure.axes[0].lines[0].get_ydata()
        assert np.all(np.isfinite(y))
        for key in data:
            np.testing.assert_array_equal(data[key], original[key])
        for ax, temperature in zip(figure.axes, (50, 100, 150), strict=True):
            for line, reference in zip(ax.lines[2:],
                                       [r for r in references if r["temperature_ev"] == temperature
                                        or (temperature == 150 and r["temperature_ev"] == 160)],
                                       strict=True):
                np.testing.assert_array_equal(line.get_ydata(), reference["z"])
    finally:
        plt.close(figure)


@pytest.mark.private_baseline
def test_be_plot_still_exposes_missing_and_marginal_points():
    """A future partial scan must not be made smooth by interpolation."""
    import matplotlib.pyplot as plt

    module = _gallery()
    data = module.load_scan()
    keep = ~((data["te_ev"] == 50) & (data["rho_g_cc"] == 65))
    data = {k: v[keep].copy() if v.shape == keep.shape else v.copy()
            for k, v in data.items()}
    data["threshold_state_status"][(data["te_ev"] == 50) & (data["rho_g_cc"] == 60)] = "marginal"
    figure = module.plot_comparison(data, module.load_reference())
    try:
        rho = figure.axes[0].lines[0].get_xdata()
        y = figure.axes[0].lines[0].get_ydata()
        assert np.all(np.isnan(y[np.isin(rho, [60, 65])]))
        assert len(figure.axes[0].collections) == 2
    finally:
        plt.close(figure)


@pytest.mark.private_baseline
def test_be_terminal_summary_comes_from_archive(monkeypatch, capsys):
    import matplotlib.pyplot as plt

    module = _gallery()
    monkeypatch.setattr(module, "save_figure", lambda *a: None)
    try:
        module.main()
        output = capsys.readouterr().out
        assert "Full SCF: 48/48; threshold: resolved=48; failed=0" in output
        assert "FAILED:" not in output
        assert "Crosses mark" not in output
    finally:
        plt.close("all")
