"""Fast, offline checks for the experimental Al IS/SC gallery pipeline."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str) -> ModuleType:
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_state() -> dict[str, np.ndarray]:
    r = np.linspace(0.1, 4.0, 8)
    k = np.linspace(0.1, 4.0, 9)
    state: dict[str, np.ndarray] = {
        "schema_version": np.asarray("otter_al_is_sc_comparison_v1"),
        "model_labels": np.asarray(("qm", "tf")),
        "model_display_labels": np.asarray(("KS-DFT", "Thomas--Fermi")),
        "structure_labels": np.asarray(("is", "sc")),
        "sc_status": np.asarray("experimental"),
        "rho_g_cc": np.asarray(8.1),
        "te_ev": np.asarray(15.0),
        "ti_ev": np.asarray(15.0),
        "r_bohr": r,
        "k_bohr_inv": k,
        "gii_r": np.ones((2, 2, r.size)),
        "sii_k": np.ones((2, 2, k.size)),
        "sc_converged": np.asarray((True, True)),
        "hnc_residual": np.full((2, 2), 1.0e-7),
        "fixed_is_mu_ha": np.asarray((0.2, 0.3)),
        "mu_ha": np.asarray(((0.2, 0.2), (0.3, 0.3))),
        "tf_discrete_ks_levels_defined": np.asarray(False),
    }
    for structure, shift in (("is", 0.0), ("sc", 0.01)):
        state[f"qm_{structure}_bound_l"] = np.asarray((0, 1), dtype=int)
        state[f"qm_{structure}_bound_n_index"] = np.asarray((1, 1), dtype=int)
        state[f"qm_{structure}_bound_energy_ha"] = np.asarray(
            (-2.0 + shift, -0.5 + shift)
        )
        state[f"qm_{structure}_bound_fd"] = np.asarray((1.0, 0.8))
        state[f"qm_{structure}_bound_occ_deg_fd"] = np.asarray((2.0, 4.8))
        for name in (
            "bound_l",
            "bound_n_index",
            "bound_energy_ha",
            "bound_fd",
            "bound_occ_deg_fd",
        ):
            dtype = int if name in ("bound_l", "bound_n_index") else float
            state[f"tf_{structure}_{name}"] = np.empty(0, dtype=dtype)
    return state


def test_producer_configuration_is_documented_production_is_start() -> None:
    producer = _load_module(
        "otter_al_is_sc_producer_test",
        "benchmarks/runners/regenerate_al_is_sc_comparison.py",
    )
    for model in ("qm", "tf"):
        cfg = producer.configuration(model)
        assert cfg.electronic_model == model
        assert cfg.elements == ["Al"]
        assert cfg.rho_g_cc == 8.1
        assert cfg.temperature_ev == 15.0
        assert cfg.ion_temperature_ev == 15.0
        assert cfg.aa_overrides.get("bound_occ_mode", "fd") == "fd"
        assert cfg.aa_overrides.get("b3_tail_model", "full") == "full"
        assert cfg.aa_overrides.get("bound_rmax_mult") is None
        assert cfg.qoz_zbar_mode == "pseudoatom_partition"
        assert cfg.qoz_renormalize_nscr_to_zbar
        assert cfg.qoz_response_chi0_model == "lindhard_fd"
        assert cfg.qoz_response_lfc_model == "chabrier1990"
    assert producer.SC_CONTROLS.require_converged
    assert producer.SC_CONTROLS.g_tol == 5.0e-4
    assert producer.SC_CONTROLS.v_corr_tol == 5.0e-4


def test_loader_validates_fixed_mu_and_tf_level_semantics() -> None:
    runner = _load_module(
        "otter_al_is_sc_runner_test",
        "benchmarks/runners/plot_al_is_sc_comparison.py",
    )
    state = _synthetic_state()
    runner.validate_state(state)
    assert [row["level"] for row in runner.level_rows(state, "is")] == [
        "1s",
        "2p",
    ]
    invalid = dict(state)
    invalid["tf_sc_bound_energy_ha"] = np.asarray((-0.1,))
    try:
        runner.validate_state(invalid)
    except ValueError as exc:
        assert "Thomas--Fermi" in str(exc)
    else:
        raise AssertionError("A TF discrete KS level must be rejected.")


def test_gallery_cites_equations_and_exposes_safe_recompute_switch() -> None:
    source = (
        ROOT / "docs" / "examples" / "plot_al_is_sc_comparison.py"
    ).read_text(encoding="utf-8")
    assert "RECOMPUTE_WITH_OTTER = True" in source
    assert "RECOMPUTE_MODEL_WORKERS = 1" in source
    assert ":cite:t:`StarrettSaumon2014`" in source
    assert "Eqs. (19)--(20)" in source
    assert "SC (experimental)" in source
    assert "Thomas--Fermi is a semiclassical density model" in source
    assert "SC extension" in source
    assert "sc_total_elapsed_s" in source
    assert "solve_plasma_workflow(" in source
    assert "solve_sc_feedback_workflow(" in source
    assert "importlib" not in source
    assert "benchmarks/runners" not in source
    assert 'style_context("thesis", palette="bing")' in source
    assert ".grid(" not in source


def test_plot_and_terminal_report_identical_total_times(monkeypatch, capsys) -> None:
    import matplotlib.pyplot as plt

    gallery = _load_module(
        "otter_al_is_sc_total_time_test", "docs/examples/plot_al_is_sc_comparison.py"
    )
    state = _synthetic_state()
    state.update(
        zbar_partition=np.full((2, 2), 3.0),
        is_elapsed_s=np.asarray((18.42, 1.0)),
        sc_extension_elapsed_s=np.asarray((83.05, 8.21)),
        sc_total_elapsed_s=np.asarray((101.47, 9.21)),
    )
    for model in ("qm", "tf"):
        state[f"{model}_sc_history_iteration"] = np.asarray((1, 2))
        state[f"{model}_sc_history_max_g_change"] = np.asarray((1e-3, 1e-5))
        state[f"{model}_sc_history_max_v_corr_residual_ha"] = np.asarray((2e-3, 2e-5))
    figures = []
    monkeypatch.setattr(gallery, "recompute_state", lambda: state)
    monkeypatch.setattr(gallery, "save_figure", lambda fig, *a, **kw: figures.append(fig))
    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        gallery.main()
        terminal = capsys.readouterr().out
        axis = figures[-1].axes[0]
        np.testing.assert_allclose(
            [bar.get_height() for bar in axis.patches], [18.42, 1.0, 101.47, 9.21]
        )
        labels = [text.get_text() for text in axis.texts]
        assert labels == ["18.42 s", "1.00 s", "101.47 s", "9.21 s"]
        rows = [line for line in terminal.splitlines()
                if line.startswith(("KS-DFT ", "Thomas--Fermi "))]
        rows = [line for line in rows if line.split()[1] in ("IS", "SC")]
        assert [float(line.split()[-1]) for line in rows] == [18.42, 101.47, 1.0, 9.21]
        assert axis.get_ylabel() == "total wall time [s]"
        assert axis.get_ylabel() in terminal
        assert "SC total = initial IS calculation + SC feedback stage." in terminal
    finally:
        for fig in figures:
            plt.close(fig)


def test_recorded_timing_figure_matches_recorded_terminal() -> None:
    import re
    import xml.etree.ElementTree as ET

    directory = ROOT / "docs/source/_static/gallery_results/plot_al_is_sc_comparison"
    terminal = (directory / "results.rst").read_text()
    rows = {}
    for line in terminal.splitlines():
        columns = line.split()
        if len(columns) == 6 and columns[1] in ("IS", "SC"):
            rows[(columns[0], columns[1])] = columns[-1] + " s"
    expected = [rows[(model, path)] for path in ("IS", "SC")
                for model in ("KS-DFT", "Thomas--Fermi")]
    svg = ET.parse(directory / "figure_003.svg")
    labels = ["".join(node.itertext()) for node in svg.iter()
              if node.tag.endswith("}text")]
    assert [label for label in labels if re.fullmatch(r"\d+\.\d{2} s", label)] == expected
    assert "total wall time [s]" in labels
    assert "total wall time [s]" in terminal
