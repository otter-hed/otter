"""Exercise the public reproduction wiring without running expensive AA solves."""
from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path
import re
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(relative: str, monkeypatch):
    name = "reproduction_test_" + Path(relative).stem
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["carbon_ionization_levels", "doppner_be_ionization"])
def test_fresh_reproduction_selects_only_requested_scan(tmp_path, monkeypatch, name):
    module = load_script("tools/recompute_all_data.py", monkeypatch)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    task = module.TASKS[name]
    for relative in task.clean_paths:
        directory = tmp_path / relative
        directory.mkdir(parents=True)
        (directory / "old-checkpoint.json").write_text("{}")
    protected = [
        tmp_path / "benchmarks/baselines/keep.npz",
        tmp_path / "benchmarks/reference_data/keep.csv",
        tmp_path / "benchmarks/outputs/unselected/keep.json",
    ]
    for path in protected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unchanged")
    calls = []

    def run(command, **kwargs):
        # The actual subprocess is replaced: no AA computation in this test.
        assert all(not (tmp_path / p).exists() for p in task.clean_paths)
        calls.append((command, kwargs))

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["recompute_all_data.py", "--only", name, "--fresh"])
    module.main()
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == [sys.executable, str(tmp_path / task.script)]
    assert kwargs["check"] is True
    assert kwargs["env"]["PYTHONPATH"].split(":")[0] == str(tmp_path / "src")
    if name == "carbon_ionization_levels":
        assert kwargs["env"]["OTTER_RECOMPUTE_CARBON_IONIZATION"] == "1"
        assert kwargs["env"]["OTTER_REUSE_ACCEPTED_CARBON_IONIZATION"] == "0"
    assert all(path.read_bytes() == b"unchanged" for path in protected)


@pytest.mark.parametrize("key", ["threshold_status", "threshold_state_status"])
def test_reproduction_audits_both_scan_schemas_and_list_failures(tmp_path, monkeypatch, key):
    module = load_script("tools/recompute_all_data.py", monkeypatch)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    directory = tmp_path / "candidate"
    directory.mkdir()
    (directory / "failures.json").write_text(json.dumps([{"error": "full SCF failed"}]))
    np.savez(directory / "state.npz", **{key: ["resolved", "unresolved"]})
    task = module.Task("unused.py", {}, ("candidate",))
    issues = module.candidate_failures(task)
    assert len(issues) == 2
    assert any("full SCF failed" in issue for issue in issues)
    assert any("1 unresolved threshold states" in issue for issue in issues)


def test_default_carbon_scan_is_serial(monkeypatch):
    module = load_script("docs/examples/plot_carbon_ionization_levels.py", monkeypatch)
    assert module.MAX_STATE_WORKERS == 1
    assert module._configuration(1.0).cont_n_jobs == 1
    assert module._configuration(1.0).run_mode == "full"


def test_be_failure_keeps_diagnostics_across_process_boundary(monkeypatch):
    module = load_script("benchmarks/runners/regenerate_doppner_2023_be_ionization.py", monkeypatch)
    failed = dict(stage2_converged=False, stage2_iters=80,
                  scf_stop_reason="continuum_matching_window", final_state_map_error=None,
                  history=[dict(iter=79, dn_rel=.002, dv_rel=.004, err=.02,
                                charge_ws=4., charge_bound=0., charge_cont=4.,
                                continuum_matching_window_full_valid=False)],
                  n_full=np.ones(4096))
    monkeypatch.setattr(module, "solve_full_only", lambda config: failed)
    with pytest.raises(module.FullSCFConvergenceError) as caught:
        module.solve_point(65, 50)
    received = pickle.loads(pickle.dumps(caught.value))
    assert received.diagnostics["history"] == failed["history"]
    assert received.diagnostics["stage2_iters"] == 80
    assert received.diagnostics["scf_stop_reason"] == "continuum_matching_window"
    assert "n_full" not in received.diagnostics
    assert "dn=0.002" in str(received)
    json.dumps(received.diagnostics)


def test_be_plot_candidate_switch_does_not_launch_solver(monkeypatch):
    monkeypatch.delenv("OTTER_USE_CANDIDATE_BE_IONIZATION", raising=False)
    baseline = load_script("benchmarks/examples/plot_doppner_2023_be_ionization.py", monkeypatch)
    assert baseline.USE_CANDIDATES is True
    assert baseline.RECOMPUTE_WITH_OTTER is True
    monkeypatch.setenv("OTTER_USE_CANDIDATE_BE_IONIZATION", "1")
    candidate = load_script("benchmarks/examples/plot_doppner_2023_be_ionization.py", monkeypatch)
    assert candidate.USE_CANDIDATES is True


def test_be_gallery_locates_checkout_without_dunder_file(monkeypatch):
    module = load_script("benchmarks/examples/plot_doppner_2023_be_ionization.py", monkeypatch)
    monkeypatch.delattr(module, "__file__")
    monkeypatch.chdir(ROOT / "benchmarks/examples")
    assert module.repository_root() == ROOT


def test_reproduction_source_is_available_in_the_benchmark_page():
    page = ROOT / "benchmarks/examples/plot_doppner_2023_be_ionization.py"
    text = page.read_text()
    links = re.findall(r":download:`[^`]*<([^>]+)>`", text)
    assert len(links) == 0
    assert "python benchmarks/examples/plot_doppner_2023_be_ionization.py" in text
    assert "subprocess.run" in text
    assert "reproducing_ionization" not in text
    assert not (ROOT / "docs/source/benchmarks/reproducing_ionization.rst").exists()


def test_be_public_description_is_concise_and_scientific():
    import ast

    page = ROOT / "benchmarks/examples/plot_doppner_2023_be_ionization.py"
    description = ast.get_docstring(ast.parse(page.read_text()))
    assert len(description.split()) < 190
    assert "160 eV" in description
    # Numerical completion/status summaries must come from the current archive,
    # not an old hard-coded paragraph that survives a data refresh.
    assert "47 full SCFs" not in description
    for phrase in ("screenshot", "supplied CSV", "not provided", "No SC outer loop"):
        assert phrase not in description
