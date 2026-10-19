"""Public reproduction tools cannot depend on excluded private studies."""
from pathlib import Path
from types import SimpleNamespace
import ast
import json
import re
import subprocess
import tomllib

import numpy as np
import pytest

from tools.export_public_review import export_review, in_public_tool_scope


def test_public_export_excludes_private_plan_and_reference_library(tmp_path, monkeypatch):
    root = tmp_path / "source"
    names = ["teos_todos.md", "literature/paper.md", "DEV_STATE.md",
             ".local/settings.json", ".github/workflows/ci.yml", ".gitignore",
             "src/otter/example.py", "applications/pamd_eos/example.py",
             "applications/tests/test_example.py", "grid.h5", "grid.hdf5"]
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("example\n")
    monkeypatch.setattr("tools.export_public_review.subprocess.check_output",
                        lambda *args, **kwargs: "\0".join(names).encode())
    monkeypatch.setattr("tools.export_public_review.subprocess.run",
                        lambda *args, **kwargs: SimpleNamespace(
                            returncode=1, stdout=b"", stderr=b""))
    destination = tmp_path / "public"
    assert export_review(root, destination, public_tools=True) == 3
    assert sorted(p.relative_to(destination).as_posix()
                  for p in destination.rglob("*") if p.is_file()) == [
                      ".github/workflows/ci.yml", ".gitignore", "src/otter/example.py"]


@pytest.mark.parametrize("public_tools", [False, True])
def test_public_export_respects_local_exclusions_for_tracked_files(
    tmp_path, public_tools,
):
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "README.md").write_text("Public documentation\n")
    (root / "notes.local.md").write_text("Private development notes\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    (root / ".git/info/exclude").write_text("/notes.local.md\n")

    destination = tmp_path / "public"
    assert export_review(root, destination, public_tools=public_tools) == 1
    assert (destination / "README.md").is_file()
    assert not (destination / "notes.local.md").exists()
    tracked = subprocess.check_output(["git", "ls-files"], cwd=root).decode()
    assert "notes.local.md" in tracked.splitlines()


def test_public_export_stops_when_local_rules_cannot_be_checked(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr("tools.export_public_review.subprocess.check_output",
                        lambda *args, **kwargs: b"README.md\0")
    monkeypatch.setattr("tools.export_public_review.subprocess.run",
                        lambda *args, **kwargs: SimpleNamespace(
                            returncode=128, stdout=b"", stderr=b"Invalid repository"))
    destination = tmp_path / "public"
    with pytest.raises(RuntimeError, match="Cannot verify local exclusion rules"):
        export_review(tmp_path, destination)
    assert not destination.exists()


def test_readme_version_and_reproduction_entry_points_are_current():
    root = Path(__file__).resolve().parents[1]
    text = (root / "README.md").read_text()
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    citation = text.split("## Citation\n", 1)[1].split("## Acknowledgements", 1)[0]
    cited_versions = re.findall(r"version (\d+\.\d+\.\d+)", citation)
    assert cited_versions == [version, version]
    assert "cached benchmark gallery" not in text
    assert "does not include precomputed Otter NPZ" in text
    assert "does not run AA or MD" in text
    commands = re.findall(r"^poetry run python ((?:docs|benchmarks)/examples/[^\s]+\.py)$",
                          text, flags=re.MULTILINE)
    assert len(commands) >= 2
    for relative in commands:
        assert (root / relative).is_file()


def test_public_tool_scope_preserves_reproduction_and_physics_tests():
    for name in ("tools/otter_lammps_md.py", "tools/reproduce_ch2_hnc_md.py",
                 "tools/recompute_all_data.py", "tests/test_ion_structure_acceptance.py",
                 "tests/test_scf_continuum_matching_guard.py", "tests/test_external_background_policy.py"):
        assert in_public_tool_scope(name)
    for name in ("tools/diagnostics/study_screening_charge_budget.py",
                 "tools/diagnostics/reports/PRIVATE.md", "tools/run_production_validation.py",
                 "tests/test_ch2_xrts_dataset_audit.py", "tests/test_library_provenance_audit.py",
                 "tests/test_background_fix_campaign.py"):
        assert not in_public_tool_scope(name)


def test_public_candidate_gate_still_rejects_failed_or_unresolved_outputs(tmp_path):
    # Keep the public-producer assertion separate from private campaign tests.
    path = Path(__file__).resolve().parents[1] / "tools/recompute_all_data.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "candidate_failures")
    namespace = {"ROOT": tmp_path, "Task": SimpleNamespace, "json": json}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    directory = tmp_path / "candidate"
    directory.mkdir()
    task = SimpleNamespace(clean_paths=("candidate",))
    (directory / "failures.json").write_text("{}")
    assert namespace["candidate_failures"](task) == []
    (directory / "failures.json").write_text('{"C": "SCF failed"}')
    np.savez(directory / "scan.npz", threshold_status=["resolved", "unresolved"])
    issues = namespace["candidate_failures"](task)
    assert len(issues) == 2
    assert any("unresolved" in issue for issue in issues)
