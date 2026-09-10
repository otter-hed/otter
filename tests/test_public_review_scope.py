"""Public reproduction tools cannot depend on excluded private studies."""
from pathlib import Path
from types import SimpleNamespace
import ast
import json
import re
import tomllib

import numpy as np

from tools.export_public_review import in_public_tool_scope


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
