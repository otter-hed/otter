"""Repository tests must not rely on python -m adding the current directory."""
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("from_checkout", [True, False], ids=["checkout", "elsewhere"])
def test_collection_without_implicit_current_directory(tmp_path, from_checkout):
    root = Path(__file__).resolve().parents[1]
    # -I removes implicit cwd/PYTHONPATH entries, like a console pytest entry
    # point. The explicit pytest configuration must supply both source roots.
    probe = """
from pathlib import Path
import sys
import pytest

root = Path(sys.argv[1])
status = pytest.main(sys.argv[2:])
if status:
    raise SystemExit(status)
import otter
import tools.gallery_notebooks as notebooks
import benchmarks.runners.ion_structure_validation as validation
for module, relative in (
    (otter, 'src/otter/__init__.py'),
    (notebooks, 'tools/gallery_notebooks.py'),
    (validation, 'benchmarks/runners/ion_structure_validation.py'),
):
    assert Path(module.__file__).resolve() == (root / relative).resolve()
"""
    files = ["test_gallery_notebooks.py", "test_ion_structure_acceptance.py",
             "test_promotion_snapshot.py", "test_public_review_scope.py"]
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(root),
         "-c", str(root / "pyproject.toml"), "--collect-only", "-q",
         *(str(root / "tests" / name) for name in files)],
        cwd=root if from_checkout else tmp_path,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
