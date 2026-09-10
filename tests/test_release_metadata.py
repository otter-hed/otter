"""Keep current package and citation versions aligned; preserve historical data."""
from pathlib import Path
import re
import tomllib

from otter import __version__


def test_current_release_versions_agree():
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    assert __version__ == version
    citation = (root / "CITATION.cff").read_text()
    assert re.findall(r"^\s*version: (\S+)$", citation, re.MULTILINE) == [version, version]
    for name in ("CITATIONS.md", "docs/source/citations.rst"):
        assert f"version {version}, computer software" in (root / name).read_text()
    changelog = (root / "CHANGELOG.md").read_text()
    release_date = re.findall(r"^date-released: (\S+)$", citation, re.MULTILINE)[0]
    assert f"## {version} — {release_date}" in changelog
