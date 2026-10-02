"""Keep current package and citation versions aligned; preserve historical data."""
from pathlib import Path
import re
import tomllib

from otter import __version__
from otter.literature import write_citations_markdown


def test_current_release_versions_agree():
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    assert __version__ == version
    citation = (root / "CITATION.cff").read_text()
    assert re.findall(r"^\s*version: (\S+)$", citation, re.MULTILINE) == [version, version]
    for name in ("CITATIONS.md", "docs/source/citations.rst"):
        assert f"version {version}, computer software" in (root / name).read_text()
    changelog = (root / "CHANGELOG.md").read_text()
    dates = re.findall(r"^\s*date-released: (\S+)$", citation, re.MULTILINE)
    if f"### {version}\n" in changelog.split("\n## ", 2)[1]:
        # An unreleased candidate must not inherit the previous release date.
        assert dates == []
    else:
        assert len(dates) == 2 and dates[0] == dates[1]
        assert f"## {version} — {dates[0]}" in changelog


def test_current_software_authors_agree(tmp_path):
    root = Path(__file__).resolve().parents[1]
    names = ["Chongbing Qu", "Julian Lütgert", "Dominik Kraus"]
    metadata = tomllib.loads((root / "pyproject.toml").read_text())
    assert [author["name"] for author in metadata["project"]["authors"]] == names
    citation = (root / "CITATION.cff").read_text()
    pairs = re.findall(r"family-names: (.+)\n\s+given-names: (.+)", citation)
    assert [f"{given} {family}" for family, given in pairs] == names * 2
    for filename in ("README.md", "CITATIONS.md", "docs/source/citations.rst"):
        content = (root / filename).read_text()
        assert all(name in content for name in names)
    generated = write_citations_markdown(tmp_path / "CITATIONS.md").read_text()
    assert all(name in generated for name in names)
    assert generated.split("## For contributors")[0] == (
        root / "CITATIONS.md").read_text().split("## For contributors")[0]
