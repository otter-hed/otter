"""Capture reviewed HTML figures/output for documentation without numerical caches.

Run after reviewing a calculation-backed documentation build. This command
copies presentation assets only; it never runs a solver or manufactures data.
The Python gallery programs remain the source for independent calculations.
"""
from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]


class TerminalOutput(HTMLParser):
    """Extract Sphinx-Gallery terminal text without page navigation or code."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.chunks = []

    def handle_starttag(self, tag, attrs):
        if tag == "div":
            if self.depth:
                self.depth += 1
            elif "sphx-glr-script-out" in dict(attrs).get("class", "").split():
                self.depth = 1

    def handle_endtag(self, tag):
        if tag == "div" and self.depth:
            self.depth -= 1
            if not self.depth:
                self.chunks.append("\n")

    def handle_data(self, data):
        if self.depth:
            self.chunks.append(data)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(root: Path, build: Path, destination: Path):
    records = {}
    for source_dir, html_dir in (("docs/examples", "gen_examples"),
                                 ("benchmarks/examples", "benchmarks/gen_benchmarks")):
        for source in sorted((root / source_dir).glob("plot_*.py")):
            page = build / html_dir / (source.stem + ".html")
            html = page.read_text()
            if "sphx-glr-script-out" not in html:
                raise ValueError(f"No executed gallery output in {page}; refusing to replace reviewed results")
            target = destination / source.stem
            target.mkdir(parents=True, exist_ok=True)
            names = list(dict.fromkeys(re.findall(
                rf'src="[^\"]*/(sphx_glr_{re.escape(source.stem)}_\d+\.png)"', html)))
            lines = [".. Generated from a reviewed HTML run; not a calculation input.", ""]
            assets = {}
            for name in names:
                original = build / "_images" / name
                shutil.copy2(original, target / name)
                assets[name] = digest(original)
                lines += [f".. image:: /_static/gallery_results/{source.stem}/{name}",
                          f"   :alt: Recorded result for {source.stem}", ""]
            parser = TerminalOutput()
            parser.feed(html)
            output = "".join(parser.chunks).strip()
            # Remove old archive-loading narration, never numerical rows/errors.
            output = "\n".join(line for line in output.splitlines()
                if not (line.startswith(("Using checksummed", "Using accepted", "Loaded nine checksummed",
                                         "Using precomputed")) or
                        (line.startswith("Using ") and "precomputed Otter" in line)))
            if output:
                lines += ["Recorded terminal output", "~~~~~~~~~~~~~~~~~~~~~~~~", "",
                          ".. code-block:: text", ""]
                lines += ["   " + line for line in output.splitlines()]
                lines.append("")
            (target / "results.rst").write_text("\n".join(lines) + "\n")
            records[source.relative_to(root).as_posix()] = {
                "source_sha256_at_capture": digest(source),
                "html_sha256_at_capture": digest(page),
                "figures": assets,
                "results_sha256": digest(target / "results.rst"),
                "role": "recorded_presentation_only_not_a_solver_input",
            }
    (destination / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "docs/build/html")
    args = parser.parse_args()
    records = capture(ROOT, args.build, ROOT / "docs/source/_static/gallery_results")
    print(f"Captured {len(records)} reviewed pages; no solver or NPZ loader was run.")


if __name__ == "__main__":
    main()
