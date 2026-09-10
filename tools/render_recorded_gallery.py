"""Render compact presentation assets from explicitly selected local archives.

This maintainer-only plotting command never calculates new AA or MD states.
Public gallery entry points remain compute-first. Review local numerical
archives before using this command to update the recorded website results.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "docs/source/_static/gallery_results"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forbidden(*args, **kwargs):
    raise RuntimeError("Recorded rendering must not invoke an AA/MD calculation")


def render_page(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("recorded_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name, value in {
        "RECOMPUTE_WITH_OTTER": False, "USE_PRECOMPUTED_DATA": True,
        "USE_RECOMPUTED_CANDIDATES": False, "USE_ACCEPTED_OTTER_SCAN": True,
        "USE_CANDIDATES": False,
    }.items():
        if hasattr(module, name):
            setattr(module, name, value)
    for name in ("solve_plasma_workflow", "solve_full_only", "solve_sc_feedback_workflow",
                 "_compute_and_stage", "compute_candidate", "solve_all_states"):
        if hasattr(module, name):
            setattr(module, name, forbidden)
    directory = DESTINATION / path.stem
    directory.mkdir(parents=True, exist_ok=True)
    figures = {}

    def save(fig, output, **kwargs):
        index = len([name for name in figures if name.endswith(".svg")]) + 1
        name = f"figure_{index:03d}.svg"
        target = directory / name
        # SVG path geometry preserves the scientific curves; no raster resizing.
        fig.savefig(target, format="svg", metadata={"Date": None})
        figures[name] = digest(target)
        if index == 1:
            thumbnail = directory / "thumbnail.png"
            fig.savefig(thumbnail, dpi=45, format="png")
            figures[thumbnail.name] = digest(thumbnail)
        if kwargs.get("close"):
            plt.close(fig)
        return {"svg": target, "png": target, "pdf": target}

    if hasattr(module, "save_figure"):
        module.save_figure = save
    output = io.StringIO()
    try:
        with redirect_stdout(output):
            module.main()
    finally:
        plt.close("all")
    lines = [".. Recorded local validation output; presentation only, not solver input.", ""]
    for name in figures:
        if name.endswith(".svg"):
            lines += [f".. image:: /_static/gallery_results/{path.stem}/{name}",
                      f"   :alt: Recorded result for {path.stem}", ""]
    terminal = []
    for line in output.getvalue().splitlines():
        if (line.startswith(("Using ", "Loaded nine", "[saved]", "saved PNG:", "saved PDF:"))
                or "_static/gallery_results/" in line):
            continue
        terminal.append(line.replace(str(ROOT) + "/", ""))
    lines += ["Recorded terminal output", "~~~~~~~~~~~~~~~~~~~~~~~~", "", ".. code-block:: text", ""]
    lines += ["   " + line for line in terminal]
    (directory / "results.rst").write_text("\n".join(lines) + "\n")
    return {"rendering_source_sha256": digest(path), "figures": figures,
            "results_sha256": digest(directory / "results.rst"),
            "role": "recorded_presentation_only_not_a_solver_input"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-local-archives", action="store_true", required=True)
    parser.parse_args()
    manifest = json.loads((DESTINATION / "manifest.json").read_text())
    # Hashes identify the data actually plotted, not a claimed fresh solve.
    archives = {str(p.relative_to(ROOT)): digest(p)
                for p in sorted((ROOT / "benchmarks/baselines").glob("*/*.npz"))}
    if not archives:
        raise FileNotFoundError("No local accepted archives; run the public scripts to compute data first")
    for relative in manifest:
        print(relative, flush=True)
        previous = manifest[relative]
        updated = render_page(ROOT / relative)
        updated["initial_html_capture"] = previous.get("initial_html_capture", {
            key: value for key, value in previous.items()
            if key in {"source_sha256_at_capture", "html_sha256_at_capture"}})
        updated["local_archive_inventory_sha256"] = hashlib.sha256(
            json.dumps(archives, sort_keys=True).encode()).hexdigest()
        manifest[relative] = updated
    (DESTINATION / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (DESTINATION / "archive_inventory.json").write_text(json.dumps(archives, indent=2) + "\n")


if __name__ == "__main__":
    main()
