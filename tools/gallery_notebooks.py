"""Build notebook launchers for the same file-backed gallery entry points.

Scientific functions must not be copied into an interactive ``__main__``:
spawn-based workers and producer file hashes require an actual Python file.
Notebook execution streams that file's output using the kernel interpreter.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import zipfile


def run_script(root: Path, relative: str) -> None:
    """Run a checked-in gallery script and forward failures and progress."""
    root = root.resolve()
    script = (root / relative).resolve()
    if not script.is_relative_to(root) or not script.is_file():
        raise FileNotFoundError(f"Missing gallery script in this checkout: {relative}")
    with subprocess.Popen(
        [sys.executable, "-u", str(script)], cwd=root,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        start_new_session=(os.name != "nt"),
    ) as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
            status = process.wait()
        except BaseException:
            # An interrupted MD notebook must not leave its MPI job running.
            if os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                # Windows has no POSIX process groups. Target this child's
                # process tree, not just its Python parent (AA/MPI workers).
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        check=False, timeout=5,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
            raise
    if status:
        raise subprocess.CalledProcessError(status, process.args)


def notebook_bytes(relative: str) -> bytes:
    """A small launcher, not a second copy of the scientific implementation."""
    introduction = (
        "# Reproduce this Otter gallery\n\n"
        f"This notebook runs `{relative}` from a complete Otter source checkout. "
        "Use the Otter kernel and set its working directory to the repository root "
        "(for example, `os.chdir('/path/to/otter')`).\n\n"
        "Edit the input block in the repository's `.py` script to change parameters. "
        "The cell streams progress and computes fresh results; it does not load "
        "bundled NPZ. Figures and numerical outputs are saved by the script. "
        "MD examples require LAMMPS/MPI and may take hours."
    )
    code = (
        "from pathlib import Path\nimport sys\n\n"
        "root = next((p for p in (Path.cwd(), *Path.cwd().parents)\n"
        f"             if (p / 'pyproject.toml').is_file() and (p / {relative!r}).is_file()), None)\n"
        "if root is None:\n"
        "    raise RuntimeError('Set the notebook working directory to the complete Otter checkout.')\n"
        "sys.path.insert(0, str(root))\n"
        "from tools.gallery_notebooks import run_script\n"
        f"run_script(root, {relative!r})\n"
    )
    payload = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3 (Otter environment)",
                                     "language": "python", "name": "python3"}},
        "cells": [
            {"cell_type": "markdown", "id": "instructions", "metadata": {}, "source": introduction},
            {"cell_type": "code", "id": "run-gallery", "metadata": {},
             "execution_count": None, "outputs": [], "source": code},
        ],
    }
    return (json.dumps(payload, indent=2) + "\n").encode()


def prepare_downloads(app, exception) -> None:
    """Update generated notebooks and ZIPs after Sphinx-Gallery copies them."""
    if exception is not None or app.builder.format != "html":
        return
    root = Path(app.srcdir).resolve().parents[1]
    pages = [*(root / "docs/examples").glob("plot_*.py"),
             *(root / "benchmarks/examples").glob("plot_*.py")]
    destinations = [Path(app.srcdir) / "gen_examples",
                    Path(app.srcdir) / "benchmarks/gen_benchmarks",
                    Path(app.outdir) / "_downloads"]
    for source in pages:
        payload = notebook_bytes(source.relative_to(root).as_posix())
        for directory in destinations:
            for notebook in directory.rglob(source.stem + ".ipynb"):
                notebook.write_bytes(payload)
            for archive in directory.rglob("*.zip"):
                with zipfile.ZipFile(archive) as handle:
                    if source.stem + ".ipynb" not in handle.namelist():
                        continue
                    members = [(item, payload if item.filename == source.stem + ".ipynb"
                                else handle.read(item)) for item in handle.infolist()]
                with zipfile.ZipFile(archive, "w") as handle:
                    for item, data in members:
                        handle.writestr(item, data)
