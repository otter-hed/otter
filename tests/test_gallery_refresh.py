"""Recorded gallery output updates without executing expensive source."""
from pathlib import Path
import subprocess
import sys

import pytest


def test_recorded_output_refreshes_without_solver_or_npz(tmp_path):
    pytest.importorskip("sphinx_gallery")
    pytest.importorskip("sphinx")
    source, examples, build = (tmp_path / name for name in ("source", "examples", "html"))
    source.mkdir()
    examples.mkdir()
    (source / "conf.py").write_text(
        "extensions=['sphinx_gallery.gen_gallery']\nmaster_doc='index'\n"
        "sphinx_gallery_conf={'examples_dirs':'../examples','gallery_dirs':'gallery',"
        "'plot_gallery':False,'download_all_examples':False}\n"
    )
    (source / "index.rst").write_text("Test\n====\n\n.. toctree::\n\n   gallery/index\n")
    (examples / "README.rst").write_text("Gallery\n=======\n")
    script = examples / "plot_values.py"
    script.write_text('"""Recorded result\n===============\n\n'
                      '.. include:: /recorded.rst\n"""\n'
                      'raise RuntimeError("expensive solver must not run in docs")\n')
    original_script = script.read_bytes()
    for value in (1.0, 4.0):
        (source / "recorded.rst").write_text(f".. code-block:: text\n\n   data_value={value:.3f}\n")
        run = subprocess.run([sys.executable, "-m", "sphinx", "-b", "html",
                              str(source), str(build)], capture_output=True, text=True)
        assert run.returncode == 0, run.stdout + run.stderr
        html = (build / "gallery/plot_values.html").read_text()
        assert f"data_value={value:.3f}" in html
    assert script.read_bytes() == original_script
    assert not list(tmp_path.rglob("*.npz"))
