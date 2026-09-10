"""Notebook downloads must execute real script files, not copied worker code."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import zipfile

import pytest

from tools.gallery_notebooks import notebook_bytes, prepare_downloads, run_script

ROOT = Path(__file__).resolve().parents[1]


def test_launcher_uses_the_kernel_interpreter_and_propagates_failures(tmp_path, capsys):
    path = tmp_path / "example.py"
    path.write_text("import sys\nfrom pathlib import Path\nprint('progress', Path(__file__).name, flush=True)\nprint(sys.executable)\nraise SystemExit(7)\n")
    with pytest.raises(subprocess.CalledProcessError) as exc:
        run_script(tmp_path, "example.py")
    assert exc.value.returncode == 7
    import sys
    output = capsys.readouterr().out
    assert "progress example.py" in output and sys.executable in output


@pytest.mark.parametrize("relative", ["missing.py", "../outside.py"])
def test_missing_script_does_not_start_process(tmp_path, monkeypatch, relative):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("unexpected process"))
    with pytest.raises(FileNotFoundError):
        run_script(tmp_path, relative)


def test_windows_interrupt_targets_only_the_child_process_tree(tmp_path, monkeypatch):
    import tools.gallery_notebooks as notebook
    (tmp_path / "example.py").write_text("# synthetic entry point\n")
    class Interrupted:
        def __iter__(self):
            raise KeyboardInterrupt
    class Process:
        pid = 12345
        stdout = Interrupted()
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def wait(self, **kwargs): return 0
    monkeypatch.setattr(notebook, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: Process())
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    with pytest.raises(KeyboardInterrupt):
        run_script(tmp_path, "example.py")
    assert calls == [["taskkill", "/PID", "12345", "/T", "/F"]]


def test_download_rewrite_preserves_python_and_updates_zip(tmp_path):
    source = tmp_path / "benchmarks/examples/plot_example.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('real source')\n")
    srcdir = tmp_path / "docs/source"
    outdir = tmp_path / "docs/build/html"
    downloads = outdir / "_downloads/hash"
    downloads.mkdir(parents=True)
    notebook = downloads / "plot_example.ipynb"
    notebook.write_text("old notebook")
    zipped = downloads / "plot_example.zip"
    with zipfile.ZipFile(zipped, "w") as handle:
        handle.writestr(source.name, source.read_bytes())
        handle.writestr(notebook.name, b"old notebook")
    app = SimpleNamespace(srcdir=srcdir, outdir=outdir, builder=SimpleNamespace(format="html"))
    prepare_downloads(app, None)
    with zipfile.ZipFile(zipped) as handle:
        assert handle.read(source.name) == source.read_bytes()
        assert handle.read(notebook.name) == notebook.read_bytes()
    nb = json.loads(notebook.read_bytes())
    assert "run_script(root, 'benchmarks/examples/plot_example.py')" in nb["cells"][1]["source"]
    assert "ProcessPoolExecutor" not in notebook.read_text()
    unchanged = notebook.read_bytes()
    prepare_downloads(app, RuntimeError("build failed"))
    assert notebook.read_bytes() == unchanged


@pytest.mark.parametrize("path", sorted((ROOT / "docs/examples").glob("plot_*.py")) +
                         sorted((ROOT / "benchmarks/examples").glob("plot_*.py")), ids=lambda p: p.stem)
def test_every_notebook_selects_its_original_script_without_running_it(path, monkeypatch):
    monkeypatch.chdir(ROOT)
    called = []
    monkeypatch.setattr("tools.gallery_notebooks.run_script", lambda root, rel: called.append((root, rel)))
    payload = json.loads(notebook_bytes(path.relative_to(ROOT).as_posix()))
    compile(payload["cells"][1]["source"], "<notebook>", "exec")
    exec(payload["cells"][1]["source"], {"__name__": "__main__"})
    assert called == [(ROOT, path.relative_to(ROOT).as_posix())]


def test_starrett_provenance_without_file_or_outside_checkout(monkeypatch, tmp_path):
    from test_compute_first_galleries import load
    page = load(ROOT / "benchmarks/examples/plot_starrett_saumon_2013_electronic.py", monkeypatch)
    expected = page.producer_source()
    assert len(expected["script_sha256_current"]) == 64
    monkeypatch.delattr(page, "__file__")
    assert page.producer_source()["script_sha256_current"] is None
    assert page.producer_source()["execution_mode"] == "interactive"
    outside = tmp_path / "download.py"
    outside.write_text("# downloaded script\n")
    monkeypatch.setattr(page, "__file__", str(outside), raising=False)
    result = page.producer_source()
    assert result["script_relative_path"] is None
    assert result["script_filename"] == outside.name
    assert result["script_sha256_current"] == page.sha256_file(outside)
