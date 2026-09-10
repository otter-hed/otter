"""Public pages must calculate without any distributed numerical archive."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted((ROOT / "docs/examples").glob("plot_*.py")) + sorted(
    (ROOT / "benchmarks/examples").glob("plot_*.py"))


def load(path, monkeypatch):
    name = "compute_first_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.stem)
def test_import_does_not_read_npz_and_defaults_calculate(path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Import must not load numerical archives or run a solver")
    monkeypatch.setattr(np, "load", forbidden)
    import otter
    import otter.electronic
    monkeypatch.setattr(otter, "solve_plasma_workflow", forbidden)
    monkeypatch.setattr(otter.electronic, "solve_full_only", forbidden)
    module = load(path, monkeypatch)
    assert callable(module.main)
    if hasattr(module, "USE_PRECOMPUTED_DATA"):
        assert module.USE_PRECOMPUTED_DATA is False
    if hasattr(module, "RECOMPUTE_WITH_OTTER"):
        assert module.RECOMPUTE_WITH_OTTER is True
    assert 'No bundled Otter NPZ is required.' in module.__doc__
    assert f'poetry run python {path.relative_to(ROOT).as_posix()}' in module.__doc__


def test_recorded_assets_are_complete_and_checksummed():
    root = ROOT / "docs/source/_static/gallery_results"
    manifest = json.loads((root / "manifest.json").read_text())
    assert set(manifest) == {p.relative_to(ROOT).as_posix() for p in PAGES}
    for source, entry in manifest.items():
        assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == entry["rendering_source_sha256"]
        directory = root / Path(source).stem
        assert {p.name for p in directory.glob('figure_*.svg')} == {
            name for name in entry['figures'] if name.endswith('.svg')
        }
        assert hashlib.sha256((directory / "results.rst").read_bytes()).hexdigest() == entry["results_sha256"]
        for name, digest in entry["figures"].items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest


def test_al_slide_exports_are_opt_in_and_not_recorded(monkeypatch):
    path = ROOT / 'docs/examples/plot_al_full_workflow.py'
    module = load(path, monkeypatch)
    assert module.EXPORT_SLIDE_FIGURES is False
    tree = ast.parse(path.read_text())
    slide_block = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                       and isinstance(node.test, ast.Name)
                       and node.test.id == 'EXPORT_SLIDE_FIGURES')
    def saves(node):
        return [item for item in ast.walk(node) if isinstance(item, ast.Call)
                and isinstance(item.func, ast.Name) and item.func.id == 'save_figure']
    assert len(saves(tree)) - len(saves(slide_block)) == 4
    assert len(saves(slide_block)) == 3
    recorded = ROOT / 'docs/source/_static/gallery_results/plot_al_full_workflow'
    assert len(list(recorded.glob('figure_*.svg'))) == 4
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == 'plot_bound_orbitals' for node in ast.walk(tree))


def test_ch136_documents_the_screening_cloud_workflow(monkeypatch):
    module = load(ROOT / 'docs/examples/plot_ch136_mixture_workflow.py', monkeypatch)
    steps = [r'\mathrm{AA}', r'n_{\rm C}^{\rm scr}(k)',
             r'n_{\rm H}^{\rm scr}(k)', r'V_{ab}(k)', r'\mathrm{QOZ/HNC}',
             r'g_{ab}(r)', r'S_{ab}(k)']
    positions = [module.__doc__.index(step) for step in steps]
    assert positions == sorted(positions)


def test_docs_never_execute_expensive_gallery_code():
    tree = ast.parse((ROOT / "docs/source/conf.py").read_text())
    setting = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "sphinx_gallery_conf" for t in node.targets))
    values = dict((k.value, v) for k, v in zip(setting.keys, setting.values))
    assert ast.literal_eval(values["plot_gallery"]) is False


def test_carbon_fresh_default_does_not_reuse_old_points(monkeypatch):
    module = load(ROOT / "docs/examples/plot_carbon_ionization_levels.py", monkeypatch)
    assert not module.RESUME_LOCAL_RESULTS
    assert not module.REUSE_ACCEPTED_POINTS_WHEN_RECOMPUTING


def test_be_starts_producer_before_reading_its_new_output(monkeypatch):
    module = load(ROOT / "benchmarks/examples/plot_doppner_2023_be_ionization.py", monkeypatch)
    class Stop(Exception):
        pass
    def producer(command, **kwargs):
        assert command[0] == sys.executable
        assert command[1].endswith("regenerate_doppner_2023_be_ionization.py")
        assert kwargs["check"]
        raise Stop
    monkeypatch.setattr(module.subprocess, "run", producer)
    monkeypatch.setattr(module, "load_scan", lambda: pytest.fail("Read before calculation"))
    with pytest.raises(Stop):
        module.main()


def test_public_ch2_driver_has_no_private_cache_dependency(monkeypatch):
    module = load(ROOT / "tools/reproduce_ch2_hnc_md.py", monkeypatch)
    assert module.TARGET_TE_EV == (9., 29., 99.)
    assert module.RUN_MD and not module.REUSE_COMPLETED_CASES
    assert module.REQUIRE_ALL_CASES
    source = Path(module.__file__).read_text()
    assert "applications/" not in source
    assert "baselines/" not in source
    assert "pickle.load" not in source
    assert "reuse_completed=False" in source


def test_schorner_fresh_potential_does_not_reuse_an_old_md_log(tmp_path, monkeypatch):
    module = load(ROOT / "benchmarks/examples/plot_schorner_et_al_2022_al_sii.py", monkeypatch)
    monkeypatch.setattr(module, "CANDIDATE_DIR", tmp_path)
    monkeypatch.setattr(module.shutil, "which", lambda value: value)
    workdir = tmp_path / "md_work" / "new_state"
    workdir.mkdir(parents=True)
    for name in ("rdf_blocks.dat", "trajectory.lammpstrj", "log.lammps"):
        (workdir / name).write_text("Loop time from an earlier potential\n")
    potential = np.array([3., 2., 1.])
    def table(path, r, v):
        np.testing.assert_array_equal(v, potential)
        return 100, 2.0
    monkeypatch.setattr(module, "_write_lammps_pair_table", table)
    class StartedNewMD(Exception):
        pass
    def run(command, **kwargs):
        assert kwargs["cwd"] == workdir and kwargs["check"]
        assert "in.al_md" in command
        raise StartedNewMD
    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(StartedNewMD):
        module.run_same_potential_md({
            "state_id": np.asarray("new_state"), "r_bohr": np.arange(1., 4.),
            "vii_r_ha": potential, "ion_density_bohr3": .05,
            "zbar_partition": 3., "ti_ev": 1.,
        })


def test_producers_do_not_hardcode_a_historical_otter_version():
    paths = [*PAGES, *(ROOT / "benchmarks/runners").glob("*.py")]
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if (isinstance(key, ast.Constant) and key.value in {"version", "project_version"}
                            and isinstance(value, ast.Constant) and isinstance(value.value, str)):
                        assert not value.value.startswith("0.2."), (path, value.lineno)


def test_ch2_rdf_transform_of_uncorrelated_particles(monkeypatch):
    module = load(ROOT / "tools/reproduce_ch2_hnc_md.py", monkeypatch)
    r = np.arange(.05, 2., .1)
    blocks = np.ones((20, len(r), 3))
    monkeypatch.setattr(module.md, "_read_rdf", lambda *args: (r, blocks))
    k, sii, sem = module.rdf_structure_factors(Path("unused"), {
        "md_particle_counts": np.array([100, 200]), "md_box_length_bohr": 10.})
    np.testing.assert_allclose(sii, np.tile([1., 0., 1.], (k.size, 1)))
    np.testing.assert_allclose(sem, 0.)


def test_ch2_collection_uses_only_new_run_outputs(tmp_path, monkeypatch):
    module = load(ROOT / "tools/reproduce_ch2_hnc_md.py", monkeypatch)
    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(module, "TARGET_TE_EV", (9.,))
    monkeypatch.setattr(module, "TI_OVER_TE", (1.,))
    r, k = np.linspace(.01, 11., 10), np.linspace(.01, 5., 10)
    diagonal = np.tile([1., 0., 1.], (k.size, 1))
    directory = tmp_path / "Te009.000_Ti009.000"
    (directory / "md").mkdir(parents=True)
    np.savez(directory / "hnc_results.npz", r_bohr=r, k_bohr_inv=k,
             gij_r=np.ones((2,2,r.size)),
             sij_k=np.broadcast_to(np.eye(2)[:,:,None], (2,2,k.size)))
    np.savez(directory / "md/md_results.npz", md_r_bohr=r, md_k_bohr_inv=k,
             md_gij_r=np.ones((r.size,3)), md_gij_block_sem=np.zeros((r.size,3)),
             md_sij_k=diagonal, md_sij_frame_sem=np.zeros_like(diagonal),
             md_vectors_per_k_bin=np.full(k.size, 20), md_box_length_bohr=20.)
    (tmp_path / "summary.json").write_text(json.dumps({"cases": [{
        "te_ev":9., "ti_ev":9., "status":"success", "md_energy_quality":"passed",
        "hnc_elapsed_s":1., "md_elapsed_s":60., "timestep_ps":1e-5}]}))
    monkeypatch.setattr(module, "rdf_structure_factors", lambda *args: (k, diagonal, np.zeros_like(diagonal)))
    data = module.collect_results()
    assert data["hnc_gij_r"].shape == (1,3,2001)
    assert data["md_sij_k"].shape == (1,3,k.size)
    for key in ("g_rmse", "s_rmse", "md_sij_estimator_rmse"):
        np.testing.assert_allclose(data[key], 0., atol=1e-15)
