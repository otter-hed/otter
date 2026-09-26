"""Grid export contracts; synthetic states avoid expensive AA calculations."""
from concurrent.futures import Future
from dataclasses import replace
import json
import os

import h5py
import numpy as np
import pytest

import otter.grid as grid


def config(ns=1, **kwargs):
    workflow = {"elements": ["C", "H", "O", "N"][:ns], "counts": [1.] * ns}
    return grid.GridConfig(workflow=workflow, temperature_ev=(10., 20.),
                           rho_g_cc=(2.,), k=(.25, .5, 1.), **kwargs)


def result(ns=1, *, native=None, l_count=2, radial_count=3):
    k = np.array([.1, .3, .6, 1.2]) if native is None else np.asarray(native)
    entries, factors = [], []
    for species in range(ns):
        energy = np.full((l_count, radial_count), np.inf)
        energy[0, 0] = -2. - species
        energy[-1, -1] = -.2 - species
        profile = np.zeros((*energy.shape, k.size))
        profile[0, 0] = (species + 1) * (1. + k)
        profile[-1, -1] = (species + 2) * (1. + k)
        entries.append({"element": ["C", "H", "O", "N"][species],
                        "result": {"bound_energy_ha": energy, "bound_l_list": np.arange(l_count),
                                   "factors": profile, "stage2_converged": True,
                                   "ext_status": {"converged": True}, "threshold_state_status": "resolved"}})
        factors.append(profile.sum(axis=(0, 1)))
    return {"species_symbols": [e["element"] for e in entries],
            "electronic": {"species": entries},
            "ion": {"k": k, "r": k, "hnc_converged": True, "hnc_output_residual": 1e-8,
                    "sij_k": np.broadcast_to(1 + k, (ns, ns, k.size)),
                    "q_k": np.broadcast_to(2 + k, (ns, k.size)),
                    "f_k": np.asarray(factors), "zbar": np.arange(ns) + 1.,
                    "zstar": np.arange(ns) + .7}}


@pytest.fixture
def transform(monkeypatch):
    monkeypatch.setattr(grid, "ion_orbital_form_factors", lambda aa, **kw: aa["factors"])


@pytest.mark.parametrize("ns", [1, 2, 3, 4])
def test_species_orbital_mapping_and_sums(tmp_path, transform, ns):
    cfg = config(ns)
    original = result(ns)
    payload = (*grid._pack_result(original, cfg), .2)
    writer = grid._GridFile(tmp_path / "data.h5", cfg, resume=False)
    try:
        writer.start((0, 0, 0))
        writer.success((0, 0, 0), payload)
        f = writer.file
        assert f["S_ii"].shape == (ns, ns, 3, 2, 1, 1)
        # (l=1,n_index=2) is 4p, slot 7, not the third principal shell.
        np.testing.assert_array_equal(f["axis/n"][:], [1, 2, 2, 3, 3, 3, 4, 4])
        np.testing.assert_array_equal(f["axis/l"][:], [0, 0, 1, 0, 1, 2, 0, 1])
        for species in range(ns):
            assert f["E_nl"][species, 0, 0, 0, 0] == -2 - species
            assert f["E_nl"][species, 7, 0, 0, 0] == -.2 - species
            assert f["orbital_present"][species, 0, 0, 0, 0]
            assert not f["orbital_present"][species, 1, 0, 0, 0]
        np.testing.assert_allclose(f["f_nl"][:, :, :, 0, 0, 0].sum(axis=1), f["f"][:, :, 0, 0, 0])
        assert np.isnan(f["E_nl"][:, 1, 0, 0, 0]).all()
        assert np.isnan(f["f"][:, :, 1, 0, 0]).all()
        assert f["status"][1, 0, 0] == grid.PENDING
        for name in ("Z_bar", "Z_star", "S_ii", "f_nl"):
            assert len(f[name].attrs["axis"]) == f[name].ndim
    finally:
        writer.file.close()


def test_dynamic_orbitals_exceed_ten_and_retroactive_zero(tmp_path, transform):
    cfg = config()
    writer = grid._GridFile(tmp_path / "grow.h5", cfg, resume=False)
    try:
        writer.success((0, 0, 0), (*grid._pack_result(result(l_count=1, radial_count=1), cfg), .1))
        writer.success((1, 0, 0), (*grid._pack_result(result(l_count=3, radial_count=4), cfg), .1))
        assert writer.file["f_nl"].shape[1] == 18  # 6d: n=6,l=2 => slot 17
        assert np.all(writer.file["f_nl"][0, 1:, :, 0, 0, 0] == 0)
        assert np.isnan(writer.file["E_nl"][0, 1:, 0, 0, 0]).all()
    finally:
        writer.file.close()


def test_no_bound_states_and_energy_only(tmp_path, transform):
    for outputs in (("f_nl", "E_nl"), ("E_nl",)):
        cfg = config(outputs=outputs)
        res = result()
        res["electronic"]["species"][0]["result"]["bound_energy_ha"][:] = np.inf
        res["electronic"]["species"][0]["result"]["factors"][:] = 0.
        res["ion"]["f_k"][:] = 0.
        writer = grid._GridFile(tmp_path / (outputs[0] + ".h5"), cfg, resume=False)
        try:
            writer.success((0, 0, 0), (*grid._pack_result(res, cfg), .1))
            assert writer.file["axis/n"].size == 0
            assert writer.file["status"][0, 0, 0] == grid.COMPLETE
        finally:
            writer.file.close()


def test_different_native_grids_share_physical_coordinates(transform):
    cfg = config()
    a = grid._pack_result(result(native=[.1, .3, .6, 1.2]), cfg)[0]
    b = grid._pack_result(result(native=[.2, .4, .8, 1.3, 1.8]), cfg)[0]
    for key in ("Sii", "q", "f"):
        np.testing.assert_allclose(a[key], b[key], rtol=1e-14)
    with pytest.raises(ValueError, match="extrapolation"):
        grid._pack_result(result(native=[.4, .6, 1.2]), cfg)


@pytest.mark.parametrize("key,value", [
    ("k", [0., 1.]), ("k", [1., .5]), ("alpha", [1., 1.]),
    ("temperature_ev", [np.nan]), ("rho_g_cc", []), ("outputs", ["bogus"]),
    ("outputs", ["q", "q"]),
])
def test_invalid_grid_inputs(key, value):
    options = dict(workflow={"elements": ["C"]}, temperature_ev=[10.], rho_g_cc=[2.], k=[.2, 1.])
    options[key] = value
    with pytest.raises(ValueError):
        grid.GridConfig(**options)


@pytest.mark.parametrize("change", [
    {"temperature_ev": 10.}, {"save_state_npz": True}, {"allow_unconverged_aa": True},
    {"allow_unconverged_root": True}, {"run_mode": "full"}, {"species_parallel_jobs": 2},
    {"aa_overrides": {"n_jobs": 2}}, {"species_overrides": {"C": {"n_jobs": 2}}},
])
def test_unsafe_workflow_rejected(change):
    with pytest.raises(ValueError):
        grid.GridConfig(workflow={"elements": ["C"], **change}, temperature_ev=[10.], rho_g_cc=[2.], k=[.2])


def test_tf_selection_and_non_equilibrium(transform):
    with pytest.raises(ValueError, match="TF"):
        replace(config(), workflow={"formula": "CH2", "electronic_model": "tf"})
    cfg = replace(config(), workflow={"formula": "CH2", "electronic_model": "tf"},
                  alpha=(1.2,), outputs=("Sii", "q", "f", "Zbar", "Zstar"))
    assert cfg.configuration((0, 0, 0)).ion_temperature_ev == 12.
    assert cfg.composition() == (["C", "H"], [1., 2.])


def test_selection_avoids_orbital_work(monkeypatch):
    monkeypatch.setattr(grid, "ion_orbital_form_factors",
                        lambda *a, **k: pytest.fail("Unrequested orbital calculation"))
    cfg = config(outputs=("q", "Zbar"))
    data, orbitals, _ = grid._pack_result(result(), cfg)
    assert set(data) == {"q", "Zbar"} and not orbitals


def test_reject_invalid_states_and_orbital_sum(transform):
    cfg, res = config(), result()
    res["ion"]["hnc_converged"] = False
    with pytest.raises(ValueError, match="HNC"):
        grid._pack_result(res, cfg)
    res["ion"]["hnc_converged"] = True
    res["ion"]["f_k"] *= 2
    with pytest.raises(ValueError, match="sum"):
        grid._pack_result(res, cfg)


def test_no_overwrite_and_resume_checks(tmp_path, transform):
    path, cfg = tmp_path / "data.h5", config()
    writer = grid._GridFile(path, cfg, resume=False)
    writer.success((0, 0, 0), (*grid._pack_result(result(), cfg), .1))
    writer.failure((1, 0, 0), "test failure", .2)
    writer.file.close()
    with pytest.raises(FileExistsError):
        grid._GridFile(path, cfg, resume=False)
    with pytest.raises(ValueError, match="Cannot resume"):
        grid._GridFile(path, replace(cfg, alpha=(1.2,)), resume=True)
    resumed = grid._GridFile(path, cfg, resume=True)
    try:
        assert resumed.file["status"][0, 0, 0] == grid.COMPLETE
        assert resumed.file["status"][1, 0, 0] == grid.FAILED
        assert np.isnan(resumed.file["q"][:, :, 1, 0, 0]).all()
        assert resumed.file["error"].asstr()[1, 0, 0] == "test failure"
        manifest = json.loads(resumed.file.attrs["manifest_json"])
        assert manifest["effective_workflow"]["species_parallel_jobs"] == 1
        assert manifest["source_sha256"]
        np.testing.assert_array_equal(resumed.file["axis/number_fraction"][:], [1.])
    finally:
        resumed.file.close()


class ImmediatePool:
    """Exercise orchestration without starting plasma calculations."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def __enter__(self):
        assert self.kwargs["mp_context"].get_start_method() == "spawn"
        assert os.environ["OMP_NUM_THREADS"] == "1"
        return self
    def __exit__(self, *exc):
        return False
    def submit(self, fn, *args):
        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future


def test_fail_resume_skip_and_environment(tmp_path, monkeypatch, transform):
    path, cfg, calls = tmp_path / "data.h5", config(), []
    fail = True
    def calculate(config, index, log):
        calls.append(index)
        if index[0] == 1 and fail:
            raise RuntimeError("SCF failed")
        return (*grid._pack_result(result(), config), .1)
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.setattr(grid, "ProcessPoolExecutor", ImmediatePool)
    monkeypatch.setattr(grid, "_calculate_point", calculate)
    summary = grid.run_grid(cfg, path, workers=2)
    assert summary == {"complete": 1, "failed": 1, "total": 2}
    assert os.environ["OMP_NUM_THREADS"] == "7"
    fail = False
    summary = grid.run_grid(cfg, path, workers=2, resume=True)
    assert summary == {"complete": 2, "failed": 0, "total": 2}
    assert calls == [(0, 0, 0), (1, 0, 0), (1, 0, 0)]
    with h5py.File(path) as f:
        assert f.attrs["complete"]
        np.testing.assert_array_equal(f["attempts"][:, 0, 0], [1, 2])
    grid.run_grid(cfg, path, resume=True)
    assert len(calls) == 3


def test_cli_config_and_failure_exit(tmp_path, monkeypatch):
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(dict(workflow={"elements": ["C"]}, temperature_ev=[10.],
                                    rho_g_cc=[2.], k={"min": .2, "max": 1., "points": 4})))
    seen = []
    def run(cfg, filename, **kwargs):
        seen.append(cfg)
        return {"complete": 0, "failed": 1, "total": 1}
    monkeypatch.setattr(grid, "run_grid", run)
    assert grid.main([str(path), str(tmp_path / "out.h5")]) == 1
    assert len(seen[0].k) == 4


def test_public_export_and_optional_dependency():
    import tomllib
    from pathlib import Path
    from tools.export_public_review import in_public_tool_scope
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    assert in_public_tool_scope("tools/grid_calculation.py")
    assert any("h5py" in dep for dep in project["project"]["optional-dependencies"]["grid"])
    assert all("h5py" not in dep for dep in project["project"]["dependencies"])

@pytest.mark.parametrize("status", [grid.PENDING, grid.RUNNING, grid.FAILED])
def test_interrupted_states_are_retried(tmp_path, monkeypatch, transform, status):
    cfg = replace(config(outputs=("q",)), temperature_ev=(10.,))
    path = tmp_path / "resume.h5"
    writer = grid._GridFile(path, cfg, resume=False)
    writer.file["status"][0, 0, 0] = status
    writer.file.close()
    monkeypatch.setattr(grid, "ProcessPoolExecutor", ImmediatePool)
    monkeypatch.setattr(grid, "_calculate_point",
                        lambda config, index, log: (*grid._pack_result(result(), config), .1))
    assert grid.run_grid(cfg, path, resume=True)["complete"] == 1


def test_absent_energy_cannot_hide_nonzero_orbital(transform):
    res = result()
    res["electronic"]["species"][0]["result"]["bound_energy_ha"][0, 0] = np.inf
    with pytest.raises(ValueError, match="sum"):
        grid._pack_result(res, config())


def test_worker_log_contains_solver_progress_and_elapsed(tmp_path, monkeypatch, transform):
    def solve(cfg):
        print("SCF iter 1")
        return result()
    monkeypatch.setattr(grid, "solve_plasma_workflow", solve)
    log = tmp_path / "state.log"
    payload = grid._calculate_point(config(), (0, 0, 0), str(log))
    assert payload[-1] >= 0
    assert "SCF iter 1" in log.read_text()


def test_missing_hdf5_does_not_affect_core_import(tmp_path):
    import subprocess
    import sys
    code = """
import sys
class NoHDF5:
    def find_spec(self, fullname, *args):
        if fullname == 'h5py':
            raise ImportError('HDF5 is intentionally unavailable')
sys.meta_path.insert(0, NoHDF5())
import otter
from otter.grid import GridConfig, _GridFile
c = GridConfig(workflow={'elements': ['C']}, temperature_ev=(10.,), rho_g_cc=(2.,), k=(.2,))
try:
    _GridFile(sys.argv[1], c, resume=False)
except ImportError as e:
    assert 'poetry install -E grid' in str(e)
else:
    raise AssertionError('Missing dependency should fail on export')
"""
    from pathlib import Path
    env = dict(os.environ, PYTHONPATH=str(Path(grid.__file__).parents[1]))
    run = subprocess.run([sys.executable, "-c", code, str(tmp_path / "none.h5")],
                         env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


def test_resume_rejects_changed_source(tmp_path, monkeypatch):
    cfg, path = config(), tmp_path / "source.h5"
    grid._GridFile(path, cfg, resume=False).file.close()
    original = grid._provenance
    monkeypatch.setattr(grid, "_provenance", lambda cfg: {**original(cfg), "source_sha256": "changed"})
    with pytest.raises(ValueError, match="source"):
        grid._GridFile(path, cfg, resume=True)


def test_nonuniform_composition_is_recorded(tmp_path):
    cfg = replace(config(3), workflow={"elements": ["C", "H", "O"], "number_fraction": [.2, .6, .2]})
    writer = grid._GridFile(tmp_path / "composition.h5", cfg, resume=False)
    try:
        np.testing.assert_allclose(writer.file["axis/number_fraction"][:], [.2, .6, .2])
    finally:
        writer.file.close()
