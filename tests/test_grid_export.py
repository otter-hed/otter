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
        np.testing.assert_array_equal(f["axis/orbitals/n"][:], [1, 2, 2, 3, 3, 3, 4, 4])
        np.testing.assert_array_equal(f["axis/orbitals/l"][:], [0, 0, 1, 0, 1, 2, 0, 1])
        for species in range(ns):
            assert f["E_nl"][species, 0, 0, 0, 0] == -2 - species
            assert f["E_nl"][species, 7, 0, 0, 0] == -.2 - species
            assert f["orbitals_present"][species, 0, 0, 0, 0]
            assert not f["orbitals_present"][species, 1, 0, 0, 0]
        np.testing.assert_allclose(f["f_nl"][:, :, :, 0, 0, 0].sum(axis=1), f["f"][:, :, 0, 0, 0])
        assert np.isnan(f["E_nl"][:, 1, 0, 0, 0]).all()
        assert np.isnan(f["f"][:, :, 1, 0, 0]).all()
        assert f["diagnostics/status"][1, 0, 0] == grid.PENDING
        for name in ("Z_bar", "Z_star", "S_ii", "f_nl"):
            assert len(f[name].attrs["axis"]) == f[name].ndim
    finally:
        writer.file.close()


def test_all_radial_and_angular_levels_for_four_species(tmp_path, transform):
    cfg, original = config(4), result(4, l_count=3, radial_count=4)
    for i, entry in enumerate(original["electronic"]["species"]):
        aa = entry["result"]
        aa["bound_energy_ha"][:] = -np.arange(1, 13).reshape(3, 4) - 20*i
        aa["factors"][:] = np.arange(1, 13).reshape(3, 4, 1) * (1 + original["ion"]["k"])
        original["ion"]["f_k"][i] = aa["factors"].sum(axis=(0, 1))
    writer = grid._GridFile(tmp_path / "orbitals.h5", cfg, resume=False)
    try:
        writer.success((0, 0, 0), (*grid._pack_result(original, cfg), .1))
        saved = writer.file
        assert saved["E_nl"].shape[1] == 18
        for i, entry in enumerate(original["electronic"]["species"]):
            for l in range(3):
                for radial_index in range(4):
                    n = radial_index + l + 1
                    slot = n * (n - 1) // 2 + l
                    assert saved["axis/orbitals/n"][slot] == n and saved["axis/orbitals/l"][slot] == l
                    assert saved["E_nl"][i, slot, 0, 0, 0] == entry["result"]["bound_energy_ha"][l, radial_index]
                    assert saved["orbitals_present"][i, slot, 0, 0, 0]
        np.testing.assert_allclose(saved["f_nl"][:, :, :, 0, 0, 0].sum(axis=1), saved["f"][:, :, 0, 0, 0])
    finally:
        writer.file.close()


@pytest.mark.parametrize("wrong", ["species_order", "orbital_shape"])
def test_reject_misaligned_species_and_orbitals(transform, wrong):
    cfg, original = config(3), result(3)
    if wrong == "species_order":
        original["electronic"]["species"].reverse()
    else:
        original["electronic"]["species"][0]["result"]["factors"] = np.zeros((1, 1, 4))
    with pytest.raises(ValueError, match="order|align"):
        grid._pack_result(original, cfg)


def test_scalar_unit_attributes(tmp_path, transform):
    cfg = config(save_native_spectra=True)
    writer = grid._GridFile(tmp_path / "units.h5", cfg, resume=False)
    try:
        writer.success((0, 0, 0), (*grid._pack_result(result(), cfg), .1))
        expected = {"axis/k": "1/a0", "axis/rho": "g/cm^3", "axis/T_e": "eV",
                    "axis/alpha": "1", "E_nl": "hartree",
                    "native/0_0_0/k": "1/a0", "native/0_0_0/q": "1"}
        for name in ("q", "f", "f_nl", "S_ii", "Z_bar", "Z_star"):
            expected[name] = "1"
        for name, unit in expected.items():
            ds = writer.file[name]
            assert isinstance(ds.attrs["unit"], str)
            assert ds.attrs["unit"] == unit
            assert grid._stored_unit(ds) == unit
    finally:
        writer.file.close()


@pytest.mark.parametrize("field,value", [
    ("axis/k", "1/a0"), ("axis/k", ["1/a0"]),
    ("axis/k", np.bytes_("1/a0")), ("axis/k", np.array([b"1/a0"])),
    ("E_nl", "hartree"), ("E_nl", "Ha"), ("E_nl", ["Ha"]),
])
def test_resume_accepts_scalar_and_array_unit_encodings(tmp_path, field, value):
    path, cfg = tmp_path / "unit.h5", config()
    grid._GridFile(path, cfg, resume=False).file.close()
    with h5py.File(path, "r+") as saved:
        saved[field].attrs["unit"] = value
    grid._GridFile(path, cfg, resume=True).file.close()


@pytest.mark.parametrize("value", [[], ["1/a0", "1/angstrom"], [["1/a0"]], 1.])
def test_resume_rejects_malformed_units(tmp_path, value):
    path, cfg = tmp_path / "unit.h5", config()
    grid._GridFile(path, cfg, resume=False).file.close()
    with h5py.File(path, "r+") as saved:
        saved["axis/k"].attrs["unit"] = value
    with pytest.raises(ValueError, match="unit"):
        grid._GridFile(path, cfg, resume=True)


@pytest.mark.parametrize("name", ["axis/elements", "axis/counts", "E_nl", "q"])
def test_resume_rejects_changed_composition_or_data_units(tmp_path, name):
    path, cfg = tmp_path / "composition.h5", config(3)
    grid._GridFile(path, cfg, resume=False).file.close()
    with h5py.File(path, "r+") as saved:
        if name == "axis/elements":
            saved[name][0] = "Be"
        elif name.startswith("axis/"):
            saved[name][:] *= 2
        else:
            saved[name].attrs["unit"] = "eV"
    with pytest.raises(ValueError, match="Cannot resume"):
        grid._GridFile(path, cfg, resume=True)


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
            assert writer.file["axis/orbitals/n"].size == 0
            assert writer.file["diagnostics/status"][0, 0, 0] == grid.COMPLETE
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
    with pytest.raises(ValueError, match="extrapolation"):
        grid._pack_result(result(native=[.1, .3, .6, .9]), cfg)


def test_partial_write_failure_clears_spectra_and_orbitals(tmp_path, transform):
    cfg = config(save_native_spectra=True)
    writer = grid._GridFile(tmp_path / "partial.h5", cfg, resume=False)
    index = (0, 0, 0)
    try:
        # A crash after writing the payload must not leave accepted numbers.
        writer.success(index, (*grid._pack_result(result(), cfg), .1))
        writer.failure(index, "write interrupted", .2)
        assert writer.file["diagnostics/status"][index] == grid.FAILED
        assert not writer.file.attrs["complete"]
        for name in ("q", "f", "S_ii", "Z_bar", "Z_star", "E_nl", "f_nl"):
            assert np.isnan(writer.file[name][(..., *index)]).all()
        assert not writer.file["orbitals_present"][(..., *index)].any()
        assert "native/0_0_0" not in writer.file
        assert writer.file["diagnostics/convergence"].asstr()[index] == ""
    finally:
        writer.file.close()


@pytest.mark.parametrize("ns", [1, 3, 4])
@pytest.mark.parametrize("reverse", [False, True])
def test_hdf5_coordinates_do_not_depend_on_completion_order(tmp_path, transform, ns, reverse):
    cfg = replace(config(ns, save_native_spectra=True), temperature_ev=(10.,), rho_g_cc=(2., 60.))
    # Equal lengths conceal the original bug; different lengths must work too.
    sources = [result(ns, native=[.05, .15, .4, .7, 1.5]),
               result(ns, native=[.2, .3, .5, .9, 1.6, 2.])]
    path = tmp_path / "order.h5"
    writer = grid._GridFile(path, cfg, resume=False)
    try:
        for r in ([1, 0] if reverse else [0, 1]):
            index = (0, r, 0)
            writer.start(index)
            writer.success(index, (*grid._pack_result(sources[r], cfg), .1))
    finally:
        writer.file.close()
    with h5py.File(path, "r") as saved:
        np.testing.assert_array_equal(saved["axis/k"][:], cfg.k)
        for r in (0, 1):
            k = np.asarray(cfg.k)
            np.testing.assert_allclose(saved["S_ii"][:, :, :, 0, r, 0],
                                       np.broadcast_to(1+k, (ns, ns, k.size)))
            np.testing.assert_allclose(saved["q"][:, :, 0, r, 0],
                                       np.broadcast_to(2+k, (ns, k.size)))
            np.testing.assert_allclose(saved["f"][:, :, 0, r, 0],
                                       (2*np.arange(ns)+3)[:, None] * (1+k))
            np.testing.assert_allclose(saved["f_nl"][:, :, :, 0, r, 0].sum(axis=1),
                                       saved["f"][:, :, 0, r, 0])
            native = saved[f"native/0_{r}_0"]
            np.testing.assert_array_equal(native["k"][:], sources[r]["ion"]["k"])
            for name, key in (("f", "f_k"), ("q", "q_k"), ("S_ii", "sij_k")):
                np.testing.assert_array_equal(native[name][:], sources[r]["ion"][key])
                assert native[name].attrs["k_path"] == native.name + "/k"
        for name in ("S_ii", "q", "f", "f_nl"):
            assert saved[name].attrs["k_path"] == "/axis/k"


def test_equal_length_native_grids_are_resampled_not_relabelled(tmp_path, transform):
    cfg = config(save_native_spectra=True)
    low = result(native=[.05, .15, .4, .7, 1.5])
    high = result(native=2.8661288662754503 * low["ion"]["k"])
    a, b = (grid._pack_result(state, cfg)[0] for state in (low, high))
    for name in ("Sii", "q", "f"):
        np.testing.assert_allclose(a[name], b[name], rtol=1e-14)
    # Copying spectra by index instead would produce different physical curves.
    assert not np.allclose(a["native"]["f"], b["native"]["f"])


@pytest.mark.parametrize("native,values,target", [
    ([0., 1.], [1., 2.], [.25]),
    ([-1., 1.], [1., 2.], [.25]),
    ([.1, .1], [1., 2.], [.25]),
    ([.1, np.nan], [1., 2.], [.25]),
    ([.1, 1.], 1., [.25]),
    ([.1, 1.], [1.], [.25]),
    ([.1, 1.], [1., np.inf], [.25]),
    ([.1, 1.], [1., 2.], [np.nan]),
    ([.1, 1.], [1., 2.], [.5, .25]),
    ([.1, 1.], [1., 2.], []),
])
def test_resample_rejects_invalid_coordinates_and_spectra(native, values, target):
    with pytest.raises(ValueError):
        grid._resample(native, values, target)


@pytest.mark.parametrize("field,change", [
    ("axis/k", "values"), ("axis/rho", "values"), ("axis/T_e", "values"),
    ("axis/alpha", "values"), ("axis/k", "unit"), ("q", "k_path"),
])
def test_resume_checks_stored_coordinates_not_only_manifest(tmp_path, field, change):
    path, cfg = tmp_path / "tampered.h5", config()
    grid._GridFile(path, cfg, resume=False).file.close()
    with h5py.File(path, "r+") as stored:
        if change == "values":
            stored[field][:] *= 2
        else:
            stored[field].attrs[change] = "1/angstrom" if change == "unit" else "/native/0_0_0/k"
    with pytest.raises(ValueError, match="Cannot resume"):
        grid._GridFile(path, cfg, resume=True)


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
        assert resumed.file["diagnostics/status"][0, 0, 0] == grid.COMPLETE
        assert resumed.file["diagnostics/status"][1, 0, 0] == grid.FAILED
        assert np.isnan(resumed.file["q"][:, :, 1, 0, 0]).all()
        assert resumed.file["diagnostics/error"].asstr()[1, 0, 0] == "test failure"
        manifest = json.loads(resumed.file.attrs["manifest_json"])
        assert manifest["effective_workflow"]["species_parallel_jobs"] == 1
        assert manifest["source_sha256"]
        np.testing.assert_array_equal(resumed.file["axis/elements"].attrs["number_fraction"], [1.])
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
        np.testing.assert_array_equal(f["diagnostics/attempts"][:, 0, 0], [1, 2])
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
    writer.file["diagnostics/status"][0, 0, 0] = status
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
        np.testing.assert_allclose(writer.file["axis/elements"].attrs["number_fraction"], [.2, .6, .2])
    finally:
        writer.file.close()


@pytest.mark.parametrize("ns", [1, 3])
def test_native_spectra_preserve_each_grid_and_first_charge(tmp_path, transform, ns):
    cfg = config(ns, save_native_spectra=True)
    sources = [result(ns), result(ns, native=[.05, .2, .6, 1.4, 2.])]
    writer = grid._GridFile(tmp_path / "native.h5", cfg, resume=False)
    try:
        for t, source in enumerate(sources):
            writer.success((t, 0, 0), (*grid._pack_result(source, cfg), .1))
            state = writer.file[f"native/{t}_0_0"]
            np.testing.assert_array_equal(state["k"][:], source["ion"]["k"])
            for exported, key in (("f", "f_k"), ("q", "q_k"), ("S_ii", "sij_k")):
                np.testing.assert_array_equal(state[exported][:], source["ion"][key])
                assert len(state[exported].attrs["axis"]) == state[exported].ndim
            np.testing.assert_array_equal(state.attrs["state_index"], [t, 0, 0])
            assert state["k"].attrs["unit"] == "1/a0"
            diagnostics = json.loads(writer.file["diagnostics/convergence"][t, 0, 0])
            np.testing.assert_array_equal(diagnostics["native_first"]["N"],
                                          source["ion"]["f_k"][:, 0] + source["ion"]["q_k"][:, 0])
        np.testing.assert_array_equal(writer.file["axis/k"][:], cfg.k)
        writer.file.attrs["complete"] = True
        writer.failure((1, 0, 0), "failed retry", .1)
        assert not writer.file.attrs["complete"]
        assert "1_0_0" not in writer.file["native"]
        assert "0_0_0" in writer.file["native"]
        writer.success((1, 0, 0), (*grid._pack_result(sources[1], cfg), .1))
        writer.success((1, 0, 0), (*grid._pack_result(sources[0], cfg), .1))
        np.testing.assert_array_equal(writer.file["native/1_0_0/k"][:], sources[0]["ion"]["k"])
    finally:
        writer.file.close()


def test_native_spectra_are_optional_and_respect_selection(tmp_path, transform):
    for enabled in (False, True):
        cfg = config(outputs=("q",), save_native_spectra=enabled)
        data, orbitals, diagnostics = grid._pack_result(result(), cfg)
        assert ("native" in data) == enabled
        if enabled:
            assert set(data["native"]) == {"k", "q"}
        writer = grid._GridFile(tmp_path / f"selection-{enabled}.h5", cfg, resume=False)
        try:
            writer.success((0, 0, 0), (data, orbitals, diagnostics, .1))
            assert ("native" in writer.file) == enabled
        finally:
            writer.file.close()
    with pytest.raises(ValueError, match="boolean"):
        config(save_native_spectra="False")


@pytest.mark.parametrize("ns", [1, 2, 3, 4])
def test_grouped_layout_and_single_orbital_axis(tmp_path, transform, ns):
    cfg = config(ns, save_native_spectra=True)
    writer = grid._GridFile(tmp_path / "layout.h5", cfg, resume=False)
    try:
        writer.success((0, 0, 0), (*grid._pack_result(result(ns), cfg), .1))
        saved = writer.file
        assert saved.attrs["schema"] == "otter_grid_v2"
        assert set(saved["diagnostics"]) == {"status", "attempts", "error", "elapsed_s", "convergence"}
        assert not {"status", "attempts", "error", "elapsed_s", "diagnostics_json", "orbital_present"} & set(saved)
        assert set(saved["axis/orbitals"]) == {"n", "l"}
        assert not {"n", "l", "number_fraction"} & set(saved["axis"])
        for name in ("f_nl", "E_nl", "orbitals_present"):
            ds = saved[name]
            labels = list(ds.attrs["axis"])
            assert labels == ["i", "orbital"] + (["k"] if name == "f_nl" else []) + ["T_e", "rho", "alpha"]
            assert len(labels) == ds.ndim
            orbital_axis = saved[ds.attrs["orbital_path"]]
            assert orbital_axis["n"].size == orbital_axis["l"].size == ds.shape[1]
        for name in ("q", "f", "S_ii", "f_nl", "native/0_0_0/q"):
            ds = saved[name]
            k = saved[ds.attrs["k_path"]][:]
            assert k.size == ds.shape[list(ds.attrs["axis"]).index("k")]
    finally:
        writer.file.close()
    grid._GridFile(tmp_path / "layout.h5", cfg, resume=True).file.close()


@pytest.mark.parametrize("value", [None, [.2, .3, .5], [1.], [.2, np.nan, .8], [[1/3]*3]])
def test_resume_validates_number_fraction_attribute(tmp_path, value):
    cfg, path = config(3), tmp_path / "fractions.h5"
    grid._GridFile(path, cfg, resume=False).file.close()
    with h5py.File(path, "r+") as saved:
        del saved["axis/elements"].attrs["number_fraction"]
        if value is not None:
            saved["axis/elements"].attrs["number_fraction"] = value
    before = path.read_bytes()
    with pytest.raises(ValueError, match="number_fraction"):
        grid._GridFile(path, cfg, resume=True)
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ["missing_diagnostics", "status_values", "status_axis", "orbital_numbers", "orbital_axis", "orbital_path"])
def test_resume_rejects_malformed_layout(tmp_path, transform, change):
    cfg, path = config(), tmp_path / "layout.h5"
    writer = grid._GridFile(path, cfg, resume=False)
    writer.success((0, 0, 0), (*grid._pack_result(result(), cfg), .1))
    writer.file.close()
    with h5py.File(path, "r+") as saved:
        if change == "missing_diagnostics":
            del saved["diagnostics/attempts"]
        elif change == "status_values":
            saved["diagnostics/status"][0, 0, 0] = 4
        elif change == "status_axis":
            saved["diagnostics/status"].attrs["axis"] = ["T_e", "alpha", "rho"]
        elif change == "orbital_numbers":
            saved["axis/orbitals/l"][0] = 1
        elif change == "orbital_axis":
            saved["orbitals_present"].attrs["axis"] = ["i", "orbitals/n", "orbitals/lT_e", "rho", "alpha"]
        else:
            saved["f_nl"].attrs["orbital_path"] = "/axis/n"
    with pytest.raises(ValueError, match="Cannot resume"):
        grid._GridFile(path, cfg, resume=True)


@pytest.mark.parametrize("schema", [None, "otter_grid_v1", "unknown"])
def test_resume_rejects_other_schemas_without_rewriting(tmp_path, schema):
    cfg, path = config(), tmp_path / "legacy.h5"
    grid._GridFile(path, cfg, resume=False).file.close()
    with h5py.File(path, "r+") as saved:
        del saved.attrs["schema"]
        if schema is not None:
            saved.attrs["schema"] = schema
    before = path.read_bytes()
    with pytest.raises(ValueError, match="schema.*new output filename"):
        grid._GridFile(path, cfg, resume=True)
    assert path.read_bytes() == before


@pytest.mark.parametrize("outputs", [("f_nl",), ("E_nl",), ("f_nl", "E_nl")])
def test_orbital_presence_with_optional_energy_output(tmp_path, transform, outputs):
    cfg, path = config(outputs=outputs), tmp_path / "orbitals.h5"
    original = result()
    # A zero-energy level is still present; zero form factor is not an absence test.
    original["electronic"]["species"][0]["result"]["bound_energy_ha"][0, 0] = 0.
    original["electronic"]["species"][0]["result"]["factors"][0, 0] = 0.
    original["ion"]["f_k"][0] = original["electronic"]["species"][0]["result"]["factors"].sum(axis=(0, 1))
    writer = grid._GridFile(path, cfg, resume=False)
    try:
        writer.success((0, 0, 0), (*grid._pack_result(original, cfg), .1))
        assert writer.file["orbitals_present"][0, 0, 0, 0, 0]
        assert not writer.file["orbitals_present"][0, 1, 0, 0, 0]
    finally:
        writer.file.close()
    grid._GridFile(path, cfg, resume=True).file.close()


def test_workflow_overrides_are_forwarded_and_recorded(tmp_path):
    workflow = {"formula": "CH2", "qoz_linear_n_points": 8192,
                "aa_overrides": {"n_points": 8192},
                "species_overrides": {"C": {"bound_zero_tail_refine": True}}}
    cfg = replace(config(), workflow=workflow, alpha=(.5, 1.2))
    for index in np.ndindex(cfg.shape):
        resolved = cfg.configuration(index)
        assert resolved.aa_overrides == workflow["aa_overrides"]
        assert resolved.species_overrides == workflow["species_overrides"]
        assert resolved.qoz_linear_n_points == 8192
        assert resolved.ion_temperature_ev == cfg.temperature_ev[index[0]] * cfg.alpha[index[2]]
    writer = grid._GridFile(tmp_path / "overrides.h5", cfg, resume=False)
    try:
        effective = json.loads(writer.file.attrs["manifest_json"])["effective_workflow"]
        assert effective["aa_overrides"] == workflow["aa_overrides"]
        assert effective["species_overrides"] == workflow["species_overrides"]
        assert workflow == {"formula": "CH2", "qoz_linear_n_points": 8192,
                            "aa_overrides": {"n_points": 8192},
                            "species_overrides": {"C": {"bound_zero_tail_refine": True}}}
    finally:
        writer.file.close()


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("complete", [False, True])
def test_documented_legacy_reader(tmp_path, monkeypatch, version, complete):
    import re
    import textwrap
    from pathlib import Path

    guide = (Path(__file__).resolve().parents[1] /
             "docs/source/user_guide/parameter_grids.rst").read_text()
    blocks = re.findall(r"^\.\. code-block:: python\n\n((?: {3}[^\n]*\n|\n)+)", guide, re.M)
    example = textwrap.dedent(next(block for block in blocks if "def unit_string" in block))
    monkeypatch.chdir(tmp_path)
    with h5py.File("carbon_grid.h5", "w") as saved:
        saved.attrs["schema"] = f"otter_grid_v{version}"
        saved.create_dataset("status" if version == 1 else "diagnostics/status",
                             data=np.full((1, 1, 1), grid.COMPLETE if complete else grid.FAILED))
        saved.create_dataset("axis/k", data=[.25, .5]).attrs["unit"] = ["1/a0"] if version == 1 else "1/a0"
        saved.create_dataset("E_nl", data=[-1.]).attrs["unit"] = ["Ha"] if version == 1 else "hartree"
    namespace = {}
    if complete:
        exec(compile(example, "parameter_grids.rst", "exec"), namespace)
        assert namespace["k_unit"] == "1/a0"
        with h5py.File("carbon_grid.h5") as saved:
            assert namespace["unit_string"](saved["E_nl"]) == "hartree"
    else:
        with pytest.raises(RuntimeError, match="unfinished"):
            exec(compile(example, "parameter_grids.rst", "exec"), namespace)
