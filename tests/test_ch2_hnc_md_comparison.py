"""Fast checks for the CH2 HNC/MD comparison and Otter-only recomputation."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import pickle
import runpy
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "applications/ch2_xrts_dataset/compare_hnc_md.py"
SPEC = importlib.util.spec_from_file_location("compare_hnc_md", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_campaign_can_select_new_electronic_cache_without_enabling_md(monkeypatch) -> None:
    relative = "applications/ch2_xrts_dataset/outputs/test_run/electronic_cache"
    monkeypatch.setenv("OTTER_CH2_CACHE_DIR", relative)
    monkeypatch.setenv("OTTER_CH2_RUN_MD", "0")
    settings = runpy.run_path(str(SCRIPT))
    assert settings["CACHE_DIR"] == ROOT / relative
    assert settings["available_electronic_temperatures"].__defaults__ == (ROOT / relative,)
    assert settings["RUN_MD"] is False


def test_temperature_selection_uses_nearest_distinct_saved_states() -> None:
    available = [float(value) for value in range(5, 121, 2)]
    assert module.select_temperatures((10.0, 30.0, 100.0), available) == [
        9.0,
        29.0,
        99.0,
    ]


def test_fresh_electronic_mode_preserves_old_cache_and_uses_current_defaults(
    monkeypatch, tmp_path
) -> None:
    old_cache = tmp_path / "old"
    old_cache.mkdir()
    old_path = old_cache / "C1H2_Te009.000.pkl"
    old_path.write_bytes(b"old cache must not be read or overwritten")
    output = tmp_path / "candidate"
    monkeypatch.setattr(module, "CACHE_DIR", old_cache)
    monkeypatch.setattr(module, "OUTPUT_DIR", output)
    monkeypatch.setattr(module, "RECOMPUTE_ELECTRONIC", True)
    result = {"meta": {
        "species": ["C", "H"], "temperature_ev": 9.0,
        "final_electronic_eligible": True,
    }}

    def solve(cfg):
        assert cfg.temperature_ev == 9.0
        assert cfg.ion_temperature_ev is None
        assert cfg.aa_overrides == {}
        return {"electronic": {"kind": "mixture", "result": result}, "runtime_s": 2.0}

    monkeypatch.setattr(module, "solve_plasma_workflow", solve)
    assert module.load_electronic_cache(9.0) == ("mixture", result)
    assert old_path.read_bytes() == b"old cache must not be read or overwritten"
    with (output / "electronic" / old_path.name).open("rb") as stream:
        saved = pickle.load(stream)
    assert saved["electronic_result"] == result
    assert saved["configuration"]["temperature_ev"] == 9.0
    assert saved["runtime_s"] == 2.0


def test_md_composition_and_clock_are_physical() -> None:
    class Prepared:
        n_i = np.asarray([0.01, 0.02])
        zbar = np.asarray([4.0, 1.0])
        r = np.linspace(0.01, 20.0, 256)

    omega, timestep_ps, damping_ps = module.plasma_clock(Prepared())
    assert omega > 0.0
    assert timestep_ps > 0.0
    assert damping_ps > timestep_ps
    assert module.CH2_FORMULA_UNITS * 3 == 3072


def test_high_temperature_md_guard_reduces_step_and_extends_table() -> None:
    class Prepared:
        n_i = np.asarray([0.01, 0.02])
        zbar = np.asarray([4.0, 1.0])
        r = np.linspace(0.01, 20.0, 256)

    cold = module.md_controls(Prepared(), 5.0)
    intermediate = module.md_controls(Prepared(), 29.0)
    hot = module.md_controls(Prepared(), 100.0)
    assert not cold["thermal_limited"]
    assert not intermediate["thermal_limited"]
    assert cold["r_min_bohr"] == module.DEFAULT_MD_R_MIN_BOHR
    assert hot["thermal_limited"]
    assert hot["thermal_step_bohr"] <= module.MAX_THERMAL_STEP_BOHR
    assert hot["timestep_omega_p_inv"] < module.TIMESTEP_OMEGA_P_INV
    assert hot["r_min_bohr"] == module.HOT_MD_R_MIN_BOHR


def test_coulomb_core_repair_removes_ringing_and_preserves_outer_potential() -> None:
    r = np.linspace(0.01, 2.0, 400)
    zbar = np.asarray([4.0, 1.0])
    matrix = np.empty((2, 2, r.size))
    for left in range(2):
        for right in range(2):
            coefficient = zbar[left] * zbar[right]
            ringing = 0.4 * np.sin(80.0 * r) * np.exp(-8.0 * r)
            matrix[left, right] = coefficient / r - 0.3 + ringing

    repaired = module.regularize_coulomb_core(r, matrix, zbar)
    inner = r < module.COULOMB_CORE_BOHR
    outer = r >= module.COULOMB_BLEND_END_BOHR
    for left in range(2):
        for right in range(2):
            coefficient = zbar[left] * zbar[right]
            remainder = repaired[left, right, inner] - coefficient / r[inner]
            np.testing.assert_allclose(remainder, remainder[0], atol=1.0e-12)
            np.testing.assert_allclose(
                repaired[left, right, outer], matrix[left, right, outer]
            )


def test_script_does_not_mislabel_one_component_vmhnc_as_a_ch2_bridge() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert module.BRIDGE_STATUS == "not_implemented_for_multicomponent_ch2"
    assert "scalar hard-sphere bridge" in source
    assert 'hnc_bridge_model="rosenfeld_ashcroft"' not in source
    assert "pair_potentials_from_otter" in source
    assert "run_otter_lammps_md" in source
