"""Offline integrity and physics checks for the ion-structure gallery data."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
LIBRARY_DIR = ROOT / "benchmarks" / "baselines" / "ion_structure_library"
AL_DIR = ROOT / "benchmarks" / "baselines" / "al_full_workflow_1ev"
REFERENCE_DIR = ROOT / "benchmarks" / "reference_data" / "ion_structure_library"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_portable_archive(path: Path) -> None:
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            value = archive[key]
            assert not value.dtype.hasobject, (path, key)
            if value.dtype.kind in "fiu":
                assert np.all(np.isfinite(value)), (path, key)
            if value.dtype.kind in "SU":
                text = " ".join(str(item) for item in value.reshape(-1))
                assert "/home/" not in text
                assert "/tmp/" not in text


def test_library_manifest_hashes_and_portable_archives() -> None:
    manifest = _json(LIBRARY_DIR / "manifest.json")
    assert manifest["schema_version"] == "otter_benchmark_manifest_v1"
    assert manifest["benchmark_id"] == "ion_structure_library"
    assert manifest["producer"]["project"] == "Otter"
    assert len(manifest["producer"]["git_commit"]) == 40
    assert len(manifest["states"]) == 7
    assert manifest["data_rights"]["reference_redistribution_status"] == (
        "published_by_maintainer_with_attribution"
    )
    assert manifest["data_rights"]["public_release_gate"] == "resolved"
    carbon_state = next(item for item in manifest["states"] if item["element"] == "C")
    assert carbon_state["reference_family"] == ("StarrettPrivateCommunication")
    serialized = json.dumps(manifest)
    assert "/home/" not in serialized
    assert "/tmp/" not in serialized

    for item in manifest["states"]:
        relative = Path(item["baseline_file"])
        assert not relative.is_absolute()
        path = (LIBRARY_DIR / relative).resolve()
        assert path.parent == LIBRARY_DIR.resolve()
        assert _sha256(path) == item["baseline_sha256"]
        _assert_portable_archive(path)
        with np.load(path, allow_pickle=False) as archive:
            assert archive["schema_version"].item() in {
                "otter_ion_structure_library_state_v1",
                "otter_ion_structure_library_state_v2",
            }
            assert archive["state_id"].item() == item["state_id"]
            expected_commit = item.get(
                "producer_git_commit", manifest["producer"]["git_commit"]
            )
            assert archive["otter_git_commit"].item() == expected_commit
            signature = json.loads(str(archive["producer_signature_json"].item()))
            resolved = signature.get("resolved_configuration", signature)
            expected_model = item.get("electronic_model", "qm")
            assert resolved["electronic_model"] == expected_model
            if "electronic_model" in archive:
                assert archive["electronic_model"].item() == expected_model
            aa = resolved.get("aa", resolved.get("aa_overrides", {}))
            assert aa.get("bound_occ_mode", "fd") == "fd"
            assert aa.get("b3_tail_model", "full") == "full"
            assert resolved.get(
                "qoz_response_chi0_model",
                resolved.get("qoz", {}).get("chi0_model"),
            ) == "lindhard_fd"
            assert resolved.get(
                "qoz_response_lfc_model",
                resolved.get("qoz", {}).get("lfc_model"),
            ) == "chabrier1990"


def test_library_grid_charge_and_convergence_invariants() -> None:
    manifest = _json(LIBRARY_DIR / "manifest.json")
    for item in manifest["states"]:
        path = LIBRARY_DIR / item["baseline_file"]
        with np.load(path, allow_pickle=False) as archive:
            r = np.asarray(archive["r_bohr"], dtype=float)
            k = np.asarray(archive["k_bohr_inv"], dtype=float)
            assert np.all(np.diff(r) > 0.0)
            assert np.all(np.diff(k) > 0.0)
            assert r[-1] <= 20.0
            assert k[-1] <= 20.0
            assert archive["gii_r"].shape == r.shape
            assert archive["sii_k"].shape == k.shape
            schema = str(archive["schema_version"].item())
            if schema == "otter_ion_structure_library_state_v1":
                r_e = np.asarray(archive["r_e_bohr"], dtype=float)
                assert np.all(np.diff(r_e) > 0.0)
                assert r_e[-1] <= 20.0
                for key in (
                    "n_full_bohr3",
                    "n_free_bohr3",
                    "n_bound_bohr3",
                    "n_ext_bohr3",
                    "n_ion_bohr3",
                    "n_pa_bohr3",
                    "n_scr_bohr3",
                    "v_full_ha",
                    "v_ext_ha",
                    "v_hartree_ha",
                    "v_xc_ha",
                ):
                    assert archive[key].shape == r_e.shape
                assert archive["vii_r_ha"].shape == r.shape
                reciprocal = (
                    "vii_k_ha_bohr3",
                    "n_scr_k_electrons",
                    "chi0_k_bohr3_per_ha",
                    "gee_k",
                )
            else:
                reciprocal = ()
                excluded = {
                    "r_e_bohr",
                    "n_full_bohr3",
                    "n_scr_bohr3",
                    "v_full_ha",
                    "vii_r_ha",
                    "vii_k_ha_bohr3",
                    "n_scr_k_electrons",
                    "chi0_k_bohr3_per_ha",
                    "gee_k",
                    "f_k",
                    "q_k",
                }
                assert excluded.isdisjoint(archive.files)
            for key in reciprocal:
                assert archive[key].shape == k.shape
            assert float(archive["hnc_best_residual"]) <= 1.0e-4
            assert float(archive["hnc_closure_mismatch"]) <= float(
                archive["hnc_closure_tolerance"]
            )
            np.testing.assert_allclose(
                float(archive["zbar_qoz"]),
                float(archive["zbar_partition"]),
                rtol=0.0,
                atol=1.0e-10,
            )
            np.testing.assert_allclose(
                float(archive["q_scr_raw"]),
                float(archive["zbar_partition"]),
                rtol=0.0,
                atol=7.0e-3,
            )


def test_selective_ion_library_producer_keeps_only_analysis_fields() -> None:
    producer = _load_module(
        "otter_ion_structure_library_selective_test",
        ROOT / "benchmarks" / "runners" / "regenerate_ion_structure_library.py",
    )
    r = np.linspace(0.01, 25.0, 128)
    k = np.linspace(0.02, 25.0, 128)
    workflow = {
        "electronic": {
            "result": {
                "n0": 0.1,
                "r_ws": 2.0,
                "mu": -0.2,
                "zbar": 3.0,
                "threshold_state_status": "none",
                "threshold_state_representation": "none",
            }
        },
        "ion": {
            "r": r,
            "k": k,
            "gii_r": np.ones_like(r),
            "sii_k": np.ones_like(k),
            "f_k": np.exp(-k),
            "q_k": 3.0 * np.exp(-k),
            "zbar_partition": 3.0,
            "zbar_qoz": 3.0,
            "n_i": 0.01,
            "zbar_screening_integral_raw": 3.0,
            "hnc_best_residual": 1.0e-7,
            "closure_transform_max_abs": 1.0e-5,
            "closure_transform_tol": 1.0e-3,
            "hnc_iters": 12,
        },
    }
    state = {
        "state_id": "synthetic",
        "element": "Al",
        "rho_g_cc": 2.7,
        "te_ev": 5.0,
        "ti_ev": 5.0,
    }
    payload = producer._pack_result(workflow, state, elapsed_s=1.0)
    assert payload["schema_version"].item() == ("otter_ion_structure_library_state_v2")
    assert payload["storage_profile"].item() == "benchmark_analysis"
    assert {"gii_r", "sii_k"}.issubset(payload)
    assert {
        "n_full_bohr3",
        "v_full_ha",
        "vii_r_ha",
        "chi0_k_bohr3_per_ha",
        "f_k",
        "q_k",
        "vmhnc_r_bohr",
        "vmhnc_k_bohr_inv",
    }.isdisjoint(payload)


def test_wunsch_be_contains_hnc_vmhnc_and_same_potential_md() -> None:
    path = LIBRARY_DIR / "be_wunsch_rho5p544_te13_ti13.npz"
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "gii_r",
            "sii_k",
            "vmhnc_gii_r",
            "vmhnc_sii_k",
            "vmhnc_eta",
            "vmhnc_variational_residual",
            "md_gii_r",
            "md_gii_block_sem",
            "md_sii_k",
            "md_sii_frame_sem",
            "md_type_pairs",
            "md_nve_relative_energy_drift",
        }
        assert required.issubset(archive.files)
        assert {
            "vmhnc_r_bohr",
            "vmhnc_k_bohr_inv",
            "md_gij_r",
            "md_gij_block_sem",
            "md_snn_k",
            "md_snn_frame_sem",
            "md_sij_k",
            "md_sij_frame_sem",
            "md_vectors_per_k_bin",
        }.isdisjoint(archive.files)
        assert archive["md_type_pairs"].tolist() == [[1, 1]]
        assert np.all(archive["md_gii_block_sem"] >= 0.0)
        assert np.all(archive["md_sii_frame_sem"] >= 0.0)
        assert 0.05 <= float(archive["vmhnc_eta"]) <= 0.49


def test_reference_manifest_hashes_and_release_decision() -> None:
    manifest = _json(REFERENCE_DIR / "manifest.json")
    assert manifest["schema_version"] == "otter_reference_manifest_v1"
    assert manifest["redistribution_status"] == (
        "published_by_maintainer_with_attribution"
    )
    assert manifest["license_declared"] == "NOASSERTION"
    assert manifest["public_release_gate"] == "resolved"
    carbon = next(
        item
        for item in manifest["files"]
        if item["path"].endswith("gii_C_20gcc_50.0ev_starrett.csv")
    )
    assert carbon["origin_type"] == "author_provided_private_numerical_data"
    assert "C. E. Starrett" in carbon["attribution"]
    assert "unpublished" in carbon["attribution"]
    for item in manifest["files"]:
        relative = Path(item["path"])
        assert not relative.is_absolute()
        path = (REFERENCE_DIR / relative).resolve()
        assert path.is_relative_to(REFERENCE_DIR.resolve())
        assert _sha256(path) == item["sha256"]
        values = np.genfromtxt(path, delimiter=",", comments="#")
        assert values.ndim == 2
        assert values.shape[1] >= 2
        assert np.all(np.isfinite(values[:, :2]))


def test_offline_library_runner_recomputes_metrics() -> None:
    runner_path = ROOT / "benchmarks" / "runners" / "plot_ion_structure_library.py"
    source = runner_path.read_text(encoding="utf-8")
    assert "solve_plasma_workflow" not in source
    assert "from otter" not in source
    runner = _load_module("otter_library_offline_test", runner_path)
    states = runner.load_states(runner.load_manifest())
    rows = runner.evaluate(states)
    assert len(rows) == 34
    primary_rows = [row for row in rows if row["role"] == "primary"]
    assert len(primary_rows) == 12
    for row in rows:
        assert row["n_points"] > 5
        assert np.isfinite(row["rmse"])
        assert np.isfinite(row["mae"])
        assert np.isfinite(row["max_abs"])
    carbon = next(
        row for row in rows if row["state_id"] == "c_starrett_rho20_te50_ti50"
    )
    assert carbon["rmse"] < 0.01


def test_reference_coordinate_conversions_match_source_plot_scripts() -> None:
    runner = _load_module(
        "otter_library_units_test",
        ROOT / "benchmarks" / "runners" / "plot_ion_structure_library.py",
    )
    state = {
        "k_bohr_inv": np.asarray([1.0]),
        "sii_k": np.asarray([0.25]),
        "r_bohr": np.asarray([1.0]),
        "gii_r": np.asarray([0.75]),
    }
    k_angstrom, _ = runner._otter_curve(
        state,
        "sii",
        "angstrom^-1",
    )
    r_angstrom, _ = runner._otter_curve(state, "gii", "angstrom")
    r_bohr, _ = runner._otter_curve(state, "gii", "bohr")
    np.testing.assert_allclose(
        k_angstrom,
        [1.0 / runner.otter_constants.BOHR_TO_ANGSTROM],
    )
    np.testing.assert_allclose(
        r_angstrom,
        [runner.otter_constants.BOHR_TO_ANGSTROM],
    )
    np.testing.assert_allclose(r_bohr, [1.0])
    md_state = {
        "md_k_bohr_inv": np.asarray([0.2, 0.3]),
        "md_sii_k": np.asarray([0.4, 0.5]),
        "md_sii_vectors_per_bin": np.asarray([3, 6]),
    }
    md_k, md_sii = runner._otter_curve(
        md_state,
        "sii",
        "bohr^-1",
        "md_",
    )
    np.testing.assert_allclose(md_k, [0.3])
    np.testing.assert_allclose(md_sii, [0.5])
    assert {
        series["x_unit"]
        for series in runner.REFERENCE_SERIES["be_wunsch_rho5p544_te13_ti13"]
    } == {"angstrom^-1", "angstrom"}
    assert {
        series["x_unit"]
        for series in runner.REFERENCE_SERIES["c_starrett_rho20_te50_ti50"]
    } == {"bohr"}


def test_complete_al_workflow_manifest_levels_and_pipeline() -> None:
    manifest = _json(AL_DIR / "manifest.json")
    assert manifest["schema_version"] == "otter_benchmark_manifest_v1"
    assert manifest["benchmark_id"] == "al_full_workflow_1ev"
    assert str(manifest["status"]).startswith("accepted")
    assert manifest["producer"]["project"] == "Otter"
    assert manifest["producer"]["worktree_clean_at_generation"] is False
    assert len(manifest["producer"]["script_sha256_at_generation"]) == 64
    assert manifest["producer"]["script_sha256_current"] == _sha256(
        ROOT / manifest["producer"]["script_relative_path"]
    )
    assert manifest["producer"]["script_relative_path"] == (
        "benchmarks/runners/regenerate_al_full_workflow.py"
    )
    assert manifest["configuration"]["bound_occ_mode"] == "fd"
    assert manifest["configuration"]["bound_rmax_mult"] is None
    assert manifest["configuration"]["bound_zero_tail_refine"] is False
    assert manifest["configuration"]["b3_tail_model"] == "full"
    assert manifest["configuration"]["qoz_zbar_mode"] == ("pseudoatom_partition")
    assert manifest["configuration"]["qoz_renormalize_nscr_to_zbar"] is True
    audit = manifest["scientific_audit"]
    assert audit["q_scr_used"] == pytest.approx(audit["zbar_qoz"])
    assert audit["hnc_best_residual"] <= manifest["configuration"]["hnc_tolerance"]
    assert audit["hnc_closure_mismatch"] <= (
        manifest["configuration"]["hnc_transform_closure_tolerance"]
    )
    item = manifest["state"]
    path = AL_DIR / item["data_file"]
    assert _sha256(path) == item["data_sha256"]
    _assert_portable_archive(path)
    runner = _load_module(
        "otter_al_workflow_offline_test",
        ROOT / "benchmarks" / "runners" / "plot_al_full_workflow.py",
    )
    state = runner.load_state()
    rows = runner.bound_level_rows(state)
    assert [row["level"] for row in rows] == ["1s", "2s", "2p"]
    np.testing.assert_allclose(
        [row["occupation"] for row in rows],
        [2.0, 2.0, 6.0],
        rtol=0.0,
        atol=1.0e-8,
    )
    assert float(state["hnc_best_residual"]) <= 1.0e-4
    assert float(state["hnc_closure_mismatch"]) <= float(state["hnc_closure_tolerance"])
    assert np.max(state["r_e_bohr"]) <= 20.0
    assert np.max(state["r_bohr"]) <= 20.0
    assert np.max(state["k_bohr_inv"]) <= 20.0
    assert state["schema_version"].item() == "otter_al_full_workflow_v2"
    assert state["storage_profile"].item() == "gallery_analysis"
    assert state["n_ion_k_electrons"].shape == state["k_bohr_inv"].shape
    assert np.all(np.isfinite(state["n_ion_k_electrons"]))
    assert state["n_scr_k_electrons"].shape == state["k_bohr_inv"].shape
    assert {
        "n_cont_bohr3",
        "n_free_bohr3",
        "n_ion_bohr3",
        "n_scr_k_raw_electrons",
        "chi0_k_bohr3_per_ha",
        "gee_k",
    }.isdisjoint(state)
    assert float(state["q_scr_used"]) == pytest.approx(float(state["zbar_qoz"]))
    assert np.all(np.isfinite(state["n_scr_k_electrons"]))
