"""Integrity, attribution, and release policy for comparison datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from otter.numerics import BOHR_TO_ANGSTROM, EV_TO_KELVIN


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "benchmark_id",
    (
        "johnson_et_al_2025_two_temperature_al",
        "argha_roy_carbon_sii",
    ),
)
def test_reference_files_are_checksummed_and_release_decision_is_recorded(
    benchmark_id: str,
) -> None:
    directory = ROOT / "benchmarks" / "reference_data" / benchmark_id
    manifest = _json(directory / "manifest.json")

    assert manifest["schema_version"] == "otter_reference_manifest_v1"
    assert manifest["reference_id"] == benchmark_id
    assert manifest["redistribution_status"] == (
        "published_by_maintainer_with_attribution"
    )
    assert manifest["license_declared"] == "NOASSERTION"
    assert manifest["public_release_gate"] == "resolved"
    assert manifest["release_decision"]["decision_date"] == "2026-08-10"
    assert "does not assert" in manifest["release_decision"]["rights_note"]
    assert manifest["files"]

    for record in manifest["files"]:
        path = directory / str(record["path"])
        assert path.is_file()
        assert _sha256(path) == record["sha256"]


def test_johnson_units_and_accepted_baselines_are_explicit() -> None:
    benchmark_id = "johnson_et_al_2025_two_temperature_al"
    reference = _json(
        ROOT / "benchmarks" / "reference_data" / benchmark_id / "manifest.json"
    )
    assert reference["publication"]["doi"] == "10.1103/5c29-kdx1"
    assert reference["units"]["column_1"] == "Bohr"
    assert "r [au]" in reference["unit_audit"]["source"]
    assert len(reference["files"]) == 12
    assert sorted({float(record["te_ev"]) for record in reference["files"]}) == [
        1.0,
        3.0,
        10.0,
        30.0,
    ]

    directory = ROOT / "benchmarks" / "baselines" / benchmark_id
    manifest = _json(directory / "manifest.json")
    controller = (
        ROOT
        / "benchmarks"
        / "examples"
        / "plot_johnson_et_al_2025_two_temperature_al.py"
    )
    assert manifest["producer"]["current_controller_sha256"] == _sha256(
        controller
    )
    statuses = {record["state_id"]: record["status"] for record in manifest["states"]}
    assert statuses == {
        "al_rho2p7_te1_ti1": "accepted",
        "al_rho2p7_te3_ti1": "accepted",
        "al_rho2p7_te10_ti1": "accepted",
        "al_rho2p7_te30_ti1": "accepted",
    }
    controller = (
        ROOT
        / str(manifest["producer"]["script_relative_path"])
    )
    assert _sha256(controller) == manifest["producer"]["current_controller_sha256"]
    _check_accepted_archives(directory, manifest)


def test_johnson_baselines_compare_hnc_and_vmhnc_with_dft_md() -> None:
    """Every panel must carry two audited IS closures and a real DFT-MD overlap."""
    benchmark_id = "johnson_et_al_2025_two_temperature_al"
    baseline_dir = ROOT / "benchmarks" / "baselines" / benchmark_id
    reference_dir = ROOT / "benchmarks" / "reference_data" / benchmark_id
    manifest = _json(baseline_dir / "manifest.json")

    for record in manifest["states"]:
        with np.load(
            baseline_dir / str(record["baseline_file"]),
            allow_pickle=False,
        ) as archive:
            required = {
                "r_bohr",
                "gii_r",
                "vmhnc_gii_r",
                "vmhnc_eta",
                "vmhnc_variational_residual",
                "vmhnc_hnc_best_residual",
                "vmhnc_hnc_closure_mismatch",
                "md_r_bohr",
                "md_gii_r",
                "md_gii_block_sem",
                "md_nve_relative_energy_drift",
            }
            assert required <= set(archive.files)
            assert {
                "k_bohr_inv",
                "sii_k",
                "vmhnc_r_bohr",
                "vmhnc_k_bohr_inv",
                "vmhnc_sii_k",
                "vii_r_ha",
                "ion_density_bohr3",
                "md_k_bohr_inv",
                "md_sii_k",
                "md_sii_block_sem",
                "md_sii_vectors_per_bin",
            }.isdisjoint(archive.files)
            assert str(archive["structure_model"].item()) == "IS"
            assert str(archive["hnc_bridge_model"].item()) == "none"
            assert (
                str(archive["vmhnc_hnc_bridge_model"].item())
                == "rosenfeld_ashcroft"
            )
            assert 0.0 < float(archive["vmhnc_eta"]) < 0.5
            assert abs(float(archive["vmhnc_variational_residual"])) <= 3.0e-5
            assert float(archive["vmhnc_hnc_best_residual"]) <= 1.0e-4
            assert float(archive["vmhnc_hnc_closure_mismatch"]) <= 2.5e-3

            r_md = np.asarray(archive["md_r_bohr"], dtype=float)
            g_md = np.asarray(archive["md_gii_r"], dtype=float)
            sem_md = np.asarray(archive["md_gii_block_sem"], dtype=float)
            assert r_md.shape == g_md.shape == sem_md.shape
            assert np.all(np.isfinite(g_md))
            assert np.all(np.isfinite(sem_md))
            assert np.all(sem_md >= 0.0)
            assert int(archive["md_atoms"]) == 2048
            assert int(archive["md_rdf_blocks"]) == 20
            assert str(archive["md_ensemble_sequence"].item()) == "NVT->NVE"
            assert float(archive["md_timestep_omega_p_inv"]) == pytest.approx(
                5.0e-3
            )
            assert abs(float(archive["md_nve_relative_energy_drift"])) < 5.0e-5
            assert float(archive["md_nve_mean_temperature_k"]) == pytest.approx(
                EV_TO_KELVIN,
                rel=0.2,
            )

            r_hnc = np.asarray(archive["r_bohr"], dtype=float)
            g_hnc = np.asarray(archive["gii_r"], dtype=float)
            g_vmhnc = np.asarray(archive["vmhnc_gii_r"], dtype=float)
            closure_delta = g_hnc - g_vmhnc
            assert np.all(np.isfinite(closure_delta))
            assert float(np.max(np.abs(closure_delta))) > 1.0e-4

            md_overlap = (r_md >= r_hnc[0]) & (r_md <= r_hnc[-1])
            md_hnc_delta = g_md[md_overlap] - np.interp(
                r_md[md_overlap], r_hnc, g_hnc
            )
            md_vmhnc_delta = g_md[md_overlap] - np.interp(
                r_md[md_overlap], r_hnc, g_vmhnc
            )
            hnc_md_rmse = float(np.sqrt(np.mean(md_hnc_delta**2)))
            vmhnc_md_rmse = float(np.sqrt(np.mean(md_vmhnc_delta**2)))
            assert vmhnc_md_rmse < hnc_md_rmse
            assert vmhnc_md_rmse < 2.5e-2

            dft_path = reference_dir / (
                f"Zak_2025_Al_rho2.7_Te{float(record['te_ev']):.1f}_"
                "Ti1.0_DFTMD.csv"
            )
            dft = np.asarray(np.genfromtxt(dft_path, delimiter=","), dtype=float)
            dft = dft[np.all(np.isfinite(dft[:, :2]), axis=1), :2]
            overlap = (dft[:, 0] >= r_hnc[0]) & (dft[:, 0] <= r_hnc[-1])
            for radius, pair_distribution in (
                (r_hnc, g_hnc),
                (r_hnc, g_vmhnc),
            ):
                delta = (
                    np.interp(dft[overlap, 0], radius, pair_distribution)
                    - dft[overlap, 1]
                )
                assert delta.size > 0
                assert np.isfinite(np.sqrt(np.mean(delta**2)))


def test_argha_attribution_uncertainty_and_current_otter_states_are_explicit() -> None:
    benchmark_id = "argha_roy_carbon_sii"
    reference = _json(
        ROOT / "benchmarks" / "reference_data" / benchmark_id / "manifest.json"
    )
    assert reference["origin_type"] == "author_provided_private_numerical_data"
    assert "Dr. Argha Roy" in reference["attribution"]
    assert "DFT-MD" in reference["attribution"]
    assert reference["method_label"] == "DFT-MD"
    assert reference["columns"][2] == "reported_uncertainty_dimensionless"
    assert len(reference["files"]) == 6

    directory = ROOT / "benchmarks" / "baselines" / benchmark_id
    manifest = _json(directory / "manifest.json")
    controller = (
        ROOT
        / "benchmarks"
        / "examples"
        / "plot_argha_roy_carbon_sii.py"
    )
    assert manifest["producer"]["current_controller_sha256"] == _sha256(
        controller
    )
    assert [record["te_ev"] for record in manifest["states"]] == [
        20.0,
        30.0,
        40.0,
        50.0,
        100.0,
    ]
    assert all(record["status"] == "accepted" for record in manifest["states"])
    assert all(
        record["threshold_state_status"] in {"resolved", "marginal"}
        for record in manifest["states"]
    )
    assert manifest["configuration"]["bound_zero_tail_refine"] is True
    assert manifest["configuration"]["bound_rmax_mult"] is None
    controller = (
        ROOT
        / str(manifest["producer"]["script_relative_path"])
    )
    assert _sha256(controller) == manifest["producer"]["current_controller_sha256"]
    _check_accepted_archives(directory, manifest)


def test_schorner_bridge_md_baselines_and_ordinate_correction() -> None:
    """The corrected DFT-MD overlay must carry two closures and direct MD S(k)."""
    benchmark_id = "schorner_et_al_2022_al_sii"
    reference_dir = ROOT / "benchmarks" / "reference_data" / benchmark_id
    reference = _json(reference_dir / "manifest.json")
    assert reference["publication"]["doi"] == "10.1103/PhysRevB.105.174310"
    assert reference["publication"]["figure"] == "Figure 2"
    assert reference["license_declared"] == "NOASSERTION"
    assert reference["release_decision"]["decision_date"] == "2026-08-23"

    file_record = reference["files"][0]
    csv_path = reference_dir / str(file_record["path"])
    assert _sha256(csv_path) == file_record["sha256"]
    values = np.genfromtxt(csv_path, delimiter=",", skip_header=2)
    assert file_record["row_count"] == 60
    assert values.shape == (60, 4)
    states = {record["state_id"]: record for record in reference["states"]}
    assert states["al_rho4p712_te1_ti1"]["sii_additive_correction"] == 1.5
    assert states["al_rho8p1_te5_ti5"]["sii_additive_correction"] == 0.0
    assert np.min(values[:, 1]) < 0.0
    assert np.min(values[:, 1] + 1.5) >= 0.0

    baseline_dir = ROOT / "benchmarks" / "baselines" / benchmark_id
    manifest = _json(baseline_dir / "manifest.json")
    controller = ROOT / str(manifest["producer"]["script_relative_path"])
    # This is the controller hash recorded at baseline acceptance, not a
    # demand to rewrite historical provenance whenever its runner is fixed.
    assert controller.is_file()
    recorded_hash = manifest["producer"]["current_controller_sha256"]
    assert len(recorded_hash) == 64
    assert all(character in "0123456789abcdef" for character in recorded_hash)
    assert manifest["configuration"]["structure_model"] == "IS"
    assert manifest["configuration"]["ionic_closures"] == [
        "HNC",
        "Rosenfeld--Ashcroft VMHNC",
    ]
    md_config = manifest["configuration"]["same_potential_md"]
    assert md_config["atoms"] == 2048
    assert md_config["sii_estimator"] == (
        "complete periodic reciprocal-shell average"
    )
    assert md_config["reciprocal_vectors_per_bin"] == (
        "all available half-space modes"
    )
    assert md_config["trajectory_frames"] == 21

    expected_ids = {
        "al_rho4p712_te1_ti1_lda",
        "al_rho4p712_te1_ti1_pbe",
        "al_rho8p1_te5_ti5_lda",
        "al_rho8p1_te5_ti5_pbe",
    }
    assert {record["state_id"] for record in manifest["states"]} == expected_ids
    for record in manifest["states"]:
        path = baseline_dir / str(record["baseline_file"])
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "k_bohr_inv",
                "sii_k",
                "vmhnc_sii_k",
                "vmhnc_eta",
                "vmhnc_variational_residual",
                "md_k_bohr_inv",
                "md_sii_k",
                "md_sii_block_sem",
                "md_sii_vectors_per_bin",
                "md_sii_uncertainty_definition",
                "md_trajectory_frames",
                "md_nve_relative_energy_drift",
            }
            assert required <= set(archive.files)
            assert str(archive["schema_version"].item()) == (
                "otter_schorner_2022_al_sii_v4"
            )
            assert str(archive["structure_model"].item()) == "IS"
            assert str(archive["hnc_bridge_model"].item()) == "none"
            assert str(archive["vmhnc_hnc_bridge_model"].item()) == (
                "rosenfeld_ashcroft"
            )
            assert float(archive["hnc_output_residual"]) <= 1.0e-4
            assert float(archive["vmhnc_hnc_output_residual"]) <= 1.0e-4
            assert abs(float(archive["vmhnc_variational_residual"])) <= 3.0e-5
            assert 0.0 < float(archive["vmhnc_eta"]) < 0.5

            k_md = np.asarray(archive["md_k_bohr_inv"], dtype=float)
            sii_md = np.asarray(archive["md_sii_k"], dtype=float)
            sem_md = np.asarray(archive["md_sii_block_sem"], dtype=float)
            assert k_md.shape == sii_md.shape == sem_md.shape
            assert np.all(np.diff(k_md) > 0.0)
            assert np.all(np.isfinite(sii_md)) and np.all(sii_md >= 0.0)
            assert np.all(np.isfinite(sem_md)) and np.all(sem_md >= 0.0)
            box_length = float(archive["md_box_length_bohr"])
            fundamental = 2.0 * np.pi / box_length
            n_max = int(np.floor(4.5 / fundamental))
            integers = np.arange(-n_max, n_max + 1)
            nx, ny, nz = np.meshgrid(
                integers, integers, integers, indexing="ij"
            )
            triplets = np.column_stack((nx.ravel(), ny.ravel(), nz.ravel()))
            half_space = (
                (triplets[:, 0] > 0)
                | ((triplets[:, 0] == 0) & (triplets[:, 1] > 0))
                | (
                    (triplets[:, 0] == 0)
                    & (triplets[:, 1] == 0)
                    & (triplets[:, 2] > 0)
                )
            )
            magnitudes = fundamental * np.sqrt(
                np.sum(triplets[half_space] ** 2, axis=1)
            )
            magnitudes = magnitudes[magnitudes <= 4.5]
            radial_bin = np.floor(
                magnitudes / (0.1 * BOHR_TO_ANGSTROM)
            ).astype(int)
            populated_bins, expected_counts = np.unique(
                radial_bin, return_counts=True
            )
            expected_k = np.asarray(
                [
                    np.mean(magnitudes[radial_bin == bin_index])
                    for bin_index in populated_bins
                ]
            )
            np.testing.assert_array_equal(
                archive["md_sii_vectors_per_bin"], expected_counts
            )
            np.testing.assert_allclose(k_md, expected_k, rtol=0.0, atol=1e-13)
            assert int(archive["md_trajectory_frames"]) == 21
            assert str(archive["md_sii_uncertainty_definition"].item()) == (
                "SEM across shell-averaged saved production frames"
            )
            assert int(archive["md_atoms"]) == 2048
            assert int(archive["md_rdf_blocks"]) == 20
            assert str(archive["md_ensemble_sequence"].item()) == "NVT->NVE"
            assert abs(float(archive["md_nve_relative_energy_drift"])) < 5e-5
            assert float(archive["md_nve_mean_temperature_k"]) == pytest.approx(
                float(archive["ti_ev"]) * EV_TO_KELVIN,
                rel=0.2,
            )

            assert {
                "r_bohr",
                "gii_r",
                "vmhnc_r_bohr",
                "vmhnc_gii_r",
                "vmhnc_k_bohr_inv",
                "md_r_bohr",
                "md_gii_r",
                "md_gii_block_sem",
            }.isdisjoint(archive.files)
            closure_delta = np.asarray(archive["sii_k"], dtype=float).copy()
            closure_delta -= np.asarray(archive["vmhnc_sii_k"], dtype=float)
            assert np.all(np.isfinite(closure_delta))
            assert np.max(np.abs(closure_delta)) > 1.0e-6

            provenance = json.loads(str(archive["xc_provenance_json"].item()))
            assert provenance["provider"] == "libxc"
            assert provenance["provider_version"] == "7.0.0"
    _check_accepted_archives(baseline_dir, manifest)


def _check_accepted_archives(directory: Path, manifest: dict) -> None:
    """Check only accepted files; reference-only records must not fake arrays."""
    for record in manifest["states"]:
        if record["status"] != "accepted":
            assert record.get("baseline_file") is None
            assert record.get("baseline_sha256") is None
            continue
        path = directory / str(record["baseline_file"])
        assert _sha256(path) == record["baseline_sha256"]
        with np.load(path, allow_pickle=False) as archive:
            assert not any(
                np.asarray(archive[key]).dtype.hasobject for key in archive.files
            )
            assert str(archive["state_id"].item()) == record["state_id"]
            residual_key = (
                "hnc_output_residual"
                if "hnc_output_residual" in archive
                else "hnc_best_residual"
            )
            closure_key = (
                "closure_transform_max_abs"
                if "closure_transform_max_abs" in archive
                else "hnc_closure_mismatch"
            )
            assert float(archive[residual_key]) <= 1.0e-4
            assert float(archive[closure_key]) <= 2.5e-3
