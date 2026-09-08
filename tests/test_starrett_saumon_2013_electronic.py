"""Offline data and method gates for the electronic-structure benchmark."""

from __future__ import annotations

import ast
import csv
import hashlib
import html
import json
import math
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ID = "starrett_saumon_2013_electronic"
REFERENCE_DIR = ROOT / "benchmarks" / "reference_data" / BENCHMARK_ID
BASELINE_DIR = ROOT / "benchmarks" / "baselines" / BENCHMARK_ID
SCRIPT = (
    ROOT
    / "benchmarks"
    / "examples"
    / "plot_starrett_saumon_2013_electronic.py"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_reference_tables_are_exact_attributed_and_checksummed() -> None:
    manifest = load_json(REFERENCE_DIR / "manifest.json")
    assert manifest["schema_version"] == "otter_reference_manifest_v1"
    assert manifest["publication"]["doi"] == "10.1103/PhysRevE.87.013104"
    assert manifest["origin_type"] == "published_numerical_table_transcription"
    assert manifest["license_declared"] == "NOASSERTION"
    assert manifest["public_release_gate"] == "resolved"
    assert [record["source_table"] for record in manifest["files"]] == [
        "Table I",
        "Table II",
        "Table III",
    ]
    for record in manifest["files"]:
        path = REFERENCE_DIR / str(record["path"])
        assert path.is_file()
        assert sha256_file(path) == record["sha256"]

    with (REFERENCE_DIR / "table_i_al_levels.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    al_15_3s = next(
        row
        for row in rows
        if row["temperature_ev"] == "15" and row["shell"] == "3s"
    )
    assert float(al_15_3s["energy_ha"]) == -0.0125
    assert float(al_15_3s["m_weight"]) == 0.134
    assert float(al_15_3s["gamma_ha"]) == 0.174
    reproduced_m = math.erf(
        -2.0
        * math.sqrt(math.log(2.0))
        * float(al_15_3s["energy_ha"])
        / float(al_15_3s["gamma_ha"])
    )
    assert np.isclose(reproduced_m, float(al_15_3s["m_weight"]), atol=5.0e-4)

    al = np.genfromtxt(
        REFERENCE_DIR / "table_ii_al_ionization.csv",
        delimiter=",",
        names=True,
    )
    fe = np.genfromtxt(
        REFERENCE_DIR / "table_iii_fe_hugoniot.csv",
        delimiter=",",
        names=True,
    )
    assert np.array_equal(al["temperature_ev"], (2.0, 6.0, 10.0, 15.0))
    assert np.array_equal(fe["temperature_ev"], (10.0, 100.0, 1000.0, 5000.0))
    assert np.array_equal(fe["zbar"], (8.78, 11.6, 21.7, 25.5))
    assert np.array_equal(al["gamma_ocp"], (41.0, 13.6, 8.14, 6.12))
    assert np.array_equal(al["gamma_tcp"], (5.05, 2.04, 1.60, 1.34))
    assert np.array_equal(fe["gamma_ocp"], (112.0, 22.4, 8.22, 2.18))
    assert np.array_equal(fe["gamma_tcp"], (14.3, 4.85, 3.32, 1.35))


def test_baseline_is_sc_pairs_and_pressure_weights_reproduce_eq_81() -> None:
    manifest = load_json(BASELINE_DIR / "manifest.json")
    data_path = BASELINE_DIR / str(manifest["state"]["data_file"])
    assert manifest["status"] == "accepted"
    assert manifest["configuration"]["scope"] == (
        "is_and_experimental_sc_feedback"
    )
    assert manifest["configuration"]["external_average_atom"] is True
    assert manifest["configuration"]["qoz_hnc"] is True
    assert manifest["configuration"]["bound_energy_cut_mode"] == "zero"
    assert manifest["configuration"]["bound_zero_tail_refine"] is True
    assert manifest["configuration"]["bound_zero_tail_max_binding_ha"] == 0.03
    assert manifest["configuration"]["sc_controls"]["fixed_is_mu"] is True
    # Baseline provenance identifies the historical producer; editing a
    # runner must not rewrite that checksum to pretend it produced old data.
    producer_hash = manifest["producer"]["script_sha256_current"]
    assert len(producer_hash) == 64 and int(producer_hash, 16) >= 0
    assert manifest["state"]["data_sha256"] == sha256_file(data_path)

    with np.load(data_path, allow_pickle=False) as archive:
        state = {key: np.asarray(archive[key]) for key in archive.files}
    assert not any(value.dtype.hasobject for value in state.values())
    assert state["schema_version"].item() == (
        "otter_starrett_saumon_2013_electronic_v2"
    )
    assert state["state_id"].shape == (8,)
    assert np.array_equal(state["structure_labels"], ("is", "sc"))
    assert np.array_equal(state["level_labels"], ("1s", "2s", "2p", "3s"))
    assert state["level_energy_ha"].shape == (8, 2, 4)
    assert state["zstar"].shape == (8, 2)
    assert np.all(np.isfinite(state["zstar"]))
    assert np.all(np.isfinite(state["zbar_partition"]))
    assert np.all(state["sc_converged"])
    assert np.array_equal(
        state["sc_iterations"], manifest["scientific_audit"]["sc_outer_iterations"]
    )
    assert np.all(state["sc_iterations"] <= manifest["configuration"]["sc_controls"]["max_outer"])
    assert np.all(state["sc_v_corr_residual_ha"] < 5.0e-4)
    assert np.all(state["sc_max_g_change"] < 5.0e-4)
    assert np.all(state["sc_inner_full_refined"][:4])
    assert not np.any(state["sc_inner_full_refined"][4:])
    # The experimental SC implementation deliberately fixes the converged IS
    # chemical potential, so n0/n_i remains identical along each pair.
    assert np.allclose(
        state["zstar"][:, 0], state["zstar"][:, 1], rtol=0.0, atol=5.0e-14
    )

    qm = np.asarray(state["electronic_model"], dtype=str) == "qm"
    for row in np.flatnonzero(qm):
        for structure in range(2):
            gamma = float(state["ion_gamma_ha"][row, structure])
            assert gamma > 0.0
            for energy, stored_m in zip(
                state["level_energy_ha"][row, structure],
                state["level_m"][row, structure],
                strict=True,
            ):
                if not np.isfinite(energy):
                    assert np.isnan(stored_m)
                    continue
                expected = math.erf(
                    -2.0 * math.sqrt(math.log(2.0)) * float(energy) / gamma
                )
                assert np.isclose(float(stored_m), expected, atol=2.0e-13)

    al_15 = int(np.flatnonzero(state["state_id"] == "al_rho2p7_te15_qm")[0])
    assert np.isnan(state["level_energy_ha"][al_15, 0, 3])
    assert manifest["scientific_audit"]["al_15ev_is_3s_status"] == (
        "no_negative_energy_level"
    )
    sc_energy = float(state["level_energy_ha"][al_15, 1, 3])
    sc_m = float(state["level_m"][al_15, 1, 3])
    assert sc_energy < 0.0
    assert manifest["scientific_audit"]["al_15ev_sc_3s_status"] == "resolved"
    assert sc_energy == manifest["scientific_audit"]["al_15ev_sc_3s_energy_ha"]
    assert sc_m == manifest["scientific_audit"]["al_15ev_sc_3s_m"]


def test_gallery_is_standalone_is_sc_and_uses_direct_tables() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    ast.parse(source, filename=str(SCRIPT))
    assert "USE_PRECOMPUTED_DATA = True" in source
    assert "PlasmaWorkflowConfig" in source
    assert "solve_plasma_workflow(" in source
    assert "SCFeedbackConfig" in source
    assert "solve_sc_feedback_workflow(" in source
    assert "converged IS chemical potential" in source
    assert "Appendix B" in source and "initial guess" in source
    assert 'bound_zero_tail_max_binding_ha": 3.0e-2' in source
    assert "binding energy" not in source
    assert "matplotlib" not in source
    assert "save_figure(" not in source
    assert "def plot_" not in source
    assert source.count('<table class="docutils align-default">') == 5
    assert source.count('<th colspan="3">\\(E\\) [Ha]</th>') == 2
    assert source.count('<th colspan="3">\\(M(E)\\)</th>') == 2
    assert "\\(\\bar Z\\)" in source
    assert "\\(\\Gamma_{\\rm TCP}\\)" in source
    assert "Table II versus Otter" in source
    assert "Table III versus Otter" in source
    assert "Paper coupling" in source
    assert "three significant digits" in source
    assert "Otter Eq. (81), paper inputs" not in source
    assert "all energies E are in Hartree" in source
    assert "all values are in Hartree" in source
    assert "QOZ" in source and "HNC" in source


def test_html_table_values_match_accepted_data() -> None:
    """A successful producer must not leave stale hand-written HTML cells."""
    doc = ast.get_docstring(ast.parse(SCRIPT.read_text()))
    tables = re.findall(r"<tbody>(.*?)</tbody>", doc, re.S)
    rows = [[re.findall(r"<td>(.*?)</td>", row, re.S)
             for row in re.findall(r"<tr>(.*?)</tr>", table, re.S)]
            for table in tables]
    with np.load(BASELINE_DIR / "electronic_states.npz", allow_pickle=False) as data:
        def check(cell, value):
            if not np.isfinite(value):
                assert html.unescape(cell) in ("unbound", "—")
            else:
                # Three significant digits, not a looser physics tolerance.
                decimals = max(0, 2 - int(math.floor(math.log10(abs(value))))) if value else 2
                assert cell == f"{value:.{decimals}f}"

        for table, index in ((0, 0), (1, 3)):
            for level, cells in enumerate(rows[table]):
                for structure in range(2):
                    check(cells[2 + structure], data["level_energy_ha"][index, structure, level])
                    check(cells[5 + structure], data["level_m"][index, structure, level])
        for row, index in enumerate((0, 3)):
            for structure in range(2):
                check(rows[2][row][2 + structure], data["ion_gamma_ha"][index, structure])
        for table, indices, offset in ((3, range(4), 1), (4, range(4, 8), 3)):
            for row, index in enumerate(indices):
                for field, column in (("zstar", offset + 1), ("zbar_partition", offset + 4)):
                    for structure in range(2):
                        check(rows[table][row][column + structure], data[field][index, structure])
