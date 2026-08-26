"""Audit the compact CH2 HNC--same-potential-MD benchmark package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "benchmarks" / "baselines" / "ch2_hnc_md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    manifest = json.loads((BASELINE_DIR / "manifest.json").read_text())
    baseline = manifest["baseline"]
    path = BASELINE_DIR / str(baseline["path"])
    assert _sha256(path) == baseline["sha256"]
    for media in manifest["media"]:
        media_path = ROOT / str(media["path"])
        assert _sha256(media_path) == media["sha256"]
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    return manifest, payload


def test_ch2_baseline_covers_six_two_temperature_states_and_controls() -> None:
    manifest, data = _load()
    assert manifest["status"] == "accepted"
    assert str(data["schema_version"].item()) == "otter_ch2_hnc_md_v1"
    assert data["te_ev"].shape == data["ti_ev"].shape == (9,)
    np.testing.assert_allclose(np.unique(data["te_ev"]), [9.0, 29.0, 99.0])
    np.testing.assert_allclose(np.unique(data["alpha"]), [0.2, 0.5, 1.0])
    assert np.count_nonzero(~np.isclose(data["te_ev"], data["ti_ev"])) == 6
    for te_ev in np.unique(data["te_ev"]):
        mask = np.isclose(data["te_ev"], te_ev)
        assert np.unique(data["electronic_state_id"][mask]).size == 1


def test_ch2_hnc_md_claim_is_supported_by_saved_diagnostics() -> None:
    manifest, data = _load()
    config = manifest["configuration"]
    assert config["same_qoz_pair_potential_for_hnc_and_md"] is True
    assert np.max(data["qoz_potential_max_abs_delta_within_te"]) < 1.0e-12
    assert np.max(np.abs(data["md_nve_relative_energy_drift"])) < 2.0e-3
    assert np.max(data["hnc_output_residual"]) < 1.0e-6
    assert np.max(data["g_rmse"]) < 2.0e-2
    assert np.max(data["s_rmse"]) < 2.0e-2
    assert np.count_nonzero(data["md_coulomb_core_regularized"]) == 2
    assert np.max(data["hnc_elapsed_s"]) < np.min(data["md_elapsed_s"])


def test_ch2_gallery_uses_otter_style_titles_and_embeds_project_media() -> None:
    source = (
        ROOT / "benchmarks" / "examples" / "plot_ch2_hnc_md.py"
    ).read_text(encoding="utf-8")
    index = (ROOT / "docs" / "source" / "benchmarks" / "index.rst").read_text(
        encoding="utf-8"
    )
    assert 'set_style("thesis", palette="bing")' in source
    assert r'CH$_2$: HNC vs MD $g_{ab}(r)$' in source
    assert r'CH$_2$: HNC vs MD $S_{ab}(k)$' in source
    assert "QOZ-derived" in source
    assert "More importantly" not in source
    assert "ch2_md.mp4" in source and "ch2_md.png" in source
    assert source.index("ch2_hnc_md_gab.png") < source.index(
        "from __future__ import annotations"
    )
    assert "plot_ch2_hnc_md" in index


def test_ch2_gallery_static_results_are_checksummed() -> None:
    manifest, _ = _load()
    roles = {str(record["role"]) for record in manifest["media"]}
    assert sum("gallery-first" in role for role in roles) == 3
