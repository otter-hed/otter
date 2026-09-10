"""Promotion of reviewed snapshots must never delete campaign evidence."""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np

from tools.promote_recomputed_data import Package, _compact_metadata
from tools import promote_recomputed_data as promotion


def test_snapshot_promotion_requires_explicit_retention(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "tools/promote_recomputed_data.py"),
         "--candidate-root", str(tmp_path), "--apply"],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "requires --keep-candidates" in result.stderr
    assert not list(tmp_path.iterdir())


def test_packaging_does_not_invent_or_replace_the_generation_version(tmp_path):
    package = Package("example", tmp_path)
    for producer in ({}, {"version": "0.2.4"}):
        result = _compact_metadata(
            package, manifest={"producer": producer}, baseline_name="example.npz",
            arrays={"schema_version": np.asarray("example_v1")},
        )
        assert result["producer"].get("version") == producer.get("version")
        assert result["producer"]["packaging_version"]


def test_new_archive_configuration_replaces_stale_controls_but_metadata_refresh_preserves_it(tmp_path, monkeypatch):
    monkeypatch.setattr(promotion, "BASELINES", tmp_path)
    package = Package("example", tmp_path / "candidate")
    package.baseline_dir.mkdir()
    old = {"configuration": {"continuum_workers_per_state": 6},
           "state": {"data_file": "state.npz"},
           "producer": {"git_commit": "a"*40, "git_commit_recovery_evidence": "old.json"}}
    (package.baseline_dir / "manifest.json").write_text(json.dumps(old))
    (package.baseline_dir / "old.json").write_text(json.dumps({"verified_git_commit": "a"*40}))
    signature = {"resolved_configuration": {"aa_overrides": {}, "hnc_max_iter": 160}}
    arrays = {"schema_version": np.asarray("example_v1"),
              "producer_signature_json": np.asarray(json.dumps(signature)),
              "otter_git_commit": np.asarray("b"*40)}
    result = promotion._merge_manifest(package, candidate_manifest=None,
                                        promoted={"state.npz": (arrays, "c"*64)})
    assert result["configuration"] == {"per_state": {"state.npz": signature}}
    assert "git_commit_recovery_evidence" not in result["producer"]
    assert result["producer"]["historical_git_commit_recovery_evidence"] == "old.json"
    result["configuration"] = {"models": ["qm", "tf"], "aa_overrides": {}}
    (package.baseline_dir / "manifest.json").write_text(json.dumps(result))
    refreshed = promotion._merge_manifest(package, candidate_manifest=None,
                                          promoted={"state.npz": (arrays, "c"*64)},
                                          preserve_configuration=True)
    assert refreshed["configuration"] == result["configuration"]
