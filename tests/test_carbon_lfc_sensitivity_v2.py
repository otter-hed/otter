"""Fast configuration checks for the candidate carbon LFC v2 protocol."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]


def _load(relative_path: str, module_name: str) -> ModuleType:
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_carbon_lfc_producer_uses_default_grids_and_strict_acceptance() -> None:
    from otter import PlasmaWorkflowConfig

    producer = _load(
        "benchmarks/runners/regenerate_carbon_lfc_sensitivity.py",
        "otter_carbon_lfc_v2_producer",
    )
    config = producer._configuration(
        100.0,
        ion_temperature_ev=100.0,
        lfc_model="chabrier1990",
    )

    assert producer.RHO_G_CC == 5.0
    defaults = PlasmaWorkflowConfig(elements=["C"], temperature_ev=100.0, rho_g_cc=5.0)
    assert config.qoz_linear_n_points == defaults.qoz_linear_n_points
    assert config.qoz_pad_factor == defaults.qoz_pad_factor
    assert producer.REFERENCE_LFC == "chabrier1990"
    assert config.aa_overrides == {}
    assert config.hnc_require_converged is True
    assert config.hnc_tol == producer.HNC_TOL
    assert (
        config.hnc_closure_transform_tol
        == defaults.hnc_closure_transform_tol
    )
    assert getattr(config, "allow_unconverged_aa", False) is False


def test_carbon_lfc_v2_runner_defaults_to_reviewed_package() -> None:
    runner = _load(
        "benchmarks/runners/plot_carbon_lfc_sensitivity.py",
        "otter_carbon_lfc_v2_runner",
    )

    assert runner.USE_PRECOMPUTED_DATA is True
    assert runner.EXPECTED_RHO_G_CC == 5.0
    assert runner.EXPECTED_TEMPERATURES_EV == (2.0, 100.0)
    assert runner.PRECOMPUTED_DATA_DIR is None
    assert runner.RECOMPUTED_DIR != runner.REVIEWED_DIR


def test_carbon_lfc_v2_candidate_manifest_is_explicit_in_source() -> None:
    source = (
        ROOT
        / "benchmarks"
        / "runners"
        / "regenerate_carbon_lfc_sensitivity.py"
    ).read_text(encoding="utf-8")
    assert '"status": "candidate_not_accepted"' in source
