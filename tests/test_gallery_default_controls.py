"""Ordinary gallery inputs inherit numerical defaults from Otter itself."""
from dataclasses import asdict
import ast
import dataclasses
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from otter import PlasmaWorkflowConfig
from otter.electronic import FullExternalConfig
from otter.experimental import SCFeedbackConfig

ROOT = Path(__file__).resolve().parents[1]


def load(path, monkeypatch):
    spec = importlib.util.spec_from_file_location("defaults_" + Path(path).stem, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


CASES = (
    ("docs/examples/plot_al_full_workflow.py", (), {}),
    ("docs/examples/plot_al_qm_tf.py", (15., "qm"), {}),
    ("docs/examples/plot_al_is_sc_comparison.py", ("qm",), {}),
    ("docs/examples/plot_al_rayleigh_weight.py", (8.1,), {}),
    ("docs/examples/plot_ch136_mixture_workflow.py", (), {}),
    ("docs/examples/plot_carbon_lfc_sensitivity.py", (2.,),
     {"ion_temperature_ev": 2., "lfc_model": "chabrier1990"}),
    ("benchmarks/examples/plot_argha_roy_carbon_sii.py",
     ({"element": "C", "te_ev": 30., "ti_ev": 30., "rho_g_cc": 3.51},), {}),
    ("benchmarks/examples/plot_ion_structure_library.py",
     ({"element": "C", "te_ev": 50., "rho_g_cc": 20.},), {"ion_temperature_ev": 50.}),
    ("benchmarks/examples/plot_johnson_et_al_2025_two_temperature_al.py", ({"te_ev": 30.},), {}),
    ("benchmarks/examples/plot_schorner_et_al_2022_al_sii.py",
     ({"te_ev": 5., "ti_ev": 5., "rho_g_cc": 8.1}, {"xc_model": "pbe"}), {}),
    ("benchmarks/examples/plot_starrett_et_al_2014_mixtures_fig3.py", (5., 100), {}),
    ("benchmarks/examples/plot_starrett_single_species_2013_2014.py",
     ({"element": "C", "model": "qm", "te_ev": 64.64, "ti_ev": 64.64, "rho_g_cc": 12.64},), {}),
    ("benchmarks/examples/plot_starrett_saumon_2013_electronic.py",
     ({"element": "Al", "electronic_model": "qm", "temperature_ev": 15., "rho_g_cc": 2.7},), {}),
)


@pytest.mark.parametrize("path,args,kwargs", CASES, ids=[Path(c[0]).stem for c in CASES])
def test_public_workflows_do_not_force_electronic_or_iteration_overrides(path, args, kwargs, monkeypatch):
    cfg = load(path, monkeypatch).workflow_config(*args, **kwargs)
    assert cfg.aa_overrides == {}
    for key in ("root_maxfev", "root_brent_maxiter", "hnc_max_iter", "qoz_linear_n_points"):
        assert getattr(cfg, key) == PlasmaWorkflowConfig.__dataclass_fields__[key].default
    assert not cfg.allow_unconverged_aa and not cfg.allow_unconverged_root


def test_al_recomputation_and_public_page_use_the_same_configuration(monkeypatch):
    runner = load("benchmarks/runners/regenerate_al_full_workflow.py", monkeypatch)
    page = load("docs/examples/plot_al_full_workflow.py", monkeypatch)
    assert asdict(runner._configuration(runner._load_library_regenerator())) == asdict(page.workflow_config())


def test_carbon_scan_uses_default_numerics_and_changes_cache_fingerprint(monkeypatch):
    page = load("docs/examples/plot_carbon_ionization_levels.py", monkeypatch)
    expected = FullExternalConfig(element="C", temperature_ev=100., rho_g_cc=5.2, run_mode="full")
    assert asdict(page._configuration(5.2)) == asdict(expected)
    before = page._calculation_fingerprint()
    original = page._configuration
    def changed(rho):
        cfg = original(rho)
        cfg.bound_zero_tail_refine = True
        return cfg
    monkeypatch.setattr(page, "_configuration", changed)
    page._calculation_fingerprint.cache_clear()
    assert page._calculation_fingerprint() != before


def test_hot_fe_retains_necessary_mu_bracket_not_qm_refinement(monkeypatch):
    page = load("benchmarks/examples/plot_starrett_saumon_2013_electronic.py", monkeypatch)
    cfg = page.workflow_config(dict(element="Fe", electronic_model="tf", temperature_ev=5000., rho_g_cc=34.37))
    assert cfg.aa_overrides == {"mu_bounds": (-2000., 200.)}


@pytest.mark.parametrize("refined", [False, True])
def test_manifest_distinguishes_requested_defaults_from_actual_recovery(refined, tmp_path, monkeypatch):
    runner = load("benchmarks/runners/regenerate_al_full_workflow.py", monkeypatch)
    output = tmp_path / "state.npz"
    output.write_bytes(b"synthetic checksum input")
    cfg = runner._configuration(runner._load_library_regenerator())
    wf = {"configuration": asdict(cfg), "electronic": {"result": {"meta": {
        "bound_zero_tail_refine": refined, "bound_zero_tail_scan_points": 64 if refined else 24,
        "b3_tail_target": "full", "cont_rmax_mult": 15. if refined else 7.,
        "bound_zero_tail_matched_energy_ha": float("nan"),
    }}}}
    manifest = runner._candidate_manifest(output, worktree_status="", workflow=wf)
    assert manifest["configuration"] == asdict(cfg)
    assert manifest["configuration"]["aa_overrides"] == {}
    assert manifest["aa_final_settings"]["bound_zero_tail_refine"] is refined
    json.dumps(manifest, allow_nan=False)


def test_shared_and_public_library_factories_agree(monkeypatch):
    runner = load("benchmarks/runners/regenerate_ion_structure_library.py", monkeypatch)
    page = load("benchmarks/examples/plot_ion_structure_library.py", monkeypatch)
    for bridge in ("none", "rosenfeld_ashcroft"):
        state = dict(element="Be", te_ev=13., ti_ev=13., rho_g_cc=5.544)
        assert asdict(runner._configuration(state, bridge_model=bridge)) == asdict(
            page.workflow_config(state, ion_temperature_ev=13., bridge_model=bridge))


def test_lfc_public_and_runner_factories_agree(monkeypatch):
    runner = load("benchmarks/runners/regenerate_carbon_lfc_sensitivity.py", monkeypatch)
    page = load("docs/examples/plot_carbon_lfc_sensitivity.py", monkeypatch)
    for model in page.LFC_MODELS:
        for temperature in page.TEMPERATURES_EV:
            opts = dict(ion_temperature_ev=temperature, lfc_model=model)
            assert asdict(runner._configuration(temperature, **opts)) == asdict(
                page.workflow_config(temperature, **opts))


def test_no_literal_default_values_are_repeated_in_config_calls():
    classes = {cls.__name__: cls for cls in (
        FullExternalConfig, PlasmaWorkflowConfig, SCFeedbackConfig)}
    for directory in ("examples", "docs/examples", "benchmarks/examples", "benchmarks/runners"):
        for path in (ROOT / directory).glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                cls = classes.get(node.func.id)
                if cls is None:
                    continue
                for kw in node.keywords:
                    field = cls.__dataclass_fields__.get(kw.arg)
                    if field is None or field.default is dataclasses.MISSING:
                        continue
                    try:
                        value = ast.literal_eval(kw.value)
                    except (ValueError, TypeError):
                        # User-input variables and model-sweep arguments are
                        # controls, not hardcoded copies of library defaults.
                        continue
                    assert value != field.default, (path, kw.lineno, kw.arg)
