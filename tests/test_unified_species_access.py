"""The public species view must not change native results or solver dispatch."""

from __future__ import annotations

import json

import numpy as np
import pytest

import otter.workflows as wf


def _electronic_payload(symbols):
    entries = []
    for i, symbol in enumerate(symbols):
        # Different native grids and values catch accidental species reuse.
        r = np.linspace(0.1, 4.0, 16 + i)
        aa = dict(r=r, r_ws=1.5 + i, mu=0.25, n0=0.05,
                  zbar_partition=2.0 + i, zstar=2.5 + i,
                  n_full=np.exp(-r), meta={"stage2_converged": True})
        entries.append(dict(element=symbol, count=i + 1.0,
                            x=(i + 1.0) / sum(range(1, len(symbols) + 1)),
                            r_ws_bohr=aa["r_ws"], mu_ha=aa["mu"], result=aa))
    if len(symbols) == 1:
        return "single_species", entries[0]["result"]
    return "mixture", dict(species=entries, meta={"root_success": True})


def _assert_view(result):
    electronic = result["electronic"]
    species = electronic["species"]
    assert isinstance(species, list)
    assert [entry["element"] for entry in species] == result["species_symbols"]
    assert [entry["count"] for entry in species] == result["species_counts"]
    np.testing.assert_allclose([entry["x"] for entry in species],
                               np.asarray(result["species_counts"]) /
                               sum(result["species_counts"]))
    legacy = ([electronic["result"]] if electronic["kind"] == "single_species"
              else [entry["result"] for entry in electronic["result"]["species"]])
    for entry, old in zip(species, legacy, strict=True):
        aa = entry["result"]
        assert aa is old
        assert aa["r"] is old["r"]
        assert aa["n_full"] is old["n_full"]
        assert "species" not in aa
        for key in ("n0", "zbar_partition", "zstar", "mu"):
            assert aa[key] == old[key]
    # Repeated references are fine, circular references are not.
    json.dumps(electronic, default=lambda value: value.tolist())


@pytest.mark.parametrize("symbols", [["C"], ["H", "C"]])
@pytest.mark.parametrize("model", ["qm", "tf"])
@pytest.mark.parametrize("run_mode", ["full", "full+ext"])
def test_solve_exposes_same_species_access(monkeypatch, symbols, model, run_mode):
    kind, payload = _electronic_payload(symbols)
    calls = []

    def solve(cfg):
        calls.append(cfg)
        return payload

    def wrong_solver(_cfg):
        pytest.fail("The additional access path must not change solver dispatch")

    monkeypatch.setattr(wf, "solve_full_then_external",
                        solve if kind == "single_species" else wrong_solver)
    monkeypatch.setattr(wf, "solve_mixture_full_then_ext",
                        solve if kind == "mixture" else wrong_solver)
    result = wf.solve_plasma_workflow(wf.PlasmaWorkflowConfig(
        elements=symbols, counts=list(range(1, len(symbols) + 1)),
        temperature_ev=10.0, rho_g_cc=1.3, electronic_model=model,
        run_mode=run_mode, show_progress=False, save_state_npz=False,
    ))
    assert len(calls) == 1
    assert result["electronic"]["kind"] == kind
    assert result["ion"] is None
    assert "saved_paths" not in result
    _assert_view(result)
    aa = result["electronic"]["species"][0]["result"]
    # Scalar updates through the legacy path cannot leave a stale copy.
    legacy = (result["electronic"]["result"] if kind == "single_species"
              else result["electronic"]["result"]["species"][0]["result"])
    legacy["zstar"] = 9.0
    assert aa["zstar"] == 9.0


@pytest.mark.parametrize("symbols", [["C"], ["H", "C"]])
@pytest.mark.parametrize("with_ions", [False, True])
def test_continuation_view_uses_final_payload(monkeypatch, symbols, with_ions):
    kind, payload = _electronic_payload(symbols)
    seen = []
    ion = {"sij_k": np.ones((len(symbols), len(symbols), 8))}

    def upgrade(_kind, raw):
        def updated(aa):
            return {**aa, "n0": aa["n0"] + 1.0}
        if _kind == "single_species":
            return updated(raw)
        return {**raw, "species": [
            {**entry, "result": updated(entry["result"])} for entry in raw["species"]
        ]}

    monkeypatch.setattr(wf, "_electronic_result_with_qoz_safe_threshold_tails", upgrade)
    monkeypatch.setattr(wf, "_validate_electronic_for_ion_structure",
                        lambda *args, **kwargs: seen.append("validate"))

    def ionic(_cfg, **kwargs):
        entries = (kwargs["species_entries"] if kind == "mixture"
                   else [kwargs["species_entry"]])
        assert entries[0]["result"]["n0"] > 1.0
        seen.append("ion")
        return ion

    monkeypatch.setattr(wf, "_one_component_ion_structure", ionic)
    monkeypatch.setattr(wf, "_multicomponent_ion_structure", ionic)
    cfg = wf.PlasmaWorkflowConfig(
        elements=symbols, counts=list(range(1, len(symbols) + 1)),
        temperature_ev=10.0, rho_g_cc=1.3,
        ion_temperature_ev=10.0 if with_ions else None, save_state_npz=False,
    )
    for _ in range(3):
        result = wf.continue_plasma_workflow_from_electronic_result(
            cfg, electronic_kind=kind, electronic_result=payload)
        _assert_view(result)
        assert result["ion"] is (ion if with_ions else None)
        assert result["electronic"]["species"][0]["result"]["n0"] > 1.0
        payload = result["electronic"]["result"]
    assert seen == (["validate", "ion"] * 3 if with_ions else [])
