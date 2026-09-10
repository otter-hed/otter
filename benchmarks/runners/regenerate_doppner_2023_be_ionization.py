"""Recompute the full-AA Be scan for Döppner et al. (2023), Fig. 3(a).

Writes small, resumable candidates only; never runs external AA, HNC or MD.
Each independent AA keeps Otter's one-worker default. Run from the checkout::

    python benchmarks/runners/regenerate_doppner_2023_be_ionization.py

The default starts a fresh scan, one AA at a time. Use ``--resume`` explicitly
to reuse matching local checkpoints from an interrupted run.

Reference: https://doi.org/10.1038/s41586-023-05996-8.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from otter import __version__
from otter.electronic import FullExternalConfig, solve_full_only
from otter.io._npz import save_npz_atomic

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "doppner_2023_be_ionization"
OUTPUT = ROOT / "benchmarks" / "outputs" / PACKAGE / "recomputed"
DENSITIES = (1, 3, 6, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70)
TEMPERATURES = (50, 100, 150)


class FullSCFConvergenceError(RuntimeError):
    """A picklable failed point retaining scalar SCF and recovery diagnostics."""

    def __init__(self, diagnostics: dict):
        self.diagnostics = diagnostics
        super().__init__(diagnostics)

    def __str__(self):
        d = self.diagnostics
        last = d["history"][-1] if d["history"] else {}
        return (f"Be rho={d['rho_g_cc']:g}, T={d['te_ev']:g}: full SCF failed; "
                f"reason={d['scf_stop_reason']}, iterations={d['stage2_iters']}, "
                f"dn={last.get('dn_rel')}, dv={last.get('dv_rel')}, "
                f"map_error={d['final_state_map_error']}")


def configuration(rho: float, temperature: float) -> FullExternalConfig:
    """Only physical inputs and the full-only mode differ from defaults."""
    return FullExternalConfig(
        element="Be", rho_g_cc=rho, temperature_ev=temperature, run_mode="full"
    )


def solve_point(rho: float, temperature: float) -> dict:
    """Reduce a full-AA result without keeping radial densities or potentials."""
    started = time.perf_counter()
    result = solve_full_only(configuration(rho, temperature))
    if not result["stage2_converged"]:
        keys = ("iter", "dn_rel", "dv_rel", "err", "mu", "charge_ws",
                "charge_bound", "charge_cont", "continuum_matching_window_full_valid",
                "continuum_mix_backtracks")
        raise FullSCFConvergenceError({
            "rho_g_cc": rho, "te_ev": temperature,
            "elapsed_s": time.perf_counter() - started,
            **{k: result.get(k) for k in ("scf_stop_reason", "stage2_iters",
                "final_state_map_error", "threshold_state_status",
                "continuum_matching_window_full_valid", "threshold_state_refine_retry",
                "scf_energy_refine_retry")},
            "history": [{k: h[k] for k in keys if k in h}
                        for h in result.get("history", [])],
        })
    meta = result["meta"]
    history = result["history"]
    row = {
        "rho_g_cc": rho, "te_ev": temperature,
        "zbar": float(result["zbar"]),
        "zstar": float(meta["n0_final_bohr3"]) / float(meta["n_i_bohr3"]),
        "zbar_partition": float(result["zbar_partition"]),
        "mu_ha": float(result["mu"]),
        "n_i_bohr3": float(meta["n_i_bohr3"]),
        "n0_bohr3": float(meta["n0_final_bohr3"]),
        "rws_bohr": (3.0 / (4.0 * np.pi * float(meta["n_i_bohr3"]))) ** (1.0 / 3.0),
        "bound_energy_cut_ha": float(meta["bound_energy_cut_ha"]),
        "stage2_converged": True,
        "stage2_iters": int(result["stage2_iters"]),
        "stage2_error": float(history[-1]["err"]),
        "stage2_dn_rel": float(history[-1]["dn_rel"]),
        "stage2_dv_rel": float(history[-1]["dv_rel"]),
        "final_state_map_error": float(result["final_state_map_error"]),
        "continuum_matching_window_full_valid": bool(result["continuum_matching_window_full_valid"]),
        "continuum_guarded_steps": sum(h.get("continuum_mix_backtracks", 0) > 0 for h in history),
        "threshold_state_status": str(result["threshold_state_status"]),
        "elapsed_s": time.perf_counter() - started,
    }
    for key, value in row.items():
        if isinstance(value, (float, int)) and not np.isfinite(value):
            raise ValueError(f"Non-finite {key} in Be rho={rho}, T={temperature}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # A changed solver/configuration must not reuse an old point checkpoint.
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "src" / "otter").rglob("*.py"))
    }
    fingerprint = hashlib.sha256(json.dumps({
        "source": hashes, "configuration": asdict(configuration(1, 50)),
        "runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, sort_keys=True, default=str).encode()).hexdigest()
    rows, pending = {}, []
    for t in TEMPERATURES:
        for rho in DENSITIES:
            key = (t, rho)
            checkpoint = OUTPUT / f"Be_T{t}_rho{rho}.json"
            if args.resume and checkpoint.exists():
                cached = json.loads(checkpoint.read_text())
                if cached.get("fingerprint") == fingerprint:
                    rows[key] = cached["result"]
                    continue
            pending.append(key)
    started = time.perf_counter()
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(solve_point, rho, t): (t, rho) for t, rho in pending}
        for job in as_completed(jobs):
            t, rho = jobs[job]
            try:
                row = job.result()
            except Exception as exc:
                failure = {"te_ev": t, "rho_g_cc": rho, "error": str(exc)}
                if isinstance(exc, FullSCFConvergenceError):
                    failure["diagnostics"] = exc.diagnostics
                failures.append(failure)
                print(f"[failed] T={t}, rho={rho}: {exc}", flush=True)
                continue
            rows[t, rho] = row
            (OUTPUT / f"Be_T{t}_rho{rho}.json").write_text(json.dumps(
                {"fingerprint": fingerprint, "result": row}, indent=2) + "\n")
            print(f"[{len(rows)}/48] T={t} eV rho={rho} g/cc "
                  f"Zbar={row['zbar']:.6f} Zstar={row['zstar']:.6f} "
                  f"{row['threshold_state_status']} {row['elapsed_s']:.1f} s", flush=True)
    (OUTPUT / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    save_scan(rows, failures, hashes, fingerprint)
    if failures or len(rows) != 48:
        raise RuntimeError(f"Incomplete scan: {len(rows)}/48; diagnostic candidate and checkpoints retained")
    print(f"Finished: 48/48 full SCFs; {time.perf_counter()-started:.1f} s", flush=True)


def save_scan(rows: dict, failures: list, hashes: dict, fingerprint: str) -> None:
    """Preserve successful rows; report failures separately, never as fake zeros."""
    if not rows:
        raise RuntimeError("No converged Be states to package")
    ordered = [rows[t, rho] for t in TEMPERATURES for rho in DENSITIES if (t, rho) in rows]
    arrays = {key: np.asarray([row[key] for row in ordered]) for key in ordered[0]}
    arrays.update(schema_version=np.asarray("otter_be_ionization_v1"),
                  storage_profile=np.asarray("electronic_summary"))
    metadata = {
        "schema_version": "otter_compact_archive_metadata_v1",
        "archive_role": "project_generated_example_or_benchmark_baseline",
        "configuration": asdict(configuration(1, 50)),
        "configuration_scope": "rho and temperature are supplied per row",
        "state": {"element": "Be", "requested_states": 48, "stored_states": len(rows)},
        "producer": {"project": "Otter", "version": __version__,
                     "source_sha256": hashes, "fingerprint": fingerprint},
        "citation_keys": ["DoppnerEtAl2023", "StarrettSaumon2013", "StarrettSaumon2014"],
        "convergence": {"requested_states": 48, "full_scf_converged_states": len(rows),
                        "stage2_nonconverged_states": len(failures), "failures": failures,
                        "threshold_counts": {s: int(np.count_nonzero(arrays["threshold_state_status"] == s))
                                             for s in np.unique(arrays["threshold_state_status"])}},
        "fields": sorted(arrays),
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, default=str, sort_keys=True))
    path = OUTPUT / "Be_ionization.npz"
    save_npz_atomic(path, arrays)
    status = "diagnostic_partial" if failures or len(rows) != 48 else "candidate"
    manifest = {
        "schema_version": "otter_benchmark_manifest_v1", "benchmark_id": PACKAGE,
        "status": status, "configuration": metadata["configuration"],
        "producer": metadata["producer"], "scientific_audit": metadata["convergence"],
        "data_rights": {"origin": "project_generated_numerical_output", "public_release_gate": "resolved"},
        "states": [{"state_id": "Be_ionization", "status": status,
                    "baseline_file": path.name, "baseline_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}],
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"Saved {len(rows)}/48 converged full SCFs ({status}): {path}", flush=True)


if __name__ == "__main__":
    main()
