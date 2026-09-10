"""Recompute every public gallery and benchmark dataset with current Otter.

The calculation scripts write candidates below ``benchmarks/outputs``.
Use ``--fresh`` to remove old candidates and point caches before calculation.
Use ``--replace-baselines`` to validate and promote the complete result set;
literature-derived reference data are never modified.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Task:
    script: str
    environment: dict[str, str]
    clean_paths: tuple[str, ...]


TASKS: dict[str, Task] = {
    "doppner_be_ionization": Task(
        "benchmarks/runners/regenerate_doppner_2023_be_ionization.py",
        {},
        ("benchmarks/outputs/doppner_2023_be_ionization/recomputed",),
    ),
    "al_full_workflow": Task(
        "benchmarks/runners/regenerate_al_full_workflow.py",
        {},
        ("benchmarks/outputs/al_full_workflow_1ev/recomputed",),
    ),
    "al_is_sc": Task(
        "benchmarks/runners/regenerate_al_is_sc_comparison.py",
        {},
        ("benchmarks/outputs/al_is_sc_comparison/recomputed",),
    ),
    "al_qm_tf": Task(
        "benchmarks/runners/regenerate_al_qm_tf.py",
        {},
        ("benchmarks/outputs/al_qm_tf/recomputed",),
    ),
    "al_rayleigh": Task(
        "docs/examples/plot_al_rayleigh_weight.py",
        {"OTTER_RECOMPUTE_AL_RAYLEIGH": "1"},
        ("benchmarks/outputs/al_rayleigh_weight_10ev/recomputed",),
    ),
    "argha_carbon_sii": Task(
        "benchmarks/examples/plot_argha_roy_carbon_sii.py",
        {"OTTER_RECOMPUTE_ARGHA_CARBON": "1"},
        ("benchmarks/outputs/argha_roy_carbon_sii/gallery_recomputed",),
    ),
    "carbon_ionization_levels": Task(
        "docs/examples/plot_carbon_ionization_levels.py",
        {
            "OTTER_RECOMPUTE_CARBON_IONIZATION": "1",
            "OTTER_REUSE_ACCEPTED_CARBON_IONIZATION": "0",
        },
        ("benchmarks/outputs/carbon_ionization_levels",),
    ),
    "carbon_lfc": Task(
        "benchmarks/runners/regenerate_carbon_lfc_sensitivity.py",
        {},
        ("benchmarks/outputs/carbon_lfc_sensitivity/recomputed",),
    ),
    "ch2_hnc_md": Task(
        "tools/reproduce_ch2_hnc_md.py",
        {
            "OTTER_CH2_RUN_MD": "0",
            "OTTER_CH2_RECOMPUTE_ELECTRONIC": "1",
            "OTTER_CH2_OUTPUT_DIR": (
                "benchmarks/outputs/ch2_hnc_md/hnc_recomputed"
            ),
            "OTTER_REQUIRE_ALL_CH2_HNC_MD": "1",
        },
        ("benchmarks/outputs/ch2_hnc_md/hnc_recomputed",),
    ),
    "ch136_workflow": Task(
        "docs/examples/plot_ch136_mixture_workflow.py",
        {"OTTER_RECOMPUTE_CH136_EXAMPLE": "1"},
        ("benchmarks/outputs/ch136_mixture_workflow_100kk/recomputed",),
    ),
    "ion_structure_library": Task(
        "benchmarks/runners/regenerate_ion_structure_library.py",
        {
            "OTTER_RUN_WUNSCH_SAME_POTENTIAL_MD": "0",
            "OTTER_REUSE_ION_STRUCTURE_GROUPS": "1",
        },
        ("benchmarks/outputs/ion_structure_library/recomputed",),
    ),
    "johnson_al": Task(
        "benchmarks/examples/plot_johnson_et_al_2025_two_temperature_al.py",
        {
            "OTTER_RECOMPUTE_JOHNSON_AL": "1",
            "OTTER_RUN_JOHNSON_SAME_POTENTIAL_MD": "0",
        },
        (
            "benchmarks/outputs/"
            "johnson_et_al_2025_two_temperature_al/gallery_recomputed",
        ),
    ),
    "schorner_al_sii": Task(
        "benchmarks/examples/plot_schorner_et_al_2022_al_sii.py",
        {
            "OTTER_RECOMPUTE_SCHORNER_AL": "1",
            "OTTER_RUN_SCHORNER_SAME_POTENTIAL_MD": "0",
        },
        ("benchmarks/outputs/" "schorner_et_al_2022_al_sii/gallery_recomputed",),
    ),
    "starrett_fig3": Task(
        "benchmarks/examples/plot_starrett_et_al_2014_mixtures_fig3.py",
        {"OTTER_RECOMPUTE_STARRETT_FIG3": "1"},
        ("benchmarks/outputs/" "starrett_et_al_2014_mixtures_fig3/gallery_recomputed",),
    ),
    "starrett_saumon_electronic": Task(
        "benchmarks/examples/plot_starrett_saumon_2013_electronic.py",
        {"OTTER_RECOMPUTE_STARRETT_SAUMON_ELECTRONIC": "1"},
        ("benchmarks/outputs/starrett_saumon_2013_electronic",),
    ),
    "starrett_single_species": Task(
        "benchmarks/examples/plot_starrett_single_species_2013_2014.py",
        {"OTTER_RECOMPUTE_STARRETT_SINGLE": "1"},
        ("benchmarks/outputs/" "starrett_single_species_2013_2014/gallery_recomputed",),
    ),
}


def candidate_failures(task: Task) -> list[str]:
    """Audit known per-state failure records even after a zero process exit."""
    import numpy as np

    issues = []
    for relative in task.clean_paths:
        directory = ROOT / relative
        for path in directory.glob("*.json"):
            record = json.loads(path.read_text())
            if path.name == "failures.json" and record:
                issues.append(f"{path.name}: {record}")
            if not isinstance(record, dict):
                continue
            audit = record.get("scientific_audit", {})
            if audit.get("stage2_nonconverged_states", 0):
                issues.append(f"{path.name}: {audit['stage2_nonconverged_states']} SCF failures")
            for state in record.get("states", []):
                if any(word in state.get("status", "") for word in ("failed", "rejected")):
                    issues.append(f"{state.get('state_id')}: {state.get('reason', state['status'])}")
        for path in directory.glob("*.npz"):
            with np.load(path, allow_pickle=False) as data:
                for key in ("threshold_status", "threshold_state_status"):
                    if key in data:
                        count = int(np.count_nonzero(data[key] == "unresolved"))
                        if count:
                            issues.append(f"{path.name}: {count} unresolved threshold states ({key})")
    return issues


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="delete selected candidate data and caches before recomputing",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=tuple(TASKS),
        help="run only this dataset; repeat the option to select several",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list dataset identifiers without running calculations",
    )
    parser.add_argument(
        "--replace-baselines",
        action="store_true",
        help=(
            "after a complete run, validate and atomically replace all "
            "project-generated baseline data"
        ),
    )
    args = parser.parse_args()

    if args.list:
        print("\n".join(TASKS))
        return
    if args.replace_baselines and args.only:
        raise ValueError(
            "--replace-baselines requires the complete dataset set; "
            "do not combine it with --only."
        )

    selected = tuple(args.only) if args.only else tuple(TASKS)
    print(f"Python executable: {sys.executable}", flush=True)
    if args.fresh:
        for name in selected:
            for relative in TASKS[name].clean_paths:
                path = ROOT / relative
                if path.exists():
                    print(f"[remove candidate] {path.relative_to(ROOT)}")
                    _remove(path)

    common_environment = os.environ.copy()
    common_environment.setdefault("MPLBACKEND", "Agg")
    common_environment.setdefault("MPLCONFIGDIR", "/tmp/otter-matplotlib")
    common_environment.setdefault("PYTHONUNBUFFERED", "1")
    source_path = str(ROOT / "src")
    existing_pythonpath = common_environment.get("PYTHONPATH")
    common_environment["PYTHONPATH"] = (
        source_path
        if not existing_pythonpath
        else source_path + os.pathsep + existing_pythonpath
    )

    for index, name in enumerate(selected, start=1):
        task = TASKS[name]
        environment = {**common_environment, **task.environment}
        print(f"\n[{index}/{len(selected)}] {name}: {task.script}", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / task.script)],
            cwd=ROOT,
            env=environment,
            check=True,
        )
        issues = candidate_failures(task)
        if issues:
            raise RuntimeError(f"{name}: candidate audit failed: {issues}")
    print("\nAll selected candidate calculations completed.")
    if args.replace_baselines:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "promote_recomputed_data.py"),
                "--apply",
            ],
            cwd=ROOT,
            env=common_environment,
            check=True,
        )
    else:
        print(
            "Project-generated baselines were not modified. Run "
            "tools/promote_recomputed_data.py --apply after review."
        )


if __name__ == "__main__":
    main()
