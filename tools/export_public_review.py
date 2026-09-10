"""Create an NPZ-free source snapshot for review, without touching Git history.

This is a build/reproduction artifact, NOT an automatic publication allowlist.
Existing destinations are rejected and local calculation outputs are untouched.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


# Explicitly reviewed tool scope: the existing public tools plus gallery
# reproduction/presentation helpers. Private studies and campaign controllers
# are not dependencies of the public package or benchmark producers.
PUBLIC_TOOLS = frozenset({
    "tools/check_public_release.py", "tools/otter_lammps_md.py",
    "tools/promote_recomputed_data.py", "tools/recompute_all_data.py",
    "tools/update_citations.py", "tools/reproduce_ch2_hnc_md.py",
    "tools/render_recorded_gallery.py", "tools/freeze_gallery_results.py",
    "tools/export_public_review.py",
    "tools/gallery_notebooks.py",
    "tools/diagnostics/bound_energy_partition_sensitivity.py",
    "tools/studies/carbon_xc_comparison.py",
})
PRIVATE_TESTS = frozenset({
    "tests/test_ch2_hnc_md_comparison.py",
    "tests/test_bound_energy_zero_audit.py",
    "tests/test_ch2_xrts_dataset_audit.py",
    "tests/test_ch2_xrts_saved_rws_regression.py",
    "tests/test_ch2_xrts_single_species_debug.py",
    "tests/test_electronic_validation.py", "tests/test_production_validation.py",
    "tests/test_background_fix_campaign.py",
    "tests/test_quadrature_study.py", "tests/test_scf_history_study.py",
    "tests/test_library_provenance_audit.py",
    "tests/test_recompute_candidate_failures.py",
    "tests/test_conditioned_mixer_diagnostic.py",
    "tests/test_matching_history_diagnostic.py",
    "tests/test_gallery_first_validation.py",
})


def in_public_tool_scope(name: str) -> bool:
    """Keep public tools and their tests, not private study dependencies."""
    private_study = (name.startswith("benchmarks/runners/diagnose_")
                     or name in {"benchmarks/runners/prepare_johnson_panel_d_sc_potential.py",
                                 "benchmarks/runners/compare_recomputed_to_baselines.py"}
                     or name.startswith("benchmarks/reference_data/Maximilian_et_al_2022_Al_Sii/"))
    return (not private_study and (not name.startswith("tools/") or name in PUBLIC_TOOLS)
            and name not in PRIVATE_TESTS)


def export_review(root: Path, destination: Path, *, public_tools: bool = False) -> int:
    if destination.exists():
        raise FileExistsError(f"Choose a new destination: {destination}")
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root,
    ).decode().split("\0")
    destination.mkdir(parents=True)
    count = 0
    for name in sorted(set(names)):
        path = Path(name)
        if public_tools and not in_public_tool_scope(name):
            continue
        if (not name or path.suffix in {".npz", ".pkl", ".pickle"}
                or path.parts[0] in {"applications", ".codex", ".agents", ".private"}
                or path.name == "DEV_STATE.md" or "_private" in path.parts):
            continue
        # Six obsolete raster plots have been replaced by the recorded SVGs.
        # Preserve the animation and its poster, which remain linked in HTML.
        if (path.parent.as_posix() == "docs/source/_static/benchmarks/ch2_hnc_md"
                and path.name in {"ch2_hnc_md_gab.png", "ch2_hnc_md_gab_residual.png",
                                  "ch2_hnc_md_sab.png", "ch2_hnc_md_sab_residual.png",
                                  "ch2_md_sab_rdf_minus_density.png", "ch2_md_sab_rdf_vs_density.png"}):
            continue
        source = root / path
        if not source.is_file() or source.is_symlink():
            continue
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        count += 1
    if list(destination.rglob("*.npz")):
        raise AssertionError("Numerical archive escaped the export filter")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--public-tools", action="store_true",
                        help="Apply the reviewed tool/private-study-test scope")
    args = parser.parse_args()
    count = export_review(Path(__file__).resolve().parents[1], args.destination.resolve(),
                         public_tools=args.public_tools)
    print(f"Exported {count} source/presentation files; NPZ count=0. No Git changes made.")


if __name__ == "__main__":
    main()
