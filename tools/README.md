# Repository tools

Run these commands from the repository root with `poetry run python`.
Numerical solvers are in `src/otter`; these scripts provide command-line
entry points and repository maintenance.

| Script | Purpose |
| --- | --- |
| `grid_calculation.py` | Calculate parameter grids and export HDF5. |
| `otter_lammps_md.py` | Run LAMMPS with Otter pair potentials and analyze trajectories. |
| `reproduce_ch2_hnc_md.py` | Calculate the CH2 electronic, HNC and MD benchmark. |
| `recompute_all_data.py` | Recalculate the maintained gallery and benchmark datasets. |
| `promote_recomputed_data.py` | Validate candidate data and replace local baselines. |
| `render_recorded_gallery.py` | Render website figures and tables from local baselines. |
| `freeze_gallery_results.py` | Capture figures and terminal output from an executed gallery build. |
| `gallery_notebooks.py` | Generate gallery notebook and ZIP downloads. |
| `update_citations.py` | Regenerate citation documentation. |
| `export_public_review.py` | Export a source snapshot without numerical archives or private applications. |
| `check_public_release.py` | Check reference-data release restrictions and optional NPZ exclusion. |
| `diagnostics/bound_energy_partition_sensitivity.py` | Compare bound/free partition choices. |
| `studies/carbon_xc_comparison.py` | Compare carbon results for selected XC models. |

For a single example or benchmark, run its script under `docs/examples` or
`benchmarks/examples`. For batch data recalculation:

```bash
poetry run python tools/recompute_all_data.py --list
poetry run python tools/recompute_all_data.py --only al_full_workflow
```

Results go to `benchmarks/outputs`. Baseline replacement is a separate step.
`--fresh` deletes the selected candidate outputs and caches before running;
omit it to preserve them. The batch command excludes MD by default. Run the
corresponding gallery script for a complete MD comparison.

The two scripts under `diagnostics/` and `studies/` are optional numerical
studies and are not required by the standard workflow or documentation build.
Python and Numba caches under `__pycache__/` are generated locally and are
not versioned.
