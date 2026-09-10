# Otter benchmarks

Run a gallery's Python script from an installed Otter source checkout to
calculate its states and export figures. No precomputed Otter NPZ is needed.
For example:

```bash
python benchmarks/examples/plot_doppner_2023_be_ionization.py
python benchmarks/examples/plot_ch2_hnc_md.py
```

The second command includes LAMMPS MD and can take hours. LAMMPS and MPI must
be installed; the Schörner LDA/PBE comparison also needs the Libxc extra.
Each script's input block describes physical states and numerical controls.
AA calculations retain the one-continuum-worker default.

## Directory layout

- `reference_data/`: cited literature coordinates and attribution manifests.
- `examples/`: runnable calculation-and-plotting gallery scripts.
- `runners/`: shared producers and local review utilities used by those scripts.
- `outputs/`: locally generated numerical results, logs and figures (not public).
- `baselines/`: private accepted numerical archives and provenance for development.

The public source export excludes **all NPZ**, not just ionization scans.
`tools/export_public_review.py` creates a separate review snapshot without
altering local results or Git history. `tools/check_public_release.py --root
<snapshot> --no-npz` checks that snapshot. Removing files from a new public
commit does not erase them from earlier Git history.

Documentation uses recorded figures, tables and terminal text under
`docs/source/_static/gallery_results/`. Their manifest records capture hashes.
These are display assets, never solver inputs; Sphinx does not launch AA/MD.
After a new scientific campaign, review the outputs before refreshing these
assets. A plotting/build pass is not a new scientific validation run.

Local NPZ saves still use explicit units, selective fields, physical inputs,
configuration and convergence metadata. No failed calculation falls back to
an older accepted curve. Optional cache-review modes require user-provided
local data and are disabled by default.

MD producers use `tools/otter_lammps_md.py`, supporting single and multiple
species, all pair potentials, runnable LAMMPS input, trajectories, RDF,
density-mode structure factors, uncertainty and timing records. CH2's public
driver is `tools/reproduce_ch2_hnc_md.py`.

Literature-derived data are not covered by Otter's software license unless
their manifest explicitly says otherwise. Check `reference_data/README.md`
and each dataset's attribution/reuse notice.

Solver profiling and exploratory diagnostics belong in `tools/diagnostics/`,
not the scientific benchmark gallery.
