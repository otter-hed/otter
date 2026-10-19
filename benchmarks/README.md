# Otter benchmarks

Run a gallery's Python script from an installed Otter source checkout to
calculate its states and export figures. No precomputed Otter NPZ is needed.
For example:

```bash
poetry run python benchmarks/examples/plot_doppner_2023_be_ionization.py
poetry run python benchmarks/examples/plot_ch2_hnc_md.py
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

Documentation uses recorded figures, tables and terminal text under
`docs/source/_static/gallery_results/`. Sphinx displays these assets without
running AA or MD. After recalculation, review the outputs and refresh the
figures, tables, terminal text and provenance together.

Local NPZ files contain units, selected fields, physical inputs, configuration
and convergence metadata. Failed calculations raise an error. Optional cache
modes require local data and are disabled by default.

Use `poetry run python tools/recompute_all_data.py --list` to list the batch
recalculation tasks, and `--only <name>` to select a task. This command produces
candidate datasets; gallery scripts also render their figures. See
[`tools/README.md`](../tools/README.md) for data promotion and documentation tools.

MD producers use `tools/otter_lammps_md.py`, supporting single and multiple
species, all pair potentials, runnable LAMMPS input, trajectories, RDF,
density-mode structure factors, uncertainty and timing records. CH2's public
driver is `tools/reproduce_ch2_hnc_md.py`.

Literature-derived data are not covered by Otter's software license unless
their manifest explicitly says otherwise. Check `reference_data/README.md`
and each dataset's attribution/reuse notice.

Solver profiling and exploratory diagnostics belong in `tools/diagnostics/`,
not the scientific benchmark gallery.
