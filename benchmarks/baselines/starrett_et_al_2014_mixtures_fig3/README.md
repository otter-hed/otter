# CH1.36 mixtures Figure 3 numerical results

The nine compressed NPZ files contain the arrays and diagnostics needed to
reproduce the Starrett *et al.* Figure 3 comparison without rerunning the
electronic and ionic solvers:

- radius and C-C, C-H, and H-H pair distributions;
- thermodynamic state and species metadata;
- average-atom, pseudoatom, and QOZ charges;
- common-chemical-potential and HNC convergence diagnostics; and
- producer configuration, revision, and file checksums.

Comparison metrics use `0 <= r <= 6 Bohr`. Arrays are numeric or fixed-width
Unicode and load with `allow_pickle=False`; no absolute paths are stored.

All nine states were recomputed in the September 2026 Otter campaign.
`manifest.json` records the producing-source provenance and is the authoritative
state/file map. `metrics.csv` records comparison errors against the independently
digitized publication curves (27 pair/state comparisons).

Run the downloadable benchmark with

```bash
python benchmarks/examples/plot_starrett_et_al_2014_mixtures_fig3.py
```

Set `USE_PRECOMPUTED_DATA=False` in that file to run all nine current-Otter
workflows.  New files are written below `benchmarks/outputs` and do not
overwrite the bundled results.
