# Carbon ionization/level gallery state

This directory contains project-generated Otter output for
`docs/examples/plot_carbon_ionization_levels.py`.  It is a capability example,
not third-party benchmark data.

The v3 archive stores one 4096-point full-AA density scan for carbon at
\(T_e=100\) eV: \(\bar Z=Z-Q_{\rm ion}(R_{\rm WS})\),
\(Z^*=n_e^0/n_i\), the chemical potential, and the 1s/2s/2p/3s/3p/3d energies relative
to the asymptotic continuum edge \(E_{\rm cut}=0\). These two mean-ionization
definitions are diagnostics rather than unique observables; their distinct
pressure-ionization behaviour is discussed by Starrett *et al.* (2019),
Sec. 4.2. The September 2026 scan independently recomputed all 84 AA states
from 0.1 through 450 g/cc with `bound_occ_mode="fd"`. All passed the full-SCF
check: 82 are threshold-resolved, while 4.6 and 300 g/cc are marginal; none
is unresolved. SCF convergence is not the same as threshold reliability.
The archive also stores
each displayed shell's direct contribution to \(Q_{\rm ion}(R_{\rm WS})\),
including the Starrett--Saumon pressure-ionization and radial-cutoff weights.
The 3s branch is stored through 0.45 g/cc and absent at 0.50 g/cc. This
brackets the sampled branch endpoint rather than defining an exact physical
pressure-ionization density. The 3p and 3d branches are displayed only where
the stored level mask permits them. Shallow-level values
are omitted when their numerical threshold classification is marginal or
unresolved; these classifications are stored per state in the NPZ, not drawn
as plot annotations.

When the requested grid matches this archive, the gallery loads it directly.
Adding densities to `DENSITIES_G_CC` automatically starts an incremental
extension: accepted points are reused and only missing states are calculated.
To force that path explicitly, set `RECOMPUTE_WITH_OTTER = True` or run:

```bash
OTTER_RECOMPUTE_CARBON_IONIZATION=1 \
PYTHONPATH=src python docs/examples/plot_carbon_ionization_levels.py
```

The ordinary recompute path seeds itself from this accepted 4096-point v3
baseline and calculates only requested densities that are absent there.  Set
`REUSE_ACCEPTED_POINTS_WHEN_RECOMPUTING = False` only for a deliberate
independent calculation of the full grid.  Each completed density point is
checkpointed under
`benchmarks/outputs/carbon_ionization_levels/point_cache`, so an interrupted
scan can resume without changing this accepted archive.

## Independent reproduction (no baseline or cache reuse)

For verification, use the following instead of the incremental command above.
Run from the checkout with Otter's dependencies installed:

```bash
export PYTHONPATH=src MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u tools/recompute_all_data.py --only carbon_ionization_levels --fresh
```

This runs all 84 full-AA inputs serially, bypassing baseline seeds and old
point caches. `--fresh` deletes the old candidate directory
`benchmarks/outputs/carbon_ionization_levels/`, including its figures and
checkpoints; back up any candidate you want to keep. The accepted NPZ here
is never overwritten. Omit `--fresh` to resume matching point checkpoints.

The new NPZ, checksum manifest and generated ionization/level figures are
written under `benchmarks/outputs/carbon_ionization_levels/`. The Bethkenhagen
benchmark uses that same new NPZ when run with:

```bash
OTTER_USE_CANDIDATE_CARBON_IONIZATION=1 python benchmarks/examples/plot_bethkenhagen_et_al_2020_carbon_ionization.py
```

The complete calculation is in `docs/examples/plot_carbon_ionization_levels.py`;
no separate private producer is required.
Its gallery page includes the calculation source and reproduction command.

Method context:

- Starrett and Saumon, *High Energy Density Physics* **10**, 35–42 (2014),
  DOI [10.1016/j.hedp.2013.12.001](https://doi.org/10.1016/j.hedp.2013.12.001).
- Starrett *et al.*, *Computer Physics Communications* **235**, 50–62 (2019),
  DOI [10.1016/j.cpc.2018.10.002](https://doi.org/10.1016/j.cpc.2018.10.002).
