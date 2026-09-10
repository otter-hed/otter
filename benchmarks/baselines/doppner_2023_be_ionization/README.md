# Be full-AA ionization benchmark

This package contains **48 converged full-SCF states out of 48 requested**,
all with resolved threshold diagnostics and valid continuum matching windows.
It was regenerated on 2026-09-09 by
`benchmarks/runners/regenerate_doppner_2023_be_ionization.py`.

Be densities: 1, 3, 6, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65,
70 g/cm³, at electron temperatures 50, 100 and 150 eV. The default QM
ion-sphere full-only settings (including Dirac exchange and one continuum
worker per atom) were used. The scan ran one AA at a time.
No external-AA, HNC or MD was run.

The SCF matching-window safeguard resolves the earlier failure at 50 eV,
65 g/cm³ and avoids the free-wave branch affecting 60 g/cm³. The new values
are Zbar=3.421730783 and 3.267113775, respectively. A doubled-radial-grid and
finer-energy control at 65 g/cm³ changes Zbar by 0.000139464. The resolved
labels alone are not a radial-grid convergence study of every point.

`zbar` is the full-only WS diagnostic `4 - Q_ion(Rws)`. `zstar` is the
asymptotic electron density divided by ion number density. `zbar_partition`
is retained separately; the full-only result is not relabelled as a
full/external screening-cloud integral. The compact archive retains chemical
potential, density normalization, Rws, continuum edge, SCF residuals,
matching-window validity, guarded-step counts and
measured point times, but no radial densities or potentials.

The adjacent NPZ is pickle-free and checksummed. Source hashes and the
calculation fingerprint identify its producing code. Digitized literature
data remain separately attributed under
`benchmarks/reference_data/doppner_2023_Be_ionization_fig3a`.

## Reproduce from source

From the Otter checkout with its dependencies installed:

```bash
export PYTHONPATH=src MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u tools/recompute_all_data.py --only doppner_be_ionization --fresh
```

This runs all 48 full-AA inputs serially. `--fresh` deletes only old candidate
data in `benchmarks/outputs/doppner_2023_be_ionization/recomputed/`, so back up
any candidate you want to keep. This baseline is not modified. Omit `--fresh`
to resume matching checkpoints. Failures produce a nonzero exit status and
remain recorded alongside the successfully calculated rows.

Plot the new candidate (including explicitly marked partial results) with:

```bash
OTTER_USE_CANDIDATE_BE_IONIZATION=1 python benchmarks/examples/plot_doppner_2023_be_ionization.py
```

The solver entry point is
`benchmarks/runners/regenerate_doppner_2023_be_ionization.py`.
The [benchmark page source](../../examples/plot_doppner_2023_be_ionization.py)
includes the reproduction command and calculation-source download.
