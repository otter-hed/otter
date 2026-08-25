# Otter two-temperature aluminium benchmark baselines

This package contains reviewed Otter calculations for the 1, 3, 10, and
30 eV electron-temperature states in Johnson et al. (2025).  Every state uses
the ion-sphere (IS) pseudoatom construction.  Ordinary HNC and
Rosenfeld--Ashcroft VMHNC reuse the same accepted electronic result and the
same effective ion--ion potential, so their difference isolates the ionic
closure.  Each archive also contains a 2048-ion LAMMPS calculation using that
same pair potential, with NVT equilibration followed by NVE production.

Both closures passed the raw OZ residual, positive-structure-factor, and
transform-closure gates.  VMHNC also passed the variational packing-fraction
root check.  The exact diagnostics and SHA-256 values are recorded in
`manifest.json`; the archives contain only portable, non-object NumPy arrays.
The four NVE relative energy drifts are below `1.9e-6` in magnitude.

The VMHNC bridge is the Percus--Yevick hard-sphere reference of Wertheim and
Thiele, used with the Rosenfeld--Ashcroft universality ansatz and the
Faussurier variational packing-fraction condition.  No hard-sphere parameter
was fitted to the DFT-MD curves.  The primary publication DOIs are recorded in
the manifest and in `src/otter/literature.bib`.

VMHNC is not IEMHNC.  VMHNC uses a variational hard-sphere reference.  IEMHNC
maps a simulation-derived OCP bridge to a Yukawa one-component plasma (YOCP)
along an isomorph.  An earlier panel-(d) Otter diagnostic first fitted the
non-Yukawa QOZ potential to an effective long-wavelength YOCP, so it is an
IEMHNC-inspired mapping rather than a generic closure for arbitrary QOZ
potentials.  Johnson's plotted YOCP curve uses the distinct Daughton empirical
Yukawa bridge.  The benchmark documentation gives the exact references and
keeps OCP, YOCP, IEMHNC, and VMHNC labels separate.

At low electron temperature, the spherical pseudoatom/average-atom model
cannot represent directional chemical bonding or transient molecular
structure.  A bridge closure cannot reconstruct physics missing from the
electronic pair potential, so low-temperature disagreement with DFT-MD is an
expected model limitation rather than, by itself, a failed bridge test.

The ordinary HNC solve at 30 eV required potential-strength continuation and
the Newton--Krylov fallback; all other accepted ordinary-HNC states converged
with direct Anderson iteration.  The VMHNC inner equations converged to raw
residuals below `8e-11` for all four states.  Against same-potential MD, the
four VMHNC RDF RMSE values are `0.0085--0.0163`; ordinary HNC gives
`0.0326--0.0850`.  This tests the closure for the Otter pair potential and
does not validate the pseudoatom potential against DFT-MD.

The self-contained gallery program
`benchmarks/examples/plot_johnson_et_al_2025_two_temperature_al.py` has three
explicit paths:

- `USE_PRECOMPUTED_DATA = True` loads these four accepted files and verifies
  every SHA-256 checksum.
- `USE_PRECOMPUTED_DATA = False` calls the public Otter workflow and writes
  candidate NPZ files plus a candidate manifest under `benchmarks/outputs`.
- `USE_RECOMPUTED_CANDIDATES = True` plots a complete candidate set for
  review without treating it as accepted data.

Future recalculations remain candidates until their convergence diagnostics,
curves, exact producer revision, and checksums have been reviewed.  Do not
bypass this review gate merely to make the documentation build.

The publication curves live in the adjacent reference-data package.  They
are published by maintainer decision with attribution and license status
`NOASSERTION`; Otter-generated baselines do not inherit that third-party data
status.  Consult the reference manifest before reuse.
