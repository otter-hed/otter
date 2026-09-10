:orphan:

Otter 3.0.0: numerical validation and recorded results
============================================================

The example gallery and scientific benchmarks use the September 9, 2026
conditioned-SCF calculation campaign. The two SC presentation packages were
subsequently recalculated with the SC-only zero-tail default. Numerical
convergence is not a claim of exact agreement with a publication, nor of
validity outside the tested states. SC feedback remains experimental.

Electronic and ionic checks
----------------------------

The 16 calculation groups cover all example and benchmark pages. The two
carbon-ionization pages share one 84-state scan, rather than two independent
calculations. All 84 full SCFs converge: 80 states are resolved and four are
marginal; none is unresolved. Marginal shallow levels are not displayed as
precisely determined energies. The Be scan contains 48 converged, resolved
states. Individual pages report the electronic and ionic residuals.

SC acceptance checks the unmixed correlation-potential residual as well as
changes in the ion structure. Quantum SC steps use zero-tail shallow-state
matching and tighten full-AA precision near outer convergence. Explicit user
settings are preserved; ordinary IS and TF defaults are unchanged. See
:doc:`../experimental/sc_feedback`.

An additional CH2 electronic validation covers 58 temperatures from 5 to
119 eV at 0.946 g cm\ :sup:`-3`. Every C/H full and external calculation
passes its convergence and QOZ-entry checks. One carbon state is marginal;
none is unresolved. Mean and median electronic times are 126.5 and 90.3 s
per temperature, respectively. These are measured wall times, not a hardware-
independent performance guarantee. This sweep did not include a new full
ion-temperature sweep or MD.

Current Otter curves and historical MD
---------------------------------------

The CH2 benchmark uses three newly computed electronic states and nine new
HNC curves. Johnson, Schörner and the ion-structure library also use fresh
Otter electronic/HNC/VMHNC results. HNC and VMHNC at the same state share one
electronic pair potential.

LAMMPS was not rerun for this refresh. Retained MD arrays, the CH2 animation
and its poster are historical. Their differences from current Otter curves
can include changes in the electronic pair potential; these overlays do not
isolate the HNC bridge approximation at a fixed potential. The public MD
reproduction scripts first calculate new Otter potentials, then run LAMMPS
and analyze the new trajectories.

Reproduction and presentation
-------------------------------

Run the Python source linked on each page to compute its results from physical
inputs. No bundled Otter NPZ is required. Local NPZ output and selective state
exports remain supported; these files are not distributed with the public
source. LAMMPS/MPI and, for the LDA/PBE comparison, Libxc are optional external
requirements.

HTML builds display recorded figures, tables and terminal output without
running AA or MD. These assets are regenerated together from reviewed local
results. The inventory in
``benchmarks/baselines/validation_20260909.json`` records their numerical
source hashes and validation scope. The presentation manifest also records
the rendering code hashes. Published reference coordinates are unchanged.
