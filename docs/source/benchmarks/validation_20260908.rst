:orphan:

September 2026 numerical validation and data refresh
=====================================================

The updated AA solver and selective state exports are accompanied by reviewed,
recomputed Otter-only baselines. Numerical convergence is not a claim of exact
agreement with a publication or of validity outside the tested states.

Updated results
---------------

The Al full workflow, QM/TF comparison, Rayleigh-weight example, carbon
ionization and LFC examples, CH1.36 example, Argha-Roy carbon structure-factor
comparison, Starrett mixture comparison and Starrett single-species comparison
use the completed September 8 campaign. Their producing solver files were
checked against this source revision. The SC feedback module is the only
subsequent solver change, and is not used by those IS calculations.

The Al IS/SC example and the Starrett--Saumon Al/Fe electronic tables were
recomputed with the SC precision safeguard. These archives retain the actual
outer residuals and precision-refinement indicators. Inner full/external SCF
acceptance alone does not establish convergence of the coupled AA/HNC loop.
SC now confirms QM full-AA precision and checks the current output-minus-input
correlation potential, not just its damped update. The model remains
experimental; see :doc:`../experimental/sc_feedback`.

Displayed energies and scattering widths use Hartree; displayed Otter numbers
use three significant digits. Stored arrays retain full precision. Tests
compare every Otter cell in the electronic HTML tables against those arrays.
Published reference values are unchanged.

Archived MD comparisons
-----------------------

The CH2, Johnson aluminium, Schörner aluminium and ion-structure-library
baselines remain archived comparisons. Their original HNC/VMHNC curves and MD
statistics are kept together. No LAMMPS calculations were performed during
this refresh, and no new HNC curve from a different electronic potential is
presented as a same-potential comparison with old MD.

Consequently these pages test the ionic closures for their documented archived
potentials; they do not validate every change in the current AA solver. Their
NPZ field inventories/checksums may change when metadata is refreshed, but
their numerical arrays are preserved. The current Otter-only candidates for
these comparisons have not replaced the matched MD baselines.

Reproducibility
---------------

Baselines are small pickle-free NPZ files under ``benchmarks/baselines``.
Each package has a manifest recording inputs, convergence and producing-code
provenance. The public source hash inventory and per-package update scope are
recorded in ``benchmarks/baselines/validation_20260908.json``. The ordinary
documentation build loads these checked data and regenerates its plots; it
does not run AA or MD simulations.
