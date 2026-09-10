Benchmark data and provenance
=============================

Reference data
--------------

``benchmarks/reference_data`` contains cited literature coordinates, units
and provenance manifests. These are inputs to the comparisons, not Otter
calculation outputs. Reuse conditions are recorded per dataset.

Reproduction
------------

Run the Python source linked on each gallery page from an Otter checkout.
The default computes from physical inputs and saves numerical results under
``benchmarks/outputs``. No distributed NPZ or earlier gallery run is required.
MD comparisons require LAMMPS and MPI; Schörner's LDA/PBE comparison also
requires the optional Libxc bindings. Input blocks expose state points and
non-default numerical or MD settings.

Local NPZ output remains supported, including configuration, units and
convergence information. The distribution policy does not change Otter's
state-export API. Explicit local-cache review modes are optional and are
never a fallback after a fresh calculation fails.

Documentation builds
--------------------

Gallery pages display recorded figures, tables and terminal output. These
presentation assets are not solver inputs. Their file hashes and capture
provenance are recorded in ``docs/source/_static/gallery_results/manifest.json``.
Sphinx does not execute AA or MD calculations. A new run provides results to
compare with the recorded run, not a promise of bitwise reproduction across
solver revisions or MD realizations.

Numerical archives and full development caches are excluded from the public
source export. Private accepted-result manifests can be retained for audit;
the default calculation does not read their NPZ files. The public-export
check rejects NPZ files anywhere in the exported source tree.

Validation
----------

Fast tests check solver contracts, reference parsers, computation-first entry
points and NPZ-free documentation builds. Full scientific runs separately
check electronic convergence, threshold-state resolution, ionic residuals
and MD energy drift and uncertainty. A successful documentation build does
not establish convergence of a new scientific campaign.
