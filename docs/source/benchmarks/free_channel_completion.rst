Free-channel completion: numerical controls
===========================================

These controls test the optional free-channel correction described in
:doc:`../experimental/free_channel_completion`. They compare density
implementations at fixed model settings. They do not validate a smaller
propagation radius or replace an external physical benchmark.

Analytic free spectrum
----------------------

For a zero potential, the exact spectral density, including spin and before
occupation, is :math:`k/\pi^2`, with :math:`k=\sqrt{2E}`
:cite:p:`StarrettSaumon2014,NISTDLMF`. The controls use energies
0.001, 0.02, 0.2 and 2 Ha and angular cutoffs 2, 8 and 32. A 2400-point
square-root grid extends to 30 Bohr; residuals are measured from 1 Bohr
outward. These explicit model grids and cutoffs deliberately expose angular
truncation and do not alter the AA defaults.

.. figure:: /_static/benchmarks/free_channel_completion/free_spectrum.png
   :alt: Truncated and completed free spectra with their analytic reference.
   :width: 80%

   Zero potential at 0.2 Ha with :math:`L=2`. The dotted line gives the exact
   free spectrum. Residuals are absolute differences after division by
   :math:`k/\pi^2`; values below :math:`10^{-16}` share the display floor.
   :download:`PDF </_static/benchmarks/free_channel_completion/free_spectrum.pdf>`.

Reproduce the analytic calculation and figure from the repository root:

.. code-block:: console

   python -m benchmarks.runners.free_channel_completion

Full and external SCF
---------------------

The material controls compare three fresh workflows: the 0.4.1 main
reference, the new implementation with completion disabled, and the same
implementation with completion enabled. All retain the AA defaults:
4096 radial nodes, Dirac exchange, the native angular checks and energy
integration, and three consecutive full/external SCF confirmations.

The outer radius is :math:`15R_g`, the regular-wave domain is
:math:`7R_g`, the B3 handoff is :math:`4R_g`, and the fitting upper radius
is :math:`5R_g`, with :math:`R_g=R_{\mathrm{geometry}}`. Native recovery
remains active; resolved geometry is recorded separately from these nominal
values. No radius reduction or normalization transfer is enabled.

The density-equivalence budget is :math:`10^{-5}n_0` pointwise and
:math:`4\pi\int|\Delta n|r^2dr\leq2\times10^{-4}Z` electrons. Each full,
external and PA source-charge change must be below
:math:`2\times10^{-5}` electrons. These compare two numerical results;
they are separate from absolute charge neutrality.

.. include:: /_static/benchmarks/free_channel_completion/results.rst

.. figure:: /_static/benchmarks/free_channel_completion/scf_density.png
   :alt: External densities relative to n0 and changes in full, external and PA densities.
   :width: 100%

   Upper panels show each calculation's external density divided by its own
   background :math:`n_0`, including the uniform reference explicitly.
   Lower panels show signed density changes divided by the native
   :math:`n_0`. The vertical dotted line marks the nominal B3 handoff.
   :download:`PDF </_static/benchmarks/free_channel_completion/scf_density.pdf>`.

Interpretation
--------------

The analytic control verifies the free contribution and its implementation.
Material comparisons test the complete coupled calculation at the listed
conditions. Neither a small correction nor successful SCF stopping alone
establishes source-charge convergence. Existing full/external/PA charge
errors are retained in the table; no density rescaling is applied.

The correction preserves the number of propagated channels. These controls
make no complete-workflow speedup claim and do not establish that B3 can be
removed. See :doc:`validation_policy` for the distinction between numerical
and physical validation.

A small change in the final SCF profile is expected when the native angular
sum is already accurate throughout the B3 fitting region. Restoring the
outer free contribution does not by itself demonstrate that fewer
interacting channels or a shorter propagation domain are sufficient.

Downloads:

* :download:`Material checks </_static/benchmarks/free_channel_completion/scf_checks.csv>`
* :download:`Analytic checks </_static/benchmarks/free_channel_completion/analytic_checks.json>`
* :download:`Figure provenance </_static/benchmarks/free_channel_completion/provenance.json>`
