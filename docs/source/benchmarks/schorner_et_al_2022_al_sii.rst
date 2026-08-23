Schörner et al. (2022) aluminium structure factors
===================================================

This benchmark compares equilibrium Otter PA-HNC static ion structure factors
calculated with LDA-PW92 and PBE with two curves digitized from Figure 2 of
:cite:t:`SchornerEtAl2022`:

* aluminium at :math:`T_e=T_i=1` eV and
  :math:`\rho=4.712\,\mathrm{g\,cm^{-3}}`;
* aluminium at :math:`T_e=T_i=5` eV and
  :math:`\rho=8.1\,\mathrm{g\,cm^{-3}}`.

The current digitization contains 60 wavenumber points for each curve.

Ordinate correction
-------------------

Only the 5 eV digitization was used to calibrate the common ordinate.  The raw
1 eV values are therefore displaced downward by 1.5.  Otter retains the CSV
unchanged and applies exactly

.. math::

   S_{ii}^{\mathrm{corrected}}(k)
   = S_{ii}^{\mathrm{stored}}(k) + 1.5

to ``Al_T_1p0_rho_4p712gcc`` while loading it.  No correction is applied to
the 5 eV curve.  The reference manifest records the affected state, operation,
value, and reason; the provenance tests verify both the raw checksum and the
corrected physical range.

Comparison summary
------------------

The accepted Otter PA-HNC baselines pass the electronic, external-AA,
threshold-state, HNC fixed-point, and transform-closure checks.  Interpolating
Otter onto the digitized :math:`k` points gives:

.. list-table::
   :header-rows: 1

   * - State
     - XC
     - RMSE
     - MAE
     - Maximum absolute error
   * - 1 eV, 4.712 g cm\ :sup:`-3`
     - LDA-PW92
     - 0.0506
     - 0.0313
     - 0.2072
   * - 1 eV, 4.712 g cm\ :sup:`-3`
     - PBE
     - 0.0464
     - 0.0281
     - 0.1921
   * - 5 eV, 8.1 g cm\ :sup:`-3`
     - LDA-PW92
     - 0.0253
     - 0.0186
     - 0.0721
   * - 5 eV, 8.1 g cm\ :sup:`-3`
     - PBE
     - 0.0241
     - 0.0178
     - 0.0700

Here LDA is the Libxc ``lda_x + lda_c_pw`` implementation
:cite:p:`Dirac1930,Bloch1929,PerdewWang1992`; PBE is
``gga_x_pbe + gga_c_pbe`` :cite:p:`PerdewBurkeErnzerhof1996`.  Both archives
record the Libxc 7.0.0 provider and exact functional IDs.  PBE uses Otter's
documented finite-core GGA regularization.

Provenance and reuse
--------------------

The curves were digitized by the Otter maintainer from the cited publication.
The reference package records the article title, Figure 2 attribution, DOI,
column mapping, units, checksum, and redistribution decision.  Its license
status is ``NOASSERTION``; Otter's BSD-3-Clause software license does not grant
rights in the digitized publication data.

Reproduce the comparison
------------------------

The complete, downloadable gallery program validates the reference and Otter
baseline manifests before plotting.  Install the optional backend with
``poetry install --extras libxc``, then set ``USE_PRECOMPUTED_DATA = False``
in that script to recalculate all four equilibrium state/XC cases.  Fresh
results are written below ``benchmarks/outputs`` and never silently replace
accepted baselines.

See the :doc:`runnable benchmark
<gen_benchmarks/plot_schorner_et_al_2022_al_sii>`.
