:orphan:

Starrett et al. 2014 mixture Figure 3
======================================

This benchmark compares the C--C, C--H, and H--H pair distributions of
:math:`\mathrm{CH}_{1.36}` with the solid IS-QM curves in Figure 3 of
:cite:t:`StarrettEtAl2014`.  It covers
:math:`\rho=2.94,5,15\ {\rm g\,cm^{-3}}` and
:math:`T=20,50,100\ {\rm kK}`.

The open markers are independent digitizations of the published curves.  The
bundled continuous curves are the September 2026 Otter recalculation of all
nine states, with configuration, producing-source provenance, diagnostics,
and file checksums recorded in the manifest.

The 27 pair/state comparisons have a median RMSE of ``0.00722``.  The largest
RMSE is ``0.02181`` for H--H at
:math:`2.94\ {\rm g\,cm^{-3}}` and 50 kK.  The plotting script recomputes
these metrics from the archived arrays.

Method
------

The stored calculation uses the ion-sphere average-atom/pseudoatom method of
:cite:t:`StarrettSaumon2014`, the multicomponent QOZ/HNC equations of
:cite:t:`StarrettEtAl2014`, finite-temperature Lindhard response, and the
finite-temperature jellium LFC of :cite:t:`Chabrier1990`.  The manifest is
authoritative for all numerical and model settings.

Reproduce the comparison
------------------------

Run the downloadable source from :doc:`gen_benchmarks/plot_starrett_et_al_2014_mixtures_fig3`::

    poetry run python benchmarks/examples/plot_starrett_et_al_2014_mixtures_fig3.py

The default calculates the electronic and ionic states, writes local results
under ``benchmarks/outputs``, and exports PNG and PDF figures. No precomputed
Otter NPZ is needed. Literature reference data remain separate inputs.

Data provenance
---------------

Digitized publication curves are stored under
``benchmarks/reference_data/starrett_et_al_2014_mixtures_fig3``.  Numerical
calculation results and diagnostics are stored separately under
``benchmarks/baselines/starrett_et_al_2014_mixtures_fig3``.  The digitizations
are derived from the cited article and are not covered by Otter's BSD software
license; the manifest records attribution, checksums, and license status
``NOASSERTION``.
