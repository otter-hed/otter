:orphan:

.. note::

   HNC and VMHNC use the September 2026 recalculation. MD is historical,
   using the earlier potentials; see :doc:`validation_20260908`.

Schörner et al. (2022) aluminium structure factors
===================================================

This benchmark compares equilibrium Otter static ion structure factors with
two aluminium curves digitized from Figure 2 of :cite:t:`SchornerEtAl2022`:

* :math:`T_e=T_i=1` eV and
  :math:`\rho=4.712\,\mathrm{g\,cm^{-3}}`;
* :math:`T_e=T_i=5` eV and
  :math:`\rho=8.1\,\mathrm{g\,cm^{-3}}`.

For each state, LDA-PW92 and PBE electronic calculations produce separate
IS-QOZ ion--ion potentials. Ordinary HNC and Rosenfeld--Ashcroft VMHNC reuse
the same current electronic result and potential. The retained 2048-ion MD
used earlier potentials, so comparison with it no longer isolates closure error.

Models and terminology
----------------------

* **HNC** sets the unknown bridge function to zero.
* **VMHNC** replaces it by a Percus--Yevick hard-sphere bridge using the
  Rosenfeld--Ashcroft universality ansatz :cite:p:`RosenfeldAshcroft1979`.
  The hard-sphere packing fraction is fixed by the variational condition of
  :cite:t:`Faussurier2004`; it is not fitted to the Schörner data.
* **OCP** means bare Coulomb ions in a uniform neutralizing background.
  **YOCP** means a screened Yukawa one-component plasma.  They are different
  physical models.
* **IEMHNC** maps a simulation-derived OCP bridge
  :cite:p:`IyetomiOgataIchimaru1992` to a YOCP state along an isomorph
  :cite:p:`ToliasLuccoCastello2019`.  The PA-QOZ potentials used here are not
  assumed to be Yukawa, so this page does not mislabel VMHNC or MD as IEMHNC.

Ordinate correction
-------------------

The current digitization contains 60 wavenumber points for each curve.  Only
the 5 eV digitization was used to calibrate their common ordinate.  The raw
1 eV values are therefore displaced downward by 1.5.  Otter retains the CSV
unchanged and applies exactly

.. math::

   S_{ii}^{\mathrm{corrected}}(k)
   = S_{ii}^{\mathrm{stored}}(k) + 1.5

to ``Al_T_1p0_rho_4p712gcc`` while loading it.  No correction is applied to
the 5 eV curve.  The reference manifest records and tests this operation.

Historical molecular dynamics
---------------------------------

The MD curves use LAMMPS :cite:p:`ThompsonEtAl2022`, 2048 ions, a shifted-
force table ending at 20 Bohr, and an ``NVT -> NVE`` sequence.  In plasma-
frequency units, the timestep is :math:`0.005\,\omega_p^{-1}`, equilibration
lasts :math:`50\,\omega_p^{-1}`, and production lasts
:math:`500\,\omega_p^{-1}`.

Unlike an RDF truncated at half the simulation box, the displayed MD
:math:`S_{ii}(k)` is calculated directly from the nonzero periodic density
modes,

.. math::

   S(\mathbf{k}) = \frac{1}{N}
   \left|\sum_j \exp(i\mathbf{k}\cdot\mathbf{r}_j)\right|^2.

Every available half-space reciprocal vector is averaged in radial bins of
:math:`0.1\,\mathrm{\AA^{-1}}`; there is no random cap on the number of
directions.  The line averages 21 saved NVE frames (the initial production
frame plus 20 production intervals), and the shaded interval is twice the SEM
across those shell-averaged frames.  This estimator is non-negative by
construction and avoids both directional subsampling noise and the
low-:math:`k` truncation artefact of a finite-RDF Fourier transform.  All four
relative NVE energy drifts are below :math:`2.5\times10^{-6}`.

Quantitative comparison
-----------------------

The :doc:`runnable gallery <gen_benchmarks/plot_schorner_et_al_2022_al_sii>`
interpolates each current curve onto the corrected DFT-MD points and prints
RMSE, mean absolute error and maximum error directly from the loaded NPZ.
Those generated values are authoritative; a separate hand-copied table is
not maintained here. The historical MD data are unchanged. They cannot certify
the current closure at a fixed potential without a matching new MD run.

At 1 eV, however, the spherical pseudoatom construction cannot represent
directional chemical bonding or transient molecular structure.  A bridge
closure can improve the classical statistics of the supplied pair potential,
but it cannot restore electronic or chemical structure absent from that
potential.  Occasional closer HNC--DFT agreement can therefore be error
cancellation and should not be interpreted as HNC being a more accurate
closure.

Here LDA is Libxc ``lda_x + lda_c_pw``
:cite:p:`Dirac1930,Bloch1929,PerdewWang1992`; PBE is
``gga_x_pbe + gga_c_pbe`` :cite:p:`PerdewBurkeErnzerhof1996`.  The archives
record the exact functional provenance, closure residuals, VMHNC variational
residual, MD protocol, energy drift, and direct-:math:`S(k)` uncertainty.

Provenance and reproduction
---------------------------

The curves were digitized by the Otter maintainer from the cited publication.
The reference package records the title, Figure 2 attribution, DOI, column
mapping, units, checksum, and redistribution decision.  Its license status is
``NOASSERTION``; Otter's BSD-3-Clause license does not grant rights in the
digitized publication data.

The downloadable gallery validates both reference and baseline manifests.
Install the optional Libxc backend, LAMMPS, and MPI, then set
``USE_PRECOMPUTED_DATA = False`` in the script to recalculate all four
state/XC cases.  Fresh files are written below ``benchmarks/outputs`` and never
silently replace accepted baselines.

See the :doc:`runnable benchmark
<gen_benchmarks/plot_schorner_et_al_2022_al_sii>`.
