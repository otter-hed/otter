:orphan:

Ion-structure literature library
================================

This benchmark compares Otter with published and author-provided
ion-structure curves for aluminium, beryllium, and carbon.

.. list-table::
   :header-rows: 1
   :widths: 18 28 18 36

   * - Material
     - State
     - Observable
     - Reference
   * - Al
     - 2.7 g cm\ :sup:`-3`, :math:`T_e=T_i=5` eV
     - :math:`S_{ii}(k)`
     - Fig. 3 of :cite:t:`GillEtAl2015`
   * - Al
     - 8.1 g cm\ :sup:`-3`, :math:`T_e=T_i=10` eV
     - :math:`S_{ii}(k)`
     - Fig. 1 of :cite:t:`ClerouinEtAl2015`
   * - Al
     - 8.1 g cm\ :sup:`-3`, :math:`T_e=10`, :math:`T_i=2` eV
     - :math:`S_{ii}(k)`
     - Fig. 1 of :cite:t:`ClerouinEtAl2015`
   * - Be
     - 5.544 g cm\ :sup:`-3`, :math:`T_e=T_i=13` eV
     - :math:`g_{ii}(r)`, :math:`S_{ii}(k)`
     - Figs. 1(c) and 2 of :cite:t:`WunschEtAl2009`
   * - C
     - 20 g cm\ :sup:`-3`, :math:`T_e=T_i=50` eV
     - :math:`g_{ii}(r)`
     - C. E. Starrett, private communication (unpublished)

Each panel states whether it is an equilibrium or two-temperature comparison.
The two Clérouin panels include independent Otter KS and Thomas--Fermi
average-atom calculations; both use the same QOZ/HNC settings.

The Wünsch Be panels contain three Otter results made from one IS-QOZ pair
potential:

* **Otter-HNC** uses the bridge-free hypernetted-chain closure.
* **Otter-VMHNC** uses the variational Rosenfeld--Ashcroft hard-sphere bridge
  :cite:p:`RosenfeldAshcroft1979,Faussurier2004`.
* **Otter-MD** is a 2048-ion LAMMPS :cite:p:`ThompsonEtAl2022` calculation
  with the same tabulated potential.  Shaded bands are twice the standard
  error across RDF blocks or saved-frame reciprocal-shell averages.

The accepted archive retains every periodic reciprocal shell.  The plotted
MD :math:`S_{ii}(k)` starts at the second shell,
:math:`k=0.5025\,\mathrm{\AA}^{-1}`: the cubic-box fundamental at
:math:`0.3553\,\mathrm{\AA}^{-1}` has only three independent half-space
vectors and is therefore excluded from the curve as a direction-starved
finite-size estimate.  No smoothing or replacement value is applied.

Thus the HNC--MD difference diagnoses the ionic closure without changing the
average atom, ionization, screening density, or pair potential.  VMHNC is not
IEMHNC: VMHNC uses a variational hard-sphere reference, whereas IEMHNC maps an
OCP bridge to a YOCP state.

Units
-----

The Gill, Clérouin, and Wünsch reciprocal-space coordinates are in
:math:`\mathrm{\AA}^{-1}`; the Wünsch real-space coordinate is in
:math:`\mathrm{\AA}`; and the Starrett carbon coordinate is in Bohr.  The
plotting script converts Otter coordinates to the reference unit without
modifying the archived reference columns.

Reproduce the comparison
------------------------

The downloadable script
:doc:`gen_benchmarks/plot_ion_structure_library` exposes one switch:

.. code-block:: python

   USE_PRECOMPUTED_DATA = True

``True`` verifies and loads checksummed Otter results.  ``False`` runs the
audited producer, including the Wünsch HNC/VMHNC and LAMMPS calculation,
writes new files
under ``benchmarks/outputs/ion_structure_library/gallery_recomputed``, and
plots those results.  Accepted files are never overwritten automatically.
Each run exports PNG and PDF figures.

The common single- and multi-species MD implementation is
``tools/otter_lammps_md.py``.  The benchmark producer
``benchmarks/runners/regenerate_ion_structure_library.py`` keeps all Wünsch
particle counts, temperatures, time scales, sampling intervals, and MPI
settings in its user-editable Python constants.  A run preserves
``atoms.data``, ``pair_potentials.table``, ``in.otter_md``, LAMMPS logs, RDF
blocks, the trajectory, statistical results, and a checksummed JSON metadata
record below ``benchmarks/outputs/ion_structure_library/recomputed/md_work``.
For mixtures the same tool requires every unordered pair potential and writes
all partial :math:`g_{ij}(r)` and :math:`S_{ij}(k)` channels.

Reference-data notice
---------------------

The publication-derived and author-provided coordinates are not covered by
Otter's BSD software license.  Their attributions, per-file checksums, units,
and license status
``NOASSERTION`` are recorded in
``benchmarks/reference_data/ion_structure_library``.  Consult those records
before redistributing the numerical values.
