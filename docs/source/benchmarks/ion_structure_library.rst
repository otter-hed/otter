:orphan:

.. note::

   All seven Otter states and Be VMHNC use the September 2026 recalculation.
   Be MD retains its earlier potential. See :doc:`validation_20260908`.

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

Reproduction uses the default AA radial resolution and SCF tolerances,
without element-specific precision overrides. The native screening-charge
integral is checked before accepting a calculated state.

The Wünsch Be panels contain two current integral-equation results and an
explicitly historical MD overlay:

* **Otter-HNC** uses the bridge-free hypernetted-chain closure.
* **Otter-VMHNC** uses the variational Rosenfeld--Ashcroft hard-sphere bridge
  :cite:p:`RosenfeldAshcroft1979,Faussurier2004`.
* **Otter-MD (old potential)** is a 2048-ion LAMMPS
  :cite:p:`ThompsonEtAl2022` calculation with the earlier tabulated potential.
  Shaded bands are twice the standard
  error across RDF blocks or saved-frame reciprocal-shell averages.

The accepted archive retains every periodic reciprocal shell.  The plotted
MD :math:`S_{ii}(k)` starts at the second shell,
:math:`k=0.5025\,\mathrm{\AA}^{-1}`: the cubic-box fundamental at
:math:`0.3553\,\mathrm{\AA}^{-1}` has only three independent half-space
vectors and is therefore excluded from the curve as a direction-starved
finite-size estimate.  No smoothing or replacement value is applied.

Only current HNC versus VMHNC holds the electronic potential fixed. The
comparison to historical MD can also include changes to that potential. VMHNC is not
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

Run the downloadable source from :doc:`gen_benchmarks/plot_ion_structure_library`::

    poetry run python benchmarks/examples/plot_ion_structure_library.py

The default calculates the electronic and ionic states, writes local results
under ``benchmarks/outputs``, and exports PNG and PDF figures. No precomputed
Otter NPZ is needed. Literature reference data remain separate inputs.

The MD curves are also recalculated and require LAMMPS and MPI. Particle
counts, timesteps, sampling intervals and CPU counts are in the input block.

Reference-data notice
---------------------

The publication-derived and author-provided coordinates are not covered by
Otter's BSD software license.  Their attributions, per-file checksums, units,
and license status
``NOASSERTION`` are recorded in
``benchmarks/reference_data/ion_structure_library``.  Consult those records
before redistributing the numerical values.
