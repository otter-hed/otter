Parameter grids for interpolation
=================================

Otter can calculate a fixed composition over electron temperature, mass
density and :math:`\alpha=T_i/T_e`, and export the results to HDF5.
Each state uses the standard full/external AA and QOZ/HNC workflow.
This tool builds on Julian Lütgert's parameter-grid contribution.

Install and run
---------------

From the repository root, install the optional HDF5 dependency:

.. code-block:: bash

   poetry install -E grid

Create a configuration file, for example ``carbon_grid.json``:

.. code-block:: json

   {
     "workflow": {"elements": ["C"]},
     "temperature_ev": [20.0, 50.0],
     "rho_g_cc": [2.0, 4.0],
     "alpha": [0.5, 1.0, 1.2],
     "k": {"min": 0.2, "max": 10.0, "points": 300}
   }

Run the calculation with two independent workers:

.. code-block:: bash

   poetry run python tools/grid_calculation.py carbon_grid.json carbon_grid.h5 --workers 2

The installed package also provides ``poetry run python -m otter.grid`` with the same
arguments. The default is one worker. Each worker uses one CPU, including
mixture calculations; memory use increases with the number of simultaneous
states. Start with one or two workers and check available memory before
increasing this number.

The terminal prints state indices, completion times and errors. Full SCF
progress is written to ``carbon_grid.h5.logs/i_j_k.log``; the indices select
the temperature, density and alpha axes, respectively.

Use ``--resume`` to retry failed or unfinished states:

.. code-block:: bash

   poetry run python tools/grid_calculation.py carbon_grid.json carbon_grid.h5 --workers 2 --resume

Completed states are retained. Resume requires identical configuration,
Otter source and numerical dependency versions. Stored state coordinates,
k coordinates, composition and units are also checked against the configuration.
Use a new filename for a different configuration or Otter version.

Composition and numerical settings
----------------------------------

The ``workflow`` object accepts unscanned
:class:`otter.PlasmaWorkflowConfig` options. For CH2, use:

.. code-block:: json

   "workflow": {"elements": ["C", "H"], "counts": [1, 2]}

Three or more species use the same interface. ``formula`` and
``number_fraction`` are also supported. One file describes one composition;
the temperature, density and alpha axes may contain any positive,
strictly increasing values, including :math:`\alpha>1`.

Physical models and numerical tolerances retain their workflow defaults.
Do not place ``temperature_ev``, ``rho_g_cc`` or ``ion_temperature_ev``
inside ``workflow``: these are determined by the scan axes.
``allow_unconverged_aa`` and ``allow_unconverged_root`` must remain ``False``.

For a TF calculation, select observables without orbitals:

.. code-block:: json

   {
     "workflow": {"formula": "CH2", "electronic_model": "tf"},
     "temperature_ev": [20.0],
     "rho_g_cc": [2.0],
     "k": {"min": 0.2, "max": 10.0, "points": 300},
     "outputs": ["Sii", "q", "f", "Zbar", "Zstar"]
   }

The common k axis
-----------------

Each AA calculation determines its native QOZ grid. Different state points
can therefore have different k coordinates even when they use the same
number of points. The exporter linearly resamples every spectrum onto the
requested physical k axis. It does not change the solver grids.

``axis/k`` is fixed when the file is created; worker completion order does
not change it. Each spectral dataset has a ``k_path`` attribute naming its
coordinate dataset. For ``S_ii``, ``q``, ``f`` and ``f_nl`` at the file root,
this is ``/axis/k``. Native spectra must be paired with their own coordinates,
even when arrays from different states have equal lengths.

The requested k values must be positive and lie within every accepted native
interval. Out-of-range states fail with their native interval recorded in
the error. The exporter neither extrapolates spectra nor evaluates their
k=0 limit. To reach smaller k, increase ``workflow.qoz_pad_factor`` as
described in :doc:`state_exports`.
Increasing ``qoz_linear_n_points`` improves radial resolution and increases
the accessible maximum k.

The ``k`` value may also be an explicit list of positive, increasing
coordinates in inverse Bohr. Resampling does not establish convergence:
check the native QOZ and output resolutions for the intended interpolation.

Set ``save_native_spectra=True`` in ``GridConfig`` (or
``"save_native_spectra": true`` in JSON) to retain each state's original
``k`` and selected ``S_ii``, ``q`` and ``f`` spectra as well. These arrays
are not interpolated or truncated to the common interval. They include
the smallest positive k produced by that state's solver, not k=0.
This option does not change the calculation or the common-grid datasets;
it increases the output size and is disabled by default.

HDF5 layout
-----------

The schema is ``otter_grid_v1``. Dataset dimensions are:

.. list-table::
   :header-rows: 1

   * - Dataset
     - Shape
   * - ``S_ii``
     - ``(Ns, Ns, Nk, NT, Nrho, Nalpha)``
   * - ``q``, ``f``
     - ``(Ns, Nk, NT, Nrho, Nalpha)``
   * - ``Z_bar``, ``Z_star``
     - ``(Ns, NT, Nrho, Nalpha)``
   * - ``f_nl``
     - ``(Ns, Norb, Nk, NT, Nrho, Nalpha)``
   * - ``E_nl``, ``orbital_present``
     - ``(Ns, Norb, NT, Nrho, Nalpha)``
   * - ``status``, ``error``, ``elapsed_s``, ``attempts``, ``diagnostics_json``
     - ``(NT, Nrho, Nalpha)``

``axis`` contains ``T_e``, ``rho``, ``alpha``, ``k``,
``elements``, ``counts`` and ``number_fraction``. Dataset attributes
give axis labels and units. Energies are in Hartree.
Read units with ``dataset.attrs["unit"][0]``; the attribute is a one-entry
string array. For ``axis/k``, the unit is ``"1/a0"`` (inverse Bohr).
The resume validator also accepts legacy scalar unit strings; configuration
and source-fingerprint checks still apply.
``Z_bar`` is the workflow's QOZ ionization; ``Z_star=n0/n_i_aa``
uses the AA-cell density, not a mixture's bulk partial ion density.

Orbital exports additionally provide ``axis/n``, ``axis/l`` and
``axis/orbitals``. The orbital axis grows as required, in the order
1s, 2s, 2p, 3s, 3p, 3d, and so on. Each element retains its own energies
and form factors. At a completed state, an absent level has zero
``f_nl``, NaN ``E_nl`` and ``orbital_present=False``.
Summing ``f_nl`` over orbitals recovers ``f`` to numerical precision.
An energy of zero is not used to represent an absent state.
Do not interpolate missing energies across a pressure-ionization boundary.

File attributes contain the schema and ``manifest_json``: the requested
scan, effective workflow defaults, package version, source fingerprint and
dependency versions. The original native k range and convergence summary
are retained in each state's ``diagnostics_json``.

When native spectra are enabled, ``native/t_r_a`` holds one state, with
zero-based indices into ``T_e``, ``rho`` and ``alpha``. Its ``k`` is a
one-dimensional array; ``f`` and ``q`` have shape ``(Ns, Nk_native)`` and
``S_ii`` has shape ``(Ns, Ns, Nk_native)``. Only requested spectra are stored.
Their ``k_path`` attributes reference ``/native/t_r_a/k``, not ``/axis/k``.
``f_nl`` remains on the common k axis. The first native spectral values,
including ``N=f+q`` when both are requested, are also recorded in
``diagnostics_json``. Failed states have no native group.

.. code-block:: python

   with h5py.File("carbon_grid.h5", "r") as grid:
       index = (0, 0, 0)
       if grid["status"][index] != 2:
           raise RuntimeError("The selected state is incomplete.")
       native = grid["native/0_0_0"]
       k_native = native["k"][:]
       N_native = native["f"][:] + native["q"][:]
       print(k_native[0], N_native[:, 0])

Existing files with mismatched coordinates must be rebuilt from the original
per-state coordinates and spectra. Changing a shared axis alone does not
resample the data. If the original coordinates are unavailable, recalculate
the affected states.

Check completion before interpolation
-------------------------------------

Status codes are 0 (pending), 1 (running), 2 (complete) and 3 (failed).
Uncomputed or failed numeric data are NaN. The file's ``complete``
attribute becomes true only when every state is complete; a run with
failures exits with a nonzero status.

.. code-block:: python

   import h5py
   import numpy as np

   with h5py.File("carbon_grid.h5", "r") as grid:
       if not np.all(grid["status"][:] == 2):
           raise RuntimeError("The parameter grid contains unfinished states.")
       k = grid["axis/k"][:]
       k_unit = grid["axis/k"].attrs["unit"][0]  # "1/a0"
       S_ab = grid["S_ii"][:]
       Zbar = grid["Z_bar"][:]
       q = grid["q"][:]
       f = grid["f"][:]

This file is a parameter-grid export, not a replacement for
:doc:`portable per-state NPZ files <state_exports>`.
Native AA restart data and radial profiles are not stored here.

Python interface
----------------

.. code-block:: python

   import numpy as np
   from otter.grid import GridConfig, run_grid

   if __name__ == "__main__":
       config = GridConfig(
           workflow={"elements": ["C"]},
           temperature_ev=(20.0, 50.0),
           rho_g_cc=(2.0, 4.0),
           alpha=(0.5, 1.0, 1.2),
           k=tuple(np.linspace(0.2, 10.0, 300)),
       )
       summary = run_grid(config, "carbon_grid.h5", workers=2)
       print(summary)

The main guard is required for spawned worker processes. In notebooks, run
the command-line entry point in a subprocess.
