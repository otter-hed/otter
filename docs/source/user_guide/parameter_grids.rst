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

Numerical overrides use the same options as a single workflow calculation.
For example, a radial-resolution check for CH2 can use:

.. code-block:: json

   "workflow": {
     "formula": "CH2",
     "aa_overrides": {"n_points": 8192},
     "species_overrides": {"C": {"bound_zero_tail_refine": true}},
     "qoz_linear_n_points": 8192
   }

``aa_overrides`` applies to all species; ``species_overrides`` sets options
for individual elements. These settings apply to every state in the file,
not to additional scan axes. Omitted parameters retain their defaults.
The requested overrides and effective workflow configuration are recorded
in ``manifest_json``. Overrides are optional; the initial example uses
the default numerical settings.

For a TF calculation, select observables without orbitals:

.. code-block:: json

   {
     "workflow": {"formula": "CH2", "electronic_model": "tf"},
     "temperature_ev": [20.0],
     "rho_g_cc": [2.0],
     "k": {"min": 0.2, "max": 10.0, "points": 300},
     "outputs": ["Sii", "q", "f", "Zbar", "Zstar"]
   }

Solver and output grids
-----------------------

Each state has a QOZ/HNC radial grid and its corresponding Fourier grid,
returned as ``result["ion"]["r"]`` and ``result["ion"]["k"]``. In a mixture,
the species' AA densities are interpolated onto this shared radial grid
before QOZ/HNC is solved. This grid is shared among species at one state,
not necessarily among different temperatures or densities.

The HDF5 exporter calls the state's QOZ/HNC k grid its *native grid*.
It linearly resamples the completed spectra onto the requested *common
output grid*, ``axis/k``, for interpolation across state points. It does
not change the solver grids or construct a new radial grid. Equal array
lengths do not imply equal physical k coordinates.

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

The schema is ``otter_grid_v2``. Dataset dimensions are:

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
   * - ``E_nl``, ``orbitals_present``
     - ``(Ns, Norb, NT, Nrho, Nalpha)``
   * - ``diagnostics/status``, ``diagnostics/error``, ``diagnostics/elapsed_s``,
       ``diagnostics/attempts``, ``diagnostics/convergence``
     - ``(NT, Nrho, Nalpha)``

``axis`` contains ``T_e``, ``rho``, ``alpha``, ``k``,
``elements`` and ``counts``. The ``number_fraction`` attribute of
``axis/elements`` gives the fixed species fractions in the same order
as the elements. It is not a scan axis.

Dataset attributes give axis labels and units. Read units with
``dataset.attrs["unit"]``; the attribute is a scalar string.
Energies are in Hartree (``"hartree"``); k is in inverse Bohr (``"1/a0"``).
For example, JaXRTS can read the coordinates as:

.. code-block:: python

   import h5py
   from jaxrts import ureg

   with h5py.File("carbon_grid.h5", "r") as grid:
       spectrum = grid["f"]
       coordinates = grid[spectrum.attrs["k_path"]]
       k = coordinates[:] * ureg(coordinates.attrs["unit"])
       k_angstrom = k.m_as(1 / ureg.angstrom)
       energy = grid["E_nl"][:] * ureg(grid["E_nl"].attrs["unit"])
       energy_ev = energy.m_as(ureg.electron_volt)

This reader runs in the JaXRTS environment; Otter is not required there.
``k_path`` selects the common or per-state coordinates without copying
the coordinate array into each spectrum.

``Z_bar`` is the workflow's QOZ ionization; ``Z_star=n0/n_i_aa``
uses the AA-cell density, not a mixture's bulk partial ion density.

Orbital exports provide ``axis/orbitals/n`` and ``axis/orbitals/l``.
These are paired quantum numbers for one orbital dimension, not two
independent dimensions. ``f_nl``, ``E_nl`` and ``orbitals_present`` use
the axis label ``orbital`` and the attribute
``orbital_path="/axis/orbitals"``. The orbital axis grows as required, in the order
1s, 2s, 2p, 3s, 3p, 3d, and so on. Each element retains its own energies
and form factors. At a completed state, an absent level has zero
``f_nl``, NaN ``E_nl`` and ``orbitals_present=False``.
Summing ``f_nl`` over orbitals recovers ``f`` to numerical precision.
An energy of zero is not used to represent an absent state.
Do not interpolate missing energies across a pressure-ionization boundary.
The presence mask is also available when energies are not requested;
a zero form factor alone does not establish that an orbital is absent.

.. code-block:: python

   with h5py.File("carbon_grid.h5", "r") as grid:
       index = (0, 0, 0)
       if grid["diagnostics/status"][index] != 2:
           raise RuntimeError("The selected state is incomplete.")
       n = grid["axis/orbitals/n"][:]
       l = grid["axis/orbitals/l"][:]
       present = grid["orbitals_present"][(0, slice(None), *index)]
       energy_ha = grid["E_nl"][(0, slice(None), *index)]
       f_nl = grid["f_nl"][(0, slice(None), slice(None), *index)]
       f_1s = f_nl[(n == 1) & (l == 0) & present]

File attributes contain the schema and ``manifest_json``: the requested
scan, effective workflow defaults, package version, source fingerprint and
dependency versions. The original native k range and convergence summary
are retained as JSON text in ``diagnostics/convergence``. This record
also contains native-grid metadata; it is not a single convergence flag.

When native spectra are enabled, ``native/t_r_a`` holds one state, with
zero-based indices into ``T_e``, ``rho`` and ``alpha``. Its ``k`` is a
one-dimensional array; ``f`` and ``q`` have shape ``(Ns, Nk_native)`` and
``S_ii`` has shape ``(Ns, Ns, Nk_native)``. Only requested spectra are stored.
Their ``k_path`` attributes reference ``/native/t_r_a/k``, not ``/axis/k``.
``f_nl`` remains on the common k axis. The first native spectral values,
including ``N=f+q`` when both are requested, are also recorded in
``diagnostics/convergence``. Failed states have no native group.

.. code-block:: python

   with h5py.File("carbon_grid.h5", "r") as grid:
       index = (0, 0, 0)
       if grid["diagnostics/status"][index] != 2:
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
       if not np.all(grid["diagnostics/status"][:] == 2):
           raise RuntimeError("The parameter grid contains unfinished states.")
       k = grid["axis/k"][:]
       k_unit = grid["axis/k"].attrs["unit"]  # "1/a0"
       S_ab = grid["S_ii"][:]
       Zbar = grid["Z_bar"][:]
       q = grid["q"][:]
       f = grid["f"][:]

This file is a parameter-grid export, not a replacement for
:doc:`portable per-state NPZ files <state_exports>`.
Native AA restart data and radial profiles are not stored here.

Earlier HDF5 files
------------------

Files using ``otter_grid_v1`` remain readable with h5py.
The physical datasets and their dimensions are unchanged; metadata paths
and unit encoding differ:

.. list-table::
   :header-rows: 1

   * - v1
     - v2
   * - Root ``status``, ``error``, ``elapsed_s``, ``attempts``
     - The same names under ``diagnostics/``
   * - ``diagnostics_json``
     - ``diagnostics/convergence``
   * - ``orbital_present``
     - ``orbitals_present``
   * - ``axis/n``, ``axis/l``
     - ``axis/orbitals/n``, ``axis/orbitals/l``
   * - ``axis/number_fraction`` dataset
     - ``axis/elements`` attribute ``number_fraction``
   * - One-entry unit arrays
     - Scalar unit strings
   * - Energy unit ``"Ha"``
     - ``"hartree"`` (same numerical values)

The old ``axis/orbitals`` string dataset is replaced by the group
containing ``n`` and ``l``; orbital names can be derived from these numbers.
A reader supporting both layouts should select paths using the file's
``schema`` attribute and accept both unit encodings:

.. code-block:: python

   import h5py
   import numpy as np

   def unit_string(dataset):
       value = np.asarray(dataset.attrs["unit"])
       if value.ndim > 1 or value.size != 1:
           raise ValueError("Expected one unit string.")
       unit = value.reshape(-1)[0]
       if isinstance(unit, bytes):
           unit = unit.decode("utf-8")
       if not isinstance(unit, str):
           raise ValueError("Expected a unit string.")
       return "hartree" if unit == "Ha" else unit

   with h5py.File("carbon_grid.h5", "r") as grid:
       schema = grid.attrs["schema"]
       if schema not in ("otter_grid_v1", "otter_grid_v2"):
           raise ValueError(f"Unsupported grid schema: {schema}")
       status_path = "status" if schema == "otter_grid_v1" else "diagnostics/status"
       if not np.all(grid[status_path][:] == 2):
           raise RuntimeError("The parameter grid contains unfinished states.")
       k_unit = unit_string(grid["axis/k"])

Old code using ``attrs["unit"][0]`` must be updated: with a scalar string,
that expression returns only its first character. No file is migrated
in place. Resuming requires v2 and the same source, configuration and
numerical dependency versions; continue an older run with the code that
created it, or start a new file with the current code. The source check
is not relaxed for a layout change.

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
