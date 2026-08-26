Use Otter pair potentials in LAMMPS
===================================

``tools/otter_lammps_md.py`` provides a reproducible path from an Otter
effective ion--ion potential to a classical LAMMPS simulation.  It supports
one or many ionic species and performs the complete sequence

.. math::

   V_{ab}(r) \longrightarrow \text{LAMMPS table and input}
   \longrightarrow \text{NVT equilibration}
   \longrightarrow \text{NVE sampling}
   \longrightarrow g_{ab}(r),\ S_{ab}(k).

LAMMPS is an external optional program and is not installed by Otter.  The
default executable names are ``lmp`` and ``mpirun``; both can be changed in
the Python configuration.  The driver has no command-line parameter parser:
all physical and numerical settings remain visible in the calling script.

Which Otter potential is used?
------------------------------

The input is the QOZ effective ion--ion pair-potential matrix ``vij_r``:

.. math::

   V_{ab}(r) = \texttt{ion["vij_r"][a,b,:]},

where :math:`a,b` are ionic-species indices.  This is not the electron--ion
channel ``v_ie_r``.  Otter stores ``vij_r`` in Hartree on a radius grid in
Bohr.  The helper ``pair_potentials_from_otter`` accepts any of these
sources:

* a complete workflow result;
* its ``result["ion"]`` dictionary;
* a portable NPZ state produced by :func:`otter.save_plasma_state`;
* a prepared multicomponent QOZ object, before an HNC solve.

For :math:`N_s` species, the helper verifies that ``vij_r`` has shape
``(N_s, N_s, N_r)``, is symmetric, and contains every unique pair.  The pair
order is ``(0,0), (0,1), ..., (N_s-1,N_s-1)``.

Complete mixture example from an Otter NPZ
------------------------------------------

The following example assumes that ``outputs/ch2_state.npz`` is a converged
Otter state with species order C, H.  Edit the settings at the beginning of
the script and run the Python file from the repository root.

.. code-block:: python

   from pathlib import Path

   import numpy as np

   from tools.otter_lammps_md import (
       MDConfig,
       MDSpecies,
       pair_potentials_from_otter,
       run_otter_lammps_md,
   )


   # ------------------------- user settings -------------------------
   STATE_PATH = Path("outputs/ch2_state.npz")
   OUTPUT_DIR = Path("outputs/ch2_lammps_md")
   TI_EV = 10.0

   # Exact CH2 stoichiometry: 1024 C + 2048 H.
   PARTICLE_COUNTS = (1024, 2048)
   MPI_PROCESSES = 8

   # Demonstration values, not universal defaults.  Validate them below.
   TIMESTEP_PS = 1.0e-5
   THERMOSTAT_DAMP_PS = 1.0e-3
   NVT_STEPS = 10_000
   NVE_STEPS = 100_000
   # -----------------------------------------------------------------


   with np.load(STATE_PATH, allow_pickle=False) as state:
       symbols = tuple(str(value) for value in state["species_symbols"])
       if symbols != ("C", "H"):
           raise ValueError(f"Expected C,H in the NPZ, found {symbols}.")

       # zbar is the charge used to construct the saved QOZ potential.
       zbar = np.asarray(state["zbar"], dtype=float)
       total_ion_density = float(np.sum(state["n_i_bohr3"]))
       potentials = pair_potentials_from_otter(state)


   config = MDConfig(
       output_dir=OUTPUT_DIR,
       species=(
           MDSpecies("C", 12.011, PARTICLE_COUNTS[0], charge_e=float(zbar[0])),
           MDSpecies("H", 1.008, PARTICLE_COUNTS[1], charge_e=float(zbar[1])),
       ),
       ion_density_bohr3=total_ion_density,
       ion_temperature_ev=TI_EV,
       timestep_ps=TIMESTEP_PS,
       thermostat_damp_ps=THERMOSTAT_DAMP_PS,
       equilibration_steps=NVT_STEPS,
       production_steps=NVE_STEPS,
       rdf_bins=500,
       rdf_every=100,
       rdf_repeat=50,
       trajectory_every=5_000,
       table_points=8_192,
       r_min_bohr=0.1,
       k_bin_width_angstrom_inv=0.1,
       k_max_angstrom_inv=8.3,
       structure_factor_workers=8,
       mpi_processes=MPI_PROCESSES,
       lammps_executable="lmp",
       mpi_launcher="mpirun",
       reuse_completed=True,
   )

   result = run_otter_lammps_md(config, potentials)

   print("pairs:", result["md_pair_labels"])
   print("NVE relative energy drift:", result["md_nve_relative_energy_drift"])
   print("saved:", OUTPUT_DIR / "md_results.npz")


Why ``charge_e=zbar`` is specified
-----------------------------------

With nonzero ``charge_e``, the driver decomposes each Otter potential as

.. math::

   V_{ab}(r) = \frac{\bar Z_a\bar Z_b}{r} + \Delta V_{ab}(r).

LAMMPS evaluates the analytic Coulomb term, while its table contains only the
finite remainder :math:`\Delta V_{ab}`.  Their sum is the original Otter
potential.  This avoids interpolating a :math:`1/r` singularity.  The combined
energy and force are shifted smoothly to zero at the finite MD cutoff.

The charge must match the ``zbar`` used by QOZ.  Do not substitute
``zbar_aa_ws`` or ``zstar`` unless the pair potential was explicitly rebuilt
with that alternative charge definition.  Setting every ``charge_e`` to zero
is supported, but then the complete potential, including its short-range
core, is tabulated directly.

The cutoff is the smaller of ``0.48 * box_length`` and the largest available
Otter radius.  The NPZ export window must therefore retain enough of
``vij_r`` for the intended box.  Increase ``state_r_max_bohr`` when generating
the Otter state if necessary.

In-memory and single-species inputs
-----------------------------------

No intermediate NPZ is required.  A finished workflow result can be passed
directly:

.. code-block:: python

   workflow = solve_plasma_workflow(otter_config)
   potentials = pair_potentials_from_otter(workflow)

For one species, use one ``MDSpecies`` and its total ion density:

.. code-block:: python

   ion = workflow["ion"]
   config = MDConfig(
       output_dir=Path("outputs/al_md"),
       species=(
           MDSpecies(
               "Al",
               26.9815385,
               2048,
               charge_e=float(ion["zbar"]),
           ),
       ),
       ion_density_bohr3=float(ion["n_i"]),
       ion_temperature_ev=15.0,
       timestep_ps=1.0e-5,
       thermostat_damp_ps=1.0e-3,
   )
   result = run_otter_lammps_md(config, potentials)

For a mixture study that should not spend time solving HNC first, extract the
potential from the reusable QOZ preparation:

.. code-block:: python

   prepared = prepare_multicomponent_ion_structure_from_electronic_result(
       config,
       electronic_kind=electronic_kind,
       electronic_result=electronic_result,
   )
   potentials = pair_potentials_from_otter(prepared)

The masses, integer MD particle counts, ion temperature, and LAMMPS controls
still belong in ``MDConfig``.  Particle counts must reproduce the intended
number fractions; they do not alter the density supplied separately through
``ion_density_bohr3``.

Generated LAMMPS files
----------------------

Before execution, the driver writes:

.. list-table:: LAMMPS input artifacts
   :header-rows: 1
   :widths: 30 70

   * - File
     - Contents
   * - ``atoms.data``
     - Reproducible randomized species placement on an FCC starting lattice,
       exact particle counts, masses, charges, and the cubic periodic box.
   * - ``pair_potentials.table``
     - One table section for every unique pair, including energy and force.
   * - ``in.otter_md``
     - Complete metal-unit LAMMPS input: velocities, NVT equilibration, NVE
       production, block RDF accumulation, and trajectory output.

After a successful run, the directory also contains ``log.lammps``,
``screen.log``, ``rdf_blocks.dat``, ``trajectory.lammpstrj``,
``md_results.npz``, and ``run_metadata.json``.  The metadata records the
configuration, LAMMPS version, box and cutoff, wall time, and hashes of the
input and output artifacts.  With ``reuse_completed=True``, results are reused
only when the previous run completed normally and the three input hashes still
match.

How :math:`g_{ab}(r)` and :math:`S_{ab}(k)` are obtained
--------------------------------------------------------

LAMMPS accumulates each partial RDF only during the NVE production segment.
``md_gij_r`` is the mean of the saved RDF blocks and
``md_gij_block_sem`` is the standard error across those blocks.  The columns
are identified by ``md_pair_labels``.

The partial structure factors are not obtained by Fourier transforming the
finite-range RDF.  For each saved NVE frame, the driver evaluates the periodic
density modes

.. math::

   \rho_a(\mathbf{k}) = \sum_{j\in a}
   \exp(i\mathbf{k}\cdot\mathbf{r}_j),
   \qquad \mathbf{k}=\frac{2\pi}{L}\mathbf{n},

and the Ashcroft--Langreth partial estimator

.. math::

   S_{ab}(k) = \left\langle
   \frac{\operatorname{Re}[\rho_a(\mathbf{k})\rho_b^*(\mathbf{k})]}
   {\sqrt{N_aN_b}}
   \right\rangle_{\text{vectors in bin, frames}}.

The relevant output fields are:

.. code-block:: python

   labels = result["md_pair_labels"]

   r = result["md_r_bohr"]
   gab = result["md_gij_r"]
   gab_sem = result["md_gij_block_sem"]

   k = result["md_k_bohr_inv"]
   Sab = result["md_sij_k"]
   Sab_sem = result["md_sij_frame_sem"]
   vectors_per_bin = result["md_vectors_per_k_bin"]

The reported :math:`S_{ab}` SEM is the standard deviation of the per-frame
shell averages divided by the square root of the saved-frame count.  It does
not correct for temporal autocorrelation, so it is a sampling diagnostic, not
automatically a rigorous confidence interval.  Small-:math:`k` bins contain
few periodic vectors and can have much larger finite-cell uncertainty; inspect
``md_vectors_per_k_bin`` before plotting them.

Numerical checks before accepting a run
---------------------------------------

At minimum, verify all of the following:

* the NVT segment is discarded and observables use only NVE production;
* ``md_nve_relative_energy_drift`` is acceptably small and decreases when the
  timestep is reduced;
* the mean NVE temperature is compatible with the target state;
* increasing NVT duration does not change the production averages;
* increasing NVE duration reduces sampling noise without shifting the mean;
* increasing particle number and the box length does not change the resolved
  peaks or the retained low-:math:`k` region;
* the MD cutoff lies in a region where the shifted potential is physically
  negligible;
* every mixture pair appears exactly once in ``md_pair_labels``.

These checks matter more than adopting one fixed timestep or step count.  Hot
light ions generally require a smaller timestep, while low-:math:`k`
structure generally requires a larger box and longer sampling.  The driver
automates data conversion and auditing, but it cannot choose a universally
converged MD protocol for every material and thermodynamic state.

LAMMPS is described by :cite:t:`ThompsonEtAl2022`.
