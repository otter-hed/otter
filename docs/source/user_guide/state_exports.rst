Workflow results and portable NPZ files
=======================================

:func:`otter.solve_plasma_workflow` returns electronic results and, when
``ion_temperature_ev`` is set, QOZ/HNC results. Convergence diagnostics are
included in each result. :func:`otter.save_plasma_state` exports selected
quantities to NPZ files containing numeric arrays and fixed-width strings,
readable with ``allow_pickle=False``.

For parameter scans with a shared output k axis, see :doc:`parameter_grids`.

:math:`V_{Ie}` and :math:`V_{ee}` denote electron--ion and electron--electron
channels. :math:`V_{ab}` denotes the effective ion--ion pair potential;
both leading axes of ``vij_k`` and ``vij_r`` index ionic species.

The current portable schema is ``otter_state_v5``. The loader also supports
schemas ``v1`` through ``v4``. Compact benchmark archives have separate
schemas and loaders; they are not inputs to :func:`otter.load_plasma_state`.

In a standard ``otter_state_v5`` archive,
``metadata_json["configuration"]`` stores the complete
``PlasmaWorkflowConfig`` snapshot, including workflow defaults.
``metadata_json["configuration_nondefault"]`` contains required inputs and
settings that differ from those defaults. Lower-level AA options and adaptive
solver decisions are not fully expanded in this snapshot; the producing
version and convergence diagnostics are recorded separately.

.. _in-memory-access:

Electronic structure
--------------------

For a system containing :math:`N_s` species, the electronic results are
stored in ``result["electronic"]["species"]``. The list contains one entry
per species and preserves the order in ``result["species_symbols"]``.
This convention includes pure elements (``N_s=1``).

.. versionadded:: 0.4.0
   The ``electronic["species"]`` access path.

.. code-block:: python

   for entry in result["electronic"]["species"]:
       aa = entry["result"]
       print(entry["element"], aa["n0"], aa["zbar_partition"], aa["zstar"])

Each entry contains ``element``, ``count``, ``x`` (number fraction),
``r_ws_bohr``, ``mu_ha`` and ``result``. The AA dictionary in ``result`` retains
the species' native radial grid, density profiles and metadata.

Cached electronic continuation and SC feedback return the same structure.

Species summaries are also available as vectors of shape ``(N_s,)``:

.. code-block:: python

   electronic = result["electronic"]
   Zbar = electronic["zbar_partition"]
   Zstar = electronic["zstar"]
   n_i_aa = electronic["n_i_aa"]
   n0 = electronic["n0"]
   mu = electronic["mu"]

These vectors are available for electronic-only calculations. They contain
snapshots of the final AA values. For older cached results, a field is omitted
if it is unavailable for any species.

For a pure aluminium plasma, the AA result is the first list entry:

.. code-block:: python

   from otter import PlasmaWorkflowConfig, solve_plasma_workflow

   config = PlasmaWorkflowConfig(
       elements=["Al"],
       temperature_ev=15.0,
       ion_temperature_ev=15.0,
       rho_g_cc=8.1,
   )
   result = solve_plasma_workflow(config)
   aa = result["electronic"]["species"][0]["result"]
   ion = result["ion"]

   r_aa = aa["r"]
   n_full = aa["n_full"]
   n_bound = aa["n_bound"]
   n_cont = aa["n_cont"]
   energies = aa["bound_energy_ha"]
   level_density = aa["bound_orbital_density_r"]
   ion_level_density = aa["ion_orbital_density_r"]
   V_eff = aa["v_full"]
   V_nuc = aa["v_nuc"]
   V_H = aa["v_H"]
   V_xc = aa["v_xc"]
   Zbar = aa["zbar_partition"]
   Zstar = aa["zstar"]
   mu = aa["mu"]

   k = ion["k"]
   q = ion["q_k"][0]       # n_scr(k) for Al
   f = ion["f_k"][0]       # n_ion(k) for Al
   G = ion["G_ee_k"]
   chi0 = ion["chi0_k"]
   chi_ee = ion["chi_ee_k"]
   V_ie = ion["v_ie_k"][0]
   V_ee = ion["v_ee_k"]
   C_ie = ion["c_ie_k"][0]
   C_ee = ion["c_ee_k"]
   V_ie_r = ion["v_ie_r"][0]
   V_ee_r = ion["v_ee_r"]
   V_ii_k = ion["vij_k"][0, 0]
   V_ii_r = ion["vij_r"][0, 0]
   g_ii = ion["gij_r"][0, 0]
   S_ii = ion["sij_k"][0, 0]

AA metadata is in ``aa["meta"]``. The example uses a quantum full/external
calculation; available profiles and levels depend on the electronic model
and completed stages. Without an ionic stage, ``result["ion"]`` is ``None``.

Element symbols can also be used to select a species:

.. code-block:: python

   aa_by_element = {
       entry["element"]: entry["result"]
       for entry in result["electronic"]["species"]
   }
   carbon = aa_by_element["C"]
   Zbar_C = carbon["zbar_partition"]
   Zstar_C = carbon["zstar"]

Compatibility
~~~~~~~~~~~~~

Otter 0.3.1 exposes ``result["electronic"]["result"]`` for a
single AA, and ``result["electronic"]["result"]["species"][i]["result"]``
for a mixture species. These paths remain supported and reference the same
AA dictionaries as ``electronic["species"]``. Treat completed results as
read-only: editing an AA dictionary changes its aliases but does not update
the summary vectors.

Mixture-wide common-mu diagnostics remain in
``result["electronic"]["result"]["meta"]``; a species' AA metadata remains in
``aa["meta"]``. For cached continuation, continue passing the raw
``result["electronic"]["result"]`` together with ``result["electronic"]["kind"]``.
Low-level AA return formats and the NPZ schema are unchanged. The ionic
array changes are described in `Migration from 0.3.1`_ below.

QOZ/HNC results
--------------------

``result["ion"]`` contains the ionic correlations, effective pair potentials
and electron-response channels on the common QOZ grids. It is ``None`` when
``ion_temperature_ev`` is not specified.

.. code-block:: python

   ion = result["ion"]
   r = ion["r"]
   k = ion["k"]
   G_ee = ion["G_ee_k"]
   chi0 = ion["chi0_k"]
   chi_ee = ion["chi_ee_k"]
   g_ab = ion["gij_r"]
   S_ab = ion["sij_k"]
   V_ab = ion["vij_k"]

The local-field correction (LFC) :math:`G_{ee}(k)` and the response functions
:math:`\chi^0_{ee}(k)` and :math:`\chi_{ee}(k)` describe the common electron
response. Each has shape ``(N_k,)``, independent of the number of ionic
species. The selected models are recorded in ``qoz_response_lfc_model`` and
``qoz_response_chi0_model``. These fields belong to the QOZ result, not to an
individual species' AA dictionary.

.. versionchanged:: 0.4.0
   Workflow species fields retain their species axis for a pure element.

The in-memory dimensions are:

.. list-table:: QOZ/HNC fields
   :header-rows: 1
   :widths: 46 27 27

   * - Keys
     - Single species
     - :math:`N_s` species
   * - ``r``, ``k``
     - ``(N_r,)``, ``(N_k,)``
     - ``(N_r,)``, ``(N_k,)``
   * - ``G_ee_k``, ``chi0_k``, ``chi_ee_k``, ``v_ee_k``, ``c_ee_k``
     - ``(N_k,)``
     - ``(N_k,)``
   * - ``v_ee_r``, ``c_ee_r``
     - ``(N_r,)``
     - ``(N_r,)``
   * - ``q_k``, ``f_k``, ``n_scr_k``, ``n_ion_k``, ``v_ie_k``, ``c_ie_k``
     - ``(1, N_k)``
     - ``(N_s, N_k)``
   * - ``n_scr_r``, ``n_ion_r``, ``v_ie_r``, ``c_ie_r``
     - ``(1, N_r)``
     - ``(N_s, N_r)``
   * - ``gij_r``, ``hij_r``, ``cij_r``, ``vij_r``
     - ``(1, 1, N_r)``
     - ``(N_s, N_s, N_r)``
   * - ``sij_k``, ``vij_k``
     - ``(1, 1, N_k)``
     - ``(N_s, N_s, N_k)``
   * - ``zbar``, ``zbar_qoz``, ``zbar_partition``, ``zbar_aa_ws``, ``n_i``
     - ``(1,)``
     - ``(N_s,)``
   * - ``zstar``, ``n_i_aa``
     - ``(1,)``
     - ``(N_s,)``
   * - ``gii_r``, ``sii_k`` (diagonal species pairs)
     - ``(1, N_r)``, ``(1, N_k)``
     - ``(N_s, N_r)``, ``(N_s, N_k)``
   * - ``bridge_r``, ``hnc_effective_potential_r`` (when provided)
     - ``(1, 1, N_r)``
     - ``(N_s, N_s, N_r)``

The species order is ``result["species_symbols"]``.
``ion["sij_k"][i, j]`` selects :math:`S_{ij}(k)`; for a pure element,
``i = j = 0``. All like- and unlike-species pairs are included.

Profiles use the same indexing for any composition:

.. code-block:: python

   for i, symbol in enumerate(result["species_symbols"]):
       q_i = ion["q_k"][i]
       f_i = ion["f_k"][i]
       V_ie_i = ion["v_ie_k"][i]
       Zbar_i = ion["zbar_partition"][i]
       Zstar_i = ion["zstar"][i]
       n_i_bulk = ion["n_i"][i]
       n_i_cell = ion["n_i_aa"][i]

``n_i`` contains bulk partial ion densities; ``n_i_aa`` contains the inverse
AA-cell volumes. They need not coincide in a mixture. ``zstar`` uses the latter.
Per-species charge-normalization diagnostics in ``charge_fix`` also have shape
``(N_s,)``. Global HNC convergence flags and residuals remain scalars.

``v_ei_k`` and ``v_ei_r`` are aliases of ``v_ie_k`` and ``v_ie_r``.
``gee_k`` and ``g_ee_k`` are compatibility aliases of ``G_ee_k``.
The older, single-component-only shortcuts ``vii_r``, ``vii_k``, ``hii_r`` and
``cii_r`` remain one-dimensional. Use ``vij_r``, ``vij_k``, ``hij_r`` and
``cij_r`` with two species indices in composition-independent code.

Convergence and solver diagnostics are available in the same dictionary:
``hnc_converged``, ``hnc_output_residual``, ``closure_transform_max_abs`` and
``stage_meta``. Model-specific diagnostics are only provided by the solver
that defines them; for example, the current VMHNC implementation supplies
single-species bridge and hard-sphere reference parameters.

Migration from 0.3.1
~~~~~~~~~~~~~~~~~~~~

The existing dictionary keys are retained, but single-species ionic profiles
now have a leading axis of length one. Select that axis before plotting or
applying a one-dimensional grid mask:

.. code-block:: python

   q_al = ion["q_k"][0]
   Zbar_al = float(ion["zbar_partition"][0])
   S_al = ion["sij_k"][0, 0]
   mask = ion["k"] < 5.0
   q_all_species = ion["q_k"][:, mask]

Code that must read both old and new single-species results can use
``np.atleast_2d(ion["q_k"])[0]`` and
``np.asarray(ion["zbar_partition"]).item()``. The latter is only appropriate
when exactly one species is expected. Multicomponent profile dimensions and
common electron-response dimensions are unchanged. Native AA scalar fields,
such as ``aa["zstar"]``, remain scalars.

Portable NPZ arrays already retain the species axes and need no migration.
The existing single-species scalar representation of charge-normalization
diagnostics inside ``metadata_json`` is also preserved.

Portable access
---------------

The standard NPZ interface is the same for one element and mixtures:

.. code-block:: python

   from otter import load_plasma_state

   state = load_plasma_state("state.npz")
   symbols = state["species_symbols"]
   Zbar = state["zbar_partition"]  # shape (number of species,)
   Zstar = state["zstar"]          # same species order

Orbital arrays remain separate under ``species_0_``, ``species_1_``, etc.,
because species can have different level counts and native radial grids.

Mean-ionization definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The AA result distinguishes three mean-ionization definitions:

* In QM, ``aa["zbar"]`` and ``aa["zbar_ws"]`` are
  :math:`Z-\int_0^{R_{\rm WS}}4\pi r^2 n_{\rm ion}(r)\,dr`.
* ``aa["zbar_partition"]`` is
  :math:`Z-\int_0^{R_{\max}}4\pi r^2 n_{\rm ion}(r)\,dr`.
* ``aa["zstar"]`` is :math:`Z^*=n_0/n_i^{\rm AA}`.
  Here :math:`n_i^{\rm AA}=1/V_{\rm WS}` is the AA-cell ion density.

For mixtures, compute the last ratio separately for each species' AA cell;
do not substitute the bulk partial densities in ``ion["n_i"]``.
``ion["zbar"]`` is the charge selected for QOZ by ``qoz_zbar_mode`` and
need not equal the AA WS diagnostic.

TF preserves its historical ``aa["zbar"] = aa["zstar"]`` convention and
does not supply a separate ``aa["zbar_ws"]`` field. Portable
``zbar_aa_ws`` falls back to that legacy ``zbar`` when ``zbar_ws`` is absent;
it represents background ionization for TF results. Use ``zbar_partition``
or ``zstar`` for model-independent field definitions.

Standard NPZ exports contain the species vector ``state["zstar"]`` in
every profile, including ``electronic_summary``. For one element:

.. code-block:: python

   Zstar = float(state["zstar"][0])
   Zbar_partition = float(state["zbar_partition"][0])
   Zbar_ws = float(state["zbar_aa_ws"][0])

The direct AA field is available from Otter 0.3.1; the standard NPZ species
vector already existed in 0.3.0.

Exports use the explicit AA ``zstar`` field when present, with a density-ratio
fallback for older results:

.. code-block:: python

   Zstar = (aa["zstar"] if "zstar" in aa else
            aa["n0"] / aa["meta"]["n_i_bohr3"])

For an older portable file without ``zstar``, when both density vectors exist:

.. code-block:: python

   Zstar = (state["zstar"] if "zstar" in state else
            state["n0_bohr3"] / state["n_i_bohr3"])

The loader returns the fields stored in the file. Portable files use
``metadata_json``. Older electronic-only exports use ``meta_json`` and require
direct NumPy loading.

Ion-orbital profiles and wavefunctions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a quantum AA result, the ionic density components are

.. math::

   n^{\rm ion}_{nl}(r) = \frac{2(2l+1)}{4\pi}
   f_{\rm FD}(E_{nl}) M(E_{nl}) f_{\rm cut}(r) |R_{nl}(r)|^2.

Their sum is ``aa["n_ion"]``. The unweighted radial wavefunctions use their
own bound grid, which can differ from the density grid:

.. code-block:: python

   from otter import bound_wavefunctions, ion_orbital_form_factors

   r_wave = aa["r_bound"]
   R_nl = bound_wavefunctions(aa)  # default: no occupation, M(E), or f_cut
   R_nl_fd = bound_wavefunctions(aa, multiply_fd=True)  # f_FD * R_nl
   r_density = aa["r"]
   n_ion_nl_r = aa["ion_orbital_density_r"]
   n_ion_nl_k = ion_orbital_form_factors(aa, r=ion["r"], k=ion["k"])

Arrays use ``(l_index, radial_state_index, grid)`` ordering, matching
``aa["bound_energy_ha"]`` and ``aa["bound_l_list"]``; unused slots are zero.
Sum the density/form-factor arrays over axes ``(0, 1)`` to recover ``n_ion``
or the species' ``f_k``, up to
floating-point error. Pass the **complete** QOZ grids to the transform; crop
only after transforming. For a mixture, pass each species' ``aa`` result
with the same ionic grids.

``multiply_fd=True`` multiplies the wavefunction amplitude by the FD factor,
not its square root. It affects the returned wavefunctions only; stored
wavefunctions, ionic densities and AA threshold status are unchanged. For matched
shallow states, normalization includes the analytic exterior tail; the
exported radial interval alone need not contain unit probability.

The standalone plotting example uses default numerical settings and shows
wavefunctions, ionic density components, and form factors:

.. code-block:: console

   poetry run python examples/bound_orbitals.py

Set ``MULTIPLY_WAVEFUNCTION_BY_FD = True`` in that script to weight the
wavefunction plot; it defaults to ``False``. TF retains its total-density
description and has no per-level wavefunctions or ionic density components.
The two real-space panels display :math:`4\pi r^2 R_{nl}(r)` (optionally
multiplied by FD) and :math:`4\pi r^2 n^{\rm ion}_{nl}(r)`, using each
quantity's own radial grid. This display scaling does not modify the raw
arrays or the input to the Fourier transform.

.. _orbital-npz-access:

Saving and reading orbital NPZ data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``orbital_densities`` group adds orbital profiles to the electronic summary:

.. code-block:: python

   from otter import StateExportOptions, save_plasma_state, load_plasma_state

   save_plasma_state(
       "Al_orbitals.npz",
       result,
       options=StateExportOptions(
           profile="electronic_summary",
           include_groups=("orbital_densities",),
       ),
   )
   state = load_plasma_state("Al_orbitals.npz")

The file stores unweighted :math:`R_{nl}`. Ionic densities include FD,
degeneracy, :math:`M(E)`, and :math:`f_{\rm cut}`. Display factors are excluded.
Per-level form factors are produced on the original full QOZ grid before
any export window is applied; loading reads the stored values directly.

Each exported row corresponds to one bound level; unlike the in-memory
``(l_index, radial_state_index, grid)`` tables, unused slots are omitted:

.. code-block:: python

   import numpy as np

   prefix = "species_0_"  # order follows state["species_symbols"]
   n = state[prefix + "bound_principal_n"]
   l = state[prefix + "bound_l"]
   energy = state[prefix + "bound_energy_ha"]

   r_wave = state[prefix + "r_bound_bohr"]
   R_nl = state[prefix + "bound_wavefunction_r"]
   r_density = state[prefix + "r_bohr"]
   n_nl = state[prefix + "ion_orbital_density_r"]
   k = state[prefix + "orbital_k_bohr_inv"]
   f_nl = state[prefix + "ion_orbital_density_k"]

   # Select 2p when this level exists in the result.
   indices = np.flatnonzero((n == 2) & (l == 1))
   if indices.size:
       j = indices[0]
       R_2p, n_2p, f_2p = R_nl[j], n_nl[j], f_nl[j]

The three profile shapes are ``(N_level, N_r_bound)``,
``(N_level, N_r_native)``, and ``(N_level, N_k)``. Use the matching grid for
each array. Mixture species use separate prefixes and may have different grids.

By default, archives retain :math:`r<20\,a_B` and :math:`k<20\,a_B^{-1}`.
To preserve all available samples for a single species, pass these additional
options to ``StateExportOptions`` (the endpoints are exclusive):

.. code-block:: python

   r_max_bohr = float(np.nextafter(max(aa["r"][-1], aa["r_bound"][-1]), np.inf))
   k_max_bohr_inv = float(np.nextafter(ion["k"][-1], np.inf))

For mixtures use the largest radial endpoint across species. Wavefunctions
are exported within their numerical bound grid. TF has no orbital fields;
per-level form factors require an ionic grid.

Plotting code is included in ``examples/bound_orbitals.py`` (in-memory arrays)
and ``docs/examples/plot_al_full_workflow.py`` (standard NPZ export).

Export profiles
---------------

The default profile is ``complete`` and preserves the previous full-state
behaviour.  Smaller profiles retain the thermodynamic inputs, species order,
convergence metadata, :math:`R_{\rm WS}`, chemical potential, background and
ion densities, and the relevant mean-ionization definitions, while omitting
large arrays that the selected analysis does not use.

.. list-table:: Built-in export profiles
   :header-rows: 1
   :widths: 24 35 41

   * - Profile
     - Retained scientific quantities
     - Required calculation stage
   * - ``electronic_summary``
     - Basic state information, :math:`\bar Z`, :math:`Z^*`, :math:`\mu`,
       :math:`R_{\rm WS}`, :math:`n_0`, and :math:`n_i`, together with the
       compact bound-level table (energies, occupations, and
       pressure-ionization weights).
     - Full average atom only; external and ion stages are not required.
   * - ``electronic_levels``
     - Compatibility alias for ``electronic_summary``.  It retains the same
       summary and bound-level quantities.
     - Full average atom only.  Radial densities and potentials are omitted.
   * - ``ion_structure``
     - The electronic summary and bound levels plus :math:`f_a(k)`,
       :math:`q_a(k)`, :math:`g_{ab}(r)`, and :math:`S_{ab}(k)`.
     - Completed QOZ/HNC calculation.
   * - ``complete``
     - Every public group, including electronic profiles and potentials,
       orbital densities, response channels, pair potentials, and solver
       history.
     - Completed QOZ/HNC calculation; this is the backward-compatible default.

Optional groups can be added without switching to ``complete``.  For example,
the following stores ion structure and the pair potential used to obtain it,
but not electronic radial profiles or response intermediates:

.. code-block:: python

   options = StateExportOptions(
       profile="ion_structure",
       include_groups=("pair_potential",),
   )

To retain the electronic response used by QOZ, including ``chi0_k``,
``chi_ee_k``, and ``G_ee_k``, add ``qoz_response``:

.. code-block:: python

   options = StateExportOptions(
       profile="ion_structure",
       include_groups=("qoz_response",),
   )

For a detailed QOZ/HNC diagnostic archive, also request the effective pair
potential and HNC direct/total correlations and residual history:

.. code-block:: python

   options = StateExportOptions(
       profile="ion_structure",
       include_groups=(
           "qoz_response",
           "pair_potential",
           "solver_history",
       ),
   )

The available groups are exported as ``otter.STATE_EXPORT_GROUPS`` and the
built-in mappings as ``otter.STATE_EXPORT_PROFILES``.  They are
``electronic_summary``, ``bound_levels``, ``electronic_profiles``,
``electronic_potentials``, ``orbital_densities``, ``electronic_spectra``,
``ion_structure``, ``qoz_response``, ``pair_potential``, and
``solver_history``.  Every archive records its resolved included and omitted
groups in ``metadata_json``.

When using ``PlasmaWorkflowConfig``, select the same options with
``state_export_profile`` and ``state_include_groups``. Selecting a save group
does not run a missing calculation stage or change the in-memory result.
``complete`` means all available public export groups, not every internal
solver object; it still applies the configured export windows.

Save and load
-------------

Set ``save_state_npz`` on a workflow.  The selected profile determines whether
an ion-structure stage is required:

.. code-block:: python

   from otter import PlasmaWorkflowConfig, load_plasma_state, solve_plasma_workflow

   config = PlasmaWorkflowConfig(
       elements=["C", "H"],
       counts=[1.0, 1.36],
       temperature_ev=8.617333,
       ion_temperature_ev=8.617333,
       rho_g_cc=2.94,
       save_state_npz=True,
       save_state_path="outputs/ch1p36_state.npz",
       state_export_profile="ion_structure",
   )
   result = solve_plasma_workflow(config)
   state = load_plasma_state(result["saved_paths"]["state_npz"])

``load_plasma_state`` checks the schema, shapes, aliases, finite values, and
the recorded export windows.  Plain NumPy access is also available when
validation is not required:

.. code-block:: python

   import numpy as np

   with np.load("outputs/ch1p36_state.npz", allow_pickle=False) as archive:
       print(archive.files)
       k = archive["k_bohr_inv"]
       q = archive["q_k"]
       f = archive["f_k"]
       S = archive["sij_k"]
       Zstar = archive["zstar"]

By default the archive retains :math:`r < 20\,a_{\rm B}` and
:math:`k < 20\,a_{\rm B}^{-1}`; both limits are exclusive.  Change them with
``state_r_max_bohr`` and ``state_k_max_bohr_inv``.  A completed in-memory
workflow can also be saved explicitly:

.. code-block:: python

   from otter import StateExportOptions, save_plasma_state

   save_plasma_state(
       "outputs/state.npz",
       result,
       options=StateExportOptions(
           profile="ion_structure",
           r_max_bohr=12.0,
           k_max_bohr_inv=15.0,
       ),
   )

Ion-stage profiles require a converged HNC result by default.
``electronic_summary`` and ``electronic_levels`` can be written directly from
a successful full-AA workflow with ``run_mode="full"`` and no
``ion_temperature_ev``.  Files are written atomically, so an interrupted write
does not replace an existing state.

Full-AA-only exports record electronic convergence diagnostics but do not
reject an unconverged electronic result automatically. Inspect
``metadata["convergence"]["electronic"]`` (final-stage SCF, external and
threshold status as applicable). A successful load validates the file, not
the physical accuracy of the calculation. ``solver_history`` exports HNC
correlations and residual history, not the complete electronic SCF trajectory.

.. _species-and-pair-axes:

Portable species and pair axes
------------------------------

For :math:`N_s` species, :math:`N_r` common radial points, and :math:`N_k`
common reciprocal points:

The table lists possible fields, not fields present in every profile:

* ``ion_structure`` supplies ``f_k``, ``q_k``, ``gij_r`` and ``sij_k``.
* ``qoz_response`` adds the common-grid ``n_ion_r``, ``n_scr_r`` and electron
  response, electron--ion and electron--electron channels.
* ``pair_potential`` supplies ``vij_k`` and ``vij_r``.
* ``solver_history`` supplies ``hij_r``, ``cij_r`` and HNC residual history.

All profiles keep the scalar species vectors such as ``zstar`` and
``zbar_partition``. ``zbar_qoz`` and the common r/k grids require an ion-stage
export group. The r and k windows are independent, so an exported pair of
cropped grids need not form a complete DST lattice or have equal lengths.

.. list-table:: Common QOZ/HNC arrays
   :header-rows: 1
   :widths: 27 23 50

   * - Key
     - Shape
     - Quantity
   * - ``species_symbols``
     - ``(N_s,)``
     - Species order for every species and pair axis.
   * - ``r_bohr``, ``k_bohr_inv``
     - ``(N_r,)``, ``(N_k,)``
     - Common QOZ/DST grids.
   * - ``n_ion_r``, ``n_scr_r``
     - ``(N_s, N_r)``
     - Ion-associated and charge-closed screening densities.
   * - ``f_k`` / ``n_ion_k``
     - ``(N_s, N_k)``
     - Identical aliases for :math:`n_{\rm ion}(k)`.
   * - ``q_k`` / ``n_scr_k``
     - ``(N_s, N_k)``
     - Identical aliases for the screening clouds used by QOZ.
   * - ``chi0_k``, ``chi_ee_k``
     - ``(N_k,)``
     - :math:`\chi^0_{ee}(k)` and :math:`\chi_{ee}(k)`.
   * - ``G_ee_k``
     - ``(N_k,)``
     - Selected local-field correction :math:`G_{ee}(k)`.
       ``gee_k`` and ``g_ee_k`` are temporary compatibility aliases.
   * - ``v_ie_k`` / ``v_ei_k``
     - ``(N_s, N_k)``
     - Electron--ion potential; the two names are explicit aliases.
   * - ``v_ee_k``
     - ``(N_k,)``
     - Electron--electron potential.
   * - ``c_ie_k``, ``c_ee_k``
     - ``(N_s, N_k)``, ``(N_k,)``
     - Electron--ion and electron--electron direct correlations.
   * - ``v_ie_r`` / ``v_ei_r``, ``v_ee_r``
     - species/common radial arrays
     - Finite-DST real-space representations of the corresponding channels.
   * - ``c_ie_r``, ``c_ee_r``
     - species/common radial arrays
     - Finite-DST real-space direct-correlation channels.
   * - ``vij_r``, ``vij_k``
     - ``(N_s, N_s, N_r/k)``
     - Effective ion--ion pair potentials :math:`V_{ab}(r)` and
       :math:`V_{ab}(k)`; :math:`a,b` are ionic-species indices.
   * - ``gij_r``, ``hij_r``, ``cij_r``
     - ``(N_s, N_s, N_r)``
     - Pair, total-correlation, and ion direct-correlation functions.
   * - ``sij_k``
     - ``(N_s, N_s, N_k)``
     - Ashcroft--Langreth partial static structure factors.
   * - ``zbar``, ``zbar_qoz``
     - ``(N_s,)``
     - QOZ charge when an ion result is present. Without ions, ``zbar`` uses
       the AA partition charge (or legacy AA ``zbar`` if unavailable).
   * - ``zbar_partition``, ``zbar_aa_ws``, ``zstar``
     - ``(N_s,)``
     - Pseudoatom-partition, AA WS/legacy diagnostic (see the TF caveat above),
       and :math:`Z^*=n_0/n_i^{\rm AA}` definitions.
   * - ``mu_ha``, ``r_ws_bohr``, ``n0_bohr3``, ``n_i_bohr3``
     - ``(N_s,)``
     - Chemical potential, WS radius, background electron density, and
       AA-cell ion density (distinct from bulk partial density in a mixture).

Pair access is direct. With ``ion_structure`` and ``pair_potential`` selected:

.. code-block:: python

   symbols = list(state["species_symbols"])
   i_c = symbols.index("C")
   i_h = symbols.index("H")
   V_ch = state["vij_k"][i_c, i_h]
   g_ch = state["gij_r"][i_c, i_h]
   s_cc = state["sij_k"][i_c, i_c]

Native average-atom arrays
--------------------------

Native electronic fields use a stable species prefix:
``species_0_*``, ``species_1_*``, and so on.  Use
``species_symbols`` to identify the prefix.
For the following example select ``electronic_profiles`` and
``electronic_potentials`` (or use ``complete``). The summary alone contains
no radial grid or density/potential arrays.

.. code-block:: python

   i_c = list(state["species_symbols"]).index("C")
   prefix = f"species_{i_c}_"
   r = state[prefix + "r_bohr"]
   n_full = state[prefix + "n_full_r"]
   n_bound = state[prefix + "n_bound_r"]
   V_eff = state[prefix + "v_full_r_ha"]
   V_nuc = state[prefix + "v_nuc_r_ha"]
   V_H = state[prefix + "v_hartree_r_ha"]
   V_xc = state[prefix + "v_xc_r_ha"]

These fields are present only when their corresponding export group is
selected and the electronic model produces them.  They include ``n_full_r``,
``n_bound_r``, ``n_cont_r``, ``n_ext_r``,
``n_pa_r``, native ``n_scr_r_native`` and ``n_ion_r_native``, TF positive-
and negative-energy densities, tail/source profiles, and repaired diagnostic
profiles.  Potential fields include the full and external effective
potentials and their nuclear, Hartree, exchange--correlation, and correlation
components.

In TF, the compatibility fields ``n_bound`` and ``n_cont`` contain
``n_ion`` and ``n_full - n_ion``, respectively. Use ``n_negative_tf`` and
``n_positive_energy_tf`` for the negative-/positive-energy density split;
the ionic partition is not the same split.

The direct positive-energy A3 density can have a shorter numerical domain.
Its grid and values are stored in ``species_i_n_free_r_bohr`` and
``species_i_n_free_r``.

Bound levels and orbital densities
----------------------------------

When ``bound_levels`` is selected, finite levels below ``bound_energy_cut_ha``
are flattened into aligned one-dimensional arrays. Inclusion does not certify
a level as ``resolved``; the separate threshold status remains authoritative.

``species_i_bound_l``, ``species_i_bound_n_index``, ``species_i_bound_principal_n``
   Angular momentum, radial index, and spectroscopic principal quantum number
   :math:`n=n_{\rm radial}+l`.
``species_i_bound_energy_ha``
   Numerical level energy.  ``species_i_bound_energy_cut_ha`` records the
   continuum edge used for the bound/free classification.
``species_i_bound_fd``, ``species_i_bound_m``, ``species_i_bound_fdm``
   Fermi--Dirac factor, pressure-ionization weight, and their product.
``species_i_bound_occ_deg_fd``, ``species_i_bound_occ_deg_fdm``
   Occupations including the :math:`2(2l+1)` degeneracy.
``species_i_bound_q_ion_ws``
   Per-level contribution to ionic charge inside the WS sphere.
``species_i_bound_orbital_density_r``
   Per-level contribution to :math:`n_{\rm bound}(r)` using the workflow's
   ``bound_occ_mode``.
``species_i_ion_orbital_density_r``
   Per-level contribution to :math:`n_{\rm ion}(r)`, including :math:`M(E)`
   and the radial partition :math:`f_{\rm cut}(r)`.
``species_i_bound_wavefunction_r``, ``species_i_r_bound_bohr``
   Raw :math:`R_{nl}(r)` (:math:`a_B^{-3/2}`) on the bound grid, without FD,
   degeneracy, :math:`M(E)`, or :math:`f_{\rm cut}` weights.
``species_i_ion_orbital_density_k``, ``species_i_orbital_k_bohr_inv``
   Ionic per-level form factors (electron number) on the QOZ grid. Available
   when the workflow contains an ionic grid. Transformation uses the complete
   profiles before the archive's radial and reciprocal windows are applied.

The profile arrays are included with ``orbital_densities`` (also part of
``complete``), not with the small ``electronic_summary`` profile. The new
wavefunction and form-factor fields can be absent in older quantum results.
TF has no per-level profiles or bound wavefunctions.

The two density tables have shape ``(N_level, N_native_r)`` and use the same
level order as ``bound_l``, ``bound_n_index``, and ``bound_energy_ha``:

.. code-block:: python

   E = state[prefix + "bound_energy_ha"]
   l = state[prefix + "bound_l"]
   n = state[prefix + "bound_principal_n"]
   n_level = state[prefix + "bound_orbital_density_r"]
   n_ion_level = state[prefix + "ion_orbital_density_r"]

The scalar strings ``species_i_bound_occ_mode``,
``species_i_threshold_state_status``, and
``species_i_threshold_state_representation`` retain the occupation and
near-threshold classification used by the electronic solve.

Metadata and discovery
----------------------

``metadata_json`` records the complete :class:`otter.PlasmaWorkflowConfig`
snapshot, citation keys, units, thermodynamic state, model choices,
electronic/common-chemical-potential/HNC convergence diagnostics, export
profile and resolved groups, computed stages, windows, definitions, and the
actual field list. These archives support analysis, not solver restarts.
Inspect the metadata with:

.. code-block:: python

   import json

   metadata = json.loads(str(state["metadata_json"].item()))
   print(metadata["configuration"])
   print(metadata["citation_keys"])
   print(metadata["model"])
   print(metadata["convergence"])
   print(metadata["export"])
   print(metadata["units"])
   print(metadata["fields"])

The schema is validated by :func:`otter.load_plasma_state`.  Direct loading
with ``numpy.load(path, allow_pickle=False)`` is also supported.

API
---

The public interface is :class:`otter.StateExportOptions`,
:func:`otter.save_plasma_state`, and :func:`otter.load_plasma_state`.
Lower-level construction and validation are in :mod:`otter.io.state`.
