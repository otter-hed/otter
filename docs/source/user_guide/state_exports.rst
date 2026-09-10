Workflow results and portable NPZ files
=======================================

The return value of :func:`otter.solve_plasma_workflow` is the primary API.
It contains the electronic result and, when ``ion_temperature_ev``
is set, the QOZ/HNC result. Check the convergence flags before using a result.
The same state can be written as a portable NPZ
archive.  NPZ files contain only numeric arrays and fixed-width strings and
are loaded with ``allow_pickle=False``.

Notation on this page distinguishes the electron channels
:math:`V_{Ie}` and :math:`V_{ee}` from the effective ion--ion pair potential
:math:`V_{ab}`, where :math:`a,b` label ionic species.  The Python keys
``vij_k`` and ``vij_r`` retain ``ij`` for compatibility, but both of their
leading axes are ionic-species axes; they are not electron--ion potentials.

This page describes the portable workflow-state schema
``otter_state_v5``.  It adds profile-selected exports while continuing to
load ``otter_state_v1`` through ``otter_state_v4`` archives.  Benchmark
baselines may instead use compact,
benchmark-specific schemas because one archive can contain several model or
thermodynamic states.  Such plotting archives are validated by their own
producer/loader and are not inputs to :func:`otter.load_plasma_state`.
Their field names and provenance follow the corresponding producer and manifest.

In a standard ``otter_state_v5`` archive,
``metadata_json["configuration"]`` stores the complete
``PlasmaWorkflowConfig`` snapshot, including workflow defaults. An archive
containing only overrides would become ambiguous if a later release changed a
default.  ``metadata_json["configuration_nondefault"]`` is the compact reader
view containing required inputs and values that differ from the producing
version's workflow defaults. This snapshot does not expand every lower-level
AA option or record every adaptive solver decision. The producer version and
available convergence diagnostics must therefore be retained too.
Calculation scripts should normally specify only intentional non-default settings.

In-memory access
----------------

For a single species:

.. code-block:: python

   from otter import PlasmaWorkflowConfig, solve_plasma_workflow

   config = PlasmaWorkflowConfig(
       elements=["Al"],
       temperature_ev=15.0,
       ion_temperature_ev=15.0,
       rho_g_cc=8.1,
   )
   result = solve_plasma_workflow(config)
   aa = result["electronic"]["result"]
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
   q = ion["q_k"]          # n_scr(k)
   f = ion["f_k"]          # n_ion(k)
   G = ion["G_ee_k"]
   chi0 = ion["chi0_k"]
   chi_ee = ion["chi_ee_k"]
   V_ie = ion["v_ie_k"]
   V_ee = ion["v_ee_k"]
   C_ie = ion["c_ie_k"]
   C_ee = ion["c_ee_k"]
   V_ie_r = ion["v_ie_r"]
   V_ee_r = ion["v_ee_r"]
   V_ii_k = ion["vij_k"][0, 0]
   V_ii_r = ion["vij_r"][0, 0]
   g_ii = ion["gij_r"][0, 0]
   S_ii = ion["sij_k"][0, 0]

``aa`` is the native AA dictionary; it is not the top-level workflow result.
AA metadata is in ``aa["meta"]``, not ``result["meta"]``. The example above
uses a quantum full/external calculation: full-only and TF results do not
provide all the same profile or level fields. With no ionic stage,
``result["ion"]`` is ``None``.

For a mixture, each average-atom result is in
``result["electronic"]["result"]["species"][i]["result"]``.  The order is
``result["species_symbols"]``.  Species axes in QOZ arrays use the same order.

.. code-block:: python

   species = result["electronic"]["result"]["species"]
   for symbol, entry in zip(result["species_symbols"], species):
       aa = entry["result"]
       Zbar = aa["zbar_partition"]
       Zstar = aa["zstar"]
       print(symbol, Zbar, Zstar)

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

``partition`` identifies the charge defined by the ionic density partition.
For this mean ionization, use ``Zbar = aa["zbar_partition"]`` directly;
no additional integration or export is needed.

These quantities are distinct:

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
it therefore represents background ionization for current TF results, despite
its historical key name. Use ``zbar_partition`` or ``zstar`` for explicit,
model-independent definitions.

New standard NPZ exports contain the species vector ``state["zstar"]`` in
every profile, including ``electronic_summary``. For one element:

.. code-block:: python

   Zstar = float(state["zstar"][0])
   Zbar_partition = float(state["zbar_partition"][0])
   Zbar_ws = float(state["zbar_aa_ws"][0])

The direct AA field is available from Otter 0.3.1; the standard NPZ species
vector already existed in 0.3.0.

The explicit AA ``zstar`` field is used when exporting. Older AA results
without it remain supported via the same density ratio. Existing user code
can still calculate it directly:

.. code-block:: python

   Zstar = (aa["zstar"] if "zstar" in aa else
            aa["n0"] / aa["meta"]["n_i_bohr3"])

For an older portable file without ``zstar``, when both density vectors exist:

.. code-block:: python

   Zstar = (state["zstar"] if "zstar" in state else
            state["n0_bohr3"] / state["n_i_bohr3"])

The loader does not fabricate fields absent from an older file. A portable
file uses ``metadata_json``; older electronic-only exports instead use
``meta_json`` and are read with NumPy, not ``load_plasma_state``.

Ion-orbital profiles and wavefunctions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a quantum AA result, the existing ionic density components are

.. math::

   n^{\rm ion}_{nl}(r) = \frac{2(2l+1)}{4\pi}
   f_{\rm FD}(E_{nl}) M(E_{nl}) f_{\rm cut}(r) |R_{nl}(r)|^2.

Their sum is ``aa["n_ion"]``. The unweighted radial wavefunctions use their
own bound grid, which can differ from the density grid. Access them and
transform the ionic components without running another solve:

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
not its square root. This is a display/analysis option, not a change of
orbital normalization or a prescription for rebuilding the ionic density.
It never changes the stored raw wavefunctions or the ionic components.
Near-threshold status remains the one reported by the AA solve. For matched
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

Use the completed ``result`` from the workflow above. This standard export
keeps basic state metadata and level tables, then adds orbital profiles:

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

The file stores raw :math:`R_{nl}`, not FD-weighted amplitudes and not
:math:`4\pi r^2 R_{nl}`. The ionic densities already include FD, degeneracy,
:math:`M(E)`, and :math:`f_{\rm cut}`. Saving never reruns the solver.
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
each array. For mixtures choose ``species_1_``, etc.; never assume different
species have the same bound or native radial grid.

By default, archives retain :math:`r<20\,a_B` and :math:`k<20\,a_B^{-1}`.
To preserve all available samples for a single species, pass these additional
options to ``StateExportOptions`` (the endpoints are exclusive):

.. code-block:: python

   r_max_bohr = float(np.nextafter(max(aa["r"][-1], aa["r_bound"][-1]), np.inf))
   k_max_bohr_inv = float(np.nextafter(ion["k"][-1], np.inf))

For mixtures use the largest radial endpoint across species. Wavefunctions
beyond the numerical bound grid are not invented by the export. TF does not
produce orbital fields; a full-AA-only result without an ionic grid has no
stored per-level form factors.

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

Species and pair axes
---------------------

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
     - Chemical potential, WS radius, background electron density, and ion
       AA-cell ion density (not bulk partial density in a mixture).

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
It therefore uses its own paired arrays ``species_i_n_free_r_bohr`` and
``species_i_n_free_r`` instead of inserting NaNs on the full native grid.

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
actual field list.  It also records that these analysis archives are not
solver-restart checkpoints.  Programmatic discovery does not require a
hard-coded list:

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
