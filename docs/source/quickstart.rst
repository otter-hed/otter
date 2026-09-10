Quick start
===========

Google Colab
------------

Run the aluminium workflow in a browser, with the same four figure groups
and colours as :doc:`gen_examples/plot_al_full_workflow`. The Colab default is
Al at :math:`8.1\,\mathrm{g\,cm^{-3}}` and :math:`T_e=T_i=1\,\mathrm{eV}`:

.. raw:: html

   <p><a href="https://colab.research.google.com/github/otter-hed/otter/blob/main/notebooks/00-otter_intro.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a></p>

Single-species aluminium
------------------------

Run the :download:`single-species workflow
<../../examples/single_species_workflow.py>` from the repository root:

.. code-block:: console

   $ poetry run python examples/single_species_workflow.py

The default state is Al at :math:`8.1\,\mathrm{g\,cm^{-3}}` and
:math:`T_e=T_i=15\,\mathrm{eV}`.  The input block controls the state and output.
The script plots the electronic density, effective potential,
:math:`g_{ii}(r)`, and :math:`S_{ii}(k)`.

To reproduce the four figure groups in the Al gallery and Colab instead,
including the bound-orbital panels and Rayleigh weight, run::

   poetry run python docs/examples/plot_al_full_workflow.py

That example uses :math:`T_e=T_i=1\,\mathrm{eV}` at the same density.

For a strongly coupled one-component ion fluid, the optional
Rosenfeld--Ashcroft closure is selected in the same configuration:

.. code-block:: python

   config = PlasmaWorkflowConfig(
       elements=["Al"],
       temperature_ev=5.0,
       ion_temperature_ev=5.0,
       rho_g_cc=8.1,
       hnc_bridge_model="rosenfeld_ashcroft",
   )

This runs variational modified HNC and reports the self-consistent hard-sphere
packing fraction as ``result["ion"]["vmhnc_eta"]``.  The implementation uses
the bridge-universality ansatz of :cite:t:`RosenfeldAshcroft1979`, the exact
Percus--Yevick hard-sphere reference of :cite:t:`Wertheim1963,Thiele1963`, and
the variational criterion written explicitly by :cite:t:`Faussurier2004`.
This is not IEMHNC: IEMHNC maps a simulation-derived OCP bridge onto a Yukawa
one-component plasma (YOCP) along an isomorph
:cite:p:`IyetomiOgataIchimaru1992,ToliasLuccoCastello2019`, whereas VMHNC
determines a hard-sphere packing fraction variationally.
Plain HNC remains the default.  The current implementation is one-component
only; it deliberately rejects mixtures rather than applying a scalar bridge
to unlike-species channels.

Mixtures
--------

Run the :download:`mixture workflow <../../examples/mixture_workflow.py>`:

.. code-block:: console

   $ poetry run python examples/mixture_workflow.py

Runtime output
--------------

The default report shows the state definition, Wigner--Seitz radius,
``d_n`` and ``d_v`` during SCF, the converged chemical potential, bound-level
tables, and elapsed time.  Set ``debug=True`` in
:class:`otter.PlasmaWorkflowConfig` for charge, continuum, tail-matching,
mixer, and timing diagnostics.  Set ``show_progress=False`` for quiet runs.

Access and export results
-------------------------

The workflow returns electronic and ionic results in separate dictionaries:

.. code-block:: python

   result = solve_plasma_workflow(config)
   electronic = result["electronic"]["result"]
   ion = result["ion"]

   k = ion["k"]
   q_k = ion["q_k"]
   f_k = ion["f_k"]
   G_k = ion["G_ee_k"]
   v_ie_k = ion["v_ie_k"]
   g_ii = ion["gii_r"]
   s_ii = ion["sii_k"]

Set ``save_state_npz=True`` in :class:`otter.PlasmaWorkflowConfig` to save a
portable ``.npz`` archive.  Array names, shapes, units, interaction channels,
and the standalone save/load API are documented in
:doc:`user_guide/state_exports`.

Runtime and cached electronic structure
---------------------------------------

Quantum continuum calculations can be slow near pressure ionization.  The
function
:func:`otter.continue_plasma_workflow_from_electronic_result` can reuse a
validated electronic result while changing QOZ/HNC controls.

Do not interpret solver completion alone as physical validation.  Inspect
charge closure, SCF convergence, HNC residuals, and the model's applicability;
the :doc:`benchmarks/index` pages show the expected reporting pattern.

Consistent PNG and PDF figures
------------------------------

``save_figure`` writes PNG and PDF files from the same Matplotlib figure:

.. code-block:: python

   import matplotlib.pyplot as plt
   from otter.plotting import save_figure, set_style

   set_style("docs", palette="nature")
   fig, ax = plt.subplots()
   ax.plot(ionic["k"], ionic["sii_k"])
   ax.set(xlabel=r"$k$ [Bohr$^{-1}$]", ylabel=r"$S_{ii}(k)$")

   paths = save_figure(fig, "outputs/al_sii")
   print(paths["png"], paths["pdf"])
   plt.show()
