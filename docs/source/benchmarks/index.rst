:tocdepth: 1

Scientific benchmarks
=====================

This page is the complete benchmark catalogue.  Each card opens a runnable
benchmark report containing its physical state, model choices, provenance,
quantitative diagnostics, and downloadable Python source.  Expensive quantum
or molecular-dynamics calculations are represented by compact, checksummed
accepted data; a documentation build never silently recomputes them.

The categories below are only a reading aid.  Every benchmark appears exactly
once, and the left navigation links directly to the individual reports rather
than passing through separate ``detailed`` and ``gallery`` index levels.

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   gen_benchmarks/plot_bethkenhagen_et_al_2020_carbon_ionization
   gen_benchmarks/plot_starrett_saumon_2013_electronic
   gen_benchmarks/plot_starrett_et_al_2014_mixtures_fig3
   gen_benchmarks/plot_starrett_single_species_2013_2014
   gen_benchmarks/plot_argha_roy_carbon_sii
   gen_benchmarks/plot_schorner_et_al_2022_al_sii
   gen_benchmarks/plot_johnson_et_al_2025_two_temperature_al
   gen_benchmarks/plot_ch2_hnc_md
   gen_benchmarks/plot_ion_structure_library
   validation_policy

Electronic structure and ionization
------------------------------------

.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Carbon ionization diagnostics compared with the model-dependent curves in Bethkenhagen et al. (2020).">

.. only:: html

   .. image:: /benchmarks/gen_benchmarks/images/thumb/sphx_glr_plot_bethkenhagen_et_al_2020_carbon_ionization_thumb.png
      :alt: Bethkenhagen et al. carbon-ionization benchmark

   :doc:`gen_benchmarks/plot_bethkenhagen_et_al_2020_carbon_ionization`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Bethkenhagen 2020 carbon ionization</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Otter IS and SC electronic levels, pressure-ionization weights, and mean ionization compared with Starrett and Saumon (2013).">

.. only:: html

   .. image:: /benchmarks/gen_benchmarks/images/thumb/sphx_glr_plot_starrett_saumon_2013_electronic_thumb.png
      :alt: Starrett and Saumon electronic levels and ionization benchmark

   :doc:`gen_benchmarks/plot_starrett_saumon_2013_electronic`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Starrett--Saumon electronic levels and ionization</div>
    </div>

.. thumbnail-parent-div-close

.. raw:: html

    </div>

Equilibrium ion structure
-------------------------

These reports compare :math:`g_{ab}(r)` or :math:`S_{ab}(k)` at
:math:`T_e=T_i` with published or attributed reference data.

.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Nine-state CH1.36 pair-distribution comparison with Figure 3 of Starrett et al. (2014).">

.. only:: html

   .. image:: /benchmarks/gen_benchmarks/images/thumb/sphx_glr_plot_starrett_et_al_2014_mixtures_fig3_thumb.png
      :alt: Starrett et al. CH1.36 mixture benchmark

   :doc:`gen_benchmarks/plot_starrett_et_al_2014_mixtures_fig3`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Starrett 2014 CH1.36 mixtures</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Single-species pair-distribution comparisons with Starrett and Saumon (2014).">

.. only:: html

   .. image:: /benchmarks/gen_benchmarks/images/thumb/sphx_glr_plot_starrett_single_species_2013_2014_thumb.png
      :alt: Starrett and Saumon single-species ion-structure benchmark

   :doc:`gen_benchmarks/plot_starrett_single_species_2013_2014`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Starrett--Saumon single species</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Otter PA-HNC carbon structure factors compared with DFT-MD data provided by Dr. Argha Roy.">

.. only:: html

   .. image:: /benchmarks/gen_benchmarks/images/thumb/sphx_glr_plot_argha_roy_carbon_sii_thumb.png
      :alt: Otter carbon structure factors and Argha Roy DFT-MD data

   :doc:`gen_benchmarks/plot_argha_roy_carbon_sii`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Carbon PA-HNC and DFT-MD</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Equilibrium aluminium LDA/PBE HNC, VMHNC, and same-potential MD compared with Schörner et al. (2022).">

.. only:: html

   .. image:: /benchmarks/gen_benchmarks/images/thumb/sphx_glr_plot_schorner_et_al_2022_al_sii_thumb.png
      :alt: Schörner et al. aluminium HNC VMHNC and MD benchmark

   :doc:`gen_benchmarks/plot_schorner_et_al_2022_al_sii`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Schörner 2022 aluminium structure factors</div>
    </div>

.. thumbnail-parent-div-close

.. raw:: html

    </div>

Two-temperature and method validation
-------------------------------------

These reports test explicitly labelled :math:`T_e\ne T_i` states or compare
different ionic-structure treatments while holding the electronic pair
potential fixed.

.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Two-temperature aluminium pair distributions compared with Johnson, Shaffer, and Murillo (2025).">

.. only:: html

   .. image:: /benchmarks/gen_benchmarks/images/thumb/sphx_glr_plot_johnson_et_al_2025_two_temperature_al_thumb.png
      :alt: Johnson et al. two-temperature aluminium benchmark

   :doc:`gen_benchmarks/plot_johnson_et_al_2025_two_temperature_al`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Johnson 2025 two-temperature aluminium</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Nine PP (CH2) equilibrium and two-temperature states comparing multicomponent HNC with same-potential LAMMPS MD.">

.. only:: html

   .. image:: /benchmarks/gen_benchmarks/images/thumb/sphx_glr_plot_ch2_hnc_md_thumb.png
      :alt: PP CH2 HNC and same-potential MD benchmark

   :doc:`gen_benchmarks/plot_ch2_hnc_md`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">PP (CH2): HNC versus same-potential MD</div>
    </div>

.. thumbnail-parent-div-close

.. raw:: html

    </div>

Cross-observable literature library
-----------------------------------

.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Audited Otter gii and Sii comparisons with aluminium, beryllium, and carbon literature curves.">

.. only:: html

   .. image:: /benchmarks/gen_benchmarks/images/thumb/sphx_glr_plot_ion_structure_library_thumb.png
      :alt: Cross-observable ion-structure literature library

   :doc:`gen_benchmarks/plot_ion_structure_library`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ion-structure literature library</div>
    </div>

.. thumbnail-parent-div-close

.. raw:: html

    </div>

Validation and data governance
------------------------------

The :doc:`validation policy <validation_policy>` defines acceptance metrics,
reference-data provenance, and public-release rules.
