.. Recorded validation output; presentation only, not solver input.

.. image:: /_static/gallery_results/plot_al_is_sc_comparison/figure_001.svg
   :alt: Recorded result for plot_al_is_sc_comparison

.. image:: /_static/gallery_results/plot_al_is_sc_comparison/figure_002.svg
   :alt: Recorded result for plot_al_is_sc_comparison

.. image:: /_static/gallery_results/plot_al_is_sc_comparison/figure_003.svg
   :alt: Recorded result for plot_al_is_sc_comparison

Recorded terminal output
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Al: rho=8.1 g/cc, Te=Ti=15 eV
   SC status: experimental; Starrett--Saumon (2014), Sec. 2.4, Eqs. (19)--(20).

   model             path      mu [Ha]       Zbar       HNC residual   total wall time [s]
   ---------------------------------------------------------------------------------------
   KS-DFT            IS      0.61838663   3.162604      2.560e-05           21.32
   KS-DFT            SC      0.61838663   3.121100      1.296e-05          110.70
   Thomas--Fermi     IS      0.69292041   4.975863      5.391e-06            1.10
   Thomas--Fermi     SC      0.69292041   4.765113      8.344e-07            9.65
   SC total = initial IS calculation + SC feedback stage.

   KS-DFT finite bound levels
   level    E_IS [Ha]    E_SC [Ha]    FD_IS      FD_SC
   ---------------------------------------------------------
   1s      -53.995320  -54.137834   1.000000   1.000000
   2s       -2.812971   -3.003337   0.998024   0.998600
   2p       -1.447095   -1.633291   0.976955   0.983451
   Thomas--Fermi is a semiclassical density model and has no discrete KS-level table.
