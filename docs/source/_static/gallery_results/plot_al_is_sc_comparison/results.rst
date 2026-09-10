.. Recorded local validation output; presentation only, not solver input.

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
   
   model             path      mu [Ha]       Zbar       HNC residual   wall time [s]
   ----------------------------------------------------------------------------------
   KS-DFT            IS      0.61838312   3.162606      2.568e-05           18.42
   KS-DFT            SC      0.61838312   3.121057      1.262e-05          101.47
   Thomas--Fermi     IS      0.69292041   4.975863      5.391e-06            1.00
   Thomas--Fermi     SC      0.69292041   4.765113      8.344e-07            9.21
   
   KS-DFT finite bound levels
   level    E_IS [Ha]    E_SC [Ha]    FD_IS      FD_SC
   ---------------------------------------------------------
   1s      -53.995318  -54.137957   1.000000   1.000000
   2s       -2.812967   -3.003578   0.998024   0.998601
   2p       -1.447091   -1.633521   0.976954   0.983458
   Thomas--Fermi is a semiclassical density model and has no discrete KS-level table.
   
