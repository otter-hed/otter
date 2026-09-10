.. Recorded local validation output; presentation only, not solver input.

.. image:: /_static/gallery_results/plot_johnson_et_al_2025_two_temperature_al/figure_001.svg
   :alt: Recorded result for plot_johnson_et_al_2025_two_temperature_al

Recorded terminal output
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   MD is historical (old potential); HNC and VMHNC use refreshed Otter data.
   state                    Otter closure    reference                  RMSE        MAE        max
   al_rho2p7_te1_ti1        HNC              2TTCP HNC+bridge     6.0151e-02 4.6096e-02 1.2076e-01
   al_rho2p7_te1_ti1        HNC              DFT-MD               6.2994e-02 4.5711e-02 1.7804e-01
   al_rho2p7_te1_ti1        HNC              YOCP HNC+bridge      3.0232e-01 2.5212e-01 5.4861e-01
   al_rho2p7_te1_ti1        VMHNC            2TTCP HNC+bridge     2.5222e-02 2.1012e-02 4.9068e-02
   al_rho2p7_te1_ti1        VMHNC            DFT-MD               1.4017e-01 1.1114e-01 3.2790e-01
   al_rho2p7_te1_ti1        VMHNC            YOCP HNC+bridge      3.5828e-01 3.0378e-01 6.3235e-01
   al_rho2p7_te1_ti1        historical MD    2TTCP HNC+bridge     2.5134e-02 2.0100e-02 4.8801e-02
   al_rho2p7_te1_ti1        historical MD    DFT-MD               1.3169e-01 1.0618e-01 3.1333e-01
   al_rho2p7_te1_ti1        historical MD    YOCP HNC+bridge      3.4595e-01 2.9395e-01 6.1181e-01
   al_rho2p7_te1_ti1: shared IS Zbar=3.00047801, VMHNC eta=0.31872211, variational residual=5.657e-06
   al_rho2p7_te3_ti1        HNC              2TTCP HNC+bridge     4.3355e-02 3.2647e-02 9.8181e-02
   al_rho2p7_te3_ti1        HNC              DFT-MD               5.3761e-02 3.9571e-02 1.7870e-01
   al_rho2p7_te3_ti1        HNC              YOCP HNC+bridge      3.0519e-01 2.5571e-01 5.3677e-01
   al_rho2p7_te3_ti1        VMHNC            2TTCP HNC+bridge     3.8358e-02 3.0949e-02 9.5070e-02
   al_rho2p7_te3_ti1        VMHNC            DFT-MD               1.2504e-01 9.9834e-02 3.3819e-01
   al_rho2p7_te3_ti1        VMHNC            YOCP HNC+bridge      3.6548e-01 3.0774e-01 6.2398e-01
   al_rho2p7_te3_ti1        historical MD    2TTCP HNC+bridge     4.5489e-02 3.8238e-02 9.6951e-02
   al_rho2p7_te3_ti1        historical MD    DFT-MD               1.2733e-01 1.0312e-01 3.4288e-01
   al_rho2p7_te3_ti1        historical MD    YOCP HNC+bridge      3.6172e-01 3.0500e-01 6.1846e-01
   al_rho2p7_te3_ti1: shared IS Zbar=3.00047927, VMHNC eta=0.33152966, variational residual=-1.047e-05
   al_rho2p7_te10_ti1       HNC              2TTCP HNC+bridge     6.2548e-02 5.1360e-02 1.1549e-01
   al_rho2p7_te10_ti1       HNC              DFT-MD               6.4272e-02 4.9097e-02 1.7600e-01
   al_rho2p7_te10_ti1       HNC              YOCP HNC+bridge      1.8136e-01 1.4866e-01 3.2180e-01
   al_rho2p7_te10_ti1       VMHNC            2TTCP HNC+bridge     4.5232e-02 3.8198e-02 8.0904e-02
   al_rho2p7_te10_ti1       VMHNC            DFT-MD               1.4615e-01 1.2148e-01 3.5590e-01
   al_rho2p7_te10_ti1       VMHNC            YOCP HNC+bridge      2.7036e-01 2.2694e-01 4.5774e-01
   al_rho2p7_te10_ti1       historical MD    2TTCP HNC+bridge     4.2319e-02 3.7165e-02 6.9763e-02
   al_rho2p7_te10_ti1       historical MD    DFT-MD               1.4131e-01 1.1742e-01 3.3928e-01
   al_rho2p7_te10_ti1       historical MD    YOCP HNC+bridge      2.6343e-01 2.2293e-01 4.4258e-01
   al_rho2p7_te10_ti1: shared IS Zbar=3.02668340, VMHNC eta=0.36063931, variational residual=-1.095e-05
   al_rho2p7_te30_ti1       HNC              2TTCP HNC+bridge     1.4063e-01 1.1348e-01 2.4847e-01
   al_rho2p7_te30_ti1       HNC              DFT-MD               9.4245e-02 7.2869e-02 2.4090e-01
   al_rho2p7_te30_ti1       HNC              YOCP HNC+bridge      4.0578e-02 2.8736e-02 1.1916e-01
   al_rho2p7_te30_ti1       VMHNC            2TTCP HNC+bridge     6.2254e-02 4.8280e-02 1.4931e-01
   al_rho2p7_te30_ti1       VMHNC            DFT-MD               1.1701e-01 8.8971e-02 2.5915e-01
   al_rho2p7_te30_ti1       VMHNC            YOCP HNC+bridge      2.0605e-01 1.6837e-01 3.8838e-01
   al_rho2p7_te30_ti1       historical MD    2TTCP HNC+bridge     4.5729e-02 3.8387e-02 9.3464e-02
   al_rho2p7_te30_ti1       historical MD    DFT-MD               1.0126e-01 7.6281e-02 2.3890e-01
   al_rho2p7_te30_ti1       historical MD    YOCP HNC+bridge      1.8690e-01 1.5519e-01 3.2567e-01
   al_rho2p7_te30_ti1: shared IS Zbar=4.47940116, VMHNC eta=0.44562679, variational residual=-8.298e-07
