.. Recorded validation output; presentation only, not solver input.

.. image:: /_static/gallery_results/plot_carbon_lfc_sensitivity/figure_001.svg
   :alt: Recorded result for plot_carbon_lfc_sensitivity

.. image:: /_static/gallery_results/plot_carbon_lfc_sensitivity/figure_002.svg
   :alt: Recorded result for plot_carbon_lfc_sensitivity

.. image:: /_static/gallery_results/plot_carbon_lfc_sensitivity/figure_003.svg
   :alt: Recorded result for plot_carbon_lfc_sensitivity

.. image:: /_static/gallery_results/plot_carbon_lfc_sensitivity/figure_004.svg
   :alt: Recorded result for plot_carbon_lfc_sensitivity

Recorded terminal output
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Otter 0.4.1
   [run] C | rho=5 g/cm^3 | Te=2 eV | Ti=-- | QM electronic structure
   [AA] C (Z=6) | Rws=1.859183 a0 | full + external
   [AA/full SCF: stage 1]
     iter |       d_n |       d_v
        0 |       nan | 4.740e-02
        1 | 1.650e-02 | 4.204e-02
        2 | 2.183e-02 | 3.734e-02
        3 | 2.251e-02 | 3.327e-02
        4 | 2.068e-02 | 2.982e-02
   [AA/full SCF: stage 2]
     iter |       d_n |       d_v
        0 |       nan | 2.700e-02
        1 | 1.566e-02 | 2.482e-02
        2 | 1.344e-02 | 2.336e-02
        3 | 1.157e-02 | 2.366e-02
        4 | 1.015e-02 | 3.053e-02
        5 | 1.104e-02 | 3.678e-02
        6 | 3.169e-02 | 2.515e-02
        7 | 1.281e-02 | 4.140e-02
        8 | 4.821e-02 | 3.687e-02
        9 | 2.067e-01 | 6.818e-02
       10 | 1.636e-01 | 1.196e-02
       11 | 1.397e-02 | 7.174e-03
       12 | 1.921e-02 | 2.933e-03
       13 | 1.722e-02 | 3.221e-03
       14 | 1.225e-02 | 8.055e-04
       15 | 3.578e-03 | 7.028e-04
       16 | 1.950e-03 | 1.350e-04
       17 | 2.227e-04 | 6.900e-04
       18 | 2.698e-03 | 5.339e-05
       19 | 8.840e-05 | 2.119e-04
       20 | 8.192e-04 | 2.672e-05
       21 | 1.384e-04 | 1.378e-05
       22 | 6.093e-05 | 9.861e-06
       23 | 2.251e-05 | 5.431e-06
       24 | 2.969e-05 | 5.166e-07
       25 | 1.868e-06 | 9.417e-07
       26 | 2.524e-06 | 1.553e-06
       27 | 9.172e-06 | 1.144e-07
   [AA/external SCF]
     iter |       d_n |       d_v
        0 | 2.425e-07 | 1.646e+12
        1 | 7.958e-02 | 1.079e+00
        2 | 7.570e-02 | 7.903e-01
        3 | 2.367e-02 | 3.181e-01
        4 | 8.589e-03 | 1.335e-01
        5 | 3.084e-03 | 2.328e-02
        6 | 8.076e-04 | 9.383e-03
        7 | 2.277e-04 | 3.242e-03
        8 | 1.073e-04 | 7.808e-04
        9 | 2.151e-05 | 6.749e-04
       10 | 8.312e-06 | 5.615e-05
       11 | 1.066e-06 | 1.123e-05
       12 | 2.787e-07 | 1.986e-05
       13 | 2.596e-07 | 5.371e-06
       14 | 1.034e-07 | 2.229e-06
       15 | 2.844e-08 | 1.029e-06
       16 | 1.315e-08 | 1.441e-07
   [AA] converged=True | mu=0.73168831 Ha | Zbar=4.00001133 | time=20.032 s

   Bound levels
     level |     E [Ha] |      f_FD
     ------+------------+----------
        1s |   -8.90073 |   1.00000
   [done] total time=20.033 s
   Otter 0.4.1
   [run] C | rho=5 g/cm^3 | Te=100 eV | Ti=-- | QM electronic structure
   [AA] C (Z=6) | Rws=1.859183 a0 | full + external
   [AA/full SCF: stage 1]
     iter |       d_n |       d_v
   [candidate] benchmarks/outputs/carbon_lfc_sensitivity/gallery_recomputed/C_rho5gcc_T2eV_lfc.npz
        0 |       nan | 5.150e-02
        1 | 6.308e-02 | 4.744e-02
        2 | 5.809e-02 | 4.369e-02
        3 | 5.238e-02 | 4.007e-02
        4 | 4.598e-02 | 3.632e-02
   [AA/full SCF: stage 2]
     iter |       d_n |       d_v
        0 |       nan | 3.233e-02
        1 | 6.946e-02 | 6.753e-02
        2 | 2.443e-02 | 1.741e-02
        3 | 1.578e-02 | 2.941e-02
        4 | 6.491e-02 | 3.204e-03
        5 | 1.689e-03 | 1.163e-02
        6 | 3.109e-02 | 1.931e-03
        7 | 1.419e-03 | 8.089e-04
        8 | 2.007e-03 | 3.437e-04
        9 | 6.199e-04 | 3.707e-05
       10 | 5.061e-05 | 6.452e-05
       11 | 2.832e-04 | 4.750e-06
       12 | 1.887e-05 | 3.937e-06
       13 | 3.145e-05 | 2.736e-07
       14 | 1.235e-06 | 2.175e-07
       15 | 1.118e-06 | 1.686e-07
       16 | 5.670e-07 | 1.431e-07
   [AA] unresolved threshold state; retrying with zero-tail matching through l=0; A3 extends to 15 R_geometry
   [AA] C (Z=6) | Rws=1.859183 a0 | full + external
   [AA/full SCF: stage 1]
     iter |       d_n |       d_v
   [AA/full SCF: stage 2]
     iter |       d_n |       d_v
        0 |       nan | 5.165e-02
        1 | 6.311e-02 | 4.760e-02
        2 | 5.812e-02 | 4.387e-02
        3 | 5.242e-02 | 4.026e-02
        4 | 7.941e-02 | 9.767e-02
        5 | 3.342e-02 | 1.686e-02
        6 | 3.301e-02 | 4.424e-02
        7 | 8.548e-02 | 2.012e-02
        8 | 2.524e-02 | 1.589e-02
        9 | 2.234e-02 | 6.443e-03
       10 | 8.170e-03 | 3.416e-03
       11 | 6.113e-03 | 3.004e-03
       12 | 3.570e-03 | 1.824e-04
       13 | 5.979e-04 | 4.761e-05
       14 | 2.111e-04 | 4.630e-05
       15 | 1.724e-04 | 5.439e-06
       16 | 1.724e-05 | 1.661e-06
       17 | 9.257e-06 | 1.403e-06
       18 | 8.879e-06 | 2.121e-06
       19 | 1.524e-05 | 1.604e-07
       20 | 8.815e-07 | 1.185e-07
       21 | 2.349e-07 | 4.542e-08
       22 | 1.868e-07 | 3.671e-08
   [AA/external SCF]
     iter |       d_n |       d_v
        0 | 2.913e-06 | 3.882e+12
        1 | 2.613e-02 | 8.870e-01
        2 | 1.889e-02 | 1.093e+00
        3 | 1.523e-02 | 1.898e-01
        4 | 3.328e-03 | 2.311e-02
        5 | 5.278e-04 | 1.990e-02
        6 | 2.881e-04 | 2.572e-03
        7 | 3.957e-05 | 6.897e-04
        8 | 1.543e-05 | 5.987e-05
        9 | 1.186e-06 | 1.900e-05
       10 | 2.110e-07 | 4.044e-05
       11 | 4.396e-07 | 7.532e-06
       12 | 8.582e-08 | 7.355e-06
       13 | 7.781e-08 | 2.925e-07
       14 | 4.791e-09 | 1.375e-06
   [AA] converged=True | mu=-6.53601783 Ha | Zbar=4.54732225 | time=168.976 s

   Bound levels
     level |     E [Ha] |      f_FD
     ------+------------+----------
        1s |  -10.11695 |   0.72600
        2s |   -0.00213 |   0.14455
   [done] total time=183.570 s
   [candidate] benchmarks/outputs/carbon_lfc_sensitivity/gallery_recomputed/C_rho5gcc_T100eV_lfc.npz
     T[eV]              model     max|dg|     max|dS|   max|dV|[Ha Bohr^3]
       2.0               none     0.40140     0.15411             19.55532
       2.0            hubbard     0.06154     0.06145             17.72958
       2.0     utsumiichimaru     0.00118     0.00054              0.08059
       2.0       chabrier1990     0.00000     0.00000              0.00000
       2.0        gregori2007     0.00197     0.00188              0.36033
     100.0               none     0.00430     0.00537             24.62183
     100.0            hubbard     0.00146     0.00400             19.14946
     100.0     utsumiichimaru     0.00025     0.00033              1.56503
     100.0       chabrier1990     0.00000     0.00000              0.00000
     100.0        gregori2007     0.00044     0.00031              1.34053

   Low-k audit at T=2 eV, k=0.05633257 Bohr^-1
                model         chi_ee    dchi/chi_ref[%]      G/k^2[Bohr^2]        V_LFC         V_ii
                 none -2.5214329e-04     -3.0909701e-02      0.0000000e+00      0.00000     93.71404
              hubbard -2.5229197e-04      2.8040400e-02      1.8599682e-01     37.28490    130.99895
       utsumiichimaru -2.5222157e-04      1.2742161e-04      9.7954327e-02     19.63591    113.34996
         chabrier1990 -2.5222125e-04      0.0000000e+00      9.7552303e-02     19.55532    113.26937
          gregori2007 -2.5222269e-04      5.6972064e-04      9.9349800e-02     19.91565    113.62969
