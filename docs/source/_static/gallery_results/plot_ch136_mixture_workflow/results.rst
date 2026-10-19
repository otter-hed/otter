.. Recorded validation output; presentation only, not solver input.

.. image:: /_static/gallery_results/plot_ch136_mixture_workflow/figure_001.svg
   :alt: Recorded result for plot_ch136_mixture_workflow

Recorded terminal output
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Otter 0.4.1
   [run] C:1 H:1.36 | rho=5 g/cm^3 | Te=8.61733 eV | Ti=8.61733 eV | QM -> QOZ/HNC
   [mixture] iter=  1  max|dmu|=6.127e-01 Ha
   [mixture] iter=  2  max|dmu|=2.664e-01 Ha
   [mixture] observed binary sign-change bracket: theta in [0.588395, 1.484275]
   [mixture] iter=  3  max|dmu|=1.885e-02 Ha
   [mixture] observed binary sign-change bracket: theta in [0.859866, 1.484275]
   [mixture] iter=  4  max|dmu|=2.487e-04 Ha
   [mixture] observed binary sign-change bracket: theta in [0.879920, 1.484275]
   [mixture] iter=  5  max|dmu|=2.445e-07 Ha
   [mixture] observed binary sign-change bracket: theta in [0.880187, 1.484275]
   [AA] C (Z=6) | Rws=1.716904 a0 | full + external
   [AA] H (Z=1) | Rws=1.155615 a0 | full + external
   [AA] converged=True | mu=0.96529341 Ha | Zbar=1.00000000 | time=8.157 s

   Bound levels
     none
   [AA] converged=True | mu=0.96529366 Ha | Zbar=4.00003176 | time=9.085 s

   Bound levels
     level |     E [Ha] |      f_FD
     ------+------------+----------
        1s |   -8.64210 |   1.00000
   [mixture] mu=0.96529354 Ha
   [mixture] C: Rws=1.716904 Bohr  Zbar=4.00003176
   [mixture] H: Rws=1.155615 Bohr  Zbar=1.00000000
   [ion] HNC iterations=4  QOZ=0.809 s  HNC=0.559 s
   [done] total time=47.685 s
   CH1.36  rho=5 g/cc  Te=Ti=100 kK (8.61733326 eV)
   common mu = 0.96529354 Ha; max|Delta mu| = 2.445e-07 Ha
   HNC residual = 1.248e-11; g/S closure = 1.426e-13; min eig(S) = 0.120106
    sp       mu[Ha]    Zbar(AA)   Zbar(part)   Zbar(QOZ)     Qnative    Qdst raw   Qdst used      scale
     C   0.96529366   4.0000318    4.0000290   4.0000290   4.0000609   4.0000724   4.0000290  0.9999891
     H   0.96529341   1.0000000    1.0000000   1.0000000   1.0002407   1.0002431   1.0000000  0.9997570
   Calculation wall time = 47.68 s
