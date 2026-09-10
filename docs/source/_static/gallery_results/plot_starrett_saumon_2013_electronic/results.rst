.. Recorded local validation output; presentation only, not solver input.

Recorded terminal output
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   
   Al bound levels: all energies E are in Hartree
    T[eV] shell  paper E[Ha]    IS E[Ha]    SC E[Ha]   M paper      M IS      M SC
        2    1s        -54.6       -54.6       -54.6      1.00      1.00      1.00
        2    2s        -3.41       -3.39       -3.40      1.00      1.00      1.00
        2    2p        -2.04       -2.02       -2.03      1.00      1.00      1.00
        2    3s      unbound     unbound     unbound        --        --        --
       15    1s        -54.9       -54.8       -54.8      1.00      1.00      1.00
       15    2s        -3.60       -3.51       -3.57      1.00      1.00      1.00
       15    2p        -2.23       -2.14       -2.21      1.00      1.00      1.00
       15    3s      -0.0125     unbound    -0.00835     0.134        --    0.0916
   Eq. (81) formula audit at 15 eV 3s: paper inputs give M=0.134; published M=0.134.
   
   Scattering width gamma: all values are in Hartree
    T[eV]  paper gamma[Ha]  IS gamma[Ha]  SC gamma[Ha]
        2           0.0698        0.0378        0.0630
       15            0.174         0.111         0.171
   
   Mean ionization: all values are dimensionless
                      state  Z* paper     Z* IS     Z* SC  Zbar paper    Zbar IS    Zbar SC
           al_rho2p7_te2_qm      1.98      2.06      2.06        3.00       3.00       3.00
           al_rho2p7_te6_qm      2.11      2.18      2.18        3.00       3.00       3.00
          al_rho2p7_te10_qm      2.24      2.31      2.31        3.00       3.03       3.02
          al_rho2p7_te15_qm      2.51      2.54      2.54        3.18       3.24       3.19
         fe_rho22.5_te10_tf      5.85      5.85      5.85        8.78       8.81       8.74
        fe_rho34.5_te100_tf      9.54      9.54      9.54        11.6       11.9       11.5
      fe_rho39.65_te1000_tf      20.4      20.4      20.4        21.7       21.9       21.6
      fe_rho34.37_te5000_tf      25.1      25.1      25.1        25.5       25.6       25.6
