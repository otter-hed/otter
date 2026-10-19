.. Recorded local validation output; presentation only, not solver input.

.. image:: /_static/gallery_results/plot_al_full_workflow/figure_001.svg
   :alt: Al electronic and ionic structure

.. image:: /_static/gallery_results/plot_al_full_workflow/figure_002.svg
   :alt: Al electronic and ionic structure

.. image:: /_static/gallery_results/plot_al_full_workflow/figure_003.svg
   :alt: Al electronic and ionic structure

.. image:: /_static/gallery_results/plot_al_full_workflow/figure_004.svg
   :alt: Al electronic and ionic structure

Recorded terminal output
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Otter 0.4.0
   [run] Al | rho=8.1 g/cm^3 | Te=1 eV | Ti=1 eV | QM -> QOZ/HNC
   [AA] Al (Z=13) | Rws=2.073224 a0 | full + external
   [AA/full SCF: stage 1]
     iter |       d_n |       d_v
        0 |       nan | 5.240e-02
        1 | 1.520e-02 | 4.620e-02
        2 | 2.703e-02 | 4.075e-02
        3 | 2.403e-02 | 3.584e-02
        4 | 2.034e-02 | 3.154e-02
   [AA/full SCF: stage 2]
     iter |       d_n |       d_v
        0 |       nan | 2.788e-02
        1 | 1.405e-02 | 2.492e-02
        2 | 1.153e-02 | 2.271e-02
        3 | 9.417e-03 | 2.134e-02
        4 | 7.677e-03 | 2.103e-02
        5 | 6.301e-03 | 2.248e-02
        6 | 5.816e-03 | 3.229e-02
        7 | 1.134e-02 | 1.286e-02
        8 | 3.621e-03 | 4.113e-03
        9 | 3.776e-03 | 1.244e-02
       10 | 2.613e-02 | 9.677e-03
       11 | 1.483e-02 | 1.428e-02
       12 | 3.320e-02 | 1.381e-03
       13 | 2.429e-03 | 2.561e-03
       14 | 1.430e-03 | 1.860e-03
       15 | 3.396e-03 | 2.924e-04
       16 | 7.165e-04 | 2.559e-04
       17 | 8.769e-04 | 7.320e-05
       18 | 1.407e-04 | 6.769e-06
       19 | 1.720e-05 | 3.018e-05
       20 | 2.151e-05 | 2.317e-06
       21 | 3.835e-06 | 1.815e-06
       22 | 4.060e-06 | 2.601e-06
       23 | 9.929e-06 | 2.508e-07
   [AA/external SCF]
     iter |       d_n |       d_v
        0 | 2.447e-07 | 3.853e+12
        1 | 1.171e-01 | 1.113e+00
        2 | 1.228e-01 | 7.093e-01
        3 | 3.025e-02 | 3.105e-01
        4 | 9.501e-03 | 1.207e-01
        5 | 3.736e-03 | 1.111e-02
        6 | 5.158e-04 | 1.324e-02
        7 | 3.172e-04 | 4.434e-03
        8 | 1.352e-04 | 8.572e-04
        9 | 2.621e-05 | 2.125e-04
       10 | 5.853e-06 | 9.190e-04
       11 | 2.433e-05 | 3.834e-04
       12 | 7.179e-06 | 1.790e-04
       13 | 5.440e-06 | 1.313e-05
       14 | 2.446e-07 | 3.496e-05
       15 | 8.033e-07 | 2.359e-06
       16 | 5.997e-08 | 7.449e-07
       17 | 1.648e-08 | 4.449e-07
   [AA] converged=True | mu=1.04477399 Ha | Zbar=3.02589432 | time=31.423 s

   Bound levels
     level |     E [Ha] |      f_FD
     ------+------------+----------
        1s |  -53.85600 |   1.00000
        2s |   -2.73953 |   1.00000
        2p |   -1.36869 |   1.00000
   [ion] HNC iterations=33  QOZ=1.707 s  HNC=0.080 s
   [done] total time=33.213 s
   Al: rho=8.1 g/cc, Te=Ti=1 eV
   mu=1.04477399 Ha, HNC residual=4.455e-06
   Zbar(AA-WS/partition/QOZ)=3.02589432/3.02275441/3.02275441
   Qscr(native/grid/used)=3.02279546/3.02279986/3.02275441; scale=0.99998496
     level        E [Ha]        E [eV]         FD   occupation
        1s    -53.855996  -1465.496296   1.000000     2.000000
        2s     -2.739533    -74.546490   1.000000     2.000000
        2p     -1.368686    -37.243853   1.000000     6.000000
