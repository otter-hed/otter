.. Recorded validation output; presentation only, not solver input.

.. image:: /_static/gallery_results/plot_al_qm_tf/figure_001.svg
   :alt: Recorded result for plot_al_qm_tf

.. image:: /_static/gallery_results/plot_al_qm_tf/figure_002.svg
   :alt: Recorded result for plot_al_qm_tf

Recorded terminal output
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Otter 0.4.1
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
   [AA] converged=True | mu=1.04477399 Ha | Zbar=3.02589432 | time=25.394 s

   Bound levels
     level |     E [Ha] |      f_FD
     ------+------------+----------
        1s |  -53.85600 |   1.00000
        2s |   -2.73953 |   1.00000
        2p |   -1.36869 |   1.00000
   [ion] HNC iterations=35  QOZ=0.821 s  HNC=0.060 s
   [done] total time=26.288 s
   Otter 0.4.1
   [run] Al | rho=8.1 g/cm^3 | Te=1 eV | Ti=1 eV | TF -> QOZ/HNC
   [computed] Al, T=1 eV, model=qm
   [AA/TF] Al (Z=13)  Te=1 eV  rho=8.1 g/cc  Rws=2.073224 Bohr  run_mode=full+ext
   [AA/TF full SCF]
     iter |       d_n |       d_v
        1 |       inf | 6.318e-01
       10 | 2.615e-02 | 9.893e-03
       20 | 1.234e-04 | 1.458e-04
   [AA/TF external SCF]
     iter |       d_n |       d_v
        1 |       inf | 2.407e+01
       10 | 1.603e-02 | 3.760e-01
       20 | 9.006e-06 | 3.168e-04
       30 | 5.165e-06 | 5.359e-04
   [AA/TF] converged=True  mu=0.93275746 Ha  Zbar=3.21837387  time=0.312 s
   [ion] HNC iterations=43  QOZ=0.812 s  HNC=0.075 s
   [done] total time=1.206 s
   Otter 0.4.1
   [run] Al | rho=8.1 g/cm^3 | Te=15 eV | Ti=15 eV | QM -> QOZ/HNC
   [AA] Al (Z=13) | Rws=2.073224 a0 | full + external
   [AA/full SCF: stage 1]
     iter |       d_n |       d_v
   [computed] Al, T=1 eV, model=tf
        0 |       nan | 5.240e-02
        1 | 2.837e-02 | 4.639e-02
        2 | 2.497e-02 | 4.088e-02
        3 | 2.165e-02 | 3.600e-02
        4 | 1.843e-02 | 3.188e-02
   [AA/full SCF: stage 2]
     iter |       d_n |       d_v
        0 |       nan | 2.866e-02
        1 | 1.289e-02 | 2.644e-02
        2 | 1.085e-02 | 2.544e-02
        3 | 9.586e-03 | 2.612e-02
        4 | 9.820e-03 | 2.955e-02
        5 | 8.759e-03 | 3.222e-02
        6 | 9.402e-03 | 1.203e-02
        7 | 4.096e-03 | 2.844e-02
        8 | 3.569e-02 | 8.738e-03
        9 | 7.683e-03 | 3.056e-03
       10 | 1.791e-03 | 8.456e-03
       11 | 1.607e-02 | 2.079e-03
       12 | 3.652e-03 | 1.375e-03
       13 | 2.032e-03 | 6.459e-04
       14 | 9.115e-04 | 1.945e-04
       15 | 3.451e-04 | 1.063e-04
       16 | 1.789e-04 | 2.766e-05
       17 | 5.530e-05 | 1.381e-05
       18 | 4.122e-05 | 1.040e-05
       19 | 2.273e-05 | 1.346e-06
       20 | 1.789e-06 | 2.312e-06
       21 | 4.787e-06 | 1.467e-07
       22 | 3.368e-07 | 3.287e-07
   [AA/external SCF]
     iter |       d_n |       d_v
        0 | 9.438e-07 | 3.339e+12
        1 | 8.098e-02 | 1.088e+00
        2 | 7.989e-02 | 7.904e-01
        3 | 2.613e-02 | 2.887e-01
        4 | 7.169e-03 | 1.307e-01
        5 | 2.836e-03 | 3.971e-02
        6 | 9.969e-04 | 1.028e-02
        7 | 2.248e-04 | 9.499e-03
        8 | 1.756e-04 | 9.538e-04
        9 | 2.088e-05 | 6.133e-04
       10 | 1.103e-05 | 5.305e-04
       11 | 6.929e-06 | 2.855e-05
       12 | 1.095e-06 | 3.737e-05
       13 | 8.038e-07 | 3.541e-05
       14 | 4.580e-07 | 9.104e-07
       15 | 3.025e-08 | 1.764e-06
       16 | 2.290e-08 | 1.265e-07
   [AA] converged=True | mu=0.61838663 Ha | Zbar=3.16542837 | time=19.692 s

   Bound levels
     level |     E [Ha] |      f_FD
     ------+------------+----------
        1s |  -53.99532 |   1.00000
        2s |   -2.81297 |   0.99802
        2p |   -1.44710 |   0.97695
   [ion] HNC iterations=10  QOZ=0.834 s  HNC=0.018 s
   [done] total time=20.555 s
   Otter 0.4.1
   [run] Al | rho=8.1 g/cm^3 | Te=15 eV | Ti=15 eV | TF -> QOZ/HNC
   [computed] Al, T=15 eV, model=qm
   [AA/TF] Al (Z=13)  Te=15 eV  rho=8.1 g/cc  Rws=2.073224 Bohr  run_mode=full+ext
   [AA/TF full SCF]
     iter |       d_n |       d_v
        1 |       inf | 6.288e-01
       10 | 3.439e-02 | 2.299e-02
       20 | 1.105e-04 | 6.723e-05
       30 | 4.799e-05 | 4.010e-05
   [AA/TF external SCF]
     iter |       d_n |       d_v
        1 |       inf | 2.052e+01
       10 | 6.365e-03 | 3.201e-01
       20 | 1.398e-06 | 1.169e-04
       30 | 1.246e-06 | 1.321e-04
   [AA/TF] converged=True  mu=0.69292041 Ha  Zbar=3.59741450  time=0.234 s
   [ion] HNC iterations=10  QOZ=0.820 s  HNC=0.018 s
   [done] total time=1.079 s
   Otter 0.4.1
   [run] Al | rho=8.1 g/cm^3 | Te=50 eV | Ti=50 eV | QM -> QOZ/HNC
   [AA] Al (Z=13) | Rws=2.073224 a0 | full + external
   [AA/full SCF: stage 1]
     iter |       d_n |       d_v
   [computed] Al, T=15 eV, model=tf
        0 |       nan | 5.256e-02
        1 | 3.132e-02 | 4.679e-02
        2 | 2.844e-02 | 4.157e-02
        3 | 2.592e-02 | 3.698e-02
        4 | 2.366e-02 | 3.304e-02
   [AA/full SCF: stage 2]
     iter |       d_n |       d_v
        0 |       nan | 2.982e-02
        1 | 1.924e-02 | 2.726e-02
        2 | 1.721e-02 | 2.543e-02
        3 | 1.560e-02 | 2.444e-02
        4 | 1.476e-02 | 2.424e-02
        5 | 1.148e-02 | 3.601e-02
        6 | 8.153e-03 | 1.536e-02
        7 | 4.071e-03 | 1.440e-02
        8 | 1.856e-02 | 1.748e-03
        9 | 1.485e-03 | 4.773e-03
       10 | 7.180e-03 | 2.152e-03
       11 | 5.601e-04 | 5.077e-04
       12 | 3.245e-04 | 7.360e-04
       13 | 1.092e-03 | 4.642e-05
       14 | 4.170e-05 | 5.291e-05
       15 | 6.088e-05 | 1.863e-05
       16 | 1.666e-05 | 3.504e-06
       17 | 6.040e-06 | 7.342e-07
       18 | 1.014e-06 | 1.945e-06
       19 | 3.680e-06 | 6.959e-08
   [AA/external SCF]
     iter |       d_n |       d_v
        0 | 2.520e-06 | 4.735e+12
        1 | 5.207e-02 | 1.037e+00
        2 | 4.905e-02 | 9.678e-01
        3 | 2.437e-02 | 2.432e-01
        4 | 5.829e-03 | 6.429e-02
        5 | 1.464e-03 | 1.689e-02
        6 | 6.668e-04 | 4.253e-03
        7 | 1.022e-04 | 2.774e-03
        8 | 5.167e-05 | 9.861e-05
        9 | 2.771e-06 | 1.586e-04
       10 | 2.594e-06 | 2.541e-04
       11 | 3.431e-06 | 3.437e-05
       12 | 8.005e-07 | 2.469e-05
       13 | 4.245e-07 | 6.313e-06
       14 | 8.111e-08 | 1.267e-06
       15 | 1.746e-08 | 7.473e-07
   [AA] converged=True | mu=-1.44023710 Ha | Zbar=5.45747083 | time=26.661 s

   Bound levels
     level |     E [Ha] |      f_FD
     ------+------------+----------
        1s |  -55.99999 |   1.00000
        2s |   -3.93544 |   0.79543
        2p |   -2.65265 |   0.65922
   [ion] HNC iterations=16  QOZ=0.798 s  HNC=0.027 s
   [done] total time=27.497 s
   Otter 0.4.1
   [run] Al | rho=8.1 g/cm^3 | Te=50 eV | Ti=50 eV | TF -> QOZ/HNC
   [computed] Al, T=50 eV, model=qm
   [AA/TF] Al (Z=13)  Te=50 eV  rho=8.1 g/cc  Rws=2.073224 Bohr  run_mode=full+ext
   [AA/TF full SCF]
     iter |       d_n |       d_v
        1 |       inf | 6.096e-01
       10 | 7.671e-02 | 4.045e-02
   [AA/TF external SCF]
     iter |       d_n |       d_v
        1 |       inf | 1.732e+01
       10 | 6.042e-04 | 2.952e-02
       20 | 1.620e-06 | 3.681e-04
   [AA/TF] converged=True  mu=-1.30100851 Ha  Zbar=5.01004236  time=0.201 s
   [ion] HNC iterations=20  QOZ=0.818 s  HNC=0.034 s
   [done] total time=1.060 s
   Otter 0.4.1
   [run] Al | rho=8.1 g/cm^3 | Te=100 eV | Ti=100 eV | QM -> QOZ/HNC
   [AA] Al (Z=13) | Rws=2.073224 a0 | full + external
   [AA/full SCF: stage 1]
     iter |       d_n |       d_v
   [computed] Al, T=50 eV, model=tf
        0 |       nan | 5.205e-02
        1 | 4.094e-02 | 4.620e-02
        2 | 3.744e-02 | 4.084e-02
        3 | 3.303e-02 | 3.597e-02
        4 | 2.813e-02 | 3.159e-02
   [AA/full SCF: stage 2]
     iter |       d_n |       d_v
        0 |       nan | 2.787e-02
        1 | 1.905e-02 | 2.453e-02
        2 | 1.551e-02 | 2.174e-02
        3 | 1.281e-02 | 1.954e-02
        4 | 1.099e-02 | 1.804e-02
        5 | 1.351e-02 | 4.831e-02
        6 | 7.715e-03 | 1.725e-02
        7 | 2.979e-03 | 1.236e-02
        8 | 1.122e-02 | 6.340e-03
        9 | 1.540e-03 | 9.316e-03
       10 | 6.998e-03 | 2.724e-03
       11 | 9.725e-04 | 5.961e-04
       12 | 2.253e-04 | 4.798e-04
       13 | 6.114e-04 | 5.670e-05
       14 | 6.617e-05 | 1.575e-05
       15 | 1.369e-05 | 2.467e-05
       16 | 2.017e-05 | 4.435e-06
       17 | 5.704e-06 | 2.015e-07
       18 | 2.737e-07 | 6.031e-07
       19 | 1.041e-06 | 6.425e-08
   [AA/external SCF]
     iter |       d_n |       d_v
        0 | 2.796e-06 | 3.270e+12
        1 | 2.128e-02 | 1.232e+00
        2 | 1.969e-02 | 1.053e+00
        3 | 2.174e-02 | 2.422e-01
        4 | 5.039e-03 | 3.778e-02
        5 | 7.088e-04 | 1.516e-02
        6 | 3.564e-04 | 2.367e-03
        7 | 3.920e-05 | 3.290e-04
        8 | 6.510e-06 | 1.212e-04
        9 | 1.575e-06 | 1.205e-04
       10 | 1.786e-06 | 2.379e-06
       11 | 3.731e-08 | 2.377e-06
       12 | 3.124e-08 | 2.643e-06
   [AA] converged=True | mu=-5.77548285 Ha | Zbar=7.69248485 | time=35.055 s

   Bound levels
     level |     E [Ha] |      f_FD
     ------+------------+----------
        1s |  -58.57673 |   1.00000
        2s |   -5.33777 |   0.47026
        2p |   -4.20236 |   0.39459
   [ion] HNC iterations=16  QOZ=0.812 s  HNC=0.027 s
   [done] total time=35.906 s
   Otter 0.4.1
   [run] Al | rho=8.1 g/cm^3 | Te=100 eV | Ti=100 eV | TF -> QOZ/HNC
   [computed] Al, T=100 eV, model=qm
   [AA/TF] Al (Z=13)  Te=100 eV  rho=8.1 g/cc  Rws=2.073224 Bohr  run_mode=full+ext
   [AA/TF full SCF]
     iter |       d_n |       d_v
        1 |       inf | 5.963e-01
       10 | 6.662e-03 | 3.353e-01
   [AA/TF external SCF]
     iter |       d_n |       d_v
        1 |       inf | 1.590e+01
       10 | 1.547e-04 | 5.345e-03
   [AA/TF] converged=True  mu=-5.77104591 Ha  Zbar=6.48507712  time=0.214 s
   [ion] HNC iterations=11  QOZ=0.823 s  HNC=0.019 s
   [done] total time=1.064 s
   [computed] Al, T=100 eV, model=tf
      rho   T[eV]   dZ(TF-KS)     RMSE(g)     RMSE(S)
     8.10     1.0     1.84369     0.06053     0.07202
     8.10    15.0     1.81326     0.04190     0.03435
     8.10    50.0     0.66385     0.01169     0.01082
     8.10   100.0    -0.19890     0.00280     0.00247
