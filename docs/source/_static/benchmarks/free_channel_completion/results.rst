All 12 analytic controls pass the declared :math:`2\times10^{-7}` residual budget. The largest completed residual is :math:`5.614\times10^{-8}` of the free spectrum; all retained phase arrays are unchanged.

All 24 fresh workflows reach their full and external SCF stopping conditions and retain the correct external energy-cutoff handoff. The disabled calculation matches the main reference bitwise in density, potential, chemical potential and background density. All ten enabled species comparisons pass the equivalence budgets.

.. list-table:: SCF density and source-charge changes
   :header-rows: 1
   :widths: 28 12 20 20 20

   * - Material, T [eV], ρ [g/cm³]
     - Species
     - :math:`\max|\Delta n|/n_0`
     - :math:`\max|\Delta Q|` [e]
     - PA error, native → completed [e]
   * - Be, 10, 10
     - Be
     - 0.000e+00
     - 0.000e+00
     - +6.9317e-05 → +6.9317e-05
   * - C, 100, 400
     - C
     - 0.000e+00
     - 0.000e+00
     - +2.2035e-05 → +2.2035e-05
   * - H, 99, 0.946
     - H
     - 0.000e+00
     - 0.000e+00
     - +4.2501e-03 → +4.2501e-03
   * - Al, 1, 8.1
     - Al
     - 0.000e+00
     - 0.000e+00
     - -2.8676e-06 → -2.8676e-06
   * - CH₂, 40, 20
     - C
     - 1.468e-09
     - 5.421e-11
     - +2.0316e-04 → +2.0316e-04
   * - CH₂, 40, 20
     - H
     - 4.851e-10
     - 5.181e-12
     - +3.4328e-04 → +3.4328e-04
   * - CH₂, 99, 0.946
     - C
     - 0.000e+00
     - 0.000e+00
     - +3.1913e-03 → +3.1913e-03
   * - CH₂, 99, 0.946
     - H
     - 0.000e+00
     - 0.000e+00
     - +6.8278e-03 → +6.8278e-03
   * - C, 10, 1
     - C
     - 0.000e+00
     - 0.000e+00
     - +1.5137e-03 → +1.5137e-03
   * - Si, 1, 1
     - Si
     - 0.000e+00
     - 0.000e+00
     - +1.0724e-03 → +1.0724e-03

Density and source-charge maxima include full, external and PA quantities. PA errors are absolute electron-number errors relative to the nuclear charge. They remain separate from differences between the two implementations.
