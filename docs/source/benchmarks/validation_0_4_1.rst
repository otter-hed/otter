Otter 0.4.1 numerical validation
================================

The example gallery and scientific benchmarks were recalculated with the
0.4.1 external-AA fixes. Recorded figures, tables and terminal output refer
to these calculations. Numerical arrays, trajectories and restart files
remain local outputs; documentation builds do not run the solvers.

Scope
-----

The calculations cover the electronic, ionization and ion-structure examples
listed in the gallery and benchmark catalogue. Carbon's 100 eV density scan
is calculated once by its benchmark; the example gallery shares its figure.
AA radial grids, continuum controls and SCF settings use the project defaults.
Schörner's comparison explicitly selects LDA-PW92 and PBE through Libxc.

Electronic states are checked before ion-structure or MD calculations.
The workflow retains its existing convergence, threshold-state, common
chemical-potential and screening-charge checks. The ion-structure library
also retains its native screening-charge integral check. These acceptance
conditions were not relaxed for this recalculation.

Molecular dynamics
------------------

The MD comparisons include Wünsch Be, Johnson Al, Schörner Al and nine CH₂
states. They use the recalculated electronic potentials rather than an
unverified overlay of new closures with older MD curves. Potential tables,
run parameters and complete trajectories are checked before accepting a
calculated state.

CH₂ uses 3072 ions and a fixed 0.01 fs timestep. Its equilibration and
production durations remain :math:`50\omega_p^{-1}` and
:math:`500\omega_p^{-1}`. The two hottest cases restore the analytic Coulomb
core below 0.3 Bohr and join the native QOZ potential by 0.5 Bohr; this
numerical treatment is included in the interpretation of HNC--MD differences.
The published C and CH DFT-MD timesteps provide reference scales, not a
material-independent stability criterion :cite:p:`BethkenhagenEtAl2020`.
MD uses MPI; the recorded CH₂ runs use four physical CPU cores per trajectory.

Each trajectory is checked for completion, finite output and NVE energy
drift. Structure factors are evaluated from periodic density modes, with an
additional RDF-transform comparison for CH₂. Reported frame or block SEMs
are sampling diagnostics; they do not include every autocorrelation,
finite-size or discretization error. Global temperature stability alone
does not establish convergence of every species or structural observable.

The table reports thermostat targets separately from actual NVE means.
Final energy changes use the initial NVE total energy as their reference;
this relative diagnostic depends on the potential-energy zero.

.. include:: /_static/gallery_results/plot_ch2_hnc_md/md_protocol.rst

Interpretation
--------------

The external solve now receives the full system's evaluated final continuum
energy cutoff and retains the common infinite-radius potential reference.
Three consecutive QM SCF confirmations are required. These changes repair
specific cutoff, potential-reference and stopping inconsistencies; they do
not establish global electron-number convergence at every state.

In particular, tighter source-charge diagnostics still detect residual
pseudoatom electron-number errors for some hot H and CH₂ states. Passing
the established workflow checks must not be interpreted as satisfying an
arbitrarily tighter charge budget.

Agreement with a literature curve tests the stated model at the listed
conditions. It does not validate all elements, densities, temperatures or
transport observables. The default Dirac model contains local exchange;
correlation is present only when the selected XC model includes it.

See :doc:`validation_policy` for reference-data provenance and the separation
between calculation outputs and recorded presentation assets.
