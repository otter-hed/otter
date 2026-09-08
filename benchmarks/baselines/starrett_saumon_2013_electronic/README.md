# Starrett--Saumon electronic benchmark baseline

`electronic_states.npz` contains eight project-generated IS/SC result pairs:

- quantum-orbital Al at 2.7 g cm\(^{-3}\) and 2, 6, 10, and 15 eV;
- Thomas--Fermi Fe at the four Hugoniot points in Table III of Starrett and
  Saumon (2013).

All states use the physical \(E=0\) bound/continuum split.  Each production
ion-sphere (IS) workflow is continued through the experimental self-consistent
(SC) AA/QOZ/HNC feedback implementation.  That feedback follows Sec. 2.4 of
Starrett and Saumon (2014), holds the converged IS chemical potential fixed,
and is not claimed to be numerically identical to the simultaneous 2013
QTCP/TFTCP solver.  All eight SC continuations passed the explicit ion-
structure and correlation-potential convergence tests.

The accepted archive records both
\(Z^*=n_e^0/n_I^0\) and the all-space ionic-density partition
\(\bar Z=Z-\int n_e^{\rm ion}\,d^3r\). For quantum states it also records
the 1s, 2s, 2p, and 3s energies, \(M(E)\), and the scattering-derived
\(\gamma\), separately for IS and SC.

At 15 eV the IS state contains no negative-energy 3s level.  SC feedback
produces a resolved 3s state at -0.00846 Ha with \(\gamma=0.171\) Ha and
\(M=0.0928\), compared with the published -0.0125 Ha, 0.174 Ha, and 0.134.
The all-space threshold matcher is enabled over the documented 0.03-Ha window
so this shallow state is not classified from a finite-box eigenvalue alone.
The published values live in the adjacent `reference_data` package and are
not embedded in this project-generated NPZ.

The September 2026 refresh uses adaptive QM full-AA precision within SC and
checks the current unmixed correlation-potential residual. Its eight SC
iteration counts are 9, 8, 7, 8 (Al) and 10, 10, 13, 13 (Fe). The archive
records the actual final residuals and precision-refinement flags; these
values are not inferred from the inner SCF convergence flag.
