Complete Al 1 eV workflow
=========================

The gallery :doc:`../gen_examples/plot_al_full_workflow` calculates
Al at ``rho=8.1 g/cc`` and
``Te=Ti=1 eV``.  It includes:

* the finite-temperature bound-level table;
* unweighted bound-state wavefunctions, ionic densities and form factors
  resolved by level;
* full, continuum/free, external, ionic, pseudoatom, and screening densities;
* full/external effective potentials and their nuclear, Hartree, and
  exchange-correlation components;
* :math:`q(k)=n_{\rm scr}(k)` and
  :math:`f(k)=n_{\rm ion}(k)`;
* electron--ion and ion--ion potentials in real and reciprocal space,
  shown on symmetric logarithmic axes;
* the final :math:`g_{ii}(r)`, :math:`S_{ii}(k)` and Rayleigh weight
  :math:`|f(k)+q(k)|^2 S_{ii}(k)`.

Running the source calculates this state from physical inputs by default
(``RECOMPUTE_WITH_OTTER=True``), then saves local results and figures. No
bundled Otter NPZ is required. HNC residuals and the independent finite-DST
:math:`g\leftrightarrow S` closure error are checked separately.

The gallery displays four figure groups: bound orbitals, electronic densities
and potentials, screening and pair interactions, and ionic correlations with
Rayleigh weight.

Set ``EXPORT_SLIDE_FIGURES = True`` to save three additional figures (PNG and
vector PDF) to ``benchmarks/outputs/al_full_workflow_1ev/figures``:
``al_full_workflow_electronic_densities``, ``al_full_workflow_gii``, and
``al_full_workflow_sii``. These exports use the same calculated state and are
disabled by default.

Confirming QM SCF convergence
-----------------------------

QM full and external SCF require three consecutive passes of their stopping
conditions by default. Both branches use density and potential change
tolerances of :math:`10^{-5}`. ``FullExternalConfig.scf_convergence_steps``
controls the confirmation count; the direct KS interface calls it
``KSDTFConfig.convergence_steps``. The separate TF solver retains its own
stopping policy.

Any failed condition resets the count. Full KS SCF checks finite density and
potential changes, the self-consistency map residual, and any requested B3
charge constraint, after the screened preconditioning stage has ended.
The dedicated fixed-chemical-potential external loop also requires a valid
interacting-continuum matching window for both the evaluated input and the
accepted potential. Its unmixed potential-map residual must satisfy the
existing ``ext_scf_dv_tol``; a small damped step alone cannot establish
self-consistency. The count is recorded in ``scf_convergence_steps`` and
the running streak in each SCF history row. Reusing a full result with fewer
recorded confirmations than requested is rejected.

Full and external potentials share the zero at infinite radius and the same
chemical potential (:cite:`StarrettSaumon2014`, Eqs. (4)--(7)). A residual
source charge gives a finite-box Coulomb tail. The external loop retains
that potential: subtracting its value at the outer boundary while holding
the chemical potential fixed would change the electron occupations.

Repeated passes protect against isolated crossings of the stopping thresholds.
They do not establish convergence with respect to the radial grid, energy
quadrature, partial-wave basis, or outer matching domain, and do not override
an ``unresolved`` shallow-state diagnostic. The neutral full solver's final
density refresh is also not three additional SCF iterations.

The native AA result reports ``pa_charge_residual = q_pa_all - Z`` separately.
Pseudoatom neutrality requires :math:`\int n^{\rm PA}(r)\,d^3r=Z`
(:cite:`StarrettSaumon2014`, Eq. (9)). A converged SCF map alone does not
bound this integral error; the continuum energy range and B3 matching radius
must also resolve the required charge accuracy.

The neutral-full return path checks the refreshed density's Poisson/XC map
again against the unchanged ``tol``. ``final_state_map_error`` records that
residual; a failed check revokes convergence and reports
``scf_stop_reason="final_refresh_map_residual"``. This check applies to the
neutral full solver.

SCF history regularization
--------------------------

The QM full and external Eyert mixers normalize their radially weighted
residual-history columns before solving a regularized least-squares problem
by SVD. ``scf_mixing_w0`` and ``ext_mixing_w0`` are dimensionless relative
regularization weights; the direct KS interface calls this ``mixing_w0``.
The regularized normal-matrix condition number is bounded by :math:`10^8`;
negligible columns are excluded, and a failed finite solve falls back to
linear mixing. Values tuned for older, unnormalized mixers may need adjustment.

Full and external SCF step safeguards reject a loss of the interacting-continuum
matching window. History between valid, actually evaluated potentials survives
a damped step; an invalid evaluated input clears it. A change from screened
preconditioning to ordinary Poisson, or a new SCF invocation, also resets the
history. These changes do not relax convergence or shallow-state acceptance
conditions. Observables sensitive to small residual density errors, especially
the low-:math:`k` effective potential, still require tolerance checks.

Continuum energy cutoff
-----------------------

The quantum continuum density is an energy integral weighted by the
Fermi--Dirac occupation (:cite:`StarrettSaumon2014`, Eq. (A3)). The
``prev_mu_fd`` policy chooses its upper limit from the current chemical
potential and electron temperature. ``cont_stage2_e_max_occ_tol`` sets
:math:`\epsilon`, with default :math:`10^{-5}`, and requires

.. math::

   f(E_{\max})=\frac{1}{1+\exp[(E_{\max}-\mu)/T_e]}\leq\epsilon.

This gives the lower bound
:math:`E_{\max}\geq\mu+T_e\log[(1-\epsilon)/\epsilon]`.
The solver also applies the configured energy floor and a conservative
bound on the omitted ideal continuum density relative to :math:`n_0`.
The final upper limit can therefore exceed the occupation-only bound.
The external branch uses the final full-branch energy limit.

A smaller :math:`\epsilon` extends the numerical energy integral. It is
independent of the SCF density and potential change tolerances and does not
bound the total pseudoatom charge error or the EOS pressure error. Check
energy-range and radial matching convergence separately.

Checking for missing shallow levels
-----------------------------------

QM full calculations default to ``bound_spectrum_check=True``. After SCF,
an independent zero-tail pole scout checks every configured bound angular
channel, including channels with no finite-box negative eigenvalue. It uses
the same exterior-matching machinery as the bound refinement
(:cite:`WilsonEtAl2006`, Appendix A.4; :cite:`StarrettEtAl2019`, Eqs. 21--22).
The search covers a finite binding window, at least :math:`\kappa R\leq5`.

A missing candidate marks the threshold representation ``unresolved`` and
triggers a full-SCF retry with exterior matching and a common bound/continuum
potential domain. Candidates require an asymptotic potential before they can
enter the orbital basis. Failed recovery prevents normal QOZ continuation.

``bound_state_diagnostics["spectrum_check"]`` records the searched window,
angular channels and candidates. ``no_additional_pole_in_window`` means only
that this finite sign-change search found no additional pole; it does not
establish convergence outside that window. The diagnostic and final-map
residual are included in the convergence metadata of minimal state exports.

WS boundary and numerical geometry
----------------------------------

The high-level AA workflow defaults to
``exact_ws_boundary_quadrature=True``: charge integration includes the partial
cell through the physical :math:`R_{\rm WS}`, not just the last inner node.
For the sharp ion-sphere background, the full and external potentials use
the analytic cavity at that same radius. A correlated SC background remains
sampled and is not replaced by a sharp sphere. This follows the boundary and
neutrality definitions in :cite:`StarrettSaumon2014`, Eqs. (1), (3), (4).

This choice does not change :math:`R_{\rm geometry}`, the outer grid, or B3
radii. For an explicit comparison to sampled quadrature, set the option to
``False`` (or put it in ``aa_overrides`` when using a plasma workflow).
``meta["ws_charge_quadrature"]`` and
``meta["ion_sphere_background_quadrature"]`` record the applied policy.
An existing full result with different or missing WS-policy metadata must
be recomputed before external-only reuse with this quadrature policy.
