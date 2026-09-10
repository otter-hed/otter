Complete Al 1 eV workflow
=========================

The gallery page :doc:`../gen_examples/plot_al_full_workflow` is a complete,
single-state walk-through for Al at ``rho=8.1 g/cc`` and
``Te=Ti=1 eV``.  It includes:

* the finite-temperature bound-level table;
* full, continuum/free, external, ionic, pseudoatom, and screening densities;
* full/external effective potentials and their nuclear, Hartree, and
  exchange-correlation components;
* :math:`q(k)=n_{\rm scr}(k)` and
  :math:`f(k)=n_{\rm ion}(k)`;
* :math:`V_{ii}(k)` and its inverse transform :math:`V_{ii}(r)`;
* the final :math:`g_{ii}(r)` and :math:`S_{ii}(k)`.

Running the source calculates this state from physical inputs by default
(``RECOMPUTE_WITH_OTTER=True``), then saves local results and figures. No
bundled Otter NPZ is required. The checksummed local-archive mode is only for
maintainer review. HNC residuals and the independent finite-DST
:math:`g\leftrightarrow S` closure error are checked separately.

For slides, the same script also writes three single-purpose figures (PNG and
vector PDF) to ``benchmarks/outputs/al_full_workflow_1ev/figures``:
``al_full_workflow_electronic_densities``, ``al_full_workflow_gii``, and
``al_full_workflow_sii``.  They are generated directly from the state used by
the composite gallery figures, so no second calculation or data-export script
is required.  These slide-only exports are intentionally hidden from the HTML
gallery; the page displays only the two overview figures above.

Confirming QM SCF convergence
-----------------------------

``FullExternalConfig(scf_convergence_steps=3, ...)`` requires three consecutive
passes of the existing stopping conditions for each QM full/external SCF.
The default remains ``1``. With the workflow interface, put
``"scf_convergence_steps": 3`` in ``aa_overrides``; when calling
``solve_ks_dft_is`` directly, use ``KSDTFConfig(convergence_steps=3, ...)``.
This option does not change the separate TF solver's convergence policy.

Any failed condition resets the count. Full KS SCF checks finite density and
potential changes, the self-consistency map residual, and any requested B3
charge constraint, after the screened preconditioning stage has ended.
The dedicated fixed-chemical-potential external loop retains its existing
density/potential and B3 checks; this option does not introduce a new external
map-residual tolerance. The count is recorded in ``scf_convergence_steps`` and
the running streak in each SCF history row. Reusing a full result with fewer
recorded confirmations than requested is rejected.

Repeated passes protect against isolated crossings of the stopping thresholds.
They do not establish convergence with respect to the radial grid, energy
quadrature, partial-wave basis, or outer matching domain, and do not override
an ``unresolved`` shallow-state diagnostic. The neutral full solver's final
density refresh is also not three additional SCF iterations.

The neutral-full return path checks the refreshed density's Poisson/XC map
again against the unchanged ``tol``. ``final_state_map_error`` records that
residual; a failed check revokes convergence and reports
``scf_stop_reason="final_refresh_map_residual"``. It does not update the
returned potential without recalculating its density. This additional check
does not change the separate fixed-mu/external stopping policies.

SCF history regularization
--------------------------

The QM full and external Eyert mixers normalize their radially weighted
residual-history columns before solving a regularized least-squares problem
by SVD. ``scf_mixing_w0`` and ``ext_mixing_w0`` are dimensionless relative
regularization weights; the direct KS interface calls this ``mixing_w0``.
They are no longer absolute floors on an unnormalized history matrix.
The regularized normal-matrix condition number is bounded by :math:`10^8`;
negligible columns are excluded, and a failed finite solve falls back to
linear mixing. Explicit values previously tuned for an absolute floor may
need reassessment.

Full-SCF step safeguards still reject a loss of the interacting-continuum
matching window. History between valid, actually evaluated potentials survives
a damped step; an invalid evaluated input clears it. A change from screened
preconditioning to ordinary Poisson, or a new SCF invocation, also resets the
history. These changes do not relax convergence or shallow-state acceptance
conditions. Observables sensitive to small residual density errors, especially
the low-:math:`k` effective potential, still require tolerance checks.

Checking for missing shallow levels
-----------------------------------

QM full calculations default to ``bound_spectrum_check=True``. After SCF,
an independent zero-tail pole scout checks every configured bound angular
channel, including channels with no finite-box negative eigenvalue. It uses
the same exterior-matching machinery as the bound refinement
(:cite:`WilsonEtAl2006`, Appendix A.4; :cite:`StarrettEtAl2019`, Eqs. 21--22).
The finite binding window, at least :math:`\kappa R\leq5`, is an Otter
diagnostic heuristic, not a proof of spectral completeness.

A missing candidate marks the threshold representation ``unresolved`` and
enters the existing bounded cold-retry policy. A candidate on a nonasymptotic
potential is not inserted as a physical orbital. The retry recomputes the
full SCF with exterior matching and a common bound/continuum potential
domain; it does not add electrons after convergence or modify :math:`M(E)`.
If recovery fails, the result remains ineligible for normal QOZ use.

``bound_state_diagnostics["spectrum_check"]`` records the searched window,
angular channels and candidates. ``no_additional_pole_in_window`` means only
that this finite sign-change search found no additional pole; it does not
establish convergence outside that window. The diagnostic and final-map
residual also remain in the convergence metadata of minimal state exports.
Explicitly disabling the scout is a diagnostic comparison, not evidence
that a finite-box spectrum is complete.

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
be recomputed before external-only reuse under the new policy. This is a
numerical discretization change, not a new definition of ionization, and it
does not replace energy/grid/domain refinement or threshold-state checks.
