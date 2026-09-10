# Changelog

This project follows [Semantic Versioning](https://semver.org/). Changes that
have not yet been released are collected under “Unreleased”.

## Unreleased

## 0.3.1 — 2026-09-10

- Use default numerical controls in both the Al gallery and standalone Colab;
  remove the gallery's extra HNC tolerances. Show partition Zbar and explicit
  Zstar, retain SCF progress and wall time, and test the complete configuration
  for parity.
  Document the same direct charge access for individual mixture species and
  the species-vector fields in standard NPZ exports.

- Expose `zstar` directly in QM/TF AA results and electronic exports. Portable
  exports prefer this value while retaining the n0/n_i fallback for older
  results; mixture values use each AA-cell density, not bulk partial density.
  Clarify export groups, convergence checks, configuration provenance and
  result/metadata access in the state guide. Add the Al Rayleigh-weight panel.
  Do not count disabled external-stage placeholders as an executed AA stage.
  Read nuclear charge from AA metadata when needed to export native TF results.

- Document selective orbital saving/loading and add orbital plots to the Al
  gallery and Colab notebook. Keep plotting code in the examples; use matching
  three-panel layouts, a_B units and electronic-only temperature labels.
  Correct the Al density plot to show n_ion rather than n_bound under that label.

- Expose the final, unweighted KS radial wavefunctions on their bound grid;
  optionally return/display amplitudes multiplied by FD occupation (off by
  default). Reuse existing ion-orbital densities and the QOZ radial transform
  for per-level form factors, with a runnable plotting example. Portable
  orbital exports retain raw wavefunctions and full-grid-transformed form
  factors when available. TF and solver defaults are unchanged.

## 0.3.0 — 2026-09-10

### Release scope

- Consolidate the selective-export and continuum/SCF work as a reviewed
  checkpoint. Default exports remain complete; selective profiles are opt-in.
- Public example and benchmark scripts compute from input parameters. Generated
  Otter NPZ/restart files are not part of the public source distribution.
  HTML uses reviewed figures and recorded output, without running expensive
  calculations during a documentation build. CH2 MD reproduction computes new
  electronic states and pair potentials before invoking LAMMPS.
- Document source-checkout reproduction and all three download formats on
  every gallery page. Downloaded notebooks launch the same file-backed Python
  script using the kernel interpreter, stream progress and propagate failures;
  they do not duplicate process-pool workers in an interactive namespace.
- Fresh Schörner MD reproduction no longer reuses a trajectory just because
  an old LAMMPS log reports completion. It reruns the new pair table; producer
  metadata takes the installed version instead of a hardcoded historical one.
- Refresh the gallery against the September 9 conditioned-SCF campaign, with
  separate SC checks for the latest SC-only zero-tail default. Historical MD
  is retained and labelled; it is not a same-potential validation of newly
  calculated Otter curves. SC feedback remains experimental.

### Final SCF corrections

- Use one consistent ion-sphere background in full and external AA, including
  cancellation of the uniform electronic background before quadrature.
- Normalize multisecant history and bound its conditioning with SVD/ridge
  regularization; retain valid history across safeguarded potential updates,
  but reset it after an invalid continuum matching window. This avoids stalls
  without changing the physical
  density/potential map or relaxing convergence and threshold gates.
- Enable `bound_zero_tail_refine=True` inside quantum SC feedback when the
  user has not specified it. Explicit global/species overrides remain valid;
  ordinary IS and TF defaults and the shallow-state search window are unchanged.

### Added

- Added optional consecutive confirmation of QM SCF convergence with
  `scf_convergence_steps` (`convergence_steps` in `KSDTFConfig`). The default
  remains one; a failed gate resets the streak. Full-result reuse cannot
  claim more confirmations than the saved calculation performed.
- Added an optional one-component Rosenfeld--Ashcroft VMHNC closure with a
  Percus--Yevick hard-sphere reference, variational packing-fraction search,
  strict raw-solution diagnostics, and primary-literature attribution.
- Added a four-panel Johnson et al. (2025) aluminium benchmark comparing
  ordinary HNC, VMHNC, and same-potential LAMMPS MD on the same IS electronic
  states with published 2TTCP, YOCP, and DFT-MD curves.
- Extended the Schörner et al. (2022) aluminium benchmark to compare LDA/PBE
  HNC and VMHNC with direct, same-potential LAMMPS structure factors and the
  corrected published DFT-MD curves.
- Added a reusable one- and multicomponent Otter-to-LAMMPS driver and extended
  the Wünsch beryllium benchmark with VMHNC and same-potential MD, including
  sampling uncertainties and reciprocal-shell reliability metadata.

### Changed

- Reuse grid validation and the energy-independent transformed potential
  within one shallow-bound-state energy scout/root search. Keep the existing
  Numerov recurrence, exterior matching and normalization unchanged; no
  potential or orbital cache is shared between SCF iterations.
- Recover slow bound-state Arnoldi solves with a larger working subspace after
  a bounded initial attempt. Keep the Hamiltonian, shift, eigenpair count and
  tolerance unchanged; never accept partially converged eigenpairs.
- Batch continuum matching-window index searches and select the rightmost
  sufficiently long potential-valid run once per energy. Preserve the scalar
  selector's low-kr relaxation, strict potential-tail criterion and basis
  normalization; no matching plan is carried between SCF potentials.
- Continuum angular retries reuse raw waves already propagated at the same
  energy and potential, and propagate only additional channels. Numerov,
  matching, angular-tail acceptance, SCF tolerances and physical geometry are
  unchanged. No wave cache is carried between SCF iterations.
- Automatic continuum angular selection now tries ``cont_l_max_soft=250``
  (inclusive) by default. Density/transport remainder failures automatically
  recover a larger range; this is not a hard cutoff. Users may change the
  trial limit or disable it with ``None``. Explicit non-``match`` angular
  strategies retain their requested behavior.
- Gallery and benchmark calculations inherit the single-AA worker default
  instead of overriding continuum jobs/shards. Independent state pools remain
  available. Otter-only validation can also regenerate the CH2 benchmark's
  electronic inputs into a separate candidate cache without rerunning MD.
- CH2 split exports retain electronic form factors once per electron
  temperature instead of duplicating them for every ion temperature; the
  audit checks their shapes, aliases and convergence metadata separately
  from each ionic structure file.
- Made ``G_ee_k`` the canonical public key for the electron local-field
  correction in workflow and portable state outputs; ``gee_k`` and
  ``g_ee_k`` remain temporary compatibility aliases.
- Added ``otter_state_v5`` selective export profiles for electronic summaries,
  electronic levels, ion structure, and complete states, with optional data
  groups and backward-compatible loading of earlier state schemas.
- Included compact bound-level energies, occupations, and pressure-ionization
  weights in ``electronic_summary``; ``electronic_levels`` remains a
  compatibility alias.
- Reduced newly generated ion-structure-library archives to the quantities
  used by that benchmark (state/convergence scalars, ``f(k)``, ``q(k)``,
  ``g(r)``, and ``S(k)``), instead of retaining unrelated electronic profiles
  and QOZ intermediates.

### Fixed

- Declare the checkout root and `src` in pytest's import paths so both
  `poetry run pytest` and `python -m pytest` can collect repository helpers.
  Isolated-interpreter regressions cover execution from other directories.
- Guard QM SCF updates against losing an already-valid continuum matching
  window, and reject final states that would use the free-wave window fallback.
  The bounded backtracking preserves the physical map and all existing
  tolerances; it fixes the Be 50-eV, 65-g/cm3 oscillation without relabelling
  a failed SCF. Benchmark failures now retain scalar residual histories and
  recovery diagnostics rather than only a generic exception.
- Experimental SC feedback now tightens QM full-SCF precision only near outer
  convergence or stagnation, retaining stricter user settings. Acceptance also
  checks the unmixed current correlation-potential residual, not only the
  damped update. The new `inner_full_tol_scale` control defaults to `0.01`;
  `None` disables numerical refinement for diagnosis. Ordinary IS/CH2 AA,
  TF, external, spectral and threshold-resolution defaults are unchanged.
  `SCFeedbackConvergenceError` preserves the failed last iterate and history,
  including across process workers, without accepting it as a solution.
- Use the actual WS boundary for neutral-AA charge integration and reporting,
  paired with an analytic sharp IS cavity in full/external potentials, even
  when R_geometry equals R_WS. Correlated SC backgrounds are not replaced.
  `FullExternalConfig.exact_ws_boundary_quadrature=False` exposes sampled
  quadrature for numerical comparison; saved full results with different or
  unrecorded WS quadrature cannot silently be reused under the new policy.
- Retain evanescent continuum partial waves in automatic angular selection.
  A channel whose oscillations start outside the matching window can still
  contribute to the density. Raise the matching-based workload estimate by
  the existing potential-aware turning-point margin; keep the requested
  global angular budget, soft-limit recovery and acceptance gates unchanged.
  Analytic free-density and explicit large-angular-sum regressions cover
  low-energy normalization. This does not by itself resolve every shallow
  bound-state or B3-tail sensitivity.
- Do not infer a continuum phase-root quadrature requirement merely from the
  configured bound-state search range. During domain recovery, preserve the
  requested quadrature unless an actual unresolved high-l threshold triggered
  the retry. Bound search settings and threshold acceptance remain unchanged.
- Hand a persistently stalled initial inner-neutral full-B3 SCF to the existing
  bounded energy/domain recovery instead of exhausting its entire budget.
  Require lack of both best and median residual progress, including the
  unmixed potential-map error. Standalone KS solves and recovery attempts keep
  their budgets; SCF and threshold acceptance criteria are unchanged. Retry
  metadata now records the first-pass iteration count and stop reason.
  ``scf_stagnation_recovery=False`` disables only the early handoff.
- Include rejected angular trials in continuum evaluation-time diagnostics.
  Previously their propagation time was counted but their total time was not.
- Bound the A3 partial-wave work by the density region needed by B3 while
  retaining the propagation/matching box and checking density/transport tails;
  recover the original angular range if the remainder check fails. Raw density
  outside the optimized region is not exported as a converged A3 state sum.
- Preserve accepted Simpson panels during inner-mu continuum-basis reuse
  instead of replacing them with trapezoids on all cached scout nodes. Resolve
  moving Fermi edges with product quadrature, and ignore arbitrary pi phase
  branches without removing physical resonance detection.
- Try one bounded, same-state, same-domain energy refinement before a cold
  expanded-domain retry for near-stationary full-B3 SCF failures. Preserve the
  mixer and acceptance tolerances; unresolved threshold states still take the
  existing spatial/threshold recovery. Numerov propagation is unchanged.
- Energy sharding now dispatches chunks of one global adaptive mesh, including
  global resonance panels. Worker count, shard count, and workload policy no
  longer rebuild different base grids. The single-AA worker default remains 1.
- The automatic continuum energy ceiling bounds the omitted ideal density
  relative to n0 as well as endpoint occupation. Removed the high-level fixed
  150-partial-wave cap, which could underresolve large dilute A3 domains.
  Neither change modifies the bound/free edge or forces a density normalization.
- Cancel the uniform electron/background density before radial integration in
  the analytic ion-sphere path. Its potential, source-charge targets, and saved
  Hartree decomposition now use the same quadrature; a neutral uniform test
  sphere has zero exterior potential independently of box size and grid spacing.
- Made the Appendix-B total-density path, ``b3_tail_target="full"``, the
  self-consistent full-branch default. Newly computed states no longer receive
  a hidden post-SCF total-density reclosure; that path remains available only
  for compatible legacy caches. The explicit ``"cont"`` target is retained
  for diagnostics and historical comparisons.
- Added one fail-closed zero-tail retry for an unresolved threshold state,
  including when it prevents full SCF convergence. During recovery, A3 uses
  the existing bound/SCF domain without changing the B3 handoff or relaxing
  acceptance bounds. Mixture searches retain the refined setup, invalidate
  old spectral samples, and preserve the accepted settings for external AA.
  A state that remains unresolved is still rejected by QOZ.
- The bounded continuum-domain recovery uses ``batch``. Both ``batch`` and
  the repaired ``shard`` now share the serial energy mesh; mixture/external
  continuation preserves the selected recovery domain and numerical settings.
- Extend the bounded A3-domain retry to unconverged full-density B3 SCF even
  when its surviving bound levels are resolved. A resolved orbital is not a
  certificate of SCF convergence. Deep surviving shells do not enlarge the
  shallow-state matching energy window in this recovery path.
- Skip an explicitly disabled stage 1 during cold fixed-mu SC recovery.
  Previously the zero-iteration solver accessed an uninitialized energy cache.
  Recovery now starts the real stage-2 solve from the Coulomb potential at
  the unchanged IS chemical potential, without fabricating a stage-1 result.
- Preserve successful Schörner and Starrett--Saumon candidate states before
  reporting another state's failure. Recompute queues now also inspect
  per-state manifests and unresolved-threshold records after a zero exit code.
  Carbon scan metadata follows the actual continuum-edge default, and cached
  seeds require a matching AA source/configuration fingerprint. Historical
  baselines and their recorded provenance are not rewritten by these changes.

## 0.2.1 - 2026-08-14

### Added

- Expanded ``otter_state_v3`` archives with electronic profiles, orbital
  densities, response functions, QOZ interaction channels, pair potentials,
  structure factors, and calculation metadata.
- Added a Colab introduction and optional Libxc installation extra.

### Changed

- Added a Poetry lock file and one cross-platform `poetry install` path with
  runtime, plotting, tests, documentation, and editable source installation.
- Added package-install smoke tests on all three operating systems and on
  supported CPython release lines.
- Replaced the duplicated quick-start snippets with the canonical
  ``examples/single_species_workflow.py`` example, which now exercises the
  production defaults directly.
- Kept the top-level examples focused on single-species and mixture workflows;
  numerical diagnostics and model studies now live under ``tools``.
- Stabilized mixture roots near pressure ionization and preserved validated
  external-density tails through the final electronic solve.
- Recomputed and promoted all accepted example and benchmark NPZ baselines
  with field inventories and provenance metadata.
- Simplified the software citation and refreshed the example and benchmark
  documentation.

## 0.2.0 - 2026-08-10

### Added

- Unified single-species and mixture AA → pseudoatom → QOZ/HNC workflow.
- Orbital Kohn–Sham and finite-temperature Thomas–Fermi electronic backends.
- Finite-temperature Chabrier (1990) jellium local-field correction.
- Portable, pickle-free `q(k)`, `f(k)`, `g_ij(r)`, and `S_ij(k)` state files.
- Cached, provenance-checked Starrett et al. mixture benchmark.
- Experimental AA ↔ QOZ/HNC self-consistent feedback API.
- Configurable exchange-correlation models, including dependency-free Dirac
  exchange and optional Libxc-backed LDA/PBE functionals with recorded
  software and functional provenance.
- Self-contained capability and scientific-benchmark galleries with
  results-first HTML pages and publication-ready PNG/PDF figures.

### Changed

- Physical Fermi–Dirac bound occupation is the production default.
- Pseudoatom charge closure is enforced on the QOZ/DST lattice.
- HNC production paths reject unconverged or projected nonphysical roots.
- Experimental SC feedback and production state export now fail closed on
  missing convergence status.
- Continuum threshold, phase-shift resonance, weak-bound-state, and B3/Friedel
  tail diagnostics were strengthened.
- Scientific reference datasets now carry explicit source, checksum, rights,
  and redistribution metadata separate from Otter's software license.

## 0.1.0

- Initial Otter project structure and documentation prototype.
