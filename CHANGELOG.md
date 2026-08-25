# Changelog

This project follows [Semantic Versioning](https://semver.org/). Changes that
have not yet been released are collected under “Unreleased”.

## Unreleased

## 0.2.3 - 2026-08-25

### Added

- Added an optional one-component Rosenfeld--Ashcroft VMHNC closure with a
  Percus--Yevick hard-sphere reference, variational packing-fraction search,
  strict raw-solution diagnostics, and primary-literature attribution.
- Added a reusable one- and multicomponent Otter-to-LAMMPS driver with
  shifted-force pair tables, total and partial structure diagnostics,
  reproducibility artifacts, and sampling uncertainties.
- Added HNC, VMHNC, and same-potential MD comparisons to all four Johnson et
  al. (2025) aluminium panels and the Wünsch et al. (2009) beryllium state.
- Extended the Schörner et al. (2022) aluminium benchmark to compare LDA/PBE
  HNC and VMHNC with direct, same-potential LAMMPS structure factors and the
  corrected published DFT-MD curves.
- Added a Starrett--Saumon (2013) electronic-structure benchmark comparing
  published Al/Fe levels, pressure-ionization weights, and ionization metrics
  with reproducible Otter IS and experimental SC-feedback calculations.

### Changed

- Made the bound-state sum and continuum integral meet at the asymptotic
  `E=0` potential gauge by default, while retaining local-potential
  thresholds as explicit finite-box sensitivity controls.
- Made ``G_ee_k`` the canonical public key for the electron local-field
  correction in workflow and ``otter_state_v4`` outputs; ``gee_k`` and
  ``g_ee_k`` remain temporary compatibility aliases.

### Fixed

- Reclosed diffuse threshold-state pseudoatoms through paired total-full and
  external-density B3 tails without changing the ordinary continuum-tail path.
- Detected repeated bound-charge branch crossings in nominally converged
  average-atom histories and refined them before common-chemical-potential
  root acceptance.

## 0.2.2 - 2026-08-23

### Added

- Added reusable preparation for multicomponent QOZ/HNC ion-temperature
  scans while retaining the general solver for larger mixtures.

### Changed

- Reused converged species-resolved average-atom states and chemical
  potentials when constructing external electronic states.
- Reduced repeated QOZ/HNC transforms and matrix work without changing the
  requested numerical tolerances.
- Focused the public installation guide on the locked Poetry workflow.

### Fixed

- Strengthened shallow-bound-state, continuum-tail, and charge-partition
  handling near pressure ionization.
- Prevented failed continuation states and invalid average-atom intervals
  from corrupting common-chemical-potential root searches.

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
