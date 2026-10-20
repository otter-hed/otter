Free-channel continuum completion
=================================

Free-channel completion is an optional diagnostic for the quantum continuum.
It restores the free-electron contribution omitted by the accepted angular
cutoff. The default calculation leaves this option disabled.

Usage
-----

Enable the same correction for the full and external electronic states:

.. code-block:: python

   from otter import PlasmaWorkflowConfig, solve_plasma_workflow

   cfg = PlasmaWorkflowConfig(
       formula="Be",
       temperature_ev=10.0,
       rho_g_cc=10.0,
       aa_overrides={"cont_free_completion": True},
   )
   result = solve_plasma_workflow(cfg)
   aa = result["electronic"]["species"][0]["result"]
   print(aa["meta"]["cont_free_completion"])

The correction also applies to mixtures through the shared ``aa_overrides``.
It requires the quantum electronic model. No additional dependency is needed.

The direct ``FullExternalConfig`` interface accepts the keyword-only option
``cont_free_completion=True``, preserving existing positional arguments.
Direct scattering integrations accept
``free_completion=True``; adaptive and fixed-grid basis metadata record the
resolved value under ``free_completion``. Direct KS results record
``cont_free_completion``.

Spectral density
----------------

For positive energy :math:`E` in atomic units, let :math:`k=\sqrt{2E}`.
With spin included, the free partial-wave density is

.. math::

   n_{E\ell}^{0}(r)
   =\frac{k}{\pi^2}(2\ell+1)j_\ell(kr)^2.

The energy-normalized continuum convention follows
:cite:p:`StarrettSaumon2014`, Eq. A3. Spherical Bessel completeness gives
:math:`\sum_{\ell=0}^{\infty}(2\ell+1)j_\ell(z)^2=1`
:cite:p:`NISTDLMF`, Eq. 10.60.12. After the original angular selection has
accepted a cutoff :math:`L`, the completed spectral density is

.. math::

   n_E^{\mathrm{comp}}(r)
   =\sum_{\ell=0}^{L}n_{E\ell}^{V}(r)
    +\frac{k}{\pi^2}
       \left[1-\sum_{\ell=0}^{L}(2\ell+1)j_\ell(kr)^2\right].

The free remainder can be evaluated from two orders. With :math:`z=kr`,
the recurrence in :cite:p:`NISTDLMF`, Eq. 10.51.1, gives

.. math::

   1-\sum_{\ell=0}^{L}(2\ell+1)j_\ell(z)^2
   =z^2\left[j_L(z)^2+j_{L+1}(z)^2\right]
      -2(L+1)zj_L(z)j_{L+1}(z).

The correction is added once, after any native angular retries. The same
Fermi--Dirac occupation multiplies the interacting sum and the free
remainder before energy integration. Negative roundoff is not clipped, and
no density or charge normalization is applied.

Scope
-----

Only the omitted free part is exact. The omitted interacting contribution
:math:`\sum_{\ell>L}(n_{E\ell}^{V}-n_{E\ell}^{0})` still needs an angular
convergence check. The option preserves the original angular acceptance,
phase shifts, propagation domain, matching window, energy limits and B3
policy. It does not reduce the number of propagated channels.

Completion supplies no density from omitted energy intervals and does not
remove B3. Agreement with an enlarged angular sum at the same matching
boundary does not establish convergence of that boundary. Full, external
and pseudoatom electron numbers must be checked independently after SCF.

Cache entries are specific to the completion setting. Use separate energy
caches for enabled and disabled calculations; mixing these entries raises
an error. Reusing a full state with a different completion setting is also
rejected before the external solve.

See :doc:`../benchmarks/free_channel_completion` for the numerical controls
and their scope.
