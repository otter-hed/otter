Development
===========

Otter is organized around explicit physical stages rather than around example
scripts:

.. code-block:: text

   composition and thermodynamic state
       -> electronic full/external model
       -> pseudoatom screening density
       -> electron response and effective pair potentials
       -> QOZ/HNC ionic structure
       -> validated, serializable results

Contribution principles
-----------------------

* Preserve the public ``otter`` namespace while internal modules evolve.
* Add tests for physical invariants before moving a numerical implementation.
* Keep physical model choices separate from convergence controls.
* Cite primary literature next to equations and model implementations.
* Add every new key to ``src/otter/literature.bib`` and expose selectable
  model keys through ``config.citation_keys``; see :doc:`../citations`.
* Do not silently continue from an unconverged electronic or ionic solve.
* Keep generated files, caches, and machine-specific paths out of Git.

Documentation workflow
----------------------

The documentation is generated from hand-written reStructuredText, public
docstrings, and Sphinx-Gallery examples:

.. code-block:: console

   $ poetry run make -C docs html
   $ poetry run make -C docs strict
   $ poetry run make -C docs serve

``strict`` treats Sphinx warnings as errors.  A successful build must not
modify tracked files because autosummary and gallery products are ignored.

Third-party software provenance
-------------------------------

The finite-temperature Geldart--Vosko and Gregori-2007 LFC routines were
copied from `JaXRTS <https://github.com/JaXRTS/jaxrts>`__ and adapted to
Otter's NumPy/atomic-unit API.  They retain the
upstream BSD-3-Clause notice in
:download:`THIRD_PARTY_NOTICES.md <../../../THIRD_PARTY_NOTICES.md>`.
The corresponding software paper is :cite:t:`LutgertEtAl2026`; primary
physical-model references remain cited beside the implementation in
:mod:`otter.ionic.lfc`.

Benchmark workflow
------------------

Scientific-gallery files must be directly executable: running the source
calculates from physical inputs and exports figures without requiring an old
NPZ. Shared producers may live in ``benchmarks/runners`` or ``tools`` but not
in private application directories. Fresh outputs go under ``benchmarks/outputs``.
Sphinx does not execute the scientific calculations; it displays reviewed
presentation assets with recorded provenance. See :doc:`/benchmarks/validation_policy`.

All maintained scientific plots—not only the Sphinx galleries—must use
:mod:`otter.plotting` for the shared serif/STIX typography, font sizes,
inward major/minor ticks, line defaults, and semantic palettes.  Every saved
figure is exported as both a 300 dpi PNG and a vector PDF.  The
repository-wide ``test_plot_export_policy.py`` gate prevents examples,
benchmark runners, and diagnostic plots from bypassing that style/export
path.

.. toctree::
   :maxdepth: 1

   roadmap
