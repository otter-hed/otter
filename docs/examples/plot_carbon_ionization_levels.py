r"""
Carbon ionization at 100 eV
===========================

Otter reports the carbon ionization measures
:math:`\bar Z=Z-Q_{\mathrm{ion}}(R_{\mathrm{WS}})` and
:math:`Z^*=n_e^0/n_i` across a density scan at :math:`T_e=100` eV.
The :doc:`Bethkenhagen comparison </benchmarks/gen_benchmarks/plot_bethkenhagen_et_al_2020_carbon_ionization>`
provides the calculation, orbital diagnostics and discussion of the different
electron partitions used by the published models.

This page shares that benchmark's figure. The scan is calculated once by the
benchmark; opening this example does not start another AA calculation.
No bundled Otter NPZ is required.

Reproduction
------------

Calculate the benchmark from the repository root::

    poetry run python benchmarks/examples/plot_bethkenhagen_et_al_2020_carbon_ionization.py

Then locate its saved figure with::

    poetry run python docs/examples/plot_carbon_ionization_levels.py

Recorded result
---------------

.. include:: /_static/gallery_results/plot_carbon_ionization_levels/results.rst

"""
from pathlib import Path


def main() -> None:
    """Report the shared benchmark figure without recalculating the scan."""
    root = Path(__file__).resolve().parents[2]
    figure = (root / "benchmarks/outputs/bethkenhagen_et_al_2020_carbon_ionization"
              / "figures/bethkenhagen_2020_carbon_ionization.png")
    if not figure.is_file():
        raise FileNotFoundError(
            "Calculate benchmarks/examples/"
            "plot_bethkenhagen_et_al_2020_carbon_ionization.py first."
        )
    print(figure)


if __name__ == "__main__":
    main()

# sphinx_gallery_thumbnail_path = "_static/gallery_results/plot_bethkenhagen_et_al_2020_carbon_ionization/thumbnail.png"
