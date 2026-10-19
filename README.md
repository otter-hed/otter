# Otter

Otter calculates electronic and ionic structure in warm and hot dense matter.
From composition, mass density, and temperature, it solves a quantum
average-atom or Thomas–Fermi model, constructs neutral pseudoatoms and
effective pair potentials, and solves one- or multicomponent QOZ/HNC equations.

Otter is based primarily on the pseudoatom model of
[Starrett and Saumon (2014)](https://doi.org/10.1016/j.hedp.2013.12.001).

## Capabilities

- finite-temperature quantum (KS-DFT, QM) and Thomas–Fermi (TF) electronic structure; the
  QM model provides orbital levels, occupations, and density components;
- orbital densities, form factors and KS radial wavefunctions, with partition
  ionization `Zbar` and background ionization `Zstar`;
- pseudoatom densities `n_pa(r)` and screening densities `n_scr(r)`, with form factors
  `f(k)=n_ion(k)` and `q(k)=n_scr(k)`;
- electron–ion potentials `V_ei(r)`, `V_ei(k)` and effective ion–ion
  potentials `V_ij(r)`, `V_ij(k)`;
- one- and multicomponent QOZ/HNC results `g_ij(r)` and `S_ij(k)`.

## Install

The recommended source installation uses CPython 3.12 or newer, Git,
and Poetry 2.1.3.

macOS, Linux, or WSL:

```bash
curl -sSL https://install.python-poetry.org | python3 - --version 2.1.3
```

Windows PowerShell:

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py - --version 2.1.3
```

Verify the installation:

```console
poetry --version
```

Clone and install:

```bash
git clone https://github.com/otter-hed/otter.git
cd otter
poetry install
```

Dependencies are locked by `poetry.lock`; Otter is installed in editable mode.

Dirac exchange is the default XC model and requires no optional package.
Libxc provides additional LDA and GGA functionals, including `lda_pw`,
`lda_pz`, `lda_vwn`, and `pbe`.

To enable these additional functionals:

```bash
poetry install --extras libxc
```

PyPI distributes the Libxc Python bindings as source, so this optional step
requires CMake and a C compiler. See the
[XC installation guide](https://otter-hed.github.io/otter/user_guide/xc_functionals.html#installation).

```console
poetry run python -c "import otter; print(otter.__version__)"
```

## Quick start

Run an introductory calculation in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/otter-hed/otter/blob/main/notebooks/00-otter_intro.ipynb)

Run the complete [single-species workflow](examples/single_species_workflow.py)
from the repository root:

```bash
poetry run python examples/single_species_workflow.py
```

The default state is Al at `rho=8.1 g/cm^3` and `Te=Ti=15 eV`. Edit the input
block to change the state or output controls. The script plots the electronic
density, effective potential, `g_ii(r)`, `S_ii(k)`, and `V_ii`/`V_ei` in both
r and k space, and saves PNG, PDF, and NPZ files.

For mixtures, run [mixture_workflow.py](examples/mixture_workflow.py).

### Saved workflow state

NPZ files store selected quantities with their units, input parameters and
convergence records. The default `complete` profile includes electronic
profiles and levels, electron response, potentials and ionic correlations.
Smaller profiles retain electronic summaries or ionic structure; optional
groups add selected quantities. Complete exports include:

- `q_k == n_scr_k` and `f_k == n_ion_k`;
- `G_ee_k`, `chi0_k`, `chi_ee_k`, `c_ie_k`, `v_ee_k`, and `c_ee_k`;
- `v_ei_k` and `v_ei_r`, also available as `v_ie_k` and `v_ie_r`;
- `gij_r`, `sij_k`, `vij_r`, and `vij_k`.

Species axes are retained for both pure elements and mixtures.
`gee_k` and `g_ee_k` are compatibility aliases for `G_ee_k`.

The default windows are `r < 20 Bohr` and `k < 20 Bohr^-1`. Archives load with
`allow_pickle=False` and are written atomically. See the
[state-export guide](docs/source/user_guide/state_exports.rst) for in-memory
and NPZ access.

Parameter scans export the same electron–ion potentials to HDF5, with units
and coordinate paths. See the
[parameter-grid guide](https://otter-hed.github.io/otter/user_guide/parameter_grids.html)
for common and per-state grids.

Quantum continuum calculations can be slow near pressure ionization.
`continue_plasma_workflow_from_electronic_result` reuses a validated
electronic result for subsequent QOZ/HNC calculations.

## Validation and documentation

Browse the [documentation](https://otter-hed.github.io/otter/),
[example gallery](https://otter-hed.github.io/otter/gen_examples/index.html), and
[scientific benchmarks](https://otter-hed.github.io/otter/benchmarks/index.html).

Examples and benchmarks calculate from their input parameters. From the
repository root, for example:

```bash
poetry run python docs/examples/plot_al_full_workflow.py
poetry run python benchmarks/examples/plot_doppner_2023_be_ionization.py
```

Scripts calculate and save results locally.
The repository does not include precomputed Otter NPZ files. Literature
reference data and recorded figures, tables and terminal output are included.
MD comparisons require LAMMPS/MPI and first calculate the
Otter pair potentials. See the
[reproduction guide](https://otter-hed.github.io/otter/user_guide/reproducing_galleries.html)
for each page's commands, optional dependencies, and download formats.

Build the documentation, including its recorded example and benchmark results:

```bash
poetry install
poetry run make -C docs strict
```

This build does not run AA or MD calculations. Open
`docs/build/html/index.html` after the build. Start with:

- [documentation source](docs/source/index.rst);
- [capability example gallery](docs/examples/README.rst);
- [scientific benchmark gallery](docs/source/benchmarks/index.rst);
- [validation policy](docs/source/benchmarks/validation_policy.rst);
- [portable state schema](docs/source/user_guide/state_exports.rst).

Reference datasets have separate attribution and reuse conditions; Otter's
BSD software license does not apply to them unless stated in their manifest.
Read the [reference-data notice](benchmarks/reference_data/README.md) before reuse.

## Development

The 0.4.0 development version unifies single-species and mixture result axes.
See the [migration guide](https://otter-hed.github.io/otter/user_guide/state_exports.html)
before updating scripts written for 0.3.1. Parameter-grid calculations use
the optional HDF5 dependency: `poetry install -E grid`.
The [repository tools](tools/README.md) cover batch recalculation,
documentation assets and release checks.

```bash
poetry install
poetry run pytest -q
poetry run make -C docs strict
poetry run python -m build
poetry run python -m twine check dist/*
```

Anyone interested in Otter is welcome to contribute. See
[CONTRIBUTING.md](CONTRIBUTING.md) for numerical and benchmark review
requirements and [CHANGELOG.md](CHANGELOG.md) for user-visible changes.

## Citation

If you use Otter in a scientific publication, please cite:

> Chongbing Qu, Julian Lütgert, and Dominik Kraus, *Otter*, version 0.4.0, computer software (2026),
> [https://github.com/otter-hed/otter](https://github.com/otter-hed/otter).

```bibtex
@misc{QuKraus2026Otter,
  author  = {Qu, Chongbing and Lütgert, Julian and Kraus, Dominik},
  title   = {Otter},
  year    = {2026},
  note    = {Computer software, version 0.4.0},
  url     = {https://github.com/otter-hed/otter}
}
```

The same metadata are available in [CITATION.cff](CITATION.cff) and GitHub's
**Cite this repository** menu. `config.citation()` returns the references for
the selected physical models; see [CITATIONS.md](CITATIONS.md).

## Acknowledgements

The authors thank Zachary A. Johnson and C. E. Starrett for helpful
discussions.

Julian Lütgert contributed the parameter-grid calculation and HDF5 export
tool for interpolation workflows.

Chongbing Qu gratefully acknowledges financial support from HEDI and the China
Scholarship Council (CSC).

Otter is distributed under the [BSD 3-Clause License](LICENSE).
