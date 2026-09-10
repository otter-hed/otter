Reproducing examples and benchmarks
===================================

Use a complete Otter source checkout at the version shown in the documentation,
with its dependencies installed as described in :doc:`../installing`.
Installing only the PyPI package does not provide the repository's reference
tables and gallery scripts. No precomputed Otter NPZ is required.

Run the included Python script
-------------------------------

From the repository root, run the command printed in the page's **Reproduction**
section. With the recommended Poetry installation::

   poetry run python docs/examples/plot_al_full_workflow.py
   poetry run python benchmarks/examples/plot_starrett_saumon_2013_electronic.py

If Otter was installed in an independently managed, active Python environment,
the same scripts can instead be run with that environment's ``python``.
The scripts are already in the checkout; downloading them separately is optional.
Change the input block in the corresponding repository script when needed.

The script computes its states and writes local results beneath
``benchmarks/outputs/<benchmark-name>/``. Figures are saved as PNG/PDF where
applicable; the electronic-level benchmark prints numerical tables. Failures
raise an error instead of silently substituting previously accepted results.

On a desktop, plots open through Matplotlib's configured interactive backend
(for example, TkAgg or QtAgg). On a headless server, run with ``MPLBACKEND=Agg``
and open the saved PNG/PDF files; that backend does not create windows.

The three download buttons
---------------------------

* **Python (.py)**: the page's executable source, already present in the checkout.
* **Notebook (.ipynb)**: a launcher for that same repository script. It uses the
  notebook kernel's Python interpreter and streams progress and errors.
* **ZIP (.zip)**: the Python source and notebook together, not a standalone
  package containing Otter, reference data or helper scripts.

For a notebook, select the environment containing Otter and set the kernel's
working directory to the repository root. For example, run this before the
launcher cell, replacing the path::

   import os
   os.chdir("/path/to/otter")

Jupyter and its Python kernel are optional and must be installed separately
in that environment; they are not required for the terminal commands.

Edit parameters in the repository's ``.py`` input block, then run the notebook
cell. The file-backed entry point avoids notebook-specific ``__file__`` and
multiprocessing-import problems. Calculated figures are saved to the same
locations as with terminal execution; the notebook is not a second solver.

MD and other optional dependencies
-----------------------------------

Pages containing MD require LAMMPS/MPI and may take hours. Their normal
reproduction path first calculates new Otter electronic states and pair
potentials, then runs MD and analyzes its trajectory. Install Libxc for the
LDA/PBE comparisons. Adjust the script's explicit MD CPU allocation to the
available machine resources before starting it.

Recorded website results
--------------------------

HTML contains reviewed figures, tables and recorded terminal output. Running
a script produces local numerical results; it does not automatically replace
the website's accepted results. Sphinx builds do not run AA or MD.
Historical MD overlays are labelled as such; fresh calculations with the
current electronic potential need not reproduce those old MD curves exactly.
