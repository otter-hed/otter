"""Parameter-grid calculations with HDF5 output."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from otter import PlasmaWorkflowConfig, __version__, ion_orbital_form_factors, solve_plasma_workflow
from otter.workflows import resolve_plasma_composition

SCHEMA = "otter_grid_v2"
OUTPUTS = ("Sii", "q", "f", "Zbar", "Zstar", "f_nl", "E_nl")
PENDING, RUNNING, COMPLETE, FAILED = range(4)


def _axis(values, name):
    array = np.asarray(values, dtype=float)
    if (array.ndim != 1 or not array.size or not np.all(np.isfinite(array))
            or np.any(array <= 0) or np.any(np.diff(array) <= 0)):
        raise ValueError(f"{name} must be a nonempty, positive, strictly increasing axis.")
    return tuple(float(x) for x in array)


def _stored_unit(dataset):
    """Read scalar unit strings and legacy one-entry unit arrays."""
    value = np.asarray(dataset.attrs.get("unit", []))
    if value.ndim > 1 or value.size != 1:
        raise ValueError(f"Expected one unit for {dataset.name}.")
    unit = value.reshape(-1)[0]
    if isinstance(unit, bytes):
        unit = unit.decode("utf-8")
    if not isinstance(unit, str):
        raise ValueError(f"Expected a unit string for {dataset.name}.")
    return "hartree" if unit == "Ha" else unit


@dataclass(frozen=True)
class GridConfig:
    """A fixed composition sampled over Te, rho and alpha=Ti/Te.

    Parameters
    ----------
    workflow : dict
        Unscanned PlasmaWorkflowConfig options, including the composition.
        Unconverged-state continuation and parallelism within a state are
        not supported.
    temperature_ev : sequence of float
        Positive, increasing electron temperatures in eV.
    rho_g_cc : sequence of float
        Positive, increasing mass densities in g/cm³.
    k : sequence of float
        Positive, increasing output wave numbers in inverse Bohr. All selected
        spectra are linearly resampled without extrapolation.
    alpha : sequence of float, optional
        Positive, increasing Ti/Te ratios; the default is (1.0,).
    outputs : sequence of str, optional
        Selected names from OUTPUTS. Orbital fields require QM.
    save_native_spectra : bool, optional
        Also retain each state's original k and selected Sii, q and f arrays.
        Default False; common-grid exports are unchanged.

    Notes
    -----
    Each worker uses one CPU. Shapes retain species axes, including Ns=1.
    The native solver grids, models and tolerances are unchanged.
    """

    workflow: dict
    temperature_ev: tuple[float, ...]
    rho_g_cc: tuple[float, ...]
    k: tuple[float, ...]
    alpha: tuple[float, ...] = (1.0,)
    outputs: tuple[str, ...] = OUTPUTS
    save_native_spectra: bool = False

    def __post_init__(self):
        if not isinstance(self.save_native_spectra, bool):
            raise ValueError("save_native_spectra must be a boolean.")
        for name in ("temperature_ev", "rho_g_cc", "k", "alpha"):
            object.__setattr__(self, name, _axis(getattr(self, name), name))
        outputs = tuple(self.outputs)
        if not outputs or len(set(outputs)) != len(outputs) or set(outputs) - set(OUTPUTS):
            raise ValueError(f"outputs must be a nonempty subset of {OUTPUTS} without duplicates.")
        object.__setattr__(self, "outputs", outputs)
        workflow = json.loads(json.dumps(self.workflow, allow_nan=False))
        forbidden = {"temperature_ev", "rho_g_cc", "ion_temperature_ev"} & workflow.keys()
        if forbidden:
            raise ValueError(f"Set scanned parameters on GridConfig: {sorted(forbidden)}")
        for key in ("allow_unconverged_aa", "allow_unconverged_root", "save_state_npz", "save_data"):
            if workflow.get(key, False):
                raise ValueError(f"Grid export requires {key}=False.")
        if workflow.get("run_mode", "full+ext") != "full+ext":
            raise ValueError("Grid export requires run_mode='full+ext'.")
        if workflow.get("species_parallel_jobs") not in (None, 1):
            raise ValueError("Parallelize grid points, not species within a worker.")
        for overrides in [workflow.get("aa_overrides", {}), *workflow.get("species_overrides", {}).values()]:
            if overrides.get("n_jobs", 1) != 1:
                raise ValueError("AA n_jobs must be 1 inside a grid worker.")
        workflow["species_parallel_jobs"] = 1
        object.__setattr__(self, "workflow", workflow)
        cfg = self.configuration((0, 0, 0))
        if cfg.electronic_model == "tf" and {"f_nl", "E_nl"} & set(outputs):
            raise ValueError("TF has no orbitals; omit f_nl and E_nl from outputs.")
        self.composition()

    @property
    def shape(self):
        return (len(self.temperature_ev), len(self.rho_g_cc), len(self.alpha))

    def configuration(self, index):
        t, r, a = index
        return PlasmaWorkflowConfig(
            **self.workflow, temperature_ev=self.temperature_ev[t],
            rho_g_cc=self.rho_g_cc[r], ion_temperature_ev=self.temperature_ev[t] * self.alpha[a])

    def composition(self):
        return resolve_plasma_composition(**{
            key: self.workflow.get(key) for key in ("formula", "elements", "counts", "number_fraction")})


def _resample(k_native, values, k):
    native, values = np.asarray(k_native, dtype=float), np.asarray(values, dtype=float)
    target = np.asarray(_axis(k, "Output k"))
    if (native.ndim != 1 or native.size < 2 or not np.all(np.isfinite(native))
            or np.any(native <= 0) or np.any(np.diff(native) <= 0)
            or values.ndim == 0 or values.shape[-1] != native.size
            or not np.all(np.isfinite(values))):
        raise ValueError("Invalid native spectrum or k grid.")
    if target[0] < native[0] or target[-1] > native[-1]:
        raise ValueError(f"Output k range [{target[0]:g}, {target[-1]:g}] lies outside "
                         f"native range [{native[0]:g}, {native[-1]:g}]; extrapolation is disabled.")
    rows = values.reshape((-1, native.size))
    return np.asarray([np.interp(target, native, row) for row in rows]).reshape(
        (*values.shape[:-1], target.size))


def _pack_result(result, config):
    """Extract selected observables from a converged unified workflow result."""
    ion = result["ion"]
    if ion is None or not bool(ion.get("hnc_converged", False)):
        raise ValueError("HNC did not converge; the point cannot be exported as complete.")
    symbols, _ = config.composition()
    if list(result["species_symbols"]) != symbols:
        raise ValueError("Workflow species order differs from the requested composition.")
    entries = result["electronic"]["species"]
    if [entry["element"] for entry in entries] != symbols:
        raise ValueError("Electronic species order differs from the requested composition.")
    ns, data = len(symbols), {}
    native_k = np.asarray(ion["k"])
    _resample(native_k, np.zeros_like(native_k), config.k)
    native_spectra = {"k": native_k}
    for requested, key in (("Sii", "sij_k"), ("q", "q_k"), ("f", "f_k")):
        if requested in config.outputs:
            array = np.asarray(ion[key], dtype=float)
            expected = (ns, ns, native_k.size) if requested == "Sii" else (ns, native_k.size)
            if array.shape != expected:
                raise ValueError(f"{key}: expected {expected}, got {array.shape}.")
            data[requested] = _resample(native_k, array, config.k)
            if config.save_native_spectra:
                native_spectra[{"Sii": "S_ii"}.get(requested, requested)] = array
    for requested, key in (("Zbar", "zbar"), ("Zstar", "zstar")):
        if requested in config.outputs:
            data[requested] = np.asarray(ion[key], dtype=float)
            if data[requested].shape != (ns,) or not np.all(np.isfinite(data[requested])):
                raise ValueError(f"Invalid {key} vector.")
    orbitals = []
    for entry in entries:
        aa = entry["result"]
        if not {"f_nl", "E_nl"} & set(config.outputs):
            continue
        energy = np.asarray(aa["bound_energy_ha"], dtype=float)
        l_values = np.asarray(aa["bound_l_list"], dtype=int)
        if energy.ndim != 2 or l_values.shape != (energy.shape[0],):
            raise ValueError("Bound energies and angular momenta do not align.")
        li, ni = np.nonzero(np.isfinite(energy))
        angular = l_values[li]
        block = {"n": ni + angular + 1, "l": angular, "E_nl": energy[li, ni]}
        if "f_nl" in config.outputs:
            factors = np.asarray(ion_orbital_form_factors(aa, r=ion["r"], k=native_k))
            if factors.shape != (*energy.shape, native_k.size):
                raise ValueError("Orbital form factors do not align with bound energies and k.")
            if not np.allclose(factors[li, ni].sum(axis=0), ion["f_k"][len(orbitals)],
                               rtol=1e-10, atol=1e-10):
                raise ValueError("Orbital form factors do not sum to the total ion form factor.")
            block["f_nl"] = _resample(native_k, factors[li, ni], config.k)
        orbitals.append(block)
    residual = ion.get("hnc_output_residual", ion.get("hnc_best_residual"))
    diagnostics = {
        "hnc_converged": True,
        "hnc_output_residual": (float(residual) if residual is not None and np.isfinite(residual) else None),
        "native_k_min": float(native_k[0]), "native_k_max": float(native_k[-1]),
        "native_k_points": int(native_k.size),
        "species": [{"element": e["element"],
                     "full_converged": bool(e["result"].get("stage2_converged", False)),
                     "external_converged": bool(e["result"].get("ext_status", {}).get("converged", False)),
                     "threshold": str(e["result"].get("threshold_state_status", "unknown"))}
                    for e in entries]}
    if config.save_native_spectra:
        data["native"] = native_spectra
        diagnostics["native_first"] = {
            name: values[..., 0].tolist() for name, values in native_spectra.items() if name != "k"}
        if {"f", "q"} <= native_spectra.keys():
            diagnostics["native_first"]["N"] = (native_spectra["f"][:, 0] + native_spectra["q"][:, 0]).tolist()
    return data, orbitals, diagnostics


def _calculate_point(config, index, log_path):
    started = time.perf_counter()
    with open(log_path, "a", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        print(f"\nState {index}; Otter {__version__}", flush=True)
        try:
            result = solve_plasma_workflow(config.configuration(index))
            data, orbitals, diagnostics = _pack_result(result, config)
        except Exception:
            traceback.print_exc()
            raise
    return data, orbitals, diagnostics, time.perf_counter() - started


def _provenance(config):
    import scipy
    import numba
    import h5py

    root, digest = Path(__file__).parent, hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    effective = asdict(config.configuration((0, 0, 0)))
    for key in ("temperature_ev", "rho_g_cc", "ion_temperature_ev"):
        effective.pop(key)
    return dict(schema=SCHEMA, otter_version=__version__, source_sha256=digest.hexdigest(),
                config=asdict(config), effective_workflow=effective,
                dependencies={"python": sys.version.split()[0], "numpy": np.__version__,
                              "scipy": scipy.__version__, "numba": numba.__version__, "h5py": h5py.__version__})


class _GridFile:
    """One parent-process writer; completion is recorded after payload flush."""

    def __init__(self, path, config, *, resume):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("Install HDF5 support with poetry install -E grid or pip install 'otter-hed[grid]'.") from exc
        self.config = config
        manifest = json.dumps(_provenance(config), sort_keys=True, allow_nan=False)
        self.file = h5py.File(path, "r+" if resume else "x")
        try:
            if resume:
                if self.file.attrs.get("schema") != SCHEMA:
                    raise ValueError(f"Cannot resume: expected schema {SCHEMA}. "
                                     "Retain the existing file and use a new output filename.")
                if self.file.attrs.get("manifest_json") != manifest:
                    raise ValueError("Cannot resume: configuration, source or dependency versions differ.")
                self._validate_axes()
            else:
                self._create(manifest)
        except BaseException:
            self.file.close()
            raise

    def _validate_axes(self):
        """Check stored coordinates, not just the manifest, before resuming."""
        cfg = self.config
        symbols, counts = cfg.composition()
        for name, values, unit in (("T_e", cfg.temperature_ev, "eV"),
                                   ("rho", cfg.rho_g_cc, "g/cm^3"),
                                   ("alpha", cfg.alpha, "1"), ("k", cfg.k, "1/a0")):
            ds = self.file.get(f"axis/{name}")
            if (ds is None or not np.array_equal(ds[:], values)
                    or _stored_unit(ds) != unit):
                raise ValueError(f"Cannot resume: stored axis/{name} or its unit differs from the configuration.")
        for name, values in (("elements", symbols), ("counts", counts)):
            ds = self.file.get(f"axis/{name}")
            stored = None if ds is None else ds.asstr()[:] if name == "elements" else ds[:]
            if not np.array_equal(stored, values):
                raise ValueError(f"Cannot resume: stored axis/{name} differs from the composition.")
        fractions = self.file["axis/elements"].attrs.get("number_fraction")
        if not np.array_equal(fractions, np.asarray(counts) / np.sum(counts)):
            raise ValueError("Cannot resume: number_fraction differs from the composition.")
        for key in cfg.outputs:
            name = {"Sii": "S_ii", "Zbar": "Z_bar", "Zstar": "Z_star"}.get(key, key)
            ds = self.file.get(name)
            if ds is None or _stored_unit(ds) != ("hartree" if key == "E_nl" else "1"):
                raise ValueError(f"Cannot resume: missing {name} or incompatible unit.")
        for name in ("S_ii", "q", "f", "f_nl"):
            if name in self.file and self.file[name].attrs.get("k_path") != "/axis/k":
                raise ValueError(f"Cannot resume: {name} does not reference the common /axis/k grid.")
        for name in ("status", "error", "elapsed_s", "attempts", "convergence"):
            ds = self.file.get(f"diagnostics/{name}")
            if (ds is None or ds.shape != cfg.shape
                    or list(ds.attrs.get("axis", [])) != ["T_e", "rho", "alpha"]):
                raise ValueError(f"Cannot resume: invalid diagnostics/{name} dataset.")
        status = self.file["diagnostics/status"]
        if status.dtype.kind not in "iu" or not np.isin(status[:], [PENDING, RUNNING, COMPLETE, FAILED]).all():
            raise ValueError("Cannot resume: invalid diagnostics/status values.")
        if {"f_nl", "E_nl"} & set(cfg.outputs):
            n = self.file.get("axis/orbitals/n")
            l = self.file.get("axis/orbitals/l")
            if (n is None or l is None or n.ndim != 1 or l.shape != n.shape
                    or n.dtype.kind not in "iu" or l.dtype.kind not in "iu"
                    or np.any(n[:] < 1) or np.any(l[:] < 0) or np.any(l[:] >= n[:])
                    or not np.array_equal(n[:] * (n[:] - 1) // 2 + l[:], np.arange(n.size))):
                raise ValueError("Cannot resume: invalid orbital quantum numbers.")
            for name, prefix, labels in (
                ("f_nl", (len(symbols), n.size, len(cfg.k)), ["i", "orbital", "k"]),
                ("E_nl", (len(symbols), n.size), ["i", "orbital"]),
                ("orbitals_present", (len(symbols), n.size), ["i", "orbital"]),
            ):
                if name != "orbitals_present" and name not in cfg.outputs:
                    continue
                ds = self.file.get(name)
                if (ds is None or ds.shape != prefix + cfg.shape
                        or list(ds.attrs.get("axis", [])) != labels + ["T_e", "rho", "alpha"]
                        or ds.attrs.get("orbital_path") != "/axis/orbitals"):
                    raise ValueError(f"Cannot resume: invalid {name} orbital dimensions.")

    def _create(self, manifest):
        import h5py

        f, cfg = self.file, self.config
        symbols, counts = cfg.composition()
        ns, nk, shape, text = len(symbols), len(cfg.k), cfg.shape, h5py.string_dtype("utf-8")
        f.attrs.update(schema=SCHEMA, manifest_json=manifest, complete=False)
        if cfg.save_native_spectra:
            native = f.create_group("native")
            native.attrs["layout"] = "One group per state, named Te-index_rho-index_alpha-index."
        axes = f.create_group("axis")
        diagnostics = f.create_group("diagnostics")
        for name, values, unit in (("T_e", cfg.temperature_ev, "eV"), ("rho", cfg.rho_g_cc, "g/cm^3"),
                                   ("alpha", cfg.alpha, "1"), ("k", cfg.k, "1/a0")):
            axes.create_dataset(name, data=values).attrs["unit"] = unit
        axes["alpha"].attrs["definition"] = "Ti/Te"
        axes.create_dataset("elements", data=symbols, dtype=text)
        axes["elements"].attrs["number_fraction"] = np.asarray(counts) / np.sum(counts)
        axes.create_dataset("counts", data=counts)
        fields = {"Sii": ("S_ii", (ns, ns, nk), ["i", "j", "k"]),
                  "q": ("q", (ns, nk), ["i", "k"]), "f": ("f", (ns, nk), ["i", "k"]),
                  "Zbar": ("Z_bar", (ns,), ["i"]), "Zstar": ("Z_star", (ns,), ["i"]),
                  "f_nl": ("f_nl", (ns, 0, nk), ["i", "orbital", "k"]),
                  "E_nl": ("E_nl", (ns, 0), ["i", "orbital"])}
        for output in cfg.outputs:
            name, prefix, labels = fields[output]
            ds = f.create_dataset(name, shape=prefix + shape,
                                  maxshape=tuple(None if x == 0 else x for x in prefix) + shape,
                                  dtype="f8", chunks=tuple(max(1, x) for x in prefix) + (1, 1, 1),
                                  compression="gzip", shuffle=True, fillvalue=np.nan)
            ds.attrs["axis"] = labels + ["T_e", "rho", "alpha"]
            ds.attrs["unit"] = "hartree" if output == "E_nl" else "1"
            if "k" in labels:
                ds.attrs["k_path"] = "/axis/k"
            if "orbital" in labels:
                ds.attrs["orbital_path"] = "/axis/orbitals"
        if "Z_bar" in f:
            f["Z_bar"].attrs["definition"] = "ion/zbar: mean ionization used by QOZ"
        if "Z_star" in f:
            f["Z_star"].attrs["definition"] = "n0 / n_i_aa"
        if {"f_nl", "E_nl"} & set(cfg.outputs):
            for name in ("orbitals/n", "orbitals/l"):
                axes.create_dataset(name, shape=(0,), maxshape=(None,), dtype="i8")
            ds = f.create_dataset("orbitals_present", shape=(ns, 0, *shape),
                                  maxshape=(ns, None, *shape), chunks=(ns, 1, 1, 1, 1), dtype="bool")
            ds.attrs["axis"] = ["i", "orbital", "T_e", "rho", "alpha"]
            ds.attrs["orbital_path"] = "/axis/orbitals"
        for name, dtype, fill in (("status", "u1", PENDING), ("attempts", "i4", 0),
                                  ("elapsed_s", "f8", np.nan), ("error", text, ""),
                                  ("convergence", text, "")):
            ds = diagnostics.create_dataset(name, shape=shape, dtype=dtype, fillvalue=fill)
            ds.attrs["axis"] = ["T_e", "rho", "alpha"]
        diagnostics["status"].attrs["values"] = "0=pending, 1=running, 2=complete, 3=failed"
        f.flush()

    def _extend_orbitals(self, count):
        f, previous = self.file, self.file["axis/orbitals/n"].size
        if count <= previous:
            return
        for name in ("f_nl", "E_nl", "orbitals_present"):
            if name in f:
                f[name].resize(count, axis=1)
        principal, angular = [], []
        for n in itertools.count(1):
            for l in range(n):
                principal.append(n)
                angular.append(l)
            if len(principal) >= count:
                break
        for name, values in (("orbitals/n", principal), ("orbitals/l", angular)):
            f[f"axis/{name}"].resize((count,))
            f[f"axis/{name}"][:] = values[:count]
        # Additional slots are absent in previously accepted states, not pending.
        if "f_nl" in f:
            for index in np.argwhere(f["diagnostics/status"][:] == COMPLETE):
                f["f_nl"][(slice(None), slice(previous, count), slice(None), *index)] = 0.0

    def start(self, index):
        f = self.file
        f.attrs["complete"] = False
        f["diagnostics/status"][index] = RUNNING
        f["diagnostics/attempts"][index] += 1
        f["diagnostics/error"][index] = ""
        f.flush()

    def success(self, index, payload):
        data, orbitals, diagnostics, elapsed = payload
        f = self.file
        for key, values in data.items():
            if key == "native":
                continue
            name = {"Sii": "S_ii", "Zbar": "Z_bar", "Zstar": "Z_star"}.get(key, key)
            f[name][(..., *index)] = values
        if self.config.save_native_spectra:
            group_name = "_".join(map(str, index))
            root = f["native"]
            if group_name in root:
                del root[group_name]
            state = root.create_group(group_name)
            state.attrs["state_index"] = index
            for name, values in data["native"].items():
                ds = state.create_dataset(name, data=values, compression="gzip", shuffle=True)
                ds.attrs["unit"] = "1/a0" if name == "k" else "1"
                ds.attrs["axis"] = (["k"] if name == "k" else
                                   ["i", "j", "k"] if name == "S_ii" else ["i", "k"])
                if name != "k":
                    ds.attrs["k_path"] = state.name + "/k"
        if orbitals:
            slots = [np.asarray(o["n"] * (o["n"] - 1) // 2 + o["l"], dtype=int) for o in orbitals]
            self._extend_orbitals(max((int(s.max()) + 1 for s in slots if s.size), default=0))
            for name, fill in (("f_nl", 0.), ("E_nl", np.nan), ("orbitals_present", False)):
                if name in f:
                    f[name][(..., *index)] = fill
            for species, (block, indices) in enumerate(zip(orbitals, slots, strict=True)):
                for row, slot in enumerate(indices):
                    f["orbitals_present"][(species, slot, *index)] = True
                    if "E_nl" in f:
                        f["E_nl"][(species, slot, *index)] = block["E_nl"][row]
                    if "f_nl" in f:
                        f["f_nl"][(species, slot, slice(None), *index)] = block["f_nl"][row]
        f["diagnostics/elapsed_s"][index] = elapsed
        f["diagnostics/convergence"][index] = json.dumps(diagnostics, allow_nan=False)
        f.flush()
        f["diagnostics/status"][index] = COMPLETE
        f.flush()

    def failure(self, index, error, elapsed):
        f = self.file
        f.attrs["complete"] = False
        native_path = "native/" + "_".join(map(str, index))
        if native_path in f:
            del f[native_path]
        for name in ("S_ii", "q", "f", "Z_bar", "Z_star", "f_nl", "E_nl"):
            if name in f:
                f[name][(..., *index)] = np.nan
        if "orbitals_present" in f:
            f["orbitals_present"][(..., *index)] = False
        f["diagnostics/convergence"][index] = ""
        f["diagnostics/error"][index], f["diagnostics/elapsed_s"][index], f["diagnostics/status"][index] = error, elapsed, FAILED
        f.flush()


@contextmanager
def _single_thread_workers():
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update(dict.fromkeys(names, "1"))
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_grid(config: GridConfig, filename, *, workers: int = 1, resume: bool = False):
    """Write a grid, retrying only incomplete points when resuming.

    Parameters
    ----------
    config : GridConfig
        Composition, state axes, output k grid and requested observables.
    filename : str or pathlib.Path
        HDF5 destination. Existing files require resume=True.
    workers : int, optional
        Number of simultaneous single-CPU states; default 1.
    resume : bool, optional
        Retry unfinished states with identical configuration, source and
        numerical dependency versions. Completed states are retained.

    Returns
    -------
    dict
        Counts named complete, failed and total. Only status=2 data are valid
        interpolation input; a partially completed file is not a dense grid.

    Raises
    ------
    ImportError
        If the optional h5py dependency is unavailable.
    FileExistsError
        If the destination exists and resume is False.
    ValueError
        If worker controls or resume metadata are incompatible.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer.")
    path = Path(filename).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = _GridFile(path, config, resume=resume)
    try:
        logs = path.with_suffix(path.suffix + ".logs")
        logs.mkdir(exist_ok=True)
        pending = iter(index for index in np.ndindex(config.shape) if writer.file["diagnostics/status"][index] != COMPLETE)
        with _single_thread_workers(), ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            active = {}
            while True:
                while len(active) < workers:
                    index = next(pending, None)
                    if index is None:
                        break
                    log = logs / ("_".join(map(str, index)) + ".log")
                    writer.start(index)
                    print(f"[start] {index} -> {log}", flush=True)
                    active[pool.submit(_calculate_point, config, index, str(log))] = (index, time.perf_counter())
                if not active:
                    break
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    index, started = active.pop(future)
                    try:
                        payload = future.result()
                        writer.success(index, payload)
                    except Exception as exc:
                        writer.failure(index, f"{type(exc).__name__}: {exc}", time.perf_counter() - started)
                        print(f"[failed] {index}: {exc}", flush=True)
                    else:
                        print(f"[complete] {index}: {payload[-1]:.1f} s", flush=True)
        status = writer.file["diagnostics/status"][:]
        counts = {"complete": int(np.sum(status == COMPLETE)), "failed": int(np.sum(status == FAILED)),
                  "total": int(status.size)}
        writer.file.attrs["complete"] = counts["complete"] == counts["total"]
        writer.file.flush()
        return counts
    finally:
        writer.file.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="JSON with workflow, temperature_ev, rho_g_cc, alpha and k.")
    parser.add_argument("output", type=Path, help="HDF5 output file; existing files require --resume.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent single-CPU states (default: 1).")
    parser.add_argument("--resume", action="store_true", help="Retry unfinished states in a matching file.")
    args = parser.parse_args(argv)
    specification = json.loads(args.config.read_text())
    if isinstance(specification.get("k"), dict):
        k = specification["k"]
        if (set(k) != {"min", "max", "points"} or isinstance(k["points"], bool)
                or not isinstance(k["points"], int) or k["points"] < 2):
            parser.error("k requires min, max and an integer points >= 2.")
        specification["k"] = np.linspace(k["min"], k["max"], k["points"]).tolist()
    summary = run_grid(GridConfig(**specification), args.output, workers=args.workers, resume=args.resume)
    print(f"Completed {summary['complete']}/{summary['total']}; failed={summary['failed']}", flush=True)
    return 0 if summary["complete"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
