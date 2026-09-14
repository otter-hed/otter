"""
Calculates a set of ionic static structure factors and writes them to disc.
Allows parallelization over several computations.
"""

import h5py
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from otter import PlasmaWorkflowConfig, solve_plasma_workflow, ion_orbital_form_factors

###
elements = ["C"]
number_fraction = [0.5]

# Set the k-grid with these parameters
qoz_pad_factor = 2
qoz_linear_n_points = 2**12
# the k-grid is big for calculation, but the inputs most of the time only
# require the first few k
k_cutoff = 600

# Sample parameters
T_e = np.linspace(5, 100, 1)
rho = np.linspace(-0.5, 0.2, 1) + 1.51
alpha = [1]

# Output quantities. currently implemented:
# Sii, q, f, Zbar, Zstar
output = ["Sii", "q", "f", "f_nl", "Zbar", "Zstar"]

###

k_points = qoz_linear_n_points * qoz_pad_factor if k_cutoff is None else k_cutoff


def otter_calc(T_e, rho, alpha):
    """
    Returns the full set of observables from the otter calculation
    """
    aa_overrides = {
        "bound_zero_tail_refine": True,
        "bound_zero_tail_max_binding_ha": 1e-2,
        "bound_zero_tail_scan_points": 64,
        "bound_zero_tail_edge_rel_tol": 0.1,
    }
    kwargs = {
        "qoz_pad_factor": qoz_pad_factor,
        "qoz_linear_n_points": qoz_linear_n_points,
        "xc_model": "dirac",
        # "hnc_bridge_model": "rosenfeld_ashcroft",
        "aa_overrides": aa_overrides,
    }
    cfg = PlasmaWorkflowConfig(
        elements=elements,
        number_fraction=number_fraction,
        temperature_ev=T_e,
        rho_g_cc=rho,
        ion_temperature_ev=T_e * alpha,
        save_state_npz=False,
        state_include_groups=("orbital_densities",),
        # electronic_model='tf',
        **kwargs,
    )
    res = solve_plasma_workflow(cfg)
    ion_res = res["ion"]
    e_res = res["electronic"]["result"]
    if len(elements) == 1:
        zstar = e_res["zstar"]
        f_orb = [ion_orbital_form_factors(e_res, r=ion_res["r"], k=ion_res["k"])]
        q_k = ion_res["q_k"][np.newaxis, :]
        f_k = ion_res["f_k"][np.newaxis, :]
    else:
        zstar = [
            e_res["species"][idx]["result"]["zstar"] for idx in range(len(elements))
        ]
        f_orb = [
            ion_orbital_form_factors(
                e_res["species"][idx]["result"], r=ion_res["r"], k=ion_res["k"]
            )
            for idx in range(len(elements))
        ]
        q_k = ion_res["q_k"]
        f_k = ion_res["f_k"]
    f_nl = np.zeros((len(elements), 10, f_orb[0].shape[2]))
    idx = 0
    for element in range(len(elements)):
        for n in range(f_orb[element].shape[0]):
            for l in range(n + 1):
                f_nl[element, idx, :] = f_orb[element][l, n, :]
                idx += 1
                if idx > 10:
                    break
    return (
        ion_res["k"],
        ion_res["sij_k"],
        ion_res["zbar"],
        zstar,
        q_k,
        f_k,
        f_nl,
    )


def calculate_job(n_idx, m_idx, p_idx, T_e, rho, alpha):
    """
    Worker function.
    """
    result = otter_calc(T_e, rho, alpha)
    return n_idx, m_idx, p_idx, result


def run(T_e, rho, alpha, filename, n_workers=None):
    n = len(T_e)
    m = len(rho)
    p = len(alpha)

    with h5py.File(filename, "w") as f:
        if "Sii" in output:
            Sii = f.create_dataset(
                "S_ii",
                shape=(len(elements), len(elements), k_points, n, m, p),
                dtype=np.float64,
                chunks=(len(elements), len(elements), k_points, 1, 1, 1),
            )
            Sii.attrs["axis"] = ["i", "j", "k", "T_e", "rho", "alpha"]
            Sii.attrs["unit"] = [""]
        if "f" in output:
            f_k = f.create_dataset(
                "f",
                shape=(len(elements), k_points, n, m, p),
                dtype=np.float64,
                chunks=(len(elements), k_points, 1, 1, 1),
            )
            f_k.attrs["axis"] = ["i", "k", "T_e", "rho", "alpha"]
            f_k.attrs["unit"] = [""]
        if "f_nl" in output:
            f_nl = f.create_dataset(
                "f_nl",
                shape=(len(elements), 10, k_points, n, m, p),
                dtype=np.float64,
                chunks=(len(elements), 10, k_points, 1, 1, 1),
            )
            f_nl.attrs["axis"] = ["i", "orbital", "k", "T_e", "rho", "alpha"]
            f_nl.attrs["unit"] = [""]
        if "q" in output:
            q_k = f.create_dataset(
                "q",
                shape=(len(elements), k_points, n, m, p),
                dtype=np.float64,
                chunks=(len(elements), k_points, 1, 1, 1),
            )
            q_k.attrs["axis"] = ["i", "k", "T_e", "rho", "alpha"]
            q_k.attrs["unit"] = [""]
        if "Zbar" in output:
            Zbar = f.create_dataset(
                "Z_bar",
                shape=(len(elements), n, m, p),
                dtype=np.float64,
            )
            Zbar.attrs["axis"] = ["i", "T", "rho, alpha"]
            Zbar.attrs["unit"] = [""]
        if "Zstar" in output:
            Zstar = f.create_dataset(
                "Z_star",
                shape=(len(elements), n, m, p),
                dtype=np.float64,
            )
            Zstar.attrs["axis"] = ["i", "T", "rho, alpha"]
            Zstar.attrs["unit"] = [""]

        axis = f.create_group("axis")

        T_out = axis.create_dataset(
            "T_e",
            shape=(n),
            dtype=np.float64,
        )
        T_out[:] = T_e
        T_out.attrs["unit"] = ["eV"]

        k_out = axis.create_dataset(
            "k",
            shape=(k_points),
            dtype=np.float64,
        )
        k_out.attrs["unit"] = ["1/a0"]

        rho_out = axis.create_dataset(
            "rho",
            shape=(m),
            dtype=np.float64,
        )
        rho_out[:] = rho
        rho_out.attrs["unit"] = ["g/cc"]

        alpha_out = axis.create_dataset(
            "alpha",
            shape=(p),
            dtype=np.float64,
        )
        alpha_out[:] = alpha
        alpha_out.attrs["unit"] = [""]
        alpha_out.attrs["definition"] = ["T_i/T_e"]

        element_out = axis.create_dataset(
            "elements",
            shape=(len(elements)),
            dtype=h5py.string_dtype(),
        )
        element_out[:] = elements
        if "f_nl" in output:
            orbital_out = axis.create_dataset(
                "orbitals",
                shape=(10,),
                dtype=h5py.string_dtype(),
            )
            orbital_out[:] = [
                "n=0,l=0",
                "n=1,l=0",
                "n=1,l=1",
                "n=2,l=0",
                "n=2,l=1",
                "n=2,l=2",
                "n=3,l=0",
                "n=3,l=1",
                "n=3,l=2",
                "n=3,l=3",
            ]

        f.flush()

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(calculate_job, n_idx, m_idx, p_idx, t, r, a): (
                    n_idx,
                    m_idx,
                    p_idx,
                )
                for n_idx, t in enumerate(T_e)
                for m_idx, r in enumerate(rho)
                for p_idx, a in enumerate(alpha)
            }

            for future in as_completed(futures):
                n_idx, m_idx, p_idx, result = future.result()
                res_k, res_Sii, res_Zbar, res_Zstar, res_q, res_f, res_f_nl = result

                k_out[:] = res_k[:k_cutoff]
                if "Sii" in output:
                    Sii[:, :, :, n_idx, m_idx, p_idx] = res_Sii[:, :, :k_cutoff]
                if "Zbar" in output:
                    Zbar[:, n_idx, m_idx, p_idx] = res_Zbar
                if "Zstar" in output:
                    Zstar[:, n_idx, m_idx, p_idx] = res_Zstar
                if "q" in output:
                    q_k[:, :, n_idx, m_idx, p_idx] = res_q[:, :k_cutoff]
                if "f" in output:
                    f_k[:, :, n_idx, m_idx, p_idx] = res_f[:, :k_cutoff]
                if "f_nl" in output:
                    f_nl[:, :, :, n_idx, m_idx, p_idx] = res_f_nl[:, :, :k_cutoff]
                f.flush()

                print(f"Finished [{n_idx}, {m_idx}] ({len(result)=})")


if __name__ == "__main__":
    run(T_e, rho, alpha, "output.hdf5")
