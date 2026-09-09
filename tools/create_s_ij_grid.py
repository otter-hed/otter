"""
Calculates a set of ionic static structure factors and writes them to disc.
Allows parallelization over several computations.
"""

import h5py
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from otter import PlasmaWorkflowConfig, solve_plasma_workflow

###
elements = ["C", "H"]
number_fraction = [0.5, 0.5]

# Set the k-grid with these parameters
qoz_pad_factor = 2
qoz_linear_n_points = 2**12

k_points = qoz_linear_n_points * qoz_pad_factor
###


def otter_calc(T_e, rho):
    """
    Returns S_ii with shape (no_of_ions, no_of_ions, len(k)), k, and Z
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
    }
    cfg = PlasmaWorkflowConfig(
        elements=elements,
        number_fraction=number_fraction,
        temperature_ev=T_e,
        rho_g_cc=rho,
        ion_temperature_ev=T_e,
        save_state_npz=False,
        aa_overrides=aa_overrides,
        # electronic_model='tf',
        **kwargs,
    )
    result = solve_plasma_workflow(cfg)["ion"]
    return result["sij_k"], result["k"], result["zbar"]


def calculate_job(n_idx, m_idx, T_e, rho):
    """
    Worker function.
    """
    result = otter_calc(T_e, rho)
    return n_idx, m_idx, result


def run(T_e, rho, filename, n_workers=None):
    n = len(T_e)
    m = len(rho)

    with h5py.File(filename, "w") as f:
        Sii = f.create_dataset(
            "S_ii",
            shape=(len(elements), len(elements), k_points, n, m),
            dtype=np.float64,
            chunks=(len(elements), len(elements), k_points, 1, 1),
        )
        Sii.attrs["axis"] = ["i", "j", "k", "T", "rho"]
        Sii.attrs["unit"] = [""]

        Zbar = f.create_dataset(
            "Z_bar",
            shape=(len(elements), n, m),
            dtype=np.float64,
        )
        Zbar.attrs["axis"] = ["i", "T", "rho"]
        Zbar.attrs["unit"] = [""]

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

        element_out = axis.create_dataset(
            "elements",
            shape=(len(elements)),
            dtype=h5py.string_dtype(),
        )
        element_out[:] = elements

        f.flush()

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    calculate_job,
                    n_idx,
                    m_idx,
                    t,
                    r,
                ): (n_idx, m_idx)
                for n_idx, t in enumerate(T_e)
                for m_idx, r in enumerate(rho)
            }

            for future in as_completed(futures):
                n_idx, m_idx, result = future.result()

                Sii[:, :, :, n_idx, m_idx] = result[0]
                k_out[:] = result[1]
                Zbar[:, n_idx, m_idx] = result[2]
                f.flush()

                print(f"Finished [{n_idx}, {m_idx}] ({len(result)=})")


if __name__ == "__main__":
    T_e = np.linspace(5, 100, 4)
    rho = np.linspace(-0.5, 0.2, 4) + 1.51

    run(T_e, rho, "output2.hdf5")
