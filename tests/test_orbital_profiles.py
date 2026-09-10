"""Exact post-processing contracts; no AA solve is needed for these tests."""

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from otter import bound_wavefunctions, ion_orbital_form_factors
from otter.electronic.full_external import _build_bound_tables_and_dos
from otter.electronic.ks_dft import _ion_density
from otter.numerics.transforms import precompute_dst_lattice_transform_like, radial_forward


@pytest.fixture
def orbital_state(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Use the retained final eigenpairs; do not solve again")
    monkeypatch.setattr("otter.electronic.full_external.solve_bound_states_sparse_numerov", forbidden)
    r_bound = np.linspace(1.0e-5, 6.0, 403)
    r = np.linspace(1.0e-4, 5.0, 251)
    energies = np.asarray([[-0.8, -0.002], [-0.3, np.inf]])
    radial = np.zeros((2, 2, r_bound.size))
    radial[0, 0] = 2 * np.exp(-r_bound)
    radial[0, 1] = (1 - r_bound) * np.exp(-0.4 * r_bound)
    radial[1, 0] = -r_bound * np.exp(-0.8 * r_bound)
    vectors = np.moveaxis(radial * np.sqrt(r_bound), -1, 1)
    cutoff = 1 / (1 + np.exp((r_bound - 2) / 0.1))
    tables = _build_bound_tables_and_dos(
        r=r_bound, v_full=-np.exp(-r_bound) / r_bound,
        l_list=np.array([0, 1]), n_states=2, mu=0.1, temperature_ha=0.3,
        energy_cut=0.0, gamma=0.05, n_jobs=1,
        eigvals=energies, eigvecs=vectors, ion_cutoff=cutoff, r_target=r, r_ws=2.0,
        zero_tail_bound_meta={"states": [{"l": 0, "state_index": 1, "matched_energy_ha": -0.002}]},
    )
    total = _ion_density(r_bound, energies, vectors, np.array([0, 1]),
                         0.1, 0.3, 0.0, 0.05, cutoff=cutoff, r_ws=2.0)
    tables.update(r=r, r_ws=2.0, n_ion=np.interp(r, r_bound, total), bound_energy_cut_ha=0.0)
    return tables, radial


def test_final_wavefunctions_are_raw_and_fd_is_opt_in(orbital_state):
    aa, expected = orbital_state
    np.testing.assert_allclose(bound_wavefunctions(aa), expected, rtol=1e-15, atol=1e-15)
    original = aa["bound_wavefunction_r"].copy()
    density = aa["ion_orbital_density_r"].copy()
    np.testing.assert_array_equal(bound_wavefunctions(aa, multiply_fd=True), original * aa["bound_fd"][..., None])
    np.testing.assert_array_equal(aa["bound_wavefunction_r"], original)
    np.testing.assert_array_equal(aa["ion_orbital_density_r"], density)
    # A shallow orbital remains visible outside the ion cutoff, without M(E).
    assert abs(original[0, 1, -1]) > 0.1
    assert aa["bound_m"][0, 1] < 0.1


def test_native_and_fourier_sums_close_with_different_grids(orbital_state):
    aa, _ = orbital_state
    np.testing.assert_allclose(aa["ion_orbital_density_r"].sum(axis=(0, 1)), aa["n_ion"], atol=1e-15)
    transform = precompute_dst_lattice_transform_like(np.arange(1, 513) * 0.02)
    expected = radial_forward(np.interp(transform.r, aa["r"], aa["n_ion"], right=0.0), transform)
    factors = ion_orbital_form_factors(aa, r=transform.r, k=transform.k)
    np.testing.assert_allclose(factors.sum(axis=(0, 1)), expected, rtol=1e-12, atol=1e-13)
    assert factors.shape == (2, 2, transform.k.size)
    assert np.all(factors[1, 1] == 0)


def test_zero_bound_levels(orbital_state):
    aa, _ = orbital_state
    for name in ("bound_energy_ha", "bound_fd", "bound_wavefunction_r", "ion_orbital_density_r"):
        aa[name] = aa[name][:, :0]
    transform = precompute_dst_lattice_transform_like(np.arange(1, 65) * 0.1)
    assert bound_wavefunctions(aa).shape == (2, 0, aa["r_bound"].size)
    factors = ion_orbital_form_factors(aa, r=transform.r, k=transform.k)
    assert factors.shape == (2, 0, transform.k.size)


def test_mismatched_transform_grids_are_rejected(orbital_state):
    aa, _ = orbital_state
    transform = precompute_dst_lattice_transform_like(np.arange(1, 65) * 0.1)
    with pytest.raises(ValueError, match="complete matching"):
        ion_orbital_form_factors(aa, r=transform.r, k=transform.k * 2)
    with pytest.raises(ValueError, match="aligned"):
        ion_orbital_form_factors(dict(aa, ion_orbital_density_r=np.zeros(3)), r=transform.r, k=transform.k)


def test_tf_is_not_assigned_fictitious_orbitals():
    tf = {"r": np.arange(1, 20), "n_ion": np.zeros(19)}
    with pytest.raises(ValueError, match="TF"):
        bound_wavefunctions(tf)
    with pytest.raises(ValueError, match="TF"):
        ion_orbital_form_factors(tf, r=tf["r"], k=tf["r"])


@pytest.mark.parametrize("multiply_fd", [False, True])
def test_example_plots_requested_definition(orbital_state, multiply_fd):
    aa, _ = orbital_state
    path = Path(__file__).resolve().parents[1] / "examples/bound_orbitals.py"
    spec = importlib.util.spec_from_file_location("orbital_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    transform = precompute_dst_lattice_transform_like(np.arange(1, 65) * 0.1)
    fig = module.plot_orbitals(aa, {"r": transform.r, "k": transform.k}, multiply_fd=multiply_fd)
    try:
        assert len(fig.axes) == 3
        expected = bound_wavefunctions(aa, multiply_fd=multiply_fd)
        np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(),
                                      (4 * np.pi * aa["r_bound"]**2 * expected[0, 0])[aa["r_bound"] <= 5.0])
        np.testing.assert_array_equal(fig.axes[1].lines[0].get_ydata(),
                                      (4 * np.pi * aa["r"]**2 * aa["ion_orbital_density_r"][0, 0])[aa["r"] <= 5.0])
        expected_f = ion_orbital_form_factors(aa, r=transform.r, k=transform.k)
        np.testing.assert_array_equal(fig.axes[2].lines[0].get_ydata(), expected_f[0, 0, transform.k <= 10.0])
        for ax in fig.axes[:2]:
            assert ax.get_xlim()[0] == -0.25
            ws_line = ax.lines[-1]
            np.testing.assert_array_equal(ws_line.get_xdata(), [aa["r_ws"], aa["r_ws"]])
            assert ws_line.get_linestyle() == "--"
            assert ws_line.get_color() == "0.6"
        assert r"a_B^{1/2}" in fig.axes[0].get_ylabel()
        assert fig.axes[2].get_ylabel() == r"$f_{nl}(k)=n^{\rm ion}_{nl}(k)$"
        assert all("Bohr" not in ax.get_xlabel() + ax.get_ylabel() for ax in fig.axes)
        for ax in fig.axes[1:]:
            labels = [text.get_text() for text in ax.get_legend().get_texts()]
            assert "Total" in labels and "Sum" not in labels
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_al_gallery_uses_portable_orbitals_without_transform(orbital_state, monkeypatch):
    aa, _ = orbital_state
    path = Path(__file__).resolve().parents[1] / "docs/examples/plot_al_full_workflow.py"
    spec = importlib.util.spec_from_file_location("al_orbital_gallery", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    transform = precompute_dst_lattice_transform_like(np.arange(1, 65) * 0.1)
    valid = np.isfinite(aa["bound_energy_ha"])
    li, ni = np.nonzero(valid)
    angular = aa["bound_l_list"][li]
    arrays = {
        "r_bound_bohr": aa["r_bound"], "r_bohr": aa["r"],
        "orbital_k_bohr_inv": transform.k,
        "bound_wavefunction_r": bound_wavefunctions(aa)[valid],
        "ion_orbital_density_r": aa["ion_orbital_density_r"][valid],
        "ion_orbital_density_k": ion_orbital_form_factors(aa, r=transform.r, k=transform.k)[valid],
        "bound_l": angular, "bound_principal_n": ni + angular + 1,
        "r_ws_bohr": aa["r_ws"],
    }
    state = {"species_0_" + key: value for key, value in arrays.items()}
    before = {key: np.asarray(value).copy() for key, value in state.items()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Plot the stored arrays without solving or transforming again")

    monkeypatch.setattr(module, "solve_plasma_workflow", forbidden)
    monkeypatch.setattr("otter.electronic.orbitals.radial_forward", forbidden)
    title = r"Al, $T_e=1$ eV"
    fig = module.plot_bound_orbitals(state, title=title)
    try:
        from otter.plotting import grid_figsize
        np.testing.assert_array_equal(fig.get_size_inches(), grid_figsize(1, 3))
        assert fig._suptitle.get_text() == title
        k_mask = transform.k <= 10.0
        np.testing.assert_array_equal(fig.axes[2].lines[0].get_ydata(),
                                      arrays["ion_orbital_density_k"][0, k_mask])
        for ax in fig.axes[1:]:
            labels = [text.get_text() for text in ax.get_legend().get_texts()]
            assert "Total" in labels and "Sum" not in labels
        for key in state:
            np.testing.assert_array_equal(state[key], before[key])
        fig.canvas.draw()
    finally:
        plt.close(fig)

    legacy = dict(
        rho_g_cc=8.1, te_ev=1.0, ti_ev=2.0, mu_ha=0.1,
        hnc_best_residual=1e-7, zbar_aa=3.0, zbar_partition=3.0, zbar_qoz=3.0,
        q_scr_raw=3.0, q_scr_grid_raw=3.0, q_scr_used=3.0, q_scr_scale_factor=1.0,
        bound_l=angular, bound_n_index=ni + 1,
        bound_energy_ha=aa["bound_energy_ha"][valid], bound_fd=aa["bound_fd"][valid],
        bound_occ_deg_fd=aa["bound_occ_deg_fd"][valid],
        r_e_bohr=aa["r"], r_ws_bohr=aa["r_ws"], n0_bohr3=0.1,
        n_full_bohr3=aa["n_ion"] + 0.1,
        n_bound_bohr3=np.full_like(aa["r"], 99.0),
        n_ext_bohr3=np.full_like(aa["r"], 0.1),
        n_pa_bohr3=aa["n_ion"], n_scr_bohr3=np.zeros_like(aa["r"]),
        r_bohr=transform.r, k_bohr_inv=transform.k,
        n_ion_k_electrons=arrays["ion_orbital_density_k"].sum(axis=0),
    )
    for key in ("v_full_ha", "v_ext_ha", "v_hartree_ha", "v_xc_ha"):
        legacy[key] = -np.exp(-aa["r"])
    for key in ("n_scr_k_electrons", "vii_k_ha_bohr3", "sii_k"):
        legacy[key] = np.exp(-transform.k)
    for key in ("vii_r_ha", "gii_r"):
        legacy[key] = np.exp(-transform.r)
    figures = []
    monkeypatch.setattr(module, "calculate_state", lambda: legacy)
    monkeypatch.setattr(module, "load_plasma_state", lambda path: state)
    monkeypatch.setattr(module, "RECOMPUTE_WITH_OTTER", True)
    monkeypatch.setattr(module, "save_figure", lambda fig, *a, **kw: figures.append(fig))
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(
        [plt.figure(number) for number in plt.get_fignums()]
    ))
    try:
        module.main()
        assert len(figures) == 4
        assert shown == [figures]
        assert figures[0].axes[0].get_title() == "Unweighted wavefunctions"
        assert figures[1].axes[0].get_title() == "Electronic densities"
        for figure in figures:
            np.testing.assert_array_equal(figure.get_size_inches(), grid_figsize(1, 3))
        assert figures[0]._suptitle.get_text() == figures[1]._suptitle.get_text()
        assert "T_i" not in figures[0]._suptitle.get_text()
        assert "T_i=2" in figures[3]._suptitle.get_text()
        assert [ax.get_title() for ax in figures[2].axes + figures[3].axes] == [
            r"$q(k)=n_{\rm scr}(k)$", r"$V_{ii}(k)$", r"$V_{ii}(r)$",
            r"$g_{ii}(r)$", r"$S_{ii}(k)$", "Rayleigh weight",
        ]
        weight = np.abs(legacy["n_ion_k_electrons"] + legacy["n_scr_k_electrons"])**2 * legacy["sii_k"]
        np.testing.assert_array_equal(figures[3].axes[-1].lines[0].get_ydata(), weight[transform.k <= 8.0])
        np.testing.assert_allclose(
            figures[1].axes[0].lines[1].get_ydata(), 4 * np.pi * aa["r"]**2 * aa["n_ion"],
            atol=1e-14,
        )
    finally:
        for figure in figures:
            plt.close(figure)
