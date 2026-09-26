"""Uniform data dimensions and unchanged numerical storage at the API boundary."""

import numpy as np
import pytest

from otter._workflow_results import electronic_species_vectors, ionic_species_axes


@pytest.mark.parametrize("ns", [1, 2, 3, 4])
def test_ionic_axes_and_shared_response_are_views(ns):
    k = np.linspace(0.1, 2.0, 13)
    r = np.linspace(0.1, 3.0, 17)
    ion = {"k": k, "r": r}
    profiles = {
        "k": ("q_k", "f_k", "n_scr_k", "n_ion_k", "v_ie_k", "v_ei_k", "c_ie_k", "sii_k"),
        "r": ("n_scr_r", "n_ion_r", "v_ie_r", "v_ei_r", "c_ie_r", "gii_r"),
    }
    pairs = {"k": ("sij_k", "vij_k"),
             "r": ("gij_r", "hij_r", "cij_r", "vij_r", "bridge_r", "hnc_effective_potential_r")}
    for grid, keys in profiles.items():
        for key in keys:
            value = np.arange(ns * ion[grid].size).reshape(ns, -1).astype(float)
            ion[key] = value[0] if ns == 1 else value
    for grid, keys in pairs.items():
        for key in keys:
            value = np.arange(ns * ns * ion[grid].size).reshape(ns, ns, -1).astype(float)
            ion[key] = value[0, 0] if ns == 1 else value
    for key in ("G_ee_k", "chi0_k", "chi_ee_k", "v_ee_k", "c_ee_k"):
        ion[key] = np.sin(k)
    for key in ("v_ee_r", "c_ee_r"):
        ion[key] = np.sin(r)
    scalars = ("zbar", "zbar_qoz", "zbar_partition", "zbar_aa_ws", "n_i",
               "zbar_electronic", "zbar_screening_integral_raw")
    for key in scalars:
        ion[key] = 0.25 if ns == 1 else np.arange(ns, dtype=float) + 0.25
    ion["charge_fix"] = {"q_scr_raw": ion["zbar"], "normalization_measure": "dst"}
    summary = {"zstar": np.arange(ns, dtype=float) + 1, "n_i_aa": np.ones(ns)}
    out = ionic_species_axes(ion, n_species=ns, electronic_vectors=summary)
    assert out is not ion
    assert out["zstar"] is summary["zstar"]
    assert out["n_i_aa"] is summary["n_i_aa"]
    for groups, prefix in ((profiles, (ns,)), (pairs, (ns, ns))):
        for grid, keys in groups.items():
            for key in keys:
                assert out[key].shape == (*prefix, ion[grid].size)
                np.testing.assert_array_equal(out[key].ravel(), ion[key].ravel())
                assert np.shares_memory(out[key], ion[key])
    for key in ("G_ee_k", "chi0_k", "chi_ee_k", "v_ee_k", "c_ee_k", "v_ee_r", "c_ee_r", "k", "r"):
        assert out[key] is ion[key]
    for key in scalars:
        assert out[key].shape == (ns,)
        np.testing.assert_array_equal(out[key], np.atleast_1d(ion[key]))
    assert out["charge_fix"]["q_scr_raw"].shape == (ns,)
    # Native dictionaries and arrays are not reshaped in place.
    assert np.ndim(ion["n_i"]) == (0 if ns == 1 else 1)
    assert ion["q_k"].ndim == (1 if ns == 1 else 2)
    again = ionic_species_axes(out, n_species=ns, electronic_vectors=summary)
    assert again["q_k"] is out["q_k"]


@pytest.mark.parametrize("ns,key,value", [
    (2, "q_k", np.ones(9)), (1, "f_k", np.ones((2, 9))),
    (3, "zbar", 2.0), (4, "vij_k", np.ones((4, 3, 9))),
])
def test_wrong_species_axes_are_rejected(ns, key, value):
    with pytest.raises(ValueError, match=key):
        ionic_species_axes({key: value}, n_species=ns, electronic_vectors={})


@pytest.mark.parametrize("ns", [1, 2, 3, 4])
def test_electronic_vectors_use_aa_density_and_explicit_zstar(ns):
    entries = []
    for i in range(ns):
        aa = {"n0": .1, "mu": .3, "zbar_partition": 1.0 + i, "zbar": 1.5 + i,
              "meta": {"n_i_bohr3": .02 * (i + 1)}}
        if i % 2 == 0:
            aa["zstar"] = 7.0 + i
        # Deliberately inconsistent fallback detects wrong density precedence.
        entries.append({"element": "C", "volume_bohr3": 999., "result": aa})
    summary = electronic_species_vectors(entries)
    for key in ("zbar_partition", "zbar_aa_ws", "zstar", "n0", "mu", "n_i_aa"):
        assert summary[key].shape == (ns,)
    expected = [7.0 + i if i % 2 == 0 else .1 / (.02 * (i + 1)) for i in range(ns)]
    np.testing.assert_array_equal(summary["zstar"], expected)
    np.testing.assert_array_equal(summary["n_i_aa"], [.02 * (i + 1) for i in range(ns)])
    out = ionic_species_axes({"n_i": np.full(ns, .7)}, n_species=ns,
                             electronic_vectors=summary)
    np.testing.assert_array_equal(out["n_i"], np.full(ns, .7))
    np.testing.assert_array_equal(out["zstar"], expected)
