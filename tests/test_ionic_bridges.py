"""Focused tests for the optional Rosenfeld--Ashcroft/VMHNC closure."""

from __future__ import annotations

import numpy as np
import pytest

import otter.ionic.bridges as bridges
from otter.numerics.transforms import precompute_dst_lattice_transform_like


def test_vmhnc_free_energy_correction_matches_its_analytic_derivative() -> None:
    """Keep Faussurier (2004), Eqs. (6)--(7), tied to the coded residual."""
    eta = 0.31
    step = 1.0e-6
    finite_difference = (
        bridges._hard_sphere_free_energy_correction(eta + step)
        - bridges._hard_sphere_free_energy_correction(eta - step)
    ) / (2.0 * step)

    assert finite_difference == pytest.approx(
        bridges._hard_sphere_free_energy_correction_derivative(eta),
        rel=2.0e-9,
    )


def test_hard_sphere_py_reference_has_exact_core_and_contact_value() -> None:
    reference = bridges.HardSpherePYReference(points_per_diameter=64)
    eta = 0.4
    bridge, g_r = reference.bridge_and_g(eta)
    contact = reference.contact_index

    assert np.all(g_r[:contact] == 0.0)
    assert g_r[contact] == pytest.approx(
        (1.0 + 0.5 * eta) / (1.0 - eta) ** 2,
        rel=1.0e-12,
    )
    assert np.all(np.isfinite(bridge))
    assert float(np.max(bridge)) <= 1.0e-12
    assert bridge[-1] == pytest.approx(0.0, abs=1.0e-12)


def test_vmhnc_outer_search_brackets_variational_eta(monkeypatch) -> None:
    r_seed = np.linspace(0.02, 8.0, 128)
    transform = precompute_dst_lattice_transform_like(r_seed)
    r = np.asarray(transform.r, dtype=float)
    k = np.asarray(transform.k, dtype=float)
    target_eta = 0.2873

    def fake_solve_eta(eta, _reference, **kwargs):
        zeros = np.zeros_like(r)
        result = bridges.VMHNCResult(
            g_r=np.ones_like(r),
            s_k=np.ones_like(k),
            h_r=zeros,
            c_r=zeros,
            bridge_r=zeros,
            effective_potential_r=np.asarray(kwargs["potential_r"], dtype=float),
            eta=float(eta),
            sigma_bohr=1.0,
            variational_residual=float(eta - target_eta),
            eta_history=(),
            hnc_residual_history=(1.0e-8,),
            hnc_stage_meta=(),
        )
        return bridges._EtaSolution(
            residual=float(eta - target_eta),
            result=result,
            nodal_r=np.zeros((1, 1, r.size), dtype=float),
        )

    monkeypatch.setattr(bridges, "_solve_eta", fake_solve_eta)
    result = bridges.solve_vmhnc(
        r,
        k,
        np.exp(-r),
        transform,
        ion_density_bohr3=0.02,
        ion_temperature_ha=0.1,
        points_per_diameter=32,
        eta_bounds=(0.1, 0.45),
        eta_scan_points=5,
        eta_tol=1.0e-5,
    )

    assert result.eta == pytest.approx(target_eta, abs=1.0e-5)
    assert abs(result.variational_residual) <= 1.0e-5
    assert len(result.eta_history) >= 3


def test_vmhnc_validates_physical_inputs() -> None:
    r_seed = np.linspace(0.02, 4.0, 64)
    transform = precompute_dst_lattice_transform_like(r_seed)
    r = np.asarray(transform.r, dtype=float)
    k = np.asarray(transform.k, dtype=float)
    with pytest.raises(ValueError, match="eta_bounds"):
        bridges.solve_vmhnc(
            r,
            k,
            np.zeros_like(r),
            transform,
            ion_density_bohr3=0.02,
            ion_temperature_ha=0.1,
            eta_bounds=(0.3, 0.2),
        )
