from __future__ import annotations

from pathlib import Path
import tomllib

import numpy as np

from otter import PlasmaWorkflowConfig, __version__
from otter.electronic.solvers.free import _prepare_numerov_geometry
from otter.electronic.xc import dirac_exchange_potential
from otter.ionic import QOZPotentialOptions
from otter.numerics import (
    ANGSTROM_TO_BOHR,
    ATOMIC_MASS_UNIT_TO_G,
    AVOGADRO_CONSTANT_MOL,
    BOHR_TO_ANGSTROM,
    EV_TO_KELVIN,
    HA_TO_EV,
    KELVIN_TO_EV,
    create_linear_grid,
)


def test_public_imports() -> None:
    cfg = PlasmaWorkflowConfig(elements=["C"], temperature_ev=10.0, rho_g_cc=1.0)
    assert cfg.temperature_ev == 10.0
    assert isinstance(QOZPotentialOptions(), QOZPotentialOptions)
    assert isinstance(__version__, str)
    assert __version__
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )
    assert __version__ == pyproject["project"]["version"]


def test_core_helpers_are_callable() -> None:
    grid = create_linear_grid(2.0, 16)
    v = dirac_exchange_potential(np.full_like(grid.r, 0.1))
    geom = _prepare_numerov_geometry(grid.r, v)
    assert geom["r"].shape == grid.r.shape
    assert np.all(np.isfinite(v))


def test_shared_unit_conversions_are_reciprocal() -> None:
    assert np.isclose(BOHR_TO_ANGSTROM, 0.529177210903, rtol=0.0, atol=1.0e-15)
    assert HA_TO_EV > 27.0
    assert np.isclose(
        BOHR_TO_ANGSTROM * ANGSTROM_TO_BOHR, 1.0, rtol=0.0, atol=1.0e-15
    )
    assert np.isclose(
        KELVIN_TO_EV * EV_TO_KELVIN, 1.0, rtol=0.0, atol=1.0e-15
    )
    assert AVOGADRO_CONSTANT_MOL == 6.02214076e23
    assert np.isclose(
        ATOMIC_MASS_UNIT_TO_G,
        1.66053906660e-24,
        rtol=0.0,
        atol=1.0e-36,
    )
