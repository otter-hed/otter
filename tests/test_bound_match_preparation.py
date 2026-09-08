"""Energy scouts may reuse one frozen preparation, never another SCF potential."""
import numpy as np
import pytest

from otter.electronic.solvers import bound
from otter.numerics.grids import create_sqrt_grid


@pytest.mark.parametrize("l", [0, 1, 8])
def test_prepared_residual_is_exactly_the_public_residual(l):
    grid = create_sqrt_grid(30., 1001, rmin=1e-5)
    v = -6*np.exp(-grid.r)/grid.r
    prepared = bound._prepare_zero_tail_matching(v, grid.r, grid.dxi, l, None)
    for energy in -np.geomspace(1e-8, .1, 17):
        assert bound._zero_tail_prepared_residual(energy, prepared) == (
            bound.zero_tail_bound_matching_residual(v, grid.r, grid.dxi, l, energy))


def test_scout_prepares_once_and_does_not_reuse_another_potential(monkeypatch):
    grid = create_sqrt_grid(10., 501, rmin=1e-5)
    original = bound._prepare_zero_tail_matching
    potentials = []

    def capture(v, *args):
        potentials.append(np.array(v, copy=True))
        return original(v, *args)

    monkeypatch.setattr(bound, "_prepare_zero_tail_matching", capture)
    for offset in (0., .01):
        # A nonnegative potential has no negative-energy poles.
        assert bound.find_shallowest_zero_tail_bound_state(
            np.full_like(grid.r, offset), grid.r, grid.dxi, 0, n_scan=12) is None
    assert len(potentials) == 2
    assert not np.array_equal(*potentials)


def test_preparation_rejects_nonuniform_grid_before_scout():
    r = np.linspace(.01, 10., 101)
    with pytest.raises(ValueError, match="uniform sqrt"):
        bound.find_shallowest_zero_tail_bound_state(np.zeros_like(r), r, .01, 0)
