"""Subspace recovery must keep the eigenproblem and refuse partial results."""
import numpy as np
import pytest
from scipy.sparse import eye

from otter.electronic.solvers import bound
from otter.numerics.grids import create_sqrt_grid


@pytest.mark.parametrize("size", [16, 80])
def test_stalled_subspace_restarts_same_problem_without_partial_pairs(size, monkeypatch):
    matrix, overlap = eye(size), eye(size)
    initial = np.linspace(0.1, 1., size)
    calls = []
    answer = (np.array([-2., -1.]), np.ones((size, 2)))

    def eigs(a, **kw):
        assert a is matrix and kw["M"] is overlap
        assert (kw["k"], kw["sigma"], kw["which"], kw["tol"]) == (2, -3., "LM", 1e-10)
        np.testing.assert_array_equal(kw["v0"], initial)
        calls.append(kw)
        kw["v0"][:] = 0.  # solver work must not corrupt the retry's start
        if len(calls) == 1:
            raise bound.ArpackNoConvergence("stalled", np.array([999.]), np.zeros((size, 1)))
        return answer

    monkeypatch.setattr(bound, "eigs", eigs)
    actual = bound._eigs_shift_invert(matrix, overlap, k=2, sigma=-3., v0=initial)
    assert actual is answer
    assert calls[0]["maxiter"] == 80 and "ncv" not in calls[0]
    assert calls[1]["ncv"] == min(size, 40) and "maxiter" not in calls[1]


def test_subspace_failure_is_not_accepted(monkeypatch):
    def fail(*args, **kwargs):
        raise bound.ArpackNoConvergence("still stalled", np.array([-1.]), np.zeros((20, 1)))

    monkeypatch.setattr(bound, "eigs", fail)
    with pytest.raises(bound.ArpackNoConvergence, match="still stalled"):
        bound._eigs_shift_invert(eye(20), eye(20), k=2, sigma=-2., v0=np.ones(20))


@pytest.mark.parametrize("charge,count", [(6., 3), (74., 7)])
def test_recovered_numerov_levels_and_densities_match_original_arpack(charge, count, monkeypatch):
    grid = create_sqrt_grid(rmax=30., N=768, rmin=1e-6)
    potential = -charge*np.exp(-grid.r)/grid.r
    original = bound.eigs
    arguments = (potential, grid.r, grid.dxi, 0, count)

    def legacy(matrix, **kw):
        kw.pop("maxiter", None)
        return original(matrix, **kw)

    monkeypatch.setattr(bound, "eigs", legacy)
    expected_e, expected_y = bound._solve_single_l_sparse(*arguments, nuclear_charge=charge)
    calls = 0

    def force_recovery(matrix, **kw):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise bound.ArpackNoConvergence("force recovery", np.array([]), np.empty((768, 0)))
        return original(matrix, **kw)

    monkeypatch.setattr(bound, "eigs", force_recovery)
    actual_e, actual_y = bound._solve_single_l_sparse(*arguments, nuclear_charge=charge)
    assert calls == 2
    np.testing.assert_allclose(actual_e, expected_e, rtol=2e-8, atol=2e-8)
    # Compare physical radial probabilities; arbitrary eigenvector signs do not matter.
    weights = 2.*grid.r**1.5*grid.dxi
    np.testing.assert_allclose(actual_y**2*weights[:, None], expected_y**2*weights[:, None],
                               rtol=2e-5, atol=2e-8)
