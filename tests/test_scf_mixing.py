"""Scale and conditioning regressions for the shared production SCF mixer."""
import numpy as np
import pytest

from otter.numerics import mixing


@pytest.fixture
def mixer():
    return mixing


@pytest.fixture
def sample():
    rng = np.random.default_rng(7281)
    r = np.linspace(.02, 4., 32)**2
    return rng.normal(size=(4, 32)), rng.normal(size=32), r


@pytest.mark.parametrize("scale", [1e-150, 1e-30, 1., 1e30, 1e150])
def test_common_residual_scale_does_not_change_weights(mixer, sample, scale):
    d, f, r = sample
    expected, _ = mixer.regularized_secant_weights(d, f, r, 5e-4)
    got, meta = mixer.regularized_secant_weights(d*scale, f*scale, r, 5e-4)
    np.testing.assert_allclose(got, expected, rtol=2e-12, atol=2e-14)
    assert meta["condition"] <= 1e8*(1+1e-12)


def test_radial_units_and_column_rescaling_preserve_correction(mixer, sample):
    d, f, r = sample
    scales = np.array([1e-4, 1e-2, 10., 100.])
    a, _ = mixer.regularized_secant_weights(d, f, r, 5e-4)
    b, _ = mixer.regularized_secant_weights(d*scales[:, None], f, r*1e3, 5e-4)
    np.testing.assert_allclose(b*scales, a, rtol=2e-12, atol=2e-14)


@pytest.mark.parametrize("regularization", [0., 1e-12, 5e-4])
@pytest.mark.parametrize("perturbation", [0., 1e-14, 1e-6])
def test_dependent_columns_condition_is_bounded(mixer, sample, regularization, perturbation):
    d, f, r = sample
    d[1] = d[0]+perturbation*d[1]
    result, meta = mixer.regularized_secant_weights(d, f, r, regularization, condition_limit=1e6)
    assert np.all(np.isfinite(result))
    assert meta["condition"] <= 1e6*(1+1e-12)


@pytest.mark.parametrize("count", [0, 1, 4])
def test_zero_history_and_zero_residual_have_finite_linear_fallback(mixer, sample, count):
    d, f, r = sample
    for residual in (f, np.zeros_like(f)):
        result, meta = mixer.regularized_secant_weights(np.zeros((count, r.size)), residual, r, 5e-4)
        np.testing.assert_array_equal(result, np.zeros(count))
        assert meta["active_columns"] == 0


def test_tiny_column_is_not_used_to_amplify_noise(mixer, sample):
    d, f, r = sample
    d[0] *= 1e-18
    result, meta = mixer.regularized_secant_weights(d, f, r, 5e-4)
    assert result[0] == 0 and meta["active_columns"] == 3


@pytest.mark.parametrize("target", ["history", "residual"])
def test_nonfinite_samples_use_linear_fallback(mixer, sample, target):
    d, f, r = sample
    if target == "history":
        d[0, 0] = np.nan
    else:
        f[0] = np.inf
    result, meta = mixer.regularized_secant_weights(d, f, r, 5e-4)
    assert result is None and meta["reason"] == "nonfinite_input"


def test_matches_independent_normalized_ridge_solution(mixer, sample):
    d, f, r = sample
    weights = np.r_[(r[1]-r[0])/2, (r[2:]-r[:-2])/2, (r[-1]-r[-2])/2]
    weighted_d = d.T*np.sqrt(weights[:, None])
    weighted_f = f*np.sqrt(weights)
    norms = np.linalg.norm(weighted_d, axis=0)
    normalized = weighted_d/norms
    smax = np.linalg.svd(normalized, compute_uv=False)[0]
    ridge = (5e-4*smax)**2
    expected = np.linalg.solve(normalized.T@normalized + ridge*np.eye(len(d)),
                               normalized.T@weighted_f)/norms
    got, _ = mixer.regularized_secant_weights(d, f, r, 5e-4)
    np.testing.assert_allclose(got, expected, rtol=2e-12, atol=2e-14)


def test_svd_failure_returns_linear_fallback(mixer, sample, monkeypatch):
    def fail(*a, **kw):
        raise np.linalg.LinAlgError("controlled failure")
    monkeypatch.setattr(np.linalg, "svd", fail)
    d, f, r = sample
    result, meta = mixer.regularized_secant_weights(d, f, r, 5e-4)
    assert result is None and meta["reason"] == "svd_failure"


@pytest.mark.parametrize("regularization", [-1., np.nan, np.inf])
def test_invalid_regularization_is_rejected(mixer, sample, regularization):
    d, f, r = sample
    with pytest.raises(ValueError, match="regularization"):
        mixer.regularized_secant_weights(d, f, r, regularization)


@pytest.mark.parametrize("limit", [1., 0., np.nan, np.inf])
def test_invalid_condition_limit_is_rejected(mixer, sample, limit):
    d, f, r = sample
    with pytest.raises(ValueError, match="condition_limit"):
        mixer.regularized_secant_weights(d, f, r, 5e-4, limit)


def test_overflowing_ridge_returns_linear_fallback(mixer, sample):
    d, f, r = sample
    result, meta = mixer.regularized_secant_weights(d, f, r, 1e300)
    assert result is None and meta["reason"] == "nonfinite_ridge"


@pytest.mark.parametrize("r", [[0., 0.], [1., 0.], [0., np.nan], [0., np.inf]])
def test_invalid_grid_is_rejected(mixer, r):
    with pytest.raises(ValueError, match="radial grid"):
        mixer.regularized_secant_weights([[1., 2.]], [2., 3.], r, 5e-4)


def test_more_columns_than_samples_keeps_nullspace_condition_bound(mixer):
    d = np.array([[1., 0.], [0., 1.], [1., 1.]])
    result, meta = mixer.regularized_secant_weights(d, [2., 3.], [.1, 1.], 0.)
    assert np.all(np.isfinite(result))
    assert meta["condition"] <= 1e8*(1+1e-12)


@pytest.mark.parametrize("initial", [1e-12, 1., 1e12])
def test_scale_aware_history_converges_when_linear_mixing_is_unstable(mixer, initial):
    from collections import deque
    r = np.linspace(.1, 3., 32)
    x = initial*np.exp(-r)
    x_prev = f_prev = None
    dx, df = deque(maxlen=4), deque(maxlen=4)
    for _ in range(8):
        f = -10*x  # Fixed point x=0; linear mix=.25 alone multiplies x by -1.5.
        if x_prev is not None:
            dx.append(x-x_prev)
            df.append(f-f_prev)
        correction = np.zeros_like(x)
        if df:
            coefficients, _ = mixer.regularized_secant_weights(df, f, r, 5e-4)
            assert coefficients is not None
            for w, dx_i, df_i in zip(coefficients, dx, df):
                correction += w*(dx_i+.25*df_i)
        x_prev, f_prev = x.copy(), f.copy()
        x = x+.25*f-correction
    assert np.max(np.abs(x))/initial < 1e-12
