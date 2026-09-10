"""Scale-aware, condition-bounded least squares for SCF extrapolation.

The nonlinear map, step safeguards and convergence tests belong to the caller.
This helper changes extrapolation weights only and cannot certify a physical
solution. Its regularization parameter is dimensionless, applied after history
normalization, rather than an absolute floor on the unnormalized Gram matrix.
"""
import numpy as np


def regularized_secant_weights(df_history, residual, r, regularization, condition_limit=1e8):
    """Solve column-normalized, radially weighted ridge least squares by SVD.

    The ridge follows the largest singular value; an explicit lower bound
    limits the condition number of the regularized normal matrix. Near-zero
    columns are excluded relative to the current residual/history amplitude.
    Nonfinite samples or failed finite arithmetic return None for linear
    fallback. Incompatible shapes, invalid grids or parameters raise ValueError.
    """
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative")
    if not np.isfinite(condition_limit) or condition_limit <= 1:
        raise ValueError("condition_limit must be finite and greater than one")
    r = np.asarray(r, dtype=float)
    f = np.asarray(residual, dtype=float)
    d = np.asarray(df_history, dtype=float).T
    if d.size == 0 and r.ndim == 1:
        d = d.reshape((r.size, 0))
    if r.ndim != 1 or f.shape != r.shape or d.ndim != 2 or d.shape[0] != r.size:
        raise ValueError("incompatible residual, history or radial grid")
    if r.size < 2 or not np.all(np.isfinite(r)) or not np.all(np.diff(r) > 0):
        raise ValueError("radial grid must be finite and strictly increasing")
    if not np.all(np.isfinite(d)) or not np.all(np.isfinite(f)):
        return None, dict(reason="nonfinite_input")
    amplitude = max(float(np.max(np.abs(d), initial=0)), float(np.max(np.abs(f), initial=0)))
    if amplitude == 0 or d.shape[1] == 0:
        return np.zeros(d.shape[1]), dict(reason="zero_history", active_columns=0)
    spacing = np.diff(r)
    weights = np.r_[spacing[0]/2, (spacing[:-1]+spacing[1:])/2, spacing[-1]/2]
    weight_sum = np.sum(weights)
    if not np.all(np.isfinite(weights)) or not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("radial quadrature weights must have a finite positive sum")
    weights /= weight_sum
    root_weights = np.sqrt(weights)
    d_scaled = (d/amplitude)*root_weights[:, None]
    f_scaled = (f/amplitude)*root_weights
    norms = np.linalg.norm(d_scaled, axis=0)
    active = norms > 32*np.finfo(float).eps
    if not np.any(active):
        return np.zeros(d.shape[1]), dict(reason="zero_history", active_columns=0)
    normalized = d_scaled[:, active] / norms[active]
    try:
        u, singular, vh = np.linalg.svd(normalized, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, dict(reason="svd_failure")
    largest_sq = float(singular[0]**2)
    smallest_sq = float(singular[-1]**2) if normalized.shape[1] <= normalized.shape[0] else 0.
    with np.errstate(over="ignore", invalid="ignore"):
        ridge = max((regularization*singular[0])**2,
                    (largest_sq-condition_limit*smallest_sq)/(condition_limit-1), 0.)
    if not np.isfinite(ridge):
        return None, dict(reason="nonfinite_ridge")
    factors = np.divide(singular, singular**2+ridge, out=np.zeros_like(singular),
                        where=singular**2+ridge > 0)
    coefficients = np.zeros(d.shape[1])
    coefficients[active] = (vh.T @ (factors*(u.T @ f_scaled))) / norms[active]
    if not np.all(np.isfinite(coefficients)):
        return None, dict(reason="nonfinite_weights")
    return coefficients, dict(reason="scaled_svd", active_columns=int(active.sum()),
        ridge=float(ridge), condition=float((largest_sq+ridge)/(smallest_sq+ridge)),
        largest_singular=float(singular[0]), smallest_singular=float(singular[-1]),
        coefficient_l1=float(np.sum(np.abs(coefficients))))

