"""Bridge-function closures for classical one-component ion fluids.

The Rosenfeld--Ashcroft closure replaces the unknown bridge function by that
of a hard-sphere reference fluid.  The effective hard-sphere diameter is not
fitted to simulation data: VMHNC determines it from the variational
free-energy condition used by Faussurier (2004).

The bridge-universality ansatz is due to
:cite:t:`RosenfeldAshcroft1979`.  The closure and variational condition below
follow Eqs. (5)--(7) of :cite:t:`Faussurier2004`; their free-energy origin is
also given by Eqs. (17)--(19) of :cite:t:`LadoFoilesAshcroft1983`.  The hard-
sphere pair distribution uses the exact Percus--Yevick solution of
:cite:t:`Wertheim1963,Thiele1963`.

Only the one-component closure is implemented here.  A mixture requires an
additive hard-sphere-mixture reference and several coupled diameters; applying
the scalar packing fraction below to every pair channel would not be the same
theory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import residue

from otter.ionic.qoz import (
    hnc_solver,
    hnc_solver_multicomponent,
    hnc_solver_multicomponent_continuation,
)
from otter.numerics.transforms import radial_forward, radial_inverse


def _hard_sphere_free_energy_correction(eta: float) -> float:
    """Return ``f_CS(eta) - f_PYV(eta)`` from Faussurier (2004), Eq. (7)."""
    eta = float(eta)
    return float(
        eta * (4.0 - 3.0 * eta) / (1.0 - eta) ** 2
        - 6.0 * eta / (1.0 - eta)
        - 2.0 * np.log(1.0 - eta)
    )


def _hard_sphere_free_energy_correction_derivative(eta: float) -> float:
    """Return the analytic derivative of the Eq. (7) correction."""
    eta = float(eta)
    return float(2.0 * eta**2 / (1.0 - eta) ** 3)


@dataclass(frozen=True)
class VMHNCResult:
    """Converged one-component VMHNC correlation functions and diagnostics."""

    g_r: np.ndarray
    s_k: np.ndarray
    h_r: np.ndarray
    c_r: np.ndarray
    bridge_r: np.ndarray
    effective_potential_r: np.ndarray
    eta: float
    sigma_bohr: float
    variational_residual: float
    eta_history: tuple[dict[str, float | bool], ...]
    hnc_residual_history: tuple[float, ...]
    hnc_stage_meta: tuple[dict[str, float | bool | str], ...]


class HardSpherePYReference:
    """Percus--Yevick hard-sphere reference on an ``x=r/sigma`` grid.

    The direct correlation polynomial and shell solution are the exact
    hard-sphere PY result of :cite:t:`Wertheim1963,Thiele1963`.  The bridge is
    reconstructed from the same PY ``g`` and ``c``; no empirical bridge fit is
    introduced.
    """

    def __init__(
        self,
        points_per_diameter: int = 256,
        *,
        x_max: float = 64.0,
    ) -> None:
        points = int(points_per_diameter)
        if points < 32:
            raise ValueError("points_per_diameter must be at least 32.")
        n_grid = int(round(float(x_max) * points))
        if n_grid < 8 * points:
            raise ValueError("x_max must span at least eight hard-sphere diameters.")

        self.points_per_diameter = points
        self.x = np.arange(1, n_grid, dtype=float) / float(points)

        # Import locally to keep this module independent of the concrete
        # transform class while still enforcing Otter's production DST grid.
        from otter.numerics.transforms import precompute_dst_lattice_transform_like

        self.transform = precompute_dst_lattice_transform_like(
            self.x,
            n_grid=n_grid,
        )
        self.contact_index = int(np.searchsorted(self.x, 1.0))

    def _analytic_g_below_six(self, eta: float) -> np.ndarray:
        """Evaluate the Wertheim--Thiele PY shell series for ``1 <= x < 6``."""
        x = self.x
        l_polynomial = np.array([1.0 + 0.5 * eta, 1.0 + 2.0 * eta])
        s_polynomial = np.array(
            [
                (1.0 - eta) ** 2,
                6.0 * eta * (1.0 - eta),
                18.0 * eta**2,
                -12.0 * eta * (1.0 + 2.0 * eta),
            ]
        )
        g_exact = np.zeros_like(x)
        for shell in range(1, 6):
            mask = (x >= shell) & (x < 6.0)
            offset = x[mask] - float(shell)
            numerator = np.polynomial.polynomial.polypow(
                l_polynomial[::-1], shell
            )[::-1]
            numerator = np.concatenate((numerator, [0.0]))
            denominator = np.polynomial.polynomial.polypow(
                s_polynomial[::-1], shell
            )[::-1]
            coefficients, poles, _ = residue(
                numerator,
                denominator,
                tol=5.0e-2,
                rtype="avg",
            )
            inverse = np.zeros(offset.size, dtype=complex)
            previous_pole: complex | None = None
            pole_order = 0
            for coefficient, pole in zip(coefficients, poles):
                if previous_pole is None or abs(pole - previous_pole) > 5.0e-2:
                    previous_pole = pole
                    pole_order = 1
                else:
                    pole_order += 1
                inverse += (
                    coefficient
                    * offset ** (pole_order - 1)
                    / math.factorial(pole_order - 1)
                    * np.exp(pole * offset)
                )
            g_exact[mask] += (-12.0 * eta) ** (shell - 1) * np.real(inverse)

        analytic = (x >= 1.0) & (x < 6.0)
        g_exact[analytic] /= x[analytic]
        return g_exact

    def bridge_and_g(self, eta: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the PY hard-sphere bridge and pair distribution at ``eta``."""
        eta = float(eta)
        if not np.isfinite(eta) or not 0.0 < eta < 0.5:
            raise ValueError("eta must be finite and lie in (0, 0.5).")

        x = self.x
        denominator = (1.0 - eta) ** 4
        lambda_1 = (1.0 + 2.0 * eta) ** 2 / denominator
        lambda_2 = -(1.0 + 0.5 * eta) ** 2 / denominator
        c_r = np.where(
            x <= 1.0,
            -lambda_1
            - 6.0 * eta * lambda_2 * x
            - 0.5 * eta * lambda_1 * x**3,
            0.0,
        )
        c_k = radial_forward(c_r, self.transform)
        density_sigma3 = 6.0 * eta / np.pi
        g_r = 1.0 + radial_inverse(
            c_k / (1.0 - density_sigma3 * c_k),
            self.transform,
        )

        # The numerical long-range PY transform rings at the hard contact.
        # Use the analytic shell solution there and join it with a C2 blend.
        g_exact = self._analytic_g_below_six(eta)
        analytic = (x >= 1.0) & (x < 6.0)
        weight = np.clip((x[analytic] - 5.5) / 0.5, 0.0, 1.0)
        weight = weight**3 * (10.0 + weight * (-15.0 + 6.0 * weight))
        g_r[analytic] = (
            (1.0 - weight) * g_exact[analytic] + weight * g_r[analytic]
        )

        inside = x < 1.0
        g_r[inside] = 0.0
        g_r[~inside] = np.maximum(g_r[~inside], 1.0e-12)
        bridge = np.empty_like(x)
        bridge[inside] = (
            np.log(np.maximum(-c_r[inside], 1.0e-300))
            + c_r[inside]
            + 1.0
        )
        bridge[~inside] = np.log(g_r[~inside]) - g_r[~inside] + 1.0
        return bridge, g_r


@dataclass
class _EtaSolution:
    residual: float
    result: VMHNCResult
    nodal_r: np.ndarray


def _audit_hnc_solution(
    g_r: np.ndarray,
    s_k: np.ndarray,
    residual_history: list[float],
    *,
    transform,
    ion_density_bohr3: float,
    tolerance: float,
) -> tuple[bool, float, float]:
    history = np.asarray(residual_history, dtype=float)
    best_residual = float(
        np.min(history[np.isfinite(history)])
        if np.any(np.isfinite(history))
        else np.inf
    )
    closure = float(
        np.max(
            np.abs(
                np.asarray(s_k, dtype=float)
                - (
                    1.0
                    + float(ion_density_bohr3)
                    * radial_forward(np.asarray(g_r, dtype=float) - 1.0, transform)
                )
            )
        )
    )
    converged = bool(
        best_residual <= float(tolerance)
        and np.all(np.isfinite(g_r))
        and float(np.min(g_r)) >= 0.0
        and np.all(np.isfinite(s_k))
        and float(np.min(s_k)) > 0.0
        and closure <= max(10.0 * float(tolerance), 1.0e-6)
    )
    return converged, best_residual, closure


def _solve_eta(
    eta: float,
    reference: HardSpherePYReference,
    *,
    r: np.ndarray,
    k: np.ndarray,
    potential_r: np.ndarray,
    transform,
    ion_density_bohr3: float,
    ion_temperature_ha: float,
    nodal_init_r: np.ndarray | None,
    eta_derivative_rel_step: float,
    hnc_mix: float,
    hnc_tol: float,
    hnc_max_iter: int,
    hnc_newton_max_iter: int,
    hnc_tail_points: int,
    hnc_potential_scales: tuple[float, ...],
    hnc_min_scale_step: float,
    hnc_max_stage_attempts: int,
) -> _EtaSolution:
    bridge_x, g_hs = reference.bridge_and_g(eta)
    sigma = float((6.0 * eta / (np.pi * ion_density_bohr3)) ** (1.0 / 3.0))
    bridge_r = np.interp(r / sigma, reference.x, bridge_x, right=0.0)
    effective_potential = potential_r - ion_temperature_ha * bridge_r

    common = dict(
        mix=float(hnc_mix),
        tol=float(hnc_tol),
        max_iter=int(hnc_max_iter),
        tail_points=int(hnc_tail_points),
        c_map_clip=0.0,
        enforce_h_tail_zero=False,
        s_min_floor=1.0e-8,
        s_max_ceil=1.0e6,
        s_projection_mode="none",
    )
    stage_meta: list[dict[str, float | bool | str]] = []
    if nodal_init_r is not None:
        try:
            direct = hnc_solver_multicomponent(
                r,
                k,
                effective_potential[None, None, :],
                transform,
                np.asarray([ion_density_bohr3], dtype=float),
                ion_temperature_ha,
                n_init_r=np.asarray(nodal_init_r, dtype=float),
                mixing_scheme="newton_krylov",
                **common,
            )
        except (RuntimeError, ValueError):
            direct = None
            converged = False
        else:
            converged, _, _ = _audit_hnc_solution(
                direct[0][0, 0],
                direct[1][0, 0],
                direct[4],
                transform=transform,
                ion_density_bohr3=ion_density_bohr3,
                tolerance=hnc_tol,
            )
    else:
        direct = None
        converged = False

    if not converged:
        scalar = hnc_solver(
            r,
            k,
            effective_potential,
            transform,
            ion_density_bohr3,
            ion_temperature_ha,
            mix=float(hnc_mix),
            tol=float(hnc_tol),
            max_iter=int(hnc_max_iter),
            mixing_scheme="anderson",
            tail_points=int(hnc_tail_points),
            c_map_clip=0.0,
            enforce_h_tail_zero=False,
        )
        converged, _, _ = _audit_hnc_solution(
            scalar[0],
            scalar[1],
            scalar[4],
            transform=transform,
            ion_density_bohr3=ion_density_bohr3,
            tolerance=hnc_tol,
        )
        if converged:
            direct = (
                scalar[0][None, None, :],
                scalar[1][None, None, :],
                scalar[2][None, None, :],
                scalar[3][None, None, :],
                scalar[4],
            )

    # A nearby eta normally converges from the previous nodal term.  If that
    # continuation follows a bad branch, restart through potential-strength
    # continuation so the outer eta search does not inherit the failure.
    if not converged:
        continued = hnc_solver_multicomponent_continuation(
            r,
            k,
            effective_potential[None, None, :],
            transform,
            np.asarray([ion_density_bohr3], dtype=float),
            ion_temperature_ha,
            potential_scales=tuple(float(value) for value in hnc_potential_scales),
            adaptive=True,
            min_scale_step=float(hnc_min_scale_step),
            max_stage_attempts=int(hnc_max_stage_attempts),
            require_converged=True,
            fallback_mixing_scheme="newton_krylov",
            newton_max_iter=int(hnc_newton_max_iter),
            mixing_scheme="anderson",
            **common,
        )
        g_matrix, s_matrix, h_matrix, c_matrix, history, stage_meta = continued
    else:
        assert direct is not None
        g_matrix, s_matrix, h_matrix, c_matrix, history = direct

    g_r = np.asarray(g_matrix[0, 0], dtype=float)
    s_k = np.asarray(s_matrix[0, 0], dtype=float)
    h_r = np.asarray(h_matrix[0, 0], dtype=float)
    c_r = np.asarray(c_matrix[0, 0], dtype=float)
    converged, best_hnc_residual, closure = _audit_hnc_solution(
        g_r,
        s_k,
        history,
        transform=transform,
        ion_density_bohr3=ion_density_bohr3,
        tolerance=hnc_tol,
    )
    if not converged:
        raise RuntimeError(
            "VMHNC inner OZ solve did not reach a physical fixed point at "
            f"eta={eta:.8f}: best residual={best_hnc_residual:.3e}, "
            f"closure mismatch={closure:.3e}."
        )

    delta_eta = max(float(eta_derivative_rel_step) * eta, 1.0e-6)
    delta_eta = min(delta_eta, 0.49 * eta, 0.49 * (0.5 - eta))
    bridge_plus, _ = reference.bridge_and_g(eta + delta_eta)
    bridge_minus, _ = reference.bridge_and_g(eta - delta_eta)

    # Faussurier (2004), Eq. (6), differentiates B_HSPY at fixed physical r.
    # Since sigma is proportional to eta^(1/3), the second term below converts
    # the derivative tabulated at fixed x=r/sigma into that fixed-r derivative.
    d_bridge_fixed_r = (
        (bridge_plus - bridge_minus) / (2.0 * delta_eta)
        - reference.x
        * np.gradient(bridge_x, reference.x, edge_order=2)
        / (3.0 * eta)
    )
    g_on_x = np.interp(sigma * reference.x, r, g_r, right=1.0)
    integrand = (
        reference.x**2 * (g_on_x - g_hs) * d_bridge_fixed_r
    )

    # Split at hard contact.  One trapezoid across the jump in g_HS creates a
    # spurious grid-dependent zero of the variational condition.
    contact = reference.contact_index
    upper = min(
        int(np.count_nonzero(sigma * reference.x <= r[-1])),
        reference.x.size,
    )
    if upper <= contact + 1:
        raise ValueError(
            "The radial HNC box must extend beyond the effective hard-sphere diameter."
        )
    inner = integrand[: contact + 1].copy()
    inner[-1] = (
        reference.x[contact] ** 2
        * g_on_x[contact]
        * d_bridge_fixed_r[contact]
    )
    integral_x = np.trapezoid(inner, x=reference.x[: contact + 1])
    integral_x += np.trapezoid(
        integrand[contact:upper],
        x=reference.x[contact:upper],
    )
    # This is Faussurier (2004), Eq. (6).  The first term is d(delta_phi)/deta
    # from Eq. (7), where delta_phi is the Carnahan--Starling minus PY-virial
    # hard-sphere free energy (:cite:p:`CarnahanStarling1969`).
    variational_residual = float(
        _hard_sphere_free_energy_correction_derivative(eta)
        - 0.5
        * ion_density_bohr3
        * 4.0
        * np.pi
        * sigma**3
        * integral_x
    )
    result = VMHNCResult(
        g_r=g_r,
        s_k=s_k,
        h_r=h_r,
        c_r=c_r,
        bridge_r=np.asarray(bridge_r, dtype=float),
        effective_potential_r=np.asarray(effective_potential, dtype=float),
        eta=float(eta),
        sigma_bohr=float(sigma),
        variational_residual=variational_residual,
        eta_history=(),
        hnc_residual_history=tuple(float(value) for value in history),
        hnc_stage_meta=tuple(dict(stage) for stage in stage_meta),
    )
    return _EtaSolution(
        residual=variational_residual,
        result=result,
        nodal_r=(h_r - c_r)[None, None, :],
    )


def solve_vmhnc(
    r: np.ndarray,
    k: np.ndarray,
    potential_r: np.ndarray,
    transform,
    ion_density_bohr3: float,
    ion_temperature_ha: float,
    *,
    points_per_diameter: int = 256,
    eta_bounds: tuple[float, float] = (0.05, 0.49),
    eta_scan_points: int = 9,
    eta_tol: float = 1.0e-4,
    eta_max_iter: int = 16,
    eta_derivative_rel_step: float = 1.0e-3,
    hnc_mix: float = 0.03,
    hnc_tol: float = 1.0e-4,
    hnc_max_iter: int = 160,
    hnc_newton_max_iter: int = 60,
    hnc_tail_points: int = 32,
    hnc_potential_scales: tuple[float, ...] = (0.05, 0.15, 0.35, 0.6, 0.8, 1.0),
    hnc_min_scale_step: float = 2.0e-3,
    hnc_max_stage_attempts: int = 32,
) -> VMHNCResult:
    """Solve the one-component Rosenfeld--Ashcroft/VMHNC closure.

    The inner closure is Eq. (5) and the outer ``eta`` root is Eq. (6) of
    :cite:t:`Faussurier2004`.  Thus ``eta`` is a variational hard-sphere
    packing fraction, not a parameter fitted to the target simulation.

    ``eta`` is bracketed automatically over ``eta_bounds``.  Every accepted
    outer point contains a separately audited raw OZ solution; no structure-
    factor projection or bridge parameter fit to external data is used.
    """
    r = np.asarray(r, dtype=float)
    k = np.asarray(k, dtype=float)
    potential_r = np.asarray(potential_r, dtype=float)
    if r.shape != transform.r.shape or k.shape != transform.k.shape:
        raise ValueError("r/k must match the supplied radial transform context.")
    if potential_r.shape != r.shape:
        raise ValueError("potential_r must have the same shape as r.")
    if not np.isfinite(float(ion_density_bohr3)) or ion_density_bohr3 <= 0.0:
        raise ValueError("ion_density_bohr3 must be finite and positive.")
    if not np.isfinite(float(ion_temperature_ha)) or ion_temperature_ha <= 0.0:
        raise ValueError("ion_temperature_ha must be finite and positive.")
    low, high = (float(value) for value in eta_bounds)
    if not 0.0 < low < high < 0.5:
        raise ValueError("eta_bounds must satisfy 0 < low < high < 0.5.")
    if int(eta_scan_points) < 2:
        raise ValueError("eta_scan_points must be at least 2.")
    if float(eta_tol) <= 0.0:
        raise ValueError("eta_tol must be positive.")
    if int(eta_max_iter) < 1:
        raise ValueError("eta_max_iter must be positive.")
    if float(eta_derivative_rel_step) <= 0.0:
        raise ValueError("eta_derivative_rel_step must be positive.")

    reference = HardSpherePYReference(points_per_diameter)
    history: list[dict[str, float | bool]] = []
    cache: dict[float, _EtaSolution] = {}

    def evaluate(eta: float, nodal_init_r: np.ndarray | None) -> _EtaSolution:
        eta_key = float(eta)
        if eta_key not in cache:
            solution = _solve_eta(
                eta_key,
                reference,
                r=r,
                k=k,
                potential_r=potential_r,
                transform=transform,
                ion_density_bohr3=float(ion_density_bohr3),
                ion_temperature_ha=float(ion_temperature_ha),
                nodal_init_r=nodal_init_r,
                eta_derivative_rel_step=float(eta_derivative_rel_step),
                hnc_mix=float(hnc_mix),
                hnc_tol=float(hnc_tol),
                hnc_max_iter=int(hnc_max_iter),
                hnc_newton_max_iter=int(hnc_newton_max_iter),
                hnc_tail_points=int(hnc_tail_points),
                hnc_potential_scales=tuple(hnc_potential_scales),
                hnc_min_scale_step=float(hnc_min_scale_step),
                hnc_max_stage_attempts=int(hnc_max_stage_attempts),
            )
            cache[eta_key] = solution
            history.append(
                {
                    "eta": eta_key,
                    "variational_residual": float(solution.residual),
                    "hnc_best_residual": float(
                        min(solution.result.hnc_residual_history)
                    ),
                    "cold_restart_used": bool(solution.result.hnc_stage_meta),
                }
            )
        return cache[eta_key]

    left_eta: float | None = None
    right_eta: float | None = None
    previous_eta: float | None = None
    previous_solution: _EtaSolution | None = None
    for eta in np.linspace(low, high, int(eta_scan_points)):
        solution = evaluate(
            float(eta),
            None if previous_solution is None else previous_solution.nodal_r,
        )
        if solution.residual == 0.0:
            left_eta = right_eta = float(eta)
            break
        if (
            previous_solution is not None
            and previous_solution.residual * solution.residual < 0.0
        ):
            left_eta = float(previous_eta)
            right_eta = float(eta)
            break
        previous_eta = float(eta)
        previous_solution = solution

    if left_eta is None or right_eta is None:
        residuals = [float(item["variational_residual"]) for item in history]
        raise RuntimeError(
            "VMHNC variational eta could not be bracketed over "
            f"[{low:.6f}, {high:.6f}]; sampled residual range="
            f"[{min(residuals):.6e}, {max(residuals):.6e}]."
        )

    if left_eta != right_eta:
        left = cache[left_eta]
        right = cache[right_eta]
        for _ in range(int(eta_max_iter)):
            if right_eta - left_eta <= float(eta_tol):
                break
            middle_eta = 0.5 * (left_eta + right_eta)
            seed = (
                left.nodal_r
                if middle_eta - left_eta <= right_eta - middle_eta
                else right.nodal_r
            )
            middle = evaluate(middle_eta, seed)
            if left.residual * middle.residual <= 0.0:
                right_eta, right = middle_eta, middle
            else:
                left_eta, left = middle_eta, middle
        final_eta = 0.5 * (left_eta + right_eta)
        nearest = left if final_eta - left_eta <= right_eta - final_eta else right
        final = evaluate(final_eta, nearest.nodal_r)
    else:
        final = cache[left_eta]

    return VMHNCResult(
        **{
            **final.result.__dict__,
            "eta_history": tuple(dict(item) for item in history),
        }
    )


__all__ = ["HardSpherePYReference", "VMHNCResult", "solve_vmhnc"]
