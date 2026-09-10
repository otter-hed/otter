"""
Experimental Starrett self-consistent AA <-> QOZ/HNC feedback.

The production plasma workflow is the ion-sphere (IS) construction.  This
module adds the outer iteration described by Starrett and Saumon (2014): the
IS ion structure is fed back into the full/external average-atom calculation,
new screening clouds are calculated, and QOZ/HNC is solved again.

For a mixture, the ionic background seen by a central species ``i`` is the
charge-weighted sum over every pair channel ``g_ij``.  At fixed mixture volume
partition the charge weights reduce to the AA volume weights.  The
ion-electron correlation potential uses the corresponding N-component
generalization of Eq. (19),

  V_i^C(k) = -(1/beta) sum_j n_j C_tilde_je(k) h_ij(k).

Starrett et al. (2014) used the IS approximation for their published mixture
results, so the multicomponent SC extension remains explicitly experimental.
The canonical bibliography keys are :cite:`StarrettSaumon2014,StarrettEtAl2014`.
The outer mixing and convergence controls below are Otter implementation
choices, not claims about the cited algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from otter.numerics.constants import EV_TO_HA
from otter.electronic.continuum.ideal import ideal_unbound_density
from otter.numerics.transforms import (
    precompute_dst_lattice_transform_like,
    radial_forward,
    radial_inverse,
)

from otter.electronic.full_external import (
    FullExternalConfig,
    solve_full_then_external,
    with_continuation_initial_guess,
)
from otter.workflows import (
    PlasmaWorkflowConfig,
    continue_plasma_workflow_from_electronic_result,
)


@dataclass(frozen=True)
class SCFeedbackConfig:
    """Numerical controls for the outer SC AA <-> QOZ/HNC iteration."""

    max_outer: int = 10
    g_tol: float = 5.0e-4
    v_corr_tol: float = 5.0e-4
    v_corr_mix: float = 0.35
    v_corr_scale: float = 1.0
    use_continuation: bool = True
    require_converged: bool = True
    # Inexact outer iteration: tighten only the QM full solve near convergence
    # or stagnation. Cap its tolerances at this fraction of ordinary AA defaults;
    # preserve stricter user tolerances. None disables this numerical safeguard
    # for diagnostics. IS, TF, external and spectral controls are unchanged.
    inner_full_tol_scale: float | None = 0.01

    def __post_init__(self) -> None:
        if int(self.max_outer) < 1:
            raise ValueError("max_outer must be at least 1.")
        if float(self.g_tol) <= 0.0 or float(self.v_corr_tol) <= 0.0:
            raise ValueError("SC convergence tolerances must be positive.")
        if not 0.0 < float(self.v_corr_mix) <= 1.0:
            raise ValueError("v_corr_mix must lie in (0, 1].")
        if float(self.v_corr_scale) < 0.0:
            raise ValueError("v_corr_scale must be non-negative.")
        if self.inner_full_tol_scale is not None and not (
            np.isfinite(self.inner_full_tol_scale)
            and 0.0 < self.inner_full_tol_scale <= 1.0
        ):
            raise ValueError("inner_full_tol_scale must be in (0, 1] or None.")


class SCFeedbackConvergenceError(RuntimeError):
    """Failed outer solve with an explicitly unconverged diagnostic result.

    Catching RuntimeError remains compatible. ``result`` retains the actual
    last iterate and its full feedback history, not an accepted SC solution.
    """

    def __init__(self, message: str, result: dict[str, Any]):
        super().__init__(message)
        self.result = result

    def __reduce__(self):
        # Benchmark producers use ProcessPoolExecutor. Preserve the diagnostic
        # payload across its exception transport, not only within one process.
        return type(self), (str(self), self.result)


def _refine_inner_full(cfg: FullExternalConfig, scale: float) -> FullExternalConfig:
    """Tighten numerical precision without changing the AA equations or model."""
    names = ("scf_tol", "scf_dn_tol", "scf_dv_tol")
    return replace(
        cfg,
        **{
            name: min(
                float(getattr(cfg, name)),
                float(getattr(FullExternalConfig, name)) * scale,
            )
            for name in names
        },
    )


def _needs_inner_precision(normalized_residuals: list[float]) -> bool:
    """Refine in the final decade or after three updates without progress.

    This trigger schedules work; it does not relax the acceptance tolerance.
    The refinement latches so later iterates cannot fall back to coarse AA.
    """
    return normalized_residuals[-1] <= 10.0 or (
        len(normalized_residuals) >= 4
        and min(normalized_residuals[-3:]) >= min(normalized_residuals[:-3])
    )


def mixture_ionic_background_profiles(
    gij_r: np.ndarray,
    volume_weights: np.ndarray,
) -> np.ndarray:
    """
    Return the effective ionic background profile seen by every AA species.

    For central species ``i``, the positive ionic charge background is

      n_e^0 g_i^bg(r) = sum_j n_j Z_j^* g_ij(r).

    The common-mu mixture volume closure gives
    ``n_j Z_j^* / n_e^0 = w_j``, where ``w_j`` are the AA volume weights.
    Consequently ``g_i^bg = sum_j w_j g_ij`` and its large-r limit is one.
    """
    g_arr = np.asarray(gij_r, dtype=float)
    weights = np.asarray(volume_weights, dtype=float)
    if g_arr.ndim != 3 or g_arr.shape[0] != g_arr.shape[1]:
        raise ValueError("gij_r must have shape (n_species, n_species, n_r).")
    if weights.shape != (g_arr.shape[0],):
        raise ValueError("volume_weights must contain one value per species.")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("volume_weights must be finite and positive.")
    weights = weights / float(np.sum(weights))
    profiles = np.einsum("j,ijr->ir", weights, g_arr, optimize=True)
    if not np.all(np.isfinite(profiles)):
        raise ValueError("The effective ionic background contains non-finite values.")
    return np.asarray(profiles, dtype=float)


def estimate_mixture_correlation_potentials(
    *,
    r: np.ndarray,
    k: np.ndarray,
    gij_r: np.ndarray,
    n_scr_k: np.ndarray,
    chi_ee_k: np.ndarray,
    zbar: np.ndarray,
    partial_ion_density: np.ndarray,
    field_free_electron_density: float,
    electron_temperature_ev: float,
) -> np.ndarray:
    """Estimate the per-species ``V_Ie^C(r)`` correlation potentials."""
    r_arr = np.asarray(r, dtype=float)
    k_arr = np.asarray(k, dtype=float)
    g_arr = np.asarray(gij_r, dtype=float)
    n_scr_arr = np.asarray(n_scr_k, dtype=float)
    chi_arr = np.asarray(chi_ee_k, dtype=float)
    zbar_arr = np.atleast_1d(np.asarray(zbar, dtype=float))
    n_i_arr = np.atleast_1d(np.asarray(partial_ion_density, dtype=float))
    n_species = int(zbar_arr.size)

    if r_arr.ndim != 1 or k_arr.ndim != 1 or r_arr.size != k_arr.size:
        raise ValueError(
            "r and k must be one-dimensional transform grids of equal length."
        )
    if g_arr.shape != (n_species, n_species, r_arr.size):
        raise ValueError(
            "gij_r shape is inconsistent with zbar and the transform grid."
        )
    if n_scr_arr.ndim == 1 and n_species == 1:
        n_scr_arr = n_scr_arr[np.newaxis, :]
    if n_scr_arr.shape != (n_species, k_arr.size):
        raise ValueError("n_scr_k must have shape (n_species, n_k).")
    if chi_arr.shape != k_arr.shape:
        raise ValueError("chi_ee_k must match k.")
    if n_i_arr.shape != (n_species,) or np.any(n_i_arr <= 0.0):
        raise ValueError(
            "partial_ion_density must be positive with one value per species."
        )
    if float(field_free_electron_density) <= 0.0:
        raise ValueError("field_free_electron_density must be positive.")
    if float(electron_temperature_ev) <= 0.0:
        raise ValueError("electron_temperature_ev must be positive for SC feedback.")
    if np.any(np.abs(chi_arr) < 1.0e-300):
        raise ValueError("chi_ee_k contains zero or underflow-scale entries.")

    beta = 1.0 / (float(electron_temperature_ev) * EV_TO_HA)
    mean_screening_density = float(np.sum(n_i_arr * zbar_arr))
    if mean_screening_density <= 0.0:
        raise ValueError("The mean screening-electron density must be positive.")

    c_ie_k = -beta * n_scr_arr / chi_arr[np.newaxis, :]
    coulomb = 4.0 * np.pi * beta / np.maximum(k_arr**2, 1.0e-24)
    c_tilde_bar_k = c_ie_k - zbar_arr[:, np.newaxis] * coulomb[np.newaxis, :]
    c_tilde_k = (
        float(field_free_electron_density) / mean_screening_density
    ) * c_tilde_bar_k

    transform = precompute_dst_lattice_transform_like(r_arr, n_grid=r_arr.size + 1)
    h_ij_k = np.empty_like(g_arr)
    for i in range(n_species):
        for j in range(n_species):
            h_ij_k[i, j] = radial_forward(g_arr[i, j] - 1.0, transform)

    v_corr_k = -(1.0 / beta) * np.einsum(
        "j,jk,ijk->ik",
        n_i_arr,
        c_tilde_k,
        h_ij_k,
        optimize=True,
    )
    v_corr_r = np.asarray(
        [radial_inverse(v_corr_k[i], transform) for i in range(n_species)],
        dtype=float,
    )
    v_corr_r = v_corr_r - v_corr_r[:, -1, np.newaxis]
    if not np.all(np.isfinite(v_corr_r)):
        raise ValueError(
            "The estimated correlation potential contains non-finite values."
        )
    return v_corr_r


def _electronic_species_entries(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    kind = str(workflow["electronic"]["kind"])
    electronic = dict(workflow["electronic"]["result"])
    if kind == "mixture":
        return [
            {**dict(sp), "result": dict(sp["result"])} for sp in electronic["species"]
        ]

    result = dict(electronic)
    r_ws = float(result["r_ws"])
    volume = 4.0 * np.pi * r_ws**3 / 3.0
    return [
        {
            "element": str(workflow["species_symbols"][0]),
            "Z": int(dict(result.get("meta", {})).get("Z", 0)),
            "count": float(workflow["species_counts"][0]),
            "x": 1.0,
            "volume_bohr3": float(volume),
            "r_ws_bohr": float(r_ws),
            "mu_ha": float(result["mu"]),
            "result": result,
        }
    ]


def _volume_weights(
    workflow: dict[str, Any], entries: list[dict[str, Any]]
) -> np.ndarray:
    if len(entries) == 1:
        return np.ones(1, dtype=float)
    electronic = dict(workflow["electronic"]["result"])
    saved = np.asarray(electronic.get("volume_weights", []), dtype=float)
    if saved.shape == (len(entries),) and np.all(saved > 0.0):
        return saved / float(np.sum(saved))
    x = np.asarray([float(sp["x"]) for sp in entries], dtype=float)
    volume = np.asarray([float(sp["volume_bohr3"]) for sp in entries], dtype=float)
    weights = x * volume
    return weights / float(np.sum(weights))


def _fixed_is_mu(workflow: dict[str, Any]) -> float:
    electronic = dict(workflow["electronic"]["result"])
    if str(workflow["electronic"]["kind"]) == "mixture":
        return float(electronic["mu_common_ha"])
    return float(electronic["mu"])


def _mix_correlation_potentials(
    *,
    r: np.ndarray,
    target: np.ndarray,
    old_r: np.ndarray | None,
    old_value: np.ndarray | None,
    mix: float,
) -> tuple[np.ndarray, float]:
    if old_r is None or old_value is None:
        mixed = float(mix) * np.asarray(target, dtype=float)
        return mixed, float(np.max(np.abs(mixed)))
    old_on_new = np.asarray(
        [np.interp(r, old_r, row) for row in np.asarray(old_value, dtype=float)],
        dtype=float,
    )
    mixed = (1.0 - float(mix)) * old_on_new + float(mix) * np.asarray(
        target, dtype=float
    )
    return mixed, float(np.max(np.abs(mixed - old_on_new)))


def _max_g_change(previous: dict[str, Any], current: dict[str, Any]) -> float:
    ion_old = dict(previous["ion"])
    ion_new = dict(current["ion"])
    r_old = np.asarray(ion_old["r"], dtype=float)
    r_new = np.asarray(ion_new["r"], dtype=float)
    g_old = np.asarray(ion_old["gij_r"], dtype=float)
    g_new = np.asarray(ion_new["gij_r"], dtype=float)
    if g_old.shape[:2] != g_new.shape[:2]:
        raise ValueError("Species dimensions changed during SC feedback.")
    max_change = 0.0
    for i in range(g_new.shape[0]):
        for j in range(g_new.shape[1]):
            old_on_new = np.interp(r_new, r_old, g_old[i, j])
            max_change = max(
                max_change, float(np.max(np.abs(g_new[i, j] - old_on_new)))
            )
    return float(max_change)


def _replace_electronic_results(
    workflow: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    iteration: int,
) -> tuple[str, dict[str, Any]]:
    kind = str(workflow["electronic"]["kind"])
    old = dict(workflow["electronic"]["result"])
    if kind == "single_species":
        result = dict(results[0])
        result_meta = dict(result.get("meta", {}))
        result_meta.update(
            {"structure_model": "SC", "sc_feedback_iteration": int(iteration)}
        )
        result["meta"] = result_meta
        return kind, result

    species_old = list(old["species"])
    species_new: list[dict[str, Any]] = []
    for sp, result in zip(species_old, results, strict=True):
        updated = dict(sp)
        updated["mu_ha"] = float(result["mu"])
        updated["result"] = dict(result)
        species_new.append(updated)
    electronic = dict(old)
    electronic["species"] = species_new
    meta = dict(electronic.get("meta", {}))
    meta.update({"structure_model": "SC", "sc_feedback_iteration": int(iteration)})
    electronic["meta"] = meta
    return kind, electronic


def solve_sc_feedback_workflow(
    workflow_cfg: PlasmaWorkflowConfig,
    is_workflow: dict[str, Any],
    *,
    feedback_cfg: SCFeedbackConfig | None = None,
) -> dict[str, Any]:
    """
    Continue one converged IS workflow to an SC AA <-> QOZ/HNC solution.

    The IS chemical potential and, for mixtures, the IS volume partition are
    held fixed.  Every mixture AA receives the full charge-weighted ``g_ij``
    background rather than only its diagonal channel.  Both the orbital
    Kohn--Sham and Thomas--Fermi electronic backends use the same
    Starrett--Saumon (2014), Sec. 2.4 coupling: Eqs. (19)--(20) construct
    ``V_Ie^C``, the QOZ/HNC ``g_II`` replaces the ion-sphere step in Eqs.
    (4)/(7), and the IS chemical potential remains fixed.

    Orbital SC steps default to zero-tail matching of shallow bound states.
    Explicit AA/species overrides take precedence; ordinary IS and TF defaults
    are unchanged. Matching retains the existing threshold acceptance checks.
    """
    controls = SCFeedbackConfig() if feedback_cfg is None else feedback_cfg
    if is_workflow.get("ion", None) is None:
        raise ValueError(
            "SC feedback requires an IS workflow with QOZ/HNC ion structure."
        )
    if workflow_cfg.ion_temperature_ev is None:
        raise ValueError("SC feedback requires ion_temperature_ev.")

    previous = is_workflow
    entries = _electronic_species_entries(previous)
    weights = _volume_weights(previous, entries)
    fixed_mu = _fixed_is_mu(previous)
    n0_common = float(
        ideal_unbound_density(
            float(fixed_mu),
            float(workflow_cfg.temperature_ev) * EV_TO_HA,
            method="exact",
        )
    )
    old_v_r: np.ndarray | None = None
    old_v_corr: np.ndarray | None = None
    history: list[dict[str, Any]] = []
    converged = False
    refine_enabled = (
        str(workflow_cfg.electronic_model).strip().lower() != "tf"
        and controls.inner_full_tol_scale is not None
    )
    inner_refined = False
    normalized_residuals: list[float] = []

    def correlation_target(state: dict[str, Any]) -> np.ndarray:
        state_ion = state["ion"]
        return float(controls.v_corr_scale) * estimate_mixture_correlation_potentials(
            r=state_ion["r"], k=state_ion["k"], gij_r=state_ion["gij_r"],
            n_scr_k=state_ion["n_scr_k"], chi_ee_k=state_ion["chi_ee_k"],
            zbar=np.atleast_1d(state_ion["zbar"]),
            partial_ion_density=np.atleast_1d(state_ion["n_i"]),
            field_free_electron_density=n0_common,
            electron_temperature_ev=float(workflow_cfg.temperature_ev),
        )

    v_target = correlation_target(previous)

    for outer in range(1, int(controls.max_outer) + 1):
        ion = dict(previous["ion"])
        r_ion = np.asarray(ion["r"], dtype=float)
        gij_r = np.asarray(ion["gij_r"], dtype=float)
        g_background = mixture_ionic_background_profiles(gij_r, weights)
        current_entries = _electronic_species_entries(previous)
        n0_values = np.asarray(
            [float(sp["result"]["n0"]) for sp in current_entries], dtype=float
        )
        n0_span = float(np.max(n0_values) - np.min(n0_values))
        v_mixed, d_v_corr = _mix_correlation_potentials(
            r=r_ion,
            target=v_target,
            old_r=old_v_r,
            old_value=old_v_corr,
            mix=float(controls.v_corr_mix),
        )

        results: list[dict[str, Any]] = []
        inner_tolerances: list[dict[str, float]] = []
        for idx, sp in enumerate(current_entries):
            symbol = str(sp["element"])
            overrides = dict(workflow_cfg.aa_overrides)
            overrides.update(dict(workflow_cfg.species_overrides.get(symbol, {})))
            electronic_model = str(workflow_cfg.electronic_model).strip().lower()
            if electronic_model != "tf":
                # Feedback can move an orbital through E=0 between updates.
                # Match its exterior before a finite-wall normalization jump
                # stalls the inner SCF. Preserve explicit per-user/species policy.
                overrides.setdefault("bound_zero_tail_refine", True)
            if (
                electronic_model != "tf"
                and str(overrides.get("bound_occ_mode", "fd")).strip().lower()
                != "fd"
            ):
                raise ValueError("SC feedback requires bound_occ_mode='fd'.")
            overrides.update(
                {
                    "element": symbol,
                    "temperature_ev": float(workflow_cfg.temperature_ev),
                    "rho_g_cc": float(workflow_cfg.rho_g_cc),
                    "electronic_model": electronic_model,
                    "run_mode": "full+ext",
                    "r_ws_override_bohr": float(sp["r_ws_bohr"]),
                    "n_i_override_bohr3": 1.0 / float(sp["volume_bohr3"]),
                    "full_fixed_mu_ha": float(fixed_mu),
                    "n0_mode_override": "ideal",
                    "bound_occ_mode": "fd",
                    "g_ii_override": np.asarray(g_background[idx], dtype=float),
                    "g_ii_override_r": r_ion,
                    "v_corr_full": np.asarray(v_mixed[idx], dtype=float),
                    "v_corr_full_r": r_ion,
                    "v_corr_ext": np.asarray(v_mixed[idx], dtype=float),
                    "v_corr_ext_r": r_ion,
                    "show_scf_progress": bool(workflow_cfg.show_progress),
                    "verbose": bool(workflow_cfg.verbose),
                    "save_data": False,
                }
            )
            aa_cfg = FullExternalConfig(**overrides)
            if inner_refined:
                aa_cfg = _refine_inner_full(
                    aa_cfg, float(controls.inner_full_tol_scale)
                )
            inner_tolerances.append({
                name: float(getattr(aa_cfg, name))
                for name in ("scf_tol", "scf_dn_tol", "scf_dv_tol")
            })
            if bool(controls.use_continuation):
                aa_cfg = with_continuation_initial_guess(
                    aa_cfg,
                    dict(sp["result"]),
                    enabled=True,
                    reuse_external=True,
                    stage2_from_init=True,
                )
                aa_cfg.stage1_max_iter = 0
            result = solve_full_then_external(aa_cfg)
            result_r = np.asarray(result["r"], dtype=float)
            v_corr_on_result = np.interp(
                result_r,
                r_ion,
                np.asarray(v_mixed[idx], dtype=float),
                left=float(v_mixed[idx, 0]),
                right=0.0,
            )
            g_background_on_result = np.interp(
                result_r,
                r_ion,
                np.asarray(g_background[idx], dtype=float),
                left=float(g_background[idx, 0]),
                right=1.0,
            )
            # Keep the actual additive potential used by this electronic
            # iterate in the payload.  This is needed to audit Eq. (19), and
            # is deliberately separate from V_H and V_xc.
            result["v_corr_full"] = np.asarray(v_corr_on_result, dtype=float)
            result["v_corr_ext"] = np.asarray(v_corr_on_result, dtype=float)
            result["g_ii_background"] = np.asarray(
                g_background_on_result,
                dtype=float,
            )
            result_meta = dict(result.get("meta", {}))
            result_meta.update(
                {
                    "structure_model": "SC_feedback_electronic_step",
                    "sc_reference": (
                        "Starrett_Saumon_2014_Sec2.4_Eqs19_20"
                    ),
                    "fixed_is_mu_ha": float(fixed_mu),
                    "g_ii_override_used": True,
                    "v_corr_full_used": True,
                    "v_corr_ext_used": True,
                }
            )
            result["meta"] = result_meta
            results.append(result)

        kind, electronic_result = _replace_electronic_results(
            previous,
            results,
            iteration=outer,
        )
        current = continue_plasma_workflow_from_electronic_result(
            workflow_cfg,
            electronic_kind=kind,
            electronic_result=electronic_result,
        )
        d_g = _max_g_change(previous, current)
        # Evaluate Eq. (19) on the NEW AA/HNC output against the potential
        # actually used for that AA input. A small damped step alone is not a
        # fixed-point residual, especially when v_corr_mix is very small.
        # Cache this target for the next iteration; the physical map is unchanged.
        v_target = correlation_target(current)
        r_new = np.asarray(current["ion"]["r"], dtype=float)
        used_on_new = np.asarray([np.interp(r_new, r_ion, row) for row in v_mixed])
        v_residual = float(np.max(np.abs(v_target - used_on_new)))
        normalized_residuals.append(max(
            float(d_g) / float(controls.g_tol),
            v_residual / float(controls.v_corr_tol),
        ))
        history.append(
            {
                "iteration": int(outer),
                "max_g_change": float(d_g),
                "max_v_corr_change_ha": float(d_v_corr),
                "max_v_corr_residual_ha": v_residual,
                "inner_full_refined": bool(inner_refined),
                "inner_full_tolerances": inner_tolerances,
                "ideal_common_n0_bohr3": float(n0_common),
                "species_n0_span_bohr3": float(n0_span),
                "zbar": np.atleast_1d(
                    np.asarray(current["ion"]["zbar"], dtype=float)
                ).tolist(),
            }
        )
        previous = current
        entries = _electronic_species_entries(previous)
        old_v_r = r_ion.copy()
        old_v_corr = np.asarray(v_mixed, dtype=float).copy()
        if (d_g < float(controls.g_tol)
                and d_v_corr < float(controls.v_corr_tol)
                and v_residual < float(controls.v_corr_tol)
                and (not refine_enabled or inner_refined)):
            converged = True
            break
        if (
            refine_enabled
            and not inner_refined
            and _needs_inner_precision(normalized_residuals)
        ):
            inner_refined = True

    feedback_meta = {
        "structure_model": "SC",
        "mixture_extension": bool(len(entries) > 1),
        "converged": bool(converged),
        "iterations": int(len(history)),
        "fixed_is_mu_ha": float(fixed_mu),
        "field_free_density_definition": "ideal_fermi_gas_at_fixed_is_mu_and_temperature",
        "ideal_common_n0_bohr3": float(n0_common),
        "volume_weights": np.asarray(weights, dtype=float).tolist(),
        "g_tol": float(controls.g_tol),
        "v_corr_tol_ha": float(controls.v_corr_tol),
        "v_corr_mix": float(controls.v_corr_mix),
        "v_corr_scale": float(controls.v_corr_scale),
        "inner_full_tol_scale": controls.inner_full_tol_scale,
        "potential_convergence_metric": "unmixed_current_output_minus_used_input",
        "electronic_model": str(workflow_cfg.electronic_model),
        "reference": (
            "C. E. Starrett and D. Saumon, High Energy Density Physics "
            "10, 35-42 (2014), Sec. 2.4, Eqs. (19)-(20), "
            "doi:10.1016/j.hedp.2013.12.001"
        ),
        "v_corr_r_bohr": (
            None if old_v_r is None else np.asarray(old_v_r, dtype=float)
        ),
        "v_corr_species_ha": (
            None
            if old_v_corr is None
            else np.asarray(old_v_corr, dtype=float)
        ),
        "history": history,
    }
    previous["structure_model"] = "SC"
    previous["sc_feedback"] = feedback_meta
    if previous.get("ion", None) is not None:
        previous["ion"]["structure_model"] = "SC"
        previous["ion"]["sc_feedback"] = feedback_meta
    if bool(controls.require_converged) and not converged:
        raise SCFeedbackConvergenceError(
            "SC feedback did not converge within "
            f"{int(controls.max_outer)} iterations: "
            f"max|dg|={float(history[-1]['max_g_change']):.3e}, "
            f"max|dVcorr|={float(history[-1]['max_v_corr_change_ha']):.3e} Ha; "
            "max|Vcorr_out-Vcorr_in|="
            f"{float(history[-1]['max_v_corr_residual_ha']):.3e} Ha.",
            previous,
        )
    return previous


__all__ = [
    "SCFeedbackConfig",
    "SCFeedbackConvergenceError",
    "estimate_mixture_correlation_potentials",
    "mixture_ionic_background_profiles",
    "solve_sc_feedback_workflow",
]
