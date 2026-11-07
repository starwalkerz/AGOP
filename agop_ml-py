"""
agop_ml.py — Bayesian calibration of AGOP parameters from historical decisions.

Vulnerability addressed: "Radical sensitivity to unlearned priors."
Historical decision episodes are used to learn a MAP estimate of the axis
values, replacing hand-coded priors with posterior means. A Laplace
approximation supplies a posterior standard deviation so downstream consumers
know how much confidence to place in the calibrated problem.

Model:
    Prior:     x_j(v) ~ N(mu0_j(v), sigma_prior^2)          for each axis j
    Likelihood: P(chosen = v | x) = softmax(beta * Phi(v; x))[v]
    Posterior:  approximated by Laplace at the MAP point.

Dependencies: numpy, scipy.  No external ML framework required.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np
from scipy import optimize

from agop_core import Problem, Theta, agop, reference_instance


# ─────────────────────────────────────────────────────────────────────
# Episode schema
# ─────────────────────────────────────────────────────────────────────


@dataclass
class Episode:
    """One historical decision observed in the field.

    Attributes
    ----------
    alternatives : list[str]
        Identifiers available at decision time.
    axes : dict[str, dict[str, float]]
        Observed axis values at decision time. Must cover every alternative
        in ``alternatives`` for every axis in ``axis_keys``.
    chosen : str
        The alternative that was actually chosen (must be in ``alternatives``).
    outcome : float, optional
        Realized decision value Y, if observable. Not used by the default
        likelihood but available to downstream extensions.
    """
    alternatives: list[str]
    axes: dict[str, dict[str, float]]
    chosen: str
    outcome: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────
# Internal: softmax and per-episode log-likelihood
# ─────────────────────────────────────────────────────────────────────


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def _episode_log_likelihood(episode: Episode,
                            problem: Problem,
                            beta: float) -> float:
    """log P(chosen | axes, problem) under a softmax over Phi."""
    V = episode.alternatives
    if episode.chosen not in V:
        return -1e300  # infeasible choice

    sub_axes = {k: {v: episode.axes[k][v] for v in V} for k in episode.axes}
    sub_G = problem.G.subgraph(V).copy()
    sub_theta = replace(
        problem.Theta,
        L_R={v: problem.Theta.L_R[v] for v in V},
        L_C={v: problem.Theta.L_C[v] for v in V},
    )
    sub_problem = Problem(V=V, axes=sub_axes, G=sub_G, Theta=sub_theta)

    state = agop(sub_problem)
    logits = np.array([state.Phi.get(v, -1e6) for v in V])
    p = _softmax(beta * logits)

    idx = V.index(episode.chosen)
    return float(np.log(max(p[idx], 1e-300)))


# ─────────────────────────────────────────────────────────────────────
# Numerical Hessian for the Laplace approximation
# ─────────────────────────────────────────────────────────────────────


def _numerical_hessian(f, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    n = len(x)
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            xi = x.copy(); xi[i] += eps
            xj = x.copy(); xj[j] += eps
            xij = x.copy(); xij[i] += eps; xij[j] += eps
            H[i, j] = (f(xij) - f(xi) - f(xj) + f0) / (eps * eps)
            H[j, i] = H[i, j]
    return H


# ─────────────────────────────────────────────────────────────────────
# Public calibration entry point
# ─────────────────────────────────────────────────────────────────────


def calibrate(
    episodes: Sequence[Episode],
    base_problem: Optional[Problem] = None,
    beta: float = 1.0,
    sigma_prior: float = 0.15,
    axis_keys: Sequence[str] = (
        "pain", "integration", "regulatory", "readiness", "barrier"
    ),
    M_during_optimization: int = 500,
    compute_hessian: bool = True,
) -> tuple[Problem, dict]:
    """Return a Problem whose axis values are MAP-calibrated against ``episodes``.

    Parameters
    ----------
    episodes : sequence of Episode
    base_problem : Problem, optional
        Starting point and prior mean. Defaults to ``reference_instance()``.
    beta : float
        Softmax inverse temperature. 1.0 = neutral; higher = more deterministic.
    sigma_prior : float
        Prior standard deviation on each axis value, in [0, 1].
    axis_keys : sequence of str
        Which axes to calibrate. Others are held at the base values.
    M_during_optimization : int
        Monte Carlo samples used inside the optimization loop. Reduced from
        the base value to keep L-BFGS tractable. The returned Problem uses
        the base M for downstream evaluation.
    compute_hessian : bool
        Whether to compute the Laplace approximation (O(n_params^2) AGOP runs).
        Set to False for fast iteration; True for production calibrations.

    Returns
    -------
    (calibrated, meta) : tuple[Problem, dict]
        ``calibrated`` is a Problem with MAP-calibrated axes and all other
        fields inherited from ``base_problem``.
        ``meta`` carries diagnostic fields: sigma_mean, sigma_max, n_episodes,
        n_alternatives_in_calibration, map_params, converged.
    """
    if base_problem is None:
        base_problem = reference_instance()
    if not episodes:
        return base_problem, {"error": "no episodes", "n_episodes": 0}

    V_all = sorted({v for e in episodes for v in e.alternatives})
    if not V_all:
        return base_problem, {"error": "empty alternatives", "n_episodes": len(episodes)}

    # Prior means from the base problem's axes, restricted to V_all
    mu0 = {k: np.array([base_problem.axes[k][v] for v in V_all]) for k in axis_keys}

    def unpack(x: np.ndarray) -> dict[str, np.ndarray]:
        out, off = {}, 0
        for k in axis_keys:
            out[k] = x[off:off + len(V_all)]
            off += len(V_all)
        return out

    # Reduced-M Theta for use inside the optimization loop
    base_theta_reduced = replace(base_problem.Theta, M=M_during_optimization)

    def build_problem_from_x(x: np.ndarray) -> Problem:
        axes_cal = unpack(x)
        updated_axes = {
            **base_problem.axes,
            **{
                k: {**base_problem.axes[k],
                    **dict(zip(V_all, axes_cal[k]))}
                for k in axis_keys
            },
        }
        return replace(base_problem,
                       axes=updated_axes,
                       Theta=base_theta_reduced)

    def neg_log_posterior(x: np.ndarray) -> float:
        problem = build_problem_from_x(x)
        prior_vec = np.concatenate([mu0[k] for k in axis_keys])
        log_prior = -0.5 * np.sum((x - prior_vec) ** 2) / (sigma_prior ** 2)
        log_like = sum(
            _episode_log_likelihood(e, problem, beta) for e in episodes
        )
        return -(log_prior + log_like)

    x0 = np.concatenate([mu0[k] for k in axis_keys])
    bounds = [(0.0, 1.0)] * len(x0)

    res = optimize.minimize(
        neg_log_posterior, x0, method="L-BFGS-B",
        bounds=bounds, options={"maxiter": 300, "ftol": 1e-9},
    )

    # Build the returned Problem at the MAP point, restoring the base M
    axes_cal = unpack(res.x)
    calibrated = replace(
        base_problem,
        axes={
            **base_problem.axes,
            **{
                k: {**base_problem.axes[k],
                    **dict(zip(V_all, axes_cal[k]))}
                for k in axis_keys
            },
        },
        Theta=base_problem.Theta,  # keep base M and all other Theta fields
    )

    # Laplace approximation
    sigma_mean, sigma_max = None, None
    if compute_hessian:
        try:
            H = _numerical_hessian(neg_log_posterior, res.x)
            cov = np.linalg.pinv(H)
            diag = np.diag(cov)
            diag = diag[diag > 0]
            if diag.size:
                sigma_mean = float(np.sqrt(diag.mean()))
                sigma_max = float(np.sqrt(diag.max()))
        except np.linalg.LinAlgError:
            pass  # leave sigma as None on numerical failure

    meta = {
        "n_episodes": len(episodes),
        "n_alternatives_in_calibration": len(V_all),
        "sigma_mean": sigma_mean,
        "sigma_max": sigma_max,
        "map_params": res.x.tolist(),
        "converged": bool(res.success),
        "final_nll": float(res.fun),
    }
    return calibrated, meta
