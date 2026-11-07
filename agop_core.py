"""
AGOP: Attributed-Graph Operator Pipeline — reference implementation.

Implements the algorithm specified in:
  "AGOP: A Computable Decision Procedure for Multi-Criteria Selection Under Uncertainty"

Phases 0..11 correspond to §3 of the article.
Operators correspond to §4. Conventions C1..C6 correspond to §2.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import networkx as nx
import numpy as np
from scipy import stats


# ─────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Theta:
    """Parameter vector Θ (§1.1, Appendix A)."""
    c_b: float
    c_s: float
    w: float
    ell: float
    v_bar: float

    theta_1: float
    theta_2: float
    theta_3: float

    sigma_R: float
    sigma_B: float

    alpha_1: float
    alpha_2: float
    alpha_3: float

    M: int
    alpha: float
    seed: int

    K: int
    h: int

    orthogonalize: bool = False
    delta: float = 0.1

    L_R: dict[str, tuple[float, float]] = field(default_factory=dict)
    L_C: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class Problem:
    V: list[str]
    axes: dict[str, dict[str, float]]
    G: nx.DiGraph
    Theta: Theta


@dataclass
class State:
    G: nx.DiGraph
    gamma: dict[str, float] = field(default_factory=dict)
    Pi: Optional[np.ndarray] = None
    e_1: Optional[tuple[np.ndarray, np.ndarray]] = None
    Sigma: dict[str, Any] = field(default_factory=dict)
    T: dict[str, str] = field(default_factory=dict)
    D_star: set[str] = field(default_factory=set)
    convergence_depth: int = 0
    nu_star: dict[str, int] = field(default_factory=dict)
    Y_hat: dict[str, float] = field(default_factory=dict)
    LCB: dict[str, float] = field(default_factory=dict)
    A: dict[str, float] = field(default_factory=dict)
    Phi: dict[str, float] = field(default_factory=dict)
    R: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Phase 1: Centrality C (§4.2)
# ─────────────────────────────────────────────────────────────────────


def phase_1_centrality(G: nx.DiGraph, theta: Theta) -> dict[str, float]:
    n = G.number_of_nodes()
    if n < 2:
        return {v: 0.0 for v in G.nodes}

    out_deg = {v: G.out_degree(v) / (n - 1) for v in G.nodes}
    betweenness = nx.betweenness_centrality(G, normalized=True)

    try:
        eig = nx.eigenvector_centrality_numpy(G)
    except Exception:
        try:
            eig = nx.eigenvector_centrality(G, max_iter=1000)
        except Exception:
            eig = {v: float(G.degree(v)) for v in G.nodes}

    eig_max = max(eig.values()) if eig else 1.0
    if eig_max <= 0:
        eig_max = 1.0
    eig_norm = {v: eig[v] / eig_max for v in G.nodes}

    return {
        v: theta.alpha_1 * out_deg[v]
         + theta.alpha_2 * betweenness.get(v, 0.0)
         + theta.alpha_3 * eig_norm[v]
        for v in G.nodes
    }


# ─────────────────────────────────────────────────────────────────────
# Phase 2: Payoff Γ (§4.3)
# ─────────────────────────────────────────────────────────────────────


def phase_2_payoff(theta: Theta) -> np.ndarray:
    # Action indexing:
    #   DM: 0 = bespoke, 1 = standard
    #   CP: 0 = adopt,   1 = demand
    U_DM = np.array([
        [theta.w - theta.c_b,                 -theta.c_b],
        [theta.w - theta.c_s,                 -theta.c_s],
    ], dtype=float)
    U_CP = np.array([
        [theta.v_bar - theta.w,                0.0],
        [theta.v_bar - theta.w - theta.ell,    0.0],
    ], dtype=float)
    return np.stack([U_DM, U_CP], axis=-1)  # shape (2, 2, 2)


# ─────────────────────────────────────────────────────────────────────
# Phase 3: Market Nash N_1 (§4.4)
# ─────────────────────────────────────────────────────────────────────


def _nash_2x2(U_DM: np.ndarray, U_CP: np.ndarray, tol: float = 1e-9
              ) -> list[tuple[np.ndarray, np.ndarray]]:
    equilibria = []

    for i in (0, 1):
        for j in (0, 1):
            if (U_DM[i, j] >= U_DM[1 - i, j] - tol
                    and U_CP[i, j] >= U_CP[i, 1 - j] - tol):
                sD = np.zeros(2); sD[i] = 1.0
                sC = np.zeros(2); sC[j] = 1.0
                equilibria.append((sD, sC))

    denom_q = U_DM[0, 0] - U_DM[0, 1] - U_DM[1, 0] + U_DM[1, 1]
    denom_p = U_CP[0, 0] - U_CP[1, 0] - U_CP[0, 1] + U_CP[1, 1]
    if abs(denom_q) > tol and abs(denom_p) > tol:
        q = (U_DM[1, 1] - U_DM[0, 1]) / denom_q
        p = (U_CP[1, 1] - U_CP[1, 0]) / denom_p
        if tol < q < 1 - tol and tol < p < 1 - tol:
            equilibria.append((np.array([p, 1 - p]), np.array([q, 1 - q])))

    return equilibria


def _utilitarian_select(eq: list[tuple[np.ndarray, np.ndarray]],
                        U_DM: np.ndarray, U_CP: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    U_total = U_DM + U_CP

    def key(e):
        sD, sC = e
        total = float(sD @ U_total @ sC)
        return (-total, tuple(np.round(sD, 9)), tuple(np.round(sC, 9)))

    return sorted(eq, key=key)[0]


def phase_3_market_nash(Pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    U_DM = Pi[:, :, 0]
    U_CP = Pi[:, :, 1]
    eq = _nash_2x2(U_DM, U_CP)
    if not eq:
        raise RuntimeError("Nash existence theorem violated — no equilibrium found")
    return _utilitarian_select(eq, U_DM, U_CP)


# ─────────────────────────────────────────────────────────────────────
# Phase 4: Systems Σ (§4.5)
# ─────────────────────────────────────────────────────────────────────


def phase_4_systems(G: nx.DiGraph, gamma: dict[str, float]) -> dict[str, Any]:
    cycles = list(nx.simple_cycles(G))
    signs = []
    for cyc in cycles:
        sign = 1
        for u, v in zip(cyc, cyc[1:] + [cyc[0]]):
            s = G[u][v].get("sign", 1)
            sign *= int(np.sign(s)) if s != 0 else 0
        signs.append(sign)

    UG = G.to_undirected()
    if UG.number_of_nodes() <= 1:
        bottleneck_size = 0
        bottleneck_set: set[str] = set()
    else:
        best_k = float("inf")
        best_cut: set[str] = set()
        for comp in nx.connected_components(UG):
            sub = UG.subgraph(comp).copy()
            if sub.number_of_nodes() <= 1:
                continue
            try:
                k = nx.node_connectivity(sub)
                cut = nx.minimum_node_cut(sub)
                if k < best_k:
                    best_k = k
                    best_cut = cut
            except Exception:
                continue
        bottleneck_size = int(best_k) if best_k != float("inf") else 0
        bottleneck_set = best_cut

    return {
        "loop_count": len(cycles),
        "reinforcing": sum(1 for s in signs if s > 0),
        "balancing": sum(1 for s in signs if s < 0),
        "bottleneck_size": bottleneck_size,
        "bottleneck": sorted(bottleneck_set),
    }


# ─────────────────────────────────────────────────────────────────────
# Phase 5: Topology Θ_op (§4.6)
# ─────────────────────────────────────────────────────────────────────


def phase_5_topology(G: nx.DiGraph, axes: dict[str, dict[str, float]],
                     h: int) -> dict[str, str]:
    if all("family" in G.nodes[v] for v in G.nodes):
        return {v: G.nodes[v]["family"] for v in G.nodes}

    hashes = nx.weisfeiler_lehman_subgraph_hashes(G, iterations=h)
    raw = {v: hs[-1] if hs else "unknown" for v, hs in hashes.items()}
    label_map: dict[str, str] = {}
    for v in sorted(G.nodes):
        if raw[v] not in label_map:
            label_map[raw[v]] = f"T_{chr(ord('A') + len(label_map))}"
    return {v: label_map[raw[v]] for v in G.nodes}


# ─────────────────────────────────────────────────────────────────────
# Phase 6: Deductive Δ (§4.7)
# ─────────────────────────────────────────────────────────────────────


def phase_6_deductive(problem: Problem) -> tuple[set[str], int]:
    axes = problem.axes
    theta = problem.Theta

    def f1(S: set[str]) -> set[str]:
        return {v for v in S if axes["pain"].get(v, 0.0) >= theta.theta_1}

    def f2(S: set[str]) -> set[str]:
        return {v for v in S if axes["integration"].get(v, 0.0) >= theta.theta_2}

    def f3(S: set[str]) -> set[str]:
        return {v for v in S if axes["regulatory"].get(v, 1.0) <= theta.theta_3}

    def Phi(S: set[str]) -> set[str]:
        return f3(f2(f1(S)))

    D = set(problem.V)
    for k in range(len(problem.V) + 1):
        D_next = Phi(D)
        if D_next == D:
            return D, k
        D = D_next
    return D, len(problem.V)


# ─────────────────────────────────────────────────────────────────────
# Phase 7: Adoption Nash N_2 (§4.8)
# ─────────────────────────────────────────────────────────────────────


def phase_7_adoption_nash(D_star: set[str], problem: Problem) -> dict[str, int]:
    theta = problem.Theta
    pi = {v: theta.L_R[v][0] - theta.L_C[v][0] for v in D_star}
    ranked = sorted(D_star, key=lambda v: (-pi[v], v))
    topK = set(ranked[: theta.K])
    return {v: (1 if v in topK else 0) for v in D_star}


# ─────────────────────────────────────────────────────────────────────
# Phase 8: Monte Carlo M (§4.9)
# ─────────────────────────────────────────────────────────────────────


def phase_8_monte_carlo(D_star: set[str], problem: Problem
                        ) -> tuple[dict[str, float], dict[str, float]]:
    theta = problem.Theta
    rng = np.random.default_rng(theta.seed)
    z = float(stats.norm.ppf(1 - theta.alpha))

    Y_hat: dict[str, float] = {}
    LCB: dict[str, float] = {}

    for v in D_star:
        mu_R, sd_R = theta.L_R[v]
        mu_C, sd_C = theta.L_C[v]
        R = rng.normal(mu_R, sd_R, theta.M)
        C = rng.normal(mu_C, sd_C, theta.M)
        Y = R - C
        yhat = float(Y.mean())
        sigma = float(Y.std(ddof=1))
        lcb = yhat - z * sigma / np.sqrt(theta.M)
        Y_hat[v] = yhat
        LCB[v] = lcb

    return Y_hat, LCB


# ─────────────────────────────────────────────────────────────────────
# Phase 9: Resonance H (§4.10)
# ─────────────────────────────────────────────────────────────────────


def phase_9_resonance(D_star: set[str], problem: Problem) -> dict[str, float]:
    axes = problem.axes
    theta = problem.Theta

    r = {v: axes["readiness"][v] for v in D_star}
    b = {v: axes["barrier"][v] for v in D_star}

    r_star = max(r.values())
    b_star = min(b.values())

    A = {}
    for v in D_star:
        A[v] = float(np.exp(
            -((r[v] - r_star) ** 2) / (2 * theta.sigma_R ** 2)
            -((b[v] - b_star) ** 2) / (2 * theta.sigma_B ** 2)
        ))
    return A


# ─────────────────────────────────────────────────────────────────────
# Phases 10 & 11: Composite Φ + TopK (§4.11)
# ─────────────────────────────────────────────────────────────────────


def phase_10_11_rank(D_star: set[str],
                     problem: Problem,
                     LCB: dict[str, float],
                     A: dict[str, float],
                     ) -> tuple[dict[str, float], list[str]]:
    theta = problem.Theta
    D_plus = {v for v in D_star if LCB[v] >= 0}
    Phi = {v: A[v] * LCB[v] for v in D_plus}
    ranked = sorted(D_plus, key=lambda v: (-Phi[v], v))
    return Phi, ranked[: theta.K]


# ─────────────────────────────────────────────────────────────────────
# Top-level driver
# ─────────────────────────────────────────────────────────────────────


def agop(problem: Problem) -> State:
    s = State(G=problem.G.copy())

    s.gamma = phase_1_centrality(s.G, problem.Theta)
    s.Pi = phase_2_payoff(problem.Theta)
    s.e_1 = phase_3_market_nash(s.Pi)
    s.Sigma = phase_4_systems(s.G, s.gamma)
    s.T = phase_5_topology(s.G, problem.axes, problem.Theta.h)
    s.D_star, s.convergence_depth = phase_6_deductive(problem)
    s.nu_star = phase_7_adoption_nash(s.D_star, problem)
    s.Y_hat, s.LCB = phase_8_monte_carlo(s.D_star, problem)
    s.A = phase_9_resonance(s.D_star, problem)
    s.Phi, s.R = phase_10_11_rank(s.D_star, problem, s.LCB, s.A)

    return s


# ─────────────────────────────────────────────────────────────────────
# Reference instance
# ─────────────────────────────────────────────────────────────────────


def reference_instance() -> Problem:
    V = [f"S_{i}" for i in range(1, 14)]

    G = nx.DiGraph()
    G.add_nodes_from(V)

    families = {
        "S_2": "A", "S_4": "A", "S_6": "A", "S_9": "A", "S_11": "A",
        "S_1": "B",
        "S_10": "C", "S_12": "C",
        "S_3": "other", "S_5": "other", "S_7": "other",
        "S_8": "other", "S_13": "other",
    }
    for v in V:
        G.nodes[v]["family"] = families.get(v, "other")

    edges = [
        ("S_1", "S_2"),  ("S_2", "S_1"),
        ("S_1", "S_10"), ("S_10", "S_1"),
        ("S_2", "S_10"), ("S_10", "S_2"),
        ("S_2", "S_4"),  ("S_4", "S_2"),
        ("S_2", "S_6"),  ("S_6", "S_2"),
        ("S_2", "S_9"),  ("S_9", "S_2"),
        ("S_2", "S_11"), ("S_11", "S_2"),
        ("S_1", "S_11"), ("S_11", "S_1"),
        ("S_10", "S_12"), ("S_12", "S_10"),
        ("S_3", "S_12"),
        ("S_5", "S_7"),  ("S_7", "S_5"),
        ("S_7", "S_8"),  ("S_8", "S_7"),
        ("S_8", "S_13"), ("S_13", "S_8"),
    ]
    for u, v in edges:
        G.add_edge(u, v, sign=1)
    G["S_12"]["S_3"]["sign"] = -1

    axes: dict[str, dict[str, float]] = {
        "pain":         {v: 0.0 for v in V},
        "integration":  {v: 0.0 for v in V},
        "regulatory":   {v: 1.0 for v in V},
        "readiness":    {v: 0.5 for v in V},
        "barrier":      {v: 0.3 for v in V},
        "effort":       {v: 0.5 for v in V},
    }

    for v in ("S_1", "S_2", "S_10", "S_12"):
        axes["pain"][v] = 0.9
        axes["integration"][v] = 0.9
        axes["regulatory"][v] = 0.2

    for v in ("S_7", "S_13"):
        axes["pain"][v] = 0.8
        axes["integration"][v] = 0.3
        axes["regulatory"][v] = 0.3

    axes["readiness"].update(
        {"S_1": 0.85, "S_2": 0.90, "S_10": 0.80, "S_12": 0.95}
    )
    axes["barrier"].update(
        {"S_1": 0.15, "S_2": 0.20, "S_10": 0.10, "S_12": 0.05}
    )

    theta = Theta(
        c_b=10.0, c_s=3.0, w=15.0, ell=2.0, v_bar=20.0,
        theta_1=0.5, theta_2=0.5, theta_3=0.5,
        sigma_R=0.2, sigma_B=0.2,
        alpha_1=0.4, alpha_2=0.4, alpha_3=0.2,
        M=10_000, alpha=0.05, seed=42, K=3, h=3,
        L_R={
            "S_1":  (30.0, 5.0),
            "S_2":  (40.0, 6.0),
            "S_10": (25.0, 4.0),
            "S_12": (35.0, 5.0),
            "S_3":  (0.0, 1.0), "S_4":  (0.0, 1.0), "S_5":  (0.0, 1.0),
            "S_6":  (0.0, 1.0), "S_7":  (0.0, 1.0), "S_8":  (0.0, 1.0),
            "S_9":  (0.0, 1.0), "S_11": (0.0, 1.0), "S_13": (0.0, 1.0),
        },
        L_C={
            "S_1":  (6.0, 2.0),
            "S_2":  (9.0, 3.0),
            "S_10": (7.0, 2.0),
            "S_12": (8.0, 2.0),
            "S_3":  (0.0, 1.0), "S_4":  (0.0, 1.0), "S_5":  (0.0, 1.0),
            "S_6":  (0.0, 1.0), "S_7":  (0.0, 1.0), "S_8":  (0.0, 1.0),
            "S_9":  (0.0, 1.0), "S_11": (0.0, 1.0), "S_13": (0.0, 1.0),
        },
    )

    return Problem(V=V, axes=axes, G=G, Theta=theta)