"""
Publication-quality PNG rendering of AGOP results.

Four panels in a 2×2 grid:
  1. Phase funnel (bar chart)
  2. Resonance scatter (readiness vs. barrier, target marked)
  3. Composite score bars (Φ, sorted, Top-K highlighted)
  4. Payoff matrix heatmap (U_DM, U_CP annotations, equilibrium highlighted)

Returns PNG bytes; no state is written to disk.
"""

from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from agop_core import Problem, State


# Palette
COLOR_PRIMARY   = "#0a4d8c"
COLOR_SECONDARY = "#2d6a2d"
COLOR_MUTED     = "#b8b8b8"
COLOR_TARGET    = "#d97706"
COLOR_EQUIL     = "#0a4d8c"

FONT_FAMILY = "DejaVu Sans"


# ─────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────


def phase_funnel_counts(problem: Problem, state: State) -> list[tuple[str, int]]:
    V0 = set(problem.V)
    theta = problem.Theta
    after_pain = {v for v in V0 if problem.axes["pain"][v] >= theta.theta_1}
    after_integration = {
        v for v in after_pain
        if problem.axes["integration"][v] >= theta.theta_2
    }
    after_regulatory = {
        v for v in after_integration
        if problem.axes["regulatory"][v] <= theta.theta_3
    }
    after_lcb = {v for v in after_regulatory if state.LCB.get(v, -1.0) >= 0}
    top_k = set(state.R)
    return [
        ("Input V_0",       len(V0)),
        ("Pain (f_1)",      len(after_pain)),
        ("Integration (f_2)", len(after_integration)),
        ("Regulatory (f_3)", len(after_regulatory)),
        ("LCB filter",      len(after_lcb)),
        ("Top-K (R)",       len(top_k)),
    ]


# ─────────────────────────────────────────────────────────────────────
# Panel renderers
# ─────────────────────────────────────────────────────────────────────


def _plot_phase_funnel(ax, counts):
    labels = [l for l, _ in counts]
    values = [v for _, v in counts]

    y = list(range(len(labels)))
    colors = [COLOR_MUTED] * len(labels)
    colors[-1] = COLOR_SECONDARY

    ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    for yi, v in zip(y, values):
        ax.text(v + 0.15, yi, str(v), va="center", fontsize=10)

    ax.set_xlim(0, max(values) * 1.15)
    ax.set_xlabel("number of alternatives")
    ax.set_title("Phase funnel", fontsize=12, fontweight="bold", loc="left")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)


def _plot_resonance(ax, problem, state):
    for v in sorted(state.D_star):
        r = problem.axes["readiness"][v]
        b = problem.axes["barrier"][v]
        phi = state.Phi.get(v, 0.0)
        color = COLOR_SECONDARY if v in state.R else COLOR_MUTED
        ax.scatter(
            r, b,
            s=max(80, phi * 22),
            c=color, edgecolors="black", linewidths=0.8,
            alpha=0.85, zorder=3,
        )
        ax.annotate(
            v, (r, b),
            xytext=(6, 6), textcoords="offset points",
            fontsize=9, color="black",
        )

    r_star = max(problem.axes["readiness"][v] for v in state.D_star)
    b_star = min(problem.axes["barrier"][v]   for v in state.D_star)
    ax.scatter(
        r_star, b_star,
        marker="*", s=420,
        c=COLOR_TARGET, edgecolors="black", linewidths=0.8,
        zorder=4, label="target (r*, b*)",
    )

    ax.set_xlabel("readiness")
    ax.set_ylabel("barrier (lower is better)")
    ax.set_title("Resonance: readiness vs barrier",
                 fontsize=12, fontweight="bold", loc="left")
    ax.invert_yaxis()
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9, frameon=True)


def _plot_composite(ax, state):
    items = sorted(state.Phi.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals  = [v for _, v in items]
    colors = [COLOR_SECONDARY if n in state.R else COLOR_MUTED for n in names]

    y = list(range(len(names)))
    ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()

    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.02, yi, f"{v:.2f}", va="center", fontsize=9)

    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_xlabel("Φ = A · LCB(Ŷ)")
    ax.set_title("Composite score (Top-K highlighted)",
                 fontsize=12, fontweight="bold", loc="left")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)


def _plot_payoff_matrix(ax, state):
    U_DM = state.Pi[:, :, 0]
    U_CP = state.Pi[:, :, 1]
    total = U_DM + U_CP

    vmax = max(abs(total).max(), 1.0)
    im = ax.imshow(total, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    for i in (0, 1):
        for j in (0, 1):
            ax.text(
                j, i,
                f"({U_DM[i, j]:.1f}, {U_CP[i, j]:.1f})",
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="black",
            )

    i_eq = int(state.e_1[0].argmax())
    j_eq = int(state.e_1[1].argmax())
    rect = patches.Rectangle(
        (j_eq - 0.5, i_eq - 0.5), 1, 1,
        fill=False, edgecolor=COLOR_EQUIL, linewidth=3.0,
    )
    ax.add_patch(rect)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CP: adopt", "CP: demand"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["DM: bespoke", "DM: standard"])
    ax.set_title("Payoff matrix Π = (u_DM, u_CP); total = colour",
                 fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel("equilibrium highlighted with border")


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


def render_chart(problem: Problem, state: State,
                 figsize: tuple[float, float] = (15.0, 10.0),
                 dpi: int = 110) -> bytes:
    fig = Figure(figsize=figsize, dpi=dpi, facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.24,
                          left=0.07, right=0.97, top=0.90, bottom=0.08)

    _plot_phase_funnel(fig.add_subplot(gs[0, 0]), phase_funnel_counts(problem, state))
    _plot_resonance(fig.add_subplot(gs[0, 1]), problem, state)
    _plot_composite(fig.add_subplot(gs[1, 0]), state)
    _plot_payoff_matrix(fig.add_subplot(gs[1, 1]), state)

    theta = problem.Theta
    subtitle = (f"|V_0| = {len(problem.V)}   "
                f"D* = {len(state.D_star)}   "
                f"R = {len(state.R)}   "
                f"M = {theta.M}   "
                f"α = {theta.alpha}   "
                f"K = {theta.K}   "
                f"seed = {theta.seed}")

    fig.suptitle(
        f"AGOP Decision Analysis — ranking: "
        f"{' > '.join(state.R)}",
        fontsize=15, fontweight="bold", y=0.965,
    )
    fig.text(0.5, 0.925, subtitle, ha="center", fontsize=10, color="#444")

    buf = io.BytesIO()
    FigureCanvasAgg(fig).print_png(buf)
    return buf.getvalue()