"""
AGOP REST API (FastAPI).

Endpoints:
  GET  /              serves the browser UI (agop_ui.html)
  GET  /api/health    healthcheck
  POST /api/agop      runs AGOP, returns the full JSON trace
  POST /api/agop/chart runs AGOP, returns image/png

Run with:
  python run.py
  # or
  uvicorn agop_api:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import networkx as nx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, field_validator

from agop_core import Theta, Problem, State, agop, reference_instance
from agop_charts import render_chart


# ─────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────


class ThetaSpec(BaseModel):
    c_b: float = 10.0
    c_s: float = 3.0
    w: float = 15.0
    ell: float = 2.0
    v_bar: float = 20.0
    theta_1: float = 0.5
    theta_2: float = 0.5
    theta_3: float = 0.5
    sigma_R: float = 0.2
    sigma_B: float = 0.2
    alpha_1: float = 0.4
    alpha_2: float = 0.4
    alpha_3: float = 0.2
    M: int = Field(default=10_000, ge=100, le=1_000_000)
    alpha: float = Field(default=0.05, gt=0.0, lt=0.5)
    seed: int = 42
    K: int = Field(default=3, ge=1)
    h: int = Field(default=3, ge=1, le=20)
    L_R: dict[str, tuple[float, float]]
    L_C: dict[str, tuple[float, float]]


class ProblemSpec(BaseModel):
    V: list[str]
    edges: list[tuple[str, str, int]]
    families: dict[str, str]
    axes: dict[str, dict[str, float]]
    theta: ThetaSpec

    @field_validator("V")
    @classmethod
    def _unique(cls, v):
        if len(v) != len(set(v)):
            raise ValueError("V must not contain duplicates")
        return v

    @field_validator("axes")
    @classmethod
    def _required_axes(cls, v):
        required = {"pain", "integration", "regulatory", "readiness", "barrier"}
        missing = required - v.keys()
        if missing:
            raise ValueError(f"missing axes: {sorted(missing)}")
        return v


class AgopRequest(BaseModel):
    preset: Optional[str] = "reference"
    problem: Optional[ProblemSpec] = None


# ─────────────────────────────────────────────────────────────────────
# Conversion
# ─────────────────────────────────────────────────────────────────────


def spec_to_problem(spec: ProblemSpec) -> Problem:
    if set(spec.theta.L_R.keys()) != set(spec.V):
        raise ValueError("L_R must cover every member of V")
    if set(spec.theta.L_C.keys()) != set(spec.V):
        raise ValueError("L_C must cover every member of V")

    G = nx.DiGraph()
    for v in spec.V:
        G.add_node(v, family=spec.families.get(v, "other"))
    for u, v, sign in spec.edges:
        G.add_edge(u, v, sign=int(sign))

    theta = Theta(
        c_b=spec.theta.c_b, c_s=spec.theta.c_s, w=spec.theta.w,
        ell=spec.theta.ell, v_bar=spec.theta.v_bar,
        theta_1=spec.theta.theta_1, theta_2=spec.theta.theta_2,
        theta_3=spec.theta.theta_3,
        sigma_R=spec.theta.sigma_R, sigma_B=spec.theta.sigma_B,
        alpha_1=spec.theta.alpha_1, alpha_2=spec.theta.alpha_2,
        alpha_3=spec.theta.alpha_3,
        M=spec.theta.M, alpha=spec.theta.alpha, seed=spec.theta.seed,
        K=spec.theta.K, h=spec.theta.h,
        L_R={k: tuple(v) for k, v in spec.theta.L_R.items()},
        L_C={k: tuple(v) for k, v in spec.theta.L_C.items()},
    )

    return Problem(V=list(spec.V), axes=spec.axes, G=G, Theta=theta)


def resolve_problem(req: AgopRequest) -> Problem:
    if req.problem is not None:
        return spec_to_problem(req.problem)
    if req.preset == "reference" or req.preset is None:
        return reference_instance()
    raise HTTPException(status_code=400, detail=f"unknown preset: {req.preset!r}")


def state_to_dict(state: State) -> dict:
    return {
        "ranking": state.R,
        "D_star": sorted(state.D_star),
        "convergence_depth": state.convergence_depth,
        "gamma": {k: round(float(v), 6) for k, v in state.gamma.items()},
        "payoff": {
            "U_DM": state.Pi[:, :, 0].round(4).tolist(),
            "U_CP": state.Pi[:, :, 1].round(4).tolist(),
        },
        "nash": {
            "sigma_DM": state.e_1[0].round(6).tolist(),
            "sigma_CP": state.e_1[1].round(6).tolist(),
        },
        "systems": state.Sigma,
        "topology": state.T,
        "adoption": state.nu_star,
        "monte_carlo": {
            "Y_hat": {k: round(float(v), 4) for k, v in state.Y_hat.items()},
            "LCB":   {k: round(float(v), 4) for k, v in state.LCB.items()},
        },
        "resonance": {k: round(float(v), 6) for k, v in state.A.items()},
        "composite": {k: round(float(v), 4) for k, v in state.Phi.items()},
    }


# ─────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────


app = FastAPI(
    title="AGOP Decision Engine",
    description=(
        "A computable decision procedure for multi-criteria selection under "
        "uncertainty. See the accompanying README for the full specification."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UI_FILE = Path(__file__).parent / "agop_ui.html"


@app.get("/", include_in_schema=False)
def root():
    if not UI_FILE.exists():
        return {"message": "AGOP API is running. See /docs for the OpenAPI UI."}
    return FileResponse(UI_FILE, media_type="text/html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/agop")
def api_agop(req: AgopRequest):
    try:
        problem = resolve_problem(req)
        state = agop(problem)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return state_to_dict(state)


@app.post("/api/agop/chart")
def api_agop_chart(req: AgopRequest):
    try:
        problem = resolve_problem(req)
        state = agop(problem)
        png = render_chart(problem, state)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return Response(content=png, media_type="image/png")