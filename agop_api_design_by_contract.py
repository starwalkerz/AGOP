"""
agop_api_design_by_contract.py — FastAPI boundary for the DbC AGOP core.

Imports from agop_core_design_by_contract (not from agop_core) so that:
  * capacity K is derived endogenously inside agop(),
  * the verifies_theorem_5 decorator runs on every call,
  * a contract breach surfaces as HTTP 422, not 500.

K is absent from ThetaSpec by design. Extra fields are rejected, so a
client cannot inject K accidentally or intentionally.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import networkx as nx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

# The DbC core, NOT agop_core.
from agop_core_design_by_contract import (
    Theta,
    Problem,
    State,
    agop,
    reference_instance,
    _shannon_entropy,
)


EXECUTOR = ProcessPoolExecutor(max_workers=4)


async def _run_agop(problem: Problem) -> State:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(EXECUTOR, agop, problem)


# ─────────────────────────────────────────────────────────────────────
# Schemas — K is not a valid theta field
# ─────────────────────────────────────────────────────────────────────


class ThetaSpec(BaseModel):
    model_config = {"extra": "forbid"}

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
    h: int = Field(default=3, ge=1, le=20)

    L_R: dict[str, tuple[float, float]]
    L_C: dict[str, tuple[float, float]]


class ProblemSpec(BaseModel):
    model_config = {"extra": "forbid"}

    V: list[str]
    edges: list[tuple[str, str, int]]
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
# Conversion helpers
# ─────────────────────────────────────────────────────────────────────


def spec_to_problem(spec: ProblemSpec) -> Problem:
    if set(spec.theta.L_R.keys()) != set(spec.V):
        raise ValueError("L_R must cover every member of V")
    if set(spec.theta.L_C.keys()) != set(spec.V):
        raise ValueError("L_C must cover every member of V")

    G = nx.DiGraph()
    for v in spec.V:
        G.add_node(v)
    for u, v, sign in spec.edges:
        if u not in G or v not in G:
            raise ValueError(f"edge references unknown vertex: ({u}, {v})")
        G.add_edge(u, v, sign=int(sign))

    theta = Theta(
        c_b=spec.theta.c_b,
        c_s=spec.theta.c_s,
        w=spec.theta.w,
        ell=spec.theta.ell,
        v_bar=spec.theta.v_bar,
        theta_1=spec.theta.theta_1,
        theta_2=spec.theta.theta_2,
        theta_3=spec.theta.theta_3,
        sigma_R=spec.theta.sigma_R,
        sigma_B=spec.theta.sigma_B,
        alpha_1=spec.theta.alpha_1,
        alpha_2=spec.theta.alpha_2,
        alpha_3=spec.theta.alpha_3,
        M=spec.theta.M,
        alpha=spec.theta.alpha,
        seed=spec.theta.seed,
        h=spec.theta.h,
        L_R={k: tuple(v) for k, v in spec.theta.L_R.items()},
        L_C={k: tuple(v) for k, v in spec.theta.L_C.items()},
    )
    return Problem(V=list(spec.V), axes=spec.axes, G=G, Theta=theta)


def resolve_problem(req: AgopRequest) -> Problem:
    if req.problem is not None:
        return spec_to_problem(req.problem)
    if req.preset in (None, "reference"):
        return reference_instance()
    raise HTTPException(status_code=400, detail=f"unknown preset: {req.preset!r}")


def state_to_dict(state: State) -> dict:
    sigma_DM, sigma_CP = state.e_1
    return {
        "ranking": state.R,
        "D_star": sorted(state.D_star),
        "convergence_depth": state.convergence_depth,

        "n_wl_classes": state.n_wl_classes,
        "nash_entropy": {
            "H_sigma_DM": _shannon_entropy(sigma_DM),
            "H_sigma_CP": _shannon_entropy(sigma_CP),
        },
        "K_derived": state.K_derived,

        "gamma": {k: round(float(v), 6) for k, v in state.gamma.items()},
        "payoff": {
            "U_DM": state.Pi[:, :, 0].round(4).tolist(),
            "U_CP": state.Pi[:, :, 1].round(4).tolist(),
        },
        "nash": {
            "sigma_DM": sigma_DM.round(6).tolist(),
            "sigma_CP": sigma_CP.round(6).tolist(),
        },
        "systems": state.Sigma,
        "topology": state.T,
        "adoption": state.nu_star,
        "monte_carlo": {
            "Y_hat": {k: round(float(v), 4) for k, v in state.Y_hat.items()},
            "LCB": {k: round(float(v), 4) for k, v in state.LCB.items()},
        },
        "resonance": {k: round(float(v), 6) for k, v in state.A.items()},
        "composite": {k: round(float(v), 4) for k, v in state.Phi.items()},

        "certified": [
            {
                "alternative": c.alternative,
                "lcb": round(c.lcb, 4),
                "nu_star": c.nu_star,
                "phi": round(c.phi, 4),
                "family": c.family,
            }
            for c in state.certified()
        ],
    }


# ─────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────


app = FastAPI(
    title="AGOP Decision Engine (Design-by-Contract)",
    description=(
        "Capacity K is derived endogenously from WL class cardinality and "
        "Nash entropy. The verifies_theorem_5 decorator enforces Theorem 2 "
        "and Theorem 5 on every call. A contract breach returns HTTP 422."
    ),
    version="1.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UI_FILE = Path(__file__).parent / "agop_ui_design_by_contract.html"


@app.get("/", include_in_schema=False)
async def root():
    if UI_FILE.exists():
        return FileResponse(UI_FILE, media_type="text/html")
    return {"message": "AGOP DbC API running. See /docs for the OpenAPI UI."}


@app.get("/api/health")
async def health():
    return {"status": "ok", "core": "design_by_contract", "version": "1.3.0"}


@app.post("/api/agop")
async def api_agop(req: AgopRequest):
    try:
        problem = resolve_problem(req)
        state = await _run_agop(problem)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AssertionError as e:
        raise HTTPException(status_code=422, detail=f"Contract violated: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return state_to_dict(state)


@app.post("/api/agop/chart")
async def api_agop_chart(req: AgopRequest):
    try:
        from agop_charts import render_chart
        problem = resolve_problem(req)
        state = await _run_agop(problem)
        png = render_chart(problem, state)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AssertionError as e:
        raise HTTPException(status_code=422, detail=f"Contract violated: {e}")
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="agop_charts module not installed; chart endpoint unavailable",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return Response(content=png, media_type="image/png")


@app.get("/api/agop/sse")
async def api_agop_sse():
    async def event_stream():
        problem = reference_instance()
        phases = [
            ("phase_1", "centrality"),
            ("phase_3", "market_nash"),
            ("phase_5", "topology"),
            ("phase_6", "deductive"),
            ("phase_7", "adoption_nash"),
            ("phase_8", "monte_carlo"),
            ("phase_11", "ranking"),
        ]
        for name, key in phases:
            try:
                state = await _run_agop(problem)
                payload = {"phase": name, "key": key, "data": state_to_dict(state)}
                yield f"data: {json.dumps(payload)}\n\n"
            except AssertionError as e:
                yield f"event: contract_violation\ndata: {json.dumps({'detail': str(e)})}\n\n"
                return
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'detail': f'{type(e).__name__}: {e}'})}\n\n"
                return
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.websocket("/ws/agop/stream")
async def ws_agop_stream(ws: WebSocket):
    await ws.accept()
    problem = reference_instance()
    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")
            if mtype == "axis_update":
                for axis, values in msg.get("axes", {}).items():
                    problem.axes.setdefault(axis, {}).update(values)
            elif mtype == "reset":
                problem = reference_instance()
            elif mtype == "ping":
                await ws.send_json({"pong": True})
                continue

            try:
                state = await _run_agop(problem)
            except AssertionError as e:
                await ws.send_json({
                    "contract_violated": str(e),
                    "K_derived": None,
                    "ranking": [],
                })
                continue

            await ws.send_json({
                "ranking": state.R,
                "D_star": sorted(state.D_star),
                "K_derived": state.K_derived,
                "n_wl_classes": state.n_wl_classes,
                "nash_entropy": {
                    "H_sigma_DM": _shannon_entropy(state.e_1[0]),
                    "H_sigma_CP": _shannon_entropy(state.e_1[1]),
                },
                "composite": {k: float(v) for k, v in state.Phi.items()},
            })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"error": f"{type(e).__name__}: {e}"})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
