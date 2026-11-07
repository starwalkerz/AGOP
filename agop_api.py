"""
agop_api.py — asynchronous, streaming, calibrating API for AGOP.

Changes vs. previous version:
  * All CPU-bound endpoints are `async def` and offload AGOP to a
    ProcessPoolExecutor, so the event loop never blocks.
  * WS /ws/agop/stream accepts continuous axis updates and pushes the
    ranking after each tick.
  * GET /api/agop/sse streams phase-by-phase results as Server-Sent Events.
  * POST /api/agop/calibrate ingests historical episodes, returns the
    calibrated Problem.
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
from pydantic import BaseModel, Field

from agop_core import Theta, Problem, State, agop, reference_instance
from agop_charts import render_chart
from agop_ml import Episode, calibrate


# ─────────────────────────────────────────────────────────────────────
# Executor: one process pool, reused across requests
# ─────────────────────────────────────────────────────────────────────

EXECUTOR = ProcessPoolExecutor(max_workers=4)


async def _run_agop(problem: Problem) -> State:
    """Run AGOP in a worker process without blocking the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(EXECUTOR, agop, problem)


# ─────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────


class ThetaSpec(BaseModel):
    # … identical to previous version, omitted for brevity …
    pass


class ProblemSpec(BaseModel):
    V: list[str]
    edges: list[tuple[str, str, int]]
    families: dict[str, str]
    axes: dict[str, dict[str, float]]
    theta: ThetaSpec


class AgopRequest(BaseModel):
    preset: Optional[str] = "reference"
    problem: Optional[ProblemSpec] = None


class EpisodeSpec(BaseModel):
    alternatives: list[str]
    axes: dict[str, dict[str, float]]
    chosen: str
    outcome: Optional[float] = None


class CalibrateRequest(BaseModel):
    episodes: list[EpisodeSpec]
    beta: float = 1.0
    sigma_prior: float = 0.15
    axis_keys: list[str] = Field(
        default=["pain", "integration", "regulatory", "readiness", "barrier"]
    )


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def spec_to_problem(spec: ProblemSpec) -> Problem:
    # … identical to previous version, omitted …
    pass


def resolve_problem(req: AgopRequest) -> Problem:
    if req.problem is not None:
        return spec_to_problem(req.problem)
    if req.preset in (None, "reference"):
        return reference_instance()
    raise HTTPException(status_code=400, detail=f"unknown preset: {req.preset!r}")


def state_to_dict(state: State) -> dict:
    # … identical to previous version, omitted …
    pass


# ─────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────


app = FastAPI(title="AGOP Decision Engine", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

UI_FILE = Path(__file__).parent / "agop_ui.html"


@app.get("/", include_in_schema=False)
async def root():
    if not UI_FILE.exists():
        return {"message": "AGOP API running. See /docs."}
    return FileResponse(UI_FILE, media_type="text/html")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/agop")
async def api_agop(req: AgopRequest):
    try:
        problem = resolve_problem(req)
        state = await _run_agop(problem)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return state_to_dict(state)


@app.post("/api/agop/chart")
async def api_agop_chart(req: AgopRequest):
    try:
        problem = resolve_problem(req)
        state = await _run_agop(problem)
        png = render_chart(problem, state)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return Response(content=png, media_type="image/png")


# ─────────────────────────────────────────────────────────────────────
# Streaming: SSE phase-by-phase results
# ─────────────────────────────────────────────────────────────────────


@app.get("/api/agop/sse")
async def api_agop_sse():
    """Stream the phase trace as Server-Sent Events."""
    async def event_stream():
        problem = reference_instance()
        phases = [
            ("phase_1", "centrality", problem),
            ("phase_3", "market_nash", problem),
            ("phase_6", "deductive", problem),
            ("phase_8", "monte_carlo", problem),
            ("phase_10", "composite", problem),
            ("phase_11", "ranking", problem),
        ]
        for name, key, p in phases:
            state = await _run_agop(p)
            payload = {"phase": name, "key": key,
                       "data": state_to_dict(state)}
            yield f"data: {json.dumps(payload)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─────────────────────────────────────────────────────────────────────
# Streaming: WebSocket continuous axis updates
# ─────────────────────────────────────────────────────────────────────


@app.websocket("/ws/agop/stream")
async def ws_agop_stream(ws: WebSocket):
    """Accept continuous axis updates; push the ranking after each tick."""
    await ws.accept()
    problem = reference_instance()
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "axis_update":
                for axis, values in msg["axes"].items():
                    problem.axes[axis].update(values)
            elif msg.get("type") == "reset":
                problem = reference_instance()
            state = await _run_agop(problem)
            await ws.send_json({"ranking": state.R,
                                "D_star": sorted(state.D_star),
                                "composite": state.Phi})
    except WebSocketDisconnect:
        pass


# ─────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────


@app.post("/api/agop/calibrate")
async def api_calibrate(req: CalibrateRequest):
    try:
        episodes = [
            Episode(e.alternatives, e.axes, e.chosen, e.outcome)
            for e in req.episodes
        ]
        problem = calibrate(episodes, beta=req.beta,
                            sigma_prior=req.sigma_prior,
                            axis_keys=req.axis_keys)
        state = agop(problem)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "calibrated_axes": problem.axes,
        "calibration_std": getattr(problem.Theta, "calibration_std", None),
        "ranking_after_calibration": state.R,
    }
