"""FastAPI gateway — entry point of the system.

Exposes:
- POST /translate   → full reasoning loop (LangGraph agent)
- POST /retrieve    → only Tool A + Tool B, no generation (Sprint 2 deliverable)
- GET  /health      → liveness check

The inference router is created once at lifespan-startup and closed
at shutdown.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID, uuid4

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import update as sql_update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from maggieai.agent.graph import build_graph
from maggieai.agent.state import GrammarHit, TMHit
from maggieai.agent.tools import grammar_lookup, tm_lookup
from maggieai.config import get_settings
from maggieai.db.engine import session_scope
from maggieai.db.models import ReasoningTrace
from maggieai.inference.router import InferenceRouter, RoutingMode

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Pydantic — request/response
# -------------------------------------------------------------------
_VALID_MODES: tuple[str, ...] = ("hybrid", "local-only", "claude-only", "deepseek-only")


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1, description="Latin sentence to translate")
    routing_mode: str | None = Field(
        default=None,
        description=(
            "Optional per-request override of the inference routing mode. "
            "One of: 'hybrid', 'local-only', 'claude-only', 'deepseek-only'. "
            "When unset, the gateway uses its lifespan-time mode."
        ),
    )


class RatingRequest(BaseModel):
    rating: int = Field(ge=1, le=5, description="User rating 1..5")


class TranslateResponse(BaseModel):
    translation: str
    rationale: str
    morpho_analysis: list[dict[str, Any]]
    phenomena_detected: list[str]
    citations: list[dict[str, Any]]
    iterations: int
    trace_id: str


class RetrieveRequest(BaseModel):
    text: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=20)
    phenomena: list[str] = Field(default_factory=list)


class RetrieveResponse(BaseModel):
    tm_hits: list[TMHit]
    grammar_hits: list[GrammarHit]


# -------------------------------------------------------------------
# Lifespan
# -------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    if settings.inference_routing_mode:
        mode: RoutingMode = settings.inference_routing_mode  # type: ignore[assignment]
        logger.info("Inference routing forced via INFERENCE_ROUTING_MODE=%s", mode)
    else:
        # Auto: prefer DeepSeek if available (cheap + frontier), then Claude, then local.
        if settings.deepseek_api_key:
            mode = "deepseek-only"
        elif settings.anthropic_api_key:
            mode = "hybrid"
        else:
            mode = "local-only"
            logger.warning("No cloud API key set — running in local-only mode.")
    router = InferenceRouter(mode=mode)
    app.state.router = router
    app.state.graph = build_graph(router=router)
    logger.info("Gateway ready (mode=%s)", mode)
    try:
        yield
    finally:
        await router.aclose()


app = FastAPI(title="MaggieAI Gateway", version="0.1.0", lifespan=lifespan)


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "gateway"}


@app.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest) -> TranslateResponse:
    trace_id: UUID = uuid4()
    initial_state: dict[str, Any] = {"input_text": req.text, "trace_id": trace_id, "iterations": 0}

    # Pick the graph: cached default, or a sibling-router graph when the
    # client asked for a different mode. Sibling shares cached clients.
    graph = app.state.graph
    if req.routing_mode and req.routing_mode != app.state.router.mode:
        if req.routing_mode not in _VALID_MODES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid routing_mode '{req.routing_mode}'. Expected one of {_VALID_MODES}.",
            )
        sibling = app.state.router.with_mode(req.routing_mode)
        graph = build_graph(router=sibling)

    try:
        final_state = await graph.ainvoke(initial_state)
    except Exception as exc:  # pragma: no cover — log + 500
        logger.exception("Error in reasoning loop (trace=%s)", trace_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    output = final_state.get("output")
    if not output:
        raise HTTPException(status_code=500, detail="Final output missing")

    _persist_trace(trace_id, req.text, final_state)
    return TranslateResponse(**output)


@app.patch("/traces/{trace_id}/rating")
def rate_trace(trace_id: UUID, req: RatingRequest) -> dict[str, Any]:
    """Set the user_rating for an existing trace. Returns the new value."""
    with session_scope() as session:
        result = session.execute(
            sql_update(ReasoningTrace)
            .where(ReasoningTrace.trace_id == trace_id)
            .values(user_rating=req.rating)
            .returning(ReasoningTrace.trace_id, ReasoningTrace.user_rating)
        )
        row = result.first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
    return {"trace_id": str(row.trace_id), "user_rating": int(row.user_rating)}


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    """Tool A + Tool B only (no LLM). Useful for debugging and the Streamlit UI."""
    from maggieai.ingestion.embedder import embed

    [vec] = embed([req.text])
    return RetrieveResponse(
        tm_hits=tm_lookup(vec, k=req.k),
        grammar_hits=grammar_lookup(req.phenomena),
    )


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _persist_trace(trace_id: UUID, input_text: str, state: dict[str, Any]) -> None:
    """Save the full state to `reasoning_traces`. Best-effort: do not
    block the response if persistence fails."""
    try:
        # Pydantic models inside the state are not directly JSON-serializable
        dump = json.loads(json.dumps(state, default=_json_default))
        with session_scope() as session:
            session.execute(
                pg_insert(ReasoningTrace).values(
                    trace_id=trace_id,
                    input_text=input_text,
                    state_dump=dump,
                )
            )
    except Exception:
        logger.exception("Trace persistence failed for %s (continuing)", trace_id)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, UUID):
        return str(obj)
    raise TypeError(f"Not serializable: {type(obj).__name__}")


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "maggieai.gateway.app:app",
        host="0.0.0.0",
        port=settings.gateway_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
