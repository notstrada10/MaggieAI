"""FastAPI gateway — entry point del sistema.

Espone:
- POST /translate   → reasoning loop completo (agent LangGraph)
- POST /retrieve    → solo Tool A + Tool B, senza generazione (Sprint 2 deliverable)
- GET  /health      → liveness check

Il router di inference è creato una volta sola al lifespan-startup
e chiuso a shutdown.
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
class TranslateRequest(BaseModel):
    text: str = Field(min_length=1, description="Frase latina da tradurre")


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
    mode: RoutingMode = "hybrid" if settings.anthropic_api_key else "local-only"
    if mode == "local-only":
        logger.warning("ANTHROPIC_API_KEY non impostato — modalità local-only.")
    router = InferenceRouter(mode=mode)
    app.state.router = router
    app.state.graph = build_graph(router=router)
    logger.info("Gateway pronto (mode=%s)", mode)
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
    try:
        final_state = await app.state.graph.ainvoke(initial_state)
    except Exception as exc:  # pragma: no cover — log + 500
        logger.exception("Errore nel reasoning loop (trace=%s)", trace_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    output = final_state.get("output")
    if not output:
        raise HTTPException(status_code=500, detail="Output finale mancante")

    _persist_trace(trace_id, req.text, final_state)
    return TranslateResponse(**output)


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    """Solo Tool A + Tool B (no LLM). Utile per debug e per la UI Streamlit."""
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
    """Salva l'intero state in `reasoning_traces`. Best-effort: non
    bloccare la response se la persistenza fallisce."""
    try:
        # Pydantic models nel state non sono direttamente JSON-serializable
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
        logger.exception("Fallita persistenza del trace %s (continuo)", trace_id)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, UUID):
        return str(obj)
    raise TypeError(f"Non serializzabile: {type(obj).__name__}")


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
