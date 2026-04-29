"""FastAPI microservice for Latin morphological analysis.

Containerized — runs in the `morphology` service of docker-compose.
The gateway reaches it via `MORPHOLOGY_URL` (default
`http://morphology:8002` inside the Docker network).
"""

from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from maggieai.config import get_settings
from maggieai.morphology.pipeline import SentenceAnalysis, analyze

logger = logging.getLogger(__name__)

app = FastAPI(title="MaggieAI — Morphology Service", version="0.1.0")


class AnalyzeRequest(BaseModel):
    text: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "morphology"}


@app.post("/analyze", response_model=SentenceAnalysis)
def analyze_endpoint(req: AnalyzeRequest) -> SentenceAnalysis:
    return analyze(req.text)


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "maggieai.morphology.api:app",
        host="0.0.0.0",
        port=settings.morphology_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
