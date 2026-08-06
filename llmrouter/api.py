"""
Optional FastAPI HTTP layer for llmrouter.

v1 budgets are process-local — run a single uvicorn worker only:
  uvicorn llmrouter.api:app --host 0.0.0.0 --port 8080
Do not use --workers > 1 until a shared BudgetStore (e.g. Redis) is implemented.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llmrouter.adapters.base import LLMResponse
from llmrouter.queue_worker import RouterClient

try:
    from fastapi import FastAPI, HTTPException
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install llmrouter[api] to use the HTTP layer") from exc


_client: RouterClient | None = None


class CompleteRequest(BaseModel):
    prompt: str
    capability: str = "chat"
    params: dict[str, Any] = Field(default_factory=dict)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    models_path = Path(__file__).resolve().parent / "models.yaml"
    _client = RouterClient(models_path=str(models_path))
    await _client.start()
    yield
    await _client.shutdown(graceful=True)
    _client = None


app = FastAPI(title="llmrouter", version="0.1.0", lifespan=lifespan)


def _require_client() -> RouterClient:
    if _client is None:
        raise HTTPException(status_code=503, detail="router not ready")
    return _client


@app.post("/v1/complete", response_model=LLMResponse)
async def complete(body: CompleteRequest) -> LLMResponse:
    client = _require_client()
    try:
        return await client.submit(body.prompt, body.capability, **body.params)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/v1/status")
async def status() -> dict[str, dict[str, int]]:
    return await _require_client().status()


@app.get("/v1/metrics")
async def metrics_endpoint() -> dict[str, Any]:
    return await _require_client().metrics_snapshot()
