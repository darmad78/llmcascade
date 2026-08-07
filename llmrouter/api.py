"""
Optional FastAPI HTTP layer for llmrouter.

v1 budgets are process-local — run a single uvicorn worker only:
  uvicorn llmrouter.api:app --host 0.0.0.0 --port 12000
Do not use --workers > 1 until a shared BudgetStore (e.g. Redis) is implemented.

Historical stats require MONGODB_URI (optional LLMROUTER_MONGO_DB, default llmrouter).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llmrouter.adapters.base import LLMResponse
from llmrouter.event_log import events
from llmrouter.queue_worker import RouterClient
from llmrouter.stats_store import NullStatsStore, StatsStore

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import FileResponse
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install llmrouter[api] to use the HTTP layer") from exc


_client: RouterClient | None = None
_stats: StatsStore | NullStatsStore | None = None
_STATIC = Path(__file__).resolve().parent / "static"
_DASHBOARD_HTML = _STATIC / "dashboard.html"
_STATS_HTML = _STATIC / "stats.html"


class CompleteRequest(BaseModel):
    prompt: str
    capability: str = "chat"
    params: dict[str, Any] = Field(default_factory=dict)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client, _stats
    # Load project .env so MONGODB_URI / provider keys work without manual `source`.
    try:
        from dotenv import load_dotenv

        # Prefer process CWD (deploy/pm2), then repo root next to the package.
        # override=True so a quoted .env URI wins over a broken/empty PM2 env.
        load_dotenv(Path.cwd() / ".env", override=True)
        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
    except ImportError:
        pass

    models_path = Path(__file__).resolve().parent / "models.yaml"
    try:
        _stats = await StatsStore.connect()
    except Exception as exc:  # noqa: BLE001
        detail = f"MongoDB connect failed: {exc}"
        events.record(detail, level="error")
        _stats = NullStatsStore(detail=detail)
    if _stats is None:
        _stats = NullStatsStore(
            detail="MONGODB_URI is empty — add it to .env and restart"
        )
    _client = RouterClient(models_path=str(models_path), stats=_stats)
    await _client.start()
    events.record(
        "router started",
        level="info",
        models=len(_client.registry),
        stats_configured=_stats.configured,
    )
    yield
    await _client.shutdown(graceful=True)
    await _stats.close()
    events.record("router stopped", level="info")
    _client = None
    _stats = None


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
        events.record(str(exc), level="error", capability=body.capability)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/v1/status")
async def status() -> dict[str, Any]:
    return await _require_client().status()


@app.get("/v1/status/gemini")
async def gemini_status() -> dict[str, Any]:
    return await _require_client().gemini_status()


@app.get("/v1/metrics")
async def metrics_endpoint() -> dict[str, Any]:
    return await _require_client().metrics_snapshot()


@app.get("/v1/stats")
async def stats_endpoint(range: str = Query(default="7d", pattern="^(24h|1d|7d|30d)$")) -> dict[str, Any]:
    return await _require_client().stats_snapshot(range)


@app.get("/v1/health")
async def health_endpoint(force: bool = False) -> dict[str, dict[str, Any]]:
    return await _require_client().health_snapshot(force=force)


@app.get("/v1/events")
async def events_endpoint(limit: int | None = None) -> list[dict[str, Any]]:
    return events.events(limit=limit)


@app.get("/v1/errors")
async def errors_endpoint(limit: int | None = None) -> list[dict[str, Any]]:
    return events.errors(limit=limit)


@app.get("/v1/dashboard")
async def dashboard_data(force_health: bool = False) -> dict[str, Any]:
    return await _require_client().dashboard_snapshot(force_health=force_health)


@app.get("/dashboard")
async def dashboard_page() -> FileResponse:
    if not _DASHBOARD_HTML.is_file():
        raise HTTPException(status_code=404, detail="dashboard.html missing")
    return FileResponse(_DASHBOARD_HTML, media_type="text/html")


@app.get("/stats")
async def stats_page() -> FileResponse:
    if not _STATS_HTML.is_file():
        raise HTTPException(status_code=404, detail="stats.html missing")
    return FileResponse(_STATS_HTML, media_type="text/html")
