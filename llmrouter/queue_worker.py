from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from llmrouter.adapters import get_adapter
from llmrouter.adapters.base import LLMResponse
from llmrouter.adapters.gemini_adapter import GeminiAdapter
from llmrouter.cascade import GeminiCascadeManager, cascade_manager_from_registry
from llmrouter.event_log import events
from llmrouter.exceptions import QueueFullError
from llmrouter.health import health_cache
from llmrouter.metrics import metrics
from llmrouter.rate_limiter import RateLimiter
from llmrouter.registry import ModelConfig, load_registry
from llmrouter.selector import ModelSelector, Strategy


@dataclass
class _Job:
    prompt: str
    capability: str
    params: dict[str, Any]
    future: asyncio.Future[LLMResponse]


class RouterClient:
    def __init__(
        self,
        registry: list[ModelConfig] | None = None,
        *,
        models_path: str | None = None,
        strategy: Strategy = "round_robin",
        workers: int = 4,
        max_queue: int = 100,
        rate_limiter: RateLimiter | None = None,
        gemini_cascade: GeminiCascadeManager | None = None,
    ) -> None:
        self.registry = registry if registry is not None else load_registry(models_path)
        self.gemini_cascade = gemini_cascade or cascade_manager_from_registry(self.registry)
        self.rate_limiter = rate_limiter or RateLimiter(
            self.registry, gemini_cascade=self.gemini_cascade
        )
        if rate_limiter is not None and self.gemini_cascade is not None:
            self.rate_limiter.gemini_cascade = self.gemini_cascade
        self.selector = ModelSelector(self.registry, self.rate_limiter, strategy=strategy)
        self._workers_n = workers
        self._max_queue = max_queue
        self._queue: asyncio.Queue[_Job | None] = asyncio.Queue(maxsize=max_queue)
        self._tasks: list[asyncio.Task[None]] = []
        self._client = httpx.AsyncClient(timeout=60.0)
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        for _ in range(self._workers_n):
            self._tasks.append(asyncio.create_task(self._worker()))

    async def shutdown(self, graceful: bool = True) -> None:
        if not self._started:
            return
        if graceful:
            for _ in self._tasks:
                await self._queue.put(None)
            await asyncio.gather(*self._tasks, return_exceptions=True)
        else:
            for t in self._tasks:
                t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        await self._client.aclose()
        self._started = False

    async def _execute(self, model: ModelConfig, prompt: str, **params: Any) -> LLMResponse:
        wait_for_gemini = bool(params.pop("wait_for_gemini", False))
        adapter = get_adapter(model, client=self._client)
        if (
            model.provider == "gemini"
            and self.gemini_cascade is not None
            and isinstance(adapter, GeminiAdapter)
        ):
            async def send(model_id: str, p: str) -> LLMResponse:
                return await adapter.send_model(model_id, p, **params)

            return await self.gemini_cascade.run(
                send, prompt, wait_for_gemini=wait_for_gemini
            )
        return await adapter.send(prompt, **params)

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                if job is None:
                    return
                if job.future.cancelled():
                    continue

                async def executor(model: ModelConfig, prompt: str) -> LLMResponse:
                    return await self._execute(model, prompt, **job.params)

                try:
                    result = await self.selector.dispatch_with_fallback(
                        job.prompt, job.capability, executor
                    )
                    if not job.future.done():
                        job.future.set_result(result)
                except Exception as exc:
                    if not job.future.done():
                        job.future.set_exception(exc)
            finally:
                self._queue.task_done()

    async def submit(self, prompt: str, capability: str = "chat", **params: Any) -> LLMResponse:
        """Submit a completion. Additive: wait_for_gemini=False (default) falls through
        to other free providers when the Gemini cascade is fully cooling.
        """
        if not self._started:
            await self.start()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[LLMResponse] = loop.create_future()
        job = _Job(prompt=prompt, capability=capability, params=params, future=fut)
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            events.record("request queue is full", level="error", capability=capability)
            raise QueueFullError("request queue is full") from exc
        return await fut

    async def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            m.name: await self.rate_limiter.remaining_budget(m.name) for m in self.registry
        }
        if self.gemini_cascade is not None:
            out["gemini_cascade"] = await self.gemini_cascade.status()
        return out

    async def gemini_status(self) -> dict[str, Any]:
        if self.gemini_cascade is None:
            return {"logical": None, "models": [], "available": [], "available_at": {}, "family_ready": False}
        return await self.gemini_cascade.status()

    async def metrics_snapshot(self) -> dict[str, Any]:
        snap = metrics.snapshot()
        budgets = await self.status()
        return {**snap, "current_budget": budgets}

    async def health_snapshot(self, *, force: bool = False) -> dict[str, dict[str, Any]]:
        return await health_cache.statuses(self._client, self.registry, force=force)

    async def dashboard_snapshot(self, *, force_health: bool = False) -> dict[str, Any]:
        budgets = await self.status()
        snap = metrics.snapshot()
        try:
            health = await self.health_snapshot(force=force_health)
        except Exception as exc:  # noqa: BLE001 — keep dashboard up if a probe crashes
            health = {}
            events.record(f"health probe failed: {exc}", level="error")
        models = []
        for m in self.registry:
            entry: dict[str, Any] = {
                "name": m.name,
                "provider": m.provider,
                "priority": m.priority,
                "capabilities": m.capabilities,
                "limits": m.limits.model_dump(),
                "budget": budgets.get(m.name, {}),
                "requests_total": snap["requests_total"].get(m.name, 0),
                "failures_total": snap["failures_total"].get(m.name, 0),
                "health": health.get(m.name, {"state": "unknown"}),
            }
            if m.cascade:
                entry["cascade"] = list(m.cascade)
            models.append(entry)
        gemini = budgets.get("gemini_cascade")
        if isinstance(gemini, dict):
            gemini = {
                **gemini,
                "budget": budgets.get("gemini", {}),
                "requests_total": snap["requests_total"].get("gemini", 0),
                "failures_total": snap["failures_total"].get("gemini", 0),
            }
        return {
            "models": models,
            "gemini_cascade": gemini,
            "queue": {
                "depth": self._queue.qsize(),
                "maxsize": self._max_queue,
                "workers": self._workers_n,
            },
            "events": events.events(),
            "errors": events.errors(),
        }
