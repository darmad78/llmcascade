from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from llmrouter.adapters import get_adapter
from llmrouter.adapters.base import LLMResponse
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
    ) -> None:
        self.registry = registry if registry is not None else load_registry(models_path)
        self.rate_limiter = rate_limiter or RateLimiter(self.registry)
        self.selector = ModelSelector(self.registry, self.rate_limiter, strategy=strategy)
        self._workers_n = workers
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
        adapter = get_adapter(model, client=self._client)
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

    async def status(self) -> dict[str, dict[str, int]]:
        return {m.name: await self.rate_limiter.remaining_budget(m.name) for m in self.registry}

    async def metrics_snapshot(self) -> dict[str, Any]:
        snap = metrics.snapshot()
        budgets = await self.status()
        return {**snap, "current_budget": budgets}

    async def health_snapshot(self, *, force: bool = False) -> dict[str, dict[str, Any]]:
        return await health_cache.statuses(self._client, self.registry, force=force)

    async def dashboard_snapshot(self, *, force_health: bool = False) -> dict[str, Any]:
        budgets = await self.status()
        snap = metrics.snapshot()
        health = await self.health_snapshot(force=force_health)
        models = []
        for m in self.registry:
            models.append(
                {
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
            )
        return {
            "models": models,
            "events": events.events(),
            "errors": events.errors(),
        }
