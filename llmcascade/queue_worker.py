from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from llmcascade.adapters import get_adapter
from llmcascade.adapters.base import LLMResponse
from llmcascade.adapters.gemini_adapter import GeminiAdapter
from llmcascade.cascade import GeminiCascadeManager, ModelCooldownTracker, cascade_manager_from_registry
from llmcascade.event_log import events
from llmcascade.exceptions import QueueFullError
from llmcascade.health import health_cache
from llmcascade.metrics import metrics
from llmcascade.rate_limiter import RateLimiter
from llmcascade.registry import ModelConfig, key_source, list_all_models, load_registry
from llmcascade.selector import ModelSelector, Strategy
from llmcascade.stats_store import NullStatsStore, StatsStore

try:
    from llmcascade.provider_store import get_free_paid, key_is_set
except Exception:  # pragma: no cover
    def get_free_paid(provider: str) -> str:  # type: ignore[misc]
        return "free"

    def key_is_set(provider: str) -> bool:  # type: ignore[misc]
        return False



@dataclass
class _Job:
    prompt: str
    capability: str
    params: dict[str, Any]
    future: asyncio.Future[LLMResponse]
    notes: str | None = None
    pinned_model: str | None = None


class RouterClient:
    def __init__(
        self,
        registry: list[ModelConfig] | None = None,
        *,
        models_path: str | None = None,
        strategy: Strategy = "weighted",
        workers: int = 4,
        max_queue: int = 100,
        rate_limiter: RateLimiter | None = None,
        gemini_cascade: GeminiCascadeManager | None = None,
        cooldowns: ModelCooldownTracker | None = None,
        stats: StatsStore | NullStatsStore | None = None,
        allow_empty: bool = False,
    ) -> None:
        self.registry = (
            registry
            if registry is not None
            else load_registry(models_path, allow_empty=allow_empty)
        )
        self._models_path = models_path
        self._strategy = strategy
        self.gemini_cascade = gemini_cascade or cascade_manager_from_registry(self.registry)
        self.cooldowns = cooldowns or ModelCooldownTracker()
        self.rate_limiter = rate_limiter or RateLimiter(
            self.registry,
            gemini_cascade=self.gemini_cascade,
            cooldowns=self.cooldowns,
        )
        if rate_limiter is not None:
            if self.gemini_cascade is not None:
                self.rate_limiter.gemini_cascade = self.gemini_cascade
            self.rate_limiter.cooldowns = self.cooldowns
        self.stats: StatsStore | NullStatsStore = stats or NullStatsStore()
        self.selector = ModelSelector(
            self.registry,
            self.rate_limiter,
            strategy=strategy,
            stats=self.stats,
            cooldowns=self.cooldowns,
        )
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

    def reload_registry(self, *, allow_empty: bool = True) -> int:
        """Hot-reload models from YAML + current keys without stopping workers."""
        new_registry = load_registry(self._models_path, allow_empty=allow_empty)
        self.registry = new_registry
        self.gemini_cascade = cascade_manager_from_registry(new_registry)
        self.rate_limiter.replace_models(new_registry)
        self.rate_limiter.gemini_cascade = self.gemini_cascade
        self.rate_limiter.cooldowns = self.cooldowns
        self.selector.registry = list(new_registry)
        self.selector.rate_limiter = self.rate_limiter
        self.selector.cooldowns = self.cooldowns
        self.selector.stats = self.stats
        self.selector.strategy = self._strategy
        return len(new_registry)

    async def _execute(
        self, model: ModelConfig, prompt: str, *, capability: str = "chat", **params: Any
    ) -> LLMResponse:
        wait_for_gemini = bool(params.pop("wait_for_gemini", False))
        params.pop("notes", None)
        adapter = get_adapter(model, client=self._client)
        if capability == "embed":
            return await adapter.embed(prompt, **params)
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
                    return await self._execute(
                        model, prompt, capability=job.capability, **job.params
                    )

                try:
                    result = await self.selector.dispatch_with_fallback(
                        job.prompt,
                        job.capability,
                        executor,
                        notes=job.notes,
                        pinned_model=job.pinned_model,
                    )
                    if not job.future.done():
                        job.future.set_result(result)
                except Exception as exc:
                    if not job.future.done():
                        job.future.set_exception(exc)
            finally:
                self._queue.task_done()

    async def submit(
        self,
        prompt: str,
        capability: str = "chat",
        *,
        notes: str | None = None,
        model: str | None = None,
        **params: Any,
    ) -> LLMResponse:
        """Submit a completion or embedding.

        Chat may fall through providers. Embeddings must pin ``model`` (same
        vector space); there is no cross-model cascade.
        """
        if not self._started:
            await self.start()
        # Never forward notes to provider adapters.
        if notes is None and "notes" in params:
            raw = params.pop("notes")
            notes = str(raw).strip() or None if raw is not None else None
        else:
            params.pop("notes", None)
        if notes is not None:
            notes = str(notes).strip() or None
        pin = (model or params.pop("model", None) or "").strip() or None
        if capability == "embed" and not pin:
            raise ValueError("embed requires model (same model for a corpus; no cascade)")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[LLMResponse] = loop.create_future()
        job = _Job(
            prompt=prompt,
            capability=capability,
            params=params,
            future=fut,
            notes=notes,
            pinned_model=pin,
        )
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            detail: dict[str, Any] = {"capability": capability}
            if notes:
                detail["notes"] = notes
            events.record("request queue is full", level="error", type="queue", **detail)
            raise QueueFullError("request queue is full") from exc
        return await fut

    async def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            m.name: await self.rate_limiter.remaining_budget(m.name) for m in self.registry
        }
        if self.gemini_cascade is not None:
            out["gemini_cascade"] = await self.gemini_cascade.status()
        out["model_cooldowns"] = await self.cooldowns.status()
        return out

    async def gemini_status(self) -> dict[str, Any]:
        if self.gemini_cascade is None:
            return {"logical": None, "models": [], "available": [], "available_at": {}, "family_ready": False}
        return await self.gemini_cascade.status()

    async def metrics_snapshot(self) -> dict[str, Any]:
        snap = metrics.snapshot()
        budgets = await self.status()
        return {**snap, "current_budget": budgets}

    async def stats_snapshot(self, range_key: str = "7d") -> dict[str, Any]:
        return await self.stats.snapshot(range_key)

    async def health_snapshot(self, *, force: bool = False) -> dict[str, dict[str, Any]]:
        return await health_cache.statuses(self._client, self.registry, force=force)

    async def dashboard_snapshot(self, *, force_health: bool = False) -> dict[str, Any]:
        budgets = await self.status()
        snap = metrics.snapshot()
        try:
            health = await self.health_snapshot(force=force_health)
        except Exception as exc:  # noqa: BLE001 — keep dashboard up if a probe crashes
            health = {}
            events.record(f"health probe failed: {exc}", level="error", type="health")
        next_model = await self.selector.peek("chat", tokens_estimate=1)
        next_embed = await self.selector.peek("embed", tokens_estimate=1)
        cooling = budgets.get("model_cooldowns") or {}
        if not isinstance(cooling, dict):
            cooling = {}
        models = []
        active_names = {m.name for m in self.registry}
        all_models = list_all_models(self._models_path) if self._models_path else list(self.registry)
        # Prefer live registry rows; also surface inactive models that lack keys.
        by_name = {m.name: m for m in all_models}
        by_name.update({m.name: m for m in self.registry})
        for m in by_name.values():
            budget = budgets.get(m.name, {})
            src = key_source(m.auth_env_var, provider=m.provider, key_tier=getattr(m, "key_tier", "free"))
            entry: dict[str, Any] = {
                "name": m.name,
                "provider": m.provider,
                "priority": m.priority,
                "capabilities": m.capabilities,
                "limits": m.limits.model_dump(),
                "budget": budget if m.name in active_names else {},
                "free_tier_verified": m.free_tier_verified,
                "free_tier_note": m.free_tier_note,
                "free_left": budget if m.free_tier_verified and m.name in active_names else None,
                "is_next": bool(
                    ("embed" in m.capabilities and next_embed and next_embed.name == m.name)
                    or ("chat" in m.capabilities and next_model and next_model.name == m.name)
                ),
                "requests_total": snap["requests_total"].get(m.name, 0),
                "failures_total": snap["failures_total"].get(m.name, 0),
                "health": health.get(m.name, {"state": "unknown"}),
                "cooldown": cooling.get(m.name),
                "key_set": src != "none" or key_is_set(m.provider),
                "key_source": src,
                "active": m.name in active_names,
                "free_paid": getattr(m, "key_tier", None) or get_free_paid(m.provider),
                "weight": getattr(m, "weight", 1),
                "enabled": getattr(m, "enabled", True),
                "custom": getattr(m, "custom", False),
            }
            if m.cascade:
                entry["cascade"] = list(m.cascade)
            models.append(entry)
        models.sort(key=lambda e: (e["priority"], e["name"]))
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
            "next_pick": (
                {"name": next_model.name, "provider": next_model.provider, "priority": next_model.priority}
                if next_model
                else None
            ),
            "next_embed": (
                {"name": next_embed.name, "provider": next_embed.provider, "priority": next_embed.priority}
                if next_embed
                else None
            ),
            "gemini_cascade": gemini,
            "queue": {
                "depth": self._queue.qsize(),
                "maxsize": self._max_queue,
                "workers": self._workers_n,
            },
            "events": events.events(),
            "errors": events.errors(),
        }
