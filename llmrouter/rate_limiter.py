from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any, Deque

from llmrouter.registry import ModelConfig


class BudgetStore(ABC):
    @abstractmethod
    async def get_events(self, key: str) -> list[tuple[float, int]]:
        ...

    @abstractmethod
    async def set_events(self, key: str, events: list[tuple[float, int]]) -> None:
        ...


class InMemoryBudgetStore(BudgetStore):
    def __init__(self) -> None:
        self._data: dict[str, list[tuple[float, int]]] = defaultdict(list)

    async def get_events(self, key: str) -> list[tuple[float, int]]:
        return list(self._data[key])

    async def set_events(self, key: str, events: list[tuple[float, int]]) -> None:
        self._data[key] = list(events)


def _prune(events: Deque[tuple[float, int]] | list[tuple[float, int]], now: float, window: float) -> list[tuple[float, int]]:
    cutoff = now - window
    return [(t, n) for t, n in events if t >= cutoff]


class RateLimiter:
    """Process-local rate limiter. Not safe across multiple uvicorn workers."""

    WINDOWS = {"rps": 1.0, "rpm": 60.0, "rpd": 86400.0, "tpm": 60.0}

    def __init__(
        self,
        models: list[ModelConfig],
        store: BudgetStore | None = None,
        *,
        gemini_cascade: Any | None = None,
        cooldowns: Any | None = None,
    ) -> None:
        self._models = {m.name: m for m in models}
        self._store = store or InMemoryBudgetStore()
        self._locks: dict[str, asyncio.Lock] = {m.name: asyncio.Lock() for m in models}
        self.gemini_cascade = gemini_cascade
        self.cooldowns = cooldowns

    def replace_models(self, models: list[ModelConfig]) -> None:
        """Hot-reload model limit map; preserve budget event store."""
        self._models = {m.name: m for m in models}
        for m in models:
            if m.name not in self._locks:
                self._locks[m.name] = asyncio.Lock()


    def _lock(self, model_name: str) -> asyncio.Lock:
        if model_name not in self._locks:
            self._locks[model_name] = asyncio.Lock()
        return self._locks[model_name]

    async def _count(self, model_name: str, metric: str, now: float) -> int:
        key = f"{model_name}:{metric}"
        events = _prune(await self._store.get_events(key), now, self.WINDOWS[metric])
        await self._store.set_events(key, events)
        return sum(n for _, n in events)

    async def can_proceed(self, model_name: str, tokens_estimate: int) -> bool:
        model = self._models.get(model_name)
        if model is None:
            return False
        if self.cooldowns is not None and await self.cooldowns.is_cooling(model_name):
            return False
        # Gemini family: ineligible while every cascade member is cooling.
        cascade = self.gemini_cascade
        if (
            cascade is not None
            and model.provider == "gemini"
            and getattr(cascade, "logical_name", None) == model_name
            and not await cascade.any_available()
        ):
            return False
        async with self._lock(model_name):
            now = time.monotonic()
            limits = model.limits
            if await self._count(model_name, "rps", now) >= limits.rps:
                return False
            if await self._count(model_name, "rpm", now) >= limits.rpm:
                return False
            if await self._count(model_name, "rpd", now) >= limits.rpd:
                return False
            if await self._count(model_name, "tpm", now) + tokens_estimate > limits.tpm:
                return False
            return True

    async def record_usage(self, model_name: str, tokens_used: int) -> None:
        async with self._lock(model_name):
            now = time.monotonic()
            for metric, amount in (("rps", 1), ("rpm", 1), ("rpd", 1), ("tpm", max(0, tokens_used))):
                key = f"{model_name}:{metric}"
                events = _prune(await self._store.get_events(key), now, self.WINDOWS[metric])
                events.append((now, amount))
                await self._store.set_events(key, events)

    async def remaining_budget(self, model_name: str) -> dict[str, int]:
        model = self._models.get(model_name)
        if model is None:
            return {}
        async with self._lock(model_name):
            now = time.monotonic()
            lim = model.limits
            used = {
                "rps": await self._count(model_name, "rps", now),
                "rpm": await self._count(model_name, "rpm", now),
                "rpd": await self._count(model_name, "rpd", now),
                "tpm": await self._count(model_name, "tpm", now),
            }
            return {
                "rps": max(0, lim.rps - used["rps"]),
                "rpm": max(0, lim.rpm - used["rpm"]),
                "rpd": max(0, lim.rpd - used["rpd"]),
                "tpm": max(0, lim.tpm - used["tpm"]),
            }


class ApiKeyRateLimiter:
    """Per-API-key RPM limiter (process-local). Opt-in via LLMROUTER_API_RPM."""

    def __init__(self, rpm: int, store: BudgetStore | None = None) -> None:
        if rpm < 1:
            raise ValueError("rpm must be >= 1")
        self.rpm = rpm
        self._store = store or InMemoryBudgetStore()
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, key_id: str) -> asyncio.Lock:
        if key_id not in self._locks:
            self._locks[key_id] = asyncio.Lock()
        return self._locks[key_id]

    @staticmethod
    def key_id(api_key: str) -> str:
        # Avoid storing raw keys as dict keys in logs; use a short stable id.
        import hashlib

        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]

    async def check_and_record(self, api_key: str) -> bool:
        """Return True if allowed (and record), False if over limit."""
        kid = self.key_id(api_key)
        async with self._lock(kid):
            now = time.monotonic()
            store_key = f"apikey:{kid}:rpm"
            events = _prune(await self._store.get_events(store_key), now, 60.0)
            used = sum(n for _, n in events)
            if used >= self.rpm:
                await self._store.set_events(store_key, events)
                return False
            events.append((now, 1))
            await self._store.set_events(store_key, events)
            return True

