from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Deque

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

    def __init__(self, models: list[ModelConfig], store: BudgetStore | None = None) -> None:
        self._models = {m.name: m for m in models}
        self._store = store or InMemoryBudgetStore()
        self._locks: dict[str, asyncio.Lock] = {m.name: asyncio.Lock() for m in models}

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
