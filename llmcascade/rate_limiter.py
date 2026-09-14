from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any, Deque

from llmcascade.provider_quota import parse_openrouter_key, parse_quota_headers
from llmcascade.registry import ModelConfig, resolve_auth_env


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
        self._live: dict[str, dict[str, int]] = {}
        self._live_limits: dict[str, dict[str, int]] = {}
        self._live_source: dict[str, str] = {}
        self._live_lock = asyncio.Lock()
        self._live_refresh_at = 0.0
        self._live_ttl_s = 20.0

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

    def _live_exhausted(self, model_name: str, tokens_estimate: int = 0) -> bool:
        live = self._live.get(model_name) or {}
        if live.get("rpd", 1) <= 0:
            return True
        if live.get("rpm", 1) <= 0:
            return True
        if tokens_estimate and live.get("tpm") is not None and live["tpm"] < tokens_estimate:
            return True
        return False

    async def ingest_quota(
        self,
        model_name: str,
        parsed: dict[str, int],
        *,
        provider: str | None = None,
        source: str = "headers",
        fanout: bool = True,
    ) -> None:
        if not parsed:
            return
        remaining = {k: v for k, v in parsed.items() if not k.endswith("_limit")}
        caps = {}
        for key, val in parsed.items():
            if key.endswith("_limit"):
                caps[key[: -len("_limit")]] = val
        names = [model_name]
        if fanout and provider:
            names = [n for n, m in self._models.items() if m.provider == provider] or names
        async with self._live_lock:
            for name in names:
                if remaining:
                    prev = dict(self._live.get(name) or {})
                    prev.update(remaining)
                    self._live[name] = prev
                    self._live_source[name] = source
                if caps:
                    prev_c = dict(self._live_limits.get(name) or {})
                    prev_c.update(caps)
                    self._live_limits[name] = prev_c

    async def ingest_headers(
        self,
        model_name: str,
        headers: dict[str, str] | None,
        *,
        provider: str = "",
    ) -> None:
        parsed = parse_quota_headers(headers, provider=provider)
        await self.ingest_quota(model_name, parsed, provider=provider, source="headers")

    def quota_source(self, model_name: str) -> str:
        return self._live_source.get(model_name) or "local"

    def quota_limits(self, model_name: str) -> dict[str, int]:
        return dict(self._live_limits.get(model_name) or {})

    async def refresh_live_quotas(self, client: Any, models: list[ModelConfig] | None = None) -> None:
        """Pull Groq header remaining and OpenRouter key payload. Cached ~20s."""
        now = time.monotonic()
        if now - self._live_refresh_at < self._live_ttl_s:
            return
        self._live_refresh_at = now
        roster = models if models is not None else list(self._models.values())
        seen: set[str] = set()
        for model in roster:
            if model.provider in seen:
                continue
            key = resolve_auth_env(
                model.auth_env_var,
                provider=model.provider,
                key_tier=getattr(model, "key_tier", "free"),
            )
            if not key:
                continue
            seen.add(model.provider)
            try:
                if model.provider == "groq":
                    resp = await client.get(
                        "https://api.groq.com/openai/v1/models",
                        headers={"Authorization": f"Bearer {key}"},
                        timeout=8.0,
                    )
                    await self.ingest_headers(
                        model.name, dict(resp.headers), provider="groq"
                    )
                elif model.provider == "openrouter":
                    resp = await client.get(
                        "https://openrouter.ai/api/v1/key",
                        headers={"Authorization": f"Bearer {key}"},
                        timeout=8.0,
                    )
                    if resp.status_code < 400:
                        parsed = parse_openrouter_key(resp.json())
                        await self.ingest_quota(
                            model.name,
                            parsed,
                            provider="openrouter",
                            source="openrouter_key",
                        )
                    await self.ingest_headers(
                        model.name, dict(resp.headers), provider="openrouter"
                    )
            except Exception:  # noqa: BLE001 — keep dashboard up
                continue

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
        if self._live_exhausted(model_name, tokens_estimate):
            return False
        async with self._lock(model_name):
            now = time.monotonic()
            limits = model.limits
            # YAML rpd/rpm/tpm are guesses — do not refuse until the provider 429s
            # (cooldown / live headers) or local rps would stampede.
            if await self._count(model_name, "rps", now) >= limits.rps:
                return False
            return True

    async def try_reserve(self, model_name: str, tokens_estimate: int) -> bool:
        """Atomically accept one request against rps/rpm/rpd/tpm, or reject."""
        model = self._models.get(model_name)
        if model is None:
            return False
        if self.cooldowns is not None and await self.cooldowns.is_cooling(model_name):
            return False
        cascade = self.gemini_cascade
        if (
            cascade is not None
            and model.provider == "gemini"
            and getattr(cascade, "logical_name", None) == model_name
            and not await cascade.any_available()
        ):
            return False
        if self._live_exhausted(model_name, tokens_estimate):
            return False
        async with self._lock(model_name):
            now = time.monotonic()
            limits = model.limits
            if await self._count(model_name, "rps", now) >= limits.rps:
                return False
            for metric, amount in (
                ("rps", 1),
                ("rpm", 1),
                ("rpd", 1),
                ("tpm", max(0, tokens_estimate)),
            ):
                key = f"{model_name}:{metric}"
                events = _prune(await self._store.get_events(key), now, self.WINDOWS[metric])
                events.append((now, amount))
                await self._store.set_events(key, events)
            live = self._live.get(model_name)
            if live:
                if "rpd" in live:
                    live["rpd"] = max(0, live["rpd"] - 1)
                if "rpm" in live:
                    live["rpm"] = max(0, live["rpm"] - 1)
                if "tpm" in live:
                    live["tpm"] = max(0, live["tpm"] - max(0, tokens_estimate))
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
        blocked = False
        if self.cooldowns is not None and await self.cooldowns.is_cooling(model_name):
            blocked = True
        cascade = self.gemini_cascade
        if (
            cascade is not None
            and model.provider == "gemini"
            and getattr(cascade, "logical_name", None) == model_name
            and not await cascade.any_available()
        ):
            blocked = True
        if self._live_exhausted(model_name, 1):
            blocked = True
        async with self._lock(model_name):
            now = time.monotonic()
            lim = model.limits
            used = {
                "rps": await self._count(model_name, "rps", now),
                "rpm": await self._count(model_name, "rpm", now),
                "rpd": await self._count(model_name, "rpd", now),
                "tpm": await self._count(model_name, "tpm", now),
            }
            out = {
                "rps": max(0, lim.rps - used["rps"]),
                "rpm": max(0, lim.rpm - used["rpm"]),
                "rpd": max(0, lim.rpd - used["rpd"]),
                "tpm": max(0, lim.tpm - used["tpm"]),
            }
            live = self._live.get(model_name) or {}
            for metric in ("rpm", "rpd", "tpm"):
                if metric in live:
                    out[metric] = live[metric]
            caps = self._live_limits.get(model_name) or {}
            if "rpd" in caps:
                out["limit_rpd"] = caps["rpd"]
            if "rpm" in caps:
                out["limit_rpm"] = caps["rpm"]
            if blocked:
                out["rpd"] = 0
                out["rpm"] = 0
                out["rps"] = 0
            return out


class ApiKeyRateLimiter:
    """Per-API-key RPM limiter (process-local). Opt-in via LLMCASCADE_API_RPM."""

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

