from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from llmcascade.adapters.base import LLMResponse
from llmcascade.exceptions import AllModelsExhaustedError, ProviderError, safe_error_message
from llmcascade.event_log import events
from llmcascade.metrics import log, metrics
from llmcascade.rate_limiter import RateLimiter
from llmcascade.registry import ModelConfig
from llmcascade.stats_store import NullStatsStore, StatsStore
from llmcascade.tokens import estimate_tokens

Strategy = Literal["round_robin", "least_used", "priority_first", "weighted"]
Executor = Callable[[ModelConfig, str], Awaitable[LLMResponse]]

RETRY_SLEEP_S = 0.25


def _weighted_pick(eligible: list[ModelConfig]) -> ModelConfig:
    weights = [max(1, int(getattr(m, "weight", 1) or 1)) for m in eligible]
    return random.choices(eligible, weights=weights, k=1)[0]


class ModelSelector:
    def __init__(
        self,
        registry: list[ModelConfig],
        rate_limiter: RateLimiter,
        strategy: Strategy = "round_robin",
        stats: StatsStore | NullStatsStore | None = None,
        *,
        cooldowns: Any | None = None,
    ) -> None:
        self.registry = list(registry)
        self.rate_limiter = rate_limiter
        self.strategy = strategy
        self.stats: StatsStore | NullStatsStore = stats or NullStatsStore()
        self.cooldowns = cooldowns
        self._rr_index = 0

    async def _eligible(self, capability: str, tokens_estimate: int) -> list[ModelConfig]:
        out: list[ModelConfig] = []
        for m in self.registry:
            if not getattr(m, "enabled", True):
                continue
            if capability not in m.capabilities:
                continue
            if await self.rate_limiter.can_proceed(m.name, tokens_estimate):
                out.append(m)
        return out

    async def pick(self, capability: str, tokens_estimate: int = 1) -> ModelConfig | None:
        eligible = await self._eligible(capability, tokens_estimate)
        if not eligible:
            return None
        if self.strategy == "priority_first":
            return sorted(eligible, key=lambda m: (m.priority, -m.weight))[0]
        if self.strategy == "least_used":
            budgets = []
            for m in eligible:
                rem = await self.rate_limiter.remaining_budget(m.name)
                budgets.append((rem.get("rpd", 0) + rem.get("rpm", 0), m))
            budgets.sort(key=lambda x: x[0], reverse=True)
            return budgets[0][1]
        if self.strategy == "round_robin":
            idx = self._rr_index % len(eligible)
            self._rr_index += 1
            return eligible[idx]
        # weighted (default): chance ∝ weight
        return _weighted_pick(eligible)

    async def peek(self, capability: str, tokens_estimate: int = 1) -> ModelConfig | None:
        """Next pick without advancing round-robin state (safe for dashboards)."""
        eligible = await self._eligible(capability, tokens_estimate)
        if not eligible:
            return None
        if self.strategy == "priority_first":
            return sorted(eligible, key=lambda m: (m.priority, -m.weight))[0]
        if self.strategy == "least_used":
            budgets = []
            for m in eligible:
                rem = await self.rate_limiter.remaining_budget(m.name)
                budgets.append((rem.get("rpd", 0) + rem.get("rpm", 0), m))
            budgets.sort(key=lambda x: x[0], reverse=True)
            return budgets[0][1]
        if self.strategy == "round_robin":
            return eligible[self._rr_index % len(eligible)]
        return _weighted_pick(eligible)

    async def _try_model(
        self,
        model: ModelConfig,
        prompt: str,
        executor: Executor,
    ) -> LLMResponse:
        try:
            return await executor(model, prompt)
        except ProviderError as exc:
            if exc.retryable:
                await asyncio.sleep(RETRY_SLEEP_S)
                return await executor(model, prompt)
            raise

    async def dispatch_with_fallback(
        self,
        prompt: str,
        capability: str,
        executor: Executor,
        *,
        notes: str | None = None,
        **_params: Any,
    ) -> LLMResponse:
        tokens_est = estimate_tokens(prompt)
        tried: set[str] = set()
        last_err: Exception | None = None
        note = (notes or "").strip() or None
        note_detail = {"notes": note} if note else {}

        while True:
            model = await self.pick(capability, tokens_est)
            if model is None or model.name in tried:
                # avoid infinite loop if only tried models remain eligible
                remaining = [m for m in await self._eligible(capability, tokens_est) if m.name not in tried]
                if not remaining:
                    break
                model = remaining[0]
            tried.add(model.name)
            try:
                resp = await self._try_model(model, prompt, executor)
                used = resp.tokens_used or tokens_est
                await self.rate_limiter.record_usage(model.name, used)
                metrics.record_success(model.name)
                await self.stats.record(
                    model=model.name,
                    provider=model.provider,
                    success=True,
                    latency_ms=resp.latency_ms,
                    tokens_used=used,
                    notes=note,
                )
                log.info(
                    "request ok",
                    extra={
                        "model_used": model.name,
                        "latency_ms": resp.latency_ms,
                        "success": True,
                        "tokens_used": used,
                        "provider": model.provider,
                        "capability": capability,
                        **note_detail,
                    },
                )
                events.record(
                    "request ok",
                    level="info",
                    type="request_ok",
                    model=model.name,
                    provider=model.provider,
                    success=True,
                    latency_ms=resp.latency_ms,
                    tokens_used=used,
                    capability=capability,
                    **note_detail,
                )
                return resp
            except ProviderError as exc:
                last_err = exc
                metrics.record_failure(model.name)
                await self.stats.record(
                    model=model.name,
                    provider=model.provider,
                    success=False,
                    latency_ms=0,
                    tokens_used=0,
                    notes=note,
                )
                if self.cooldowns is not None and model.provider != "gemini":
                    # Gemini cascade owns per-member cooldowns; do not pin the logical
                    # family name to permanent from cascade-exhausted status codes.
                    kind = await self.cooldowns.apply_from_error(
                        model.name,
                        status_code=exc.status_code,
                        body=str(exc),
                        headers=getattr(exc, "headers", None),
                    )
                    if kind is not None:
                        events.record(
                            f"cooldown [{kind}]",
                            level="warn",
                            type="cooldown",
                            model=model.name,
                            provider=model.provider,
                            error=safe_error_message(exc),
                            capability=capability,
                            **note_detail,
                        )
                log.info(
                    "request fail",
                    extra={
                        "model_used": model.name,
                        "latency_ms": 0,
                        "success": False,
                        "tokens_used": 0,
                        "provider": model.provider,
                        "capability": capability,
                        **note_detail,
                    },
                )
                events.record(
                    "request fail",
                    level="error",
                    type="request_fail",
                    model=model.name,
                    provider=model.provider,
                    success=False,
                    error=safe_error_message(exc),
                    capability=capability,
                    **note_detail,
                )
                continue

        msg = f"no free-tier model succeeded for capability={capability!r}"
        if last_err is not None:
            msg = f"{msg}; last error: {safe_error_message(last_err)}"
        events.record(
            f"no free-tier model succeeded for capability={capability!r}",
            level="error",
            type="request_fail",
            capability=capability,
            error=safe_error_message(last_err) if last_err else None,
            **note_detail,
        )
        raise AllModelsExhaustedError(msg)
