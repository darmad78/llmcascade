from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from llmrouter.adapters.base import LLMResponse
from llmrouter.exceptions import AllModelsExhaustedError, ProviderError
from llmrouter.event_log import events
from llmrouter.metrics import log, metrics
from llmrouter.rate_limiter import RateLimiter
from llmrouter.registry import ModelConfig
from llmrouter.tokens import estimate_tokens

Strategy = Literal["round_robin", "least_used", "priority_first"]
Executor = Callable[[ModelConfig, str], Awaitable[LLMResponse]]

RETRY_SLEEP_S = 0.25


class ModelSelector:
    def __init__(
        self,
        registry: list[ModelConfig],
        rate_limiter: RateLimiter,
        strategy: Strategy = "round_robin",
    ) -> None:
        self.registry = list(registry)
        self.rate_limiter = rate_limiter
        self.strategy = strategy
        self._rr_index = 0

    async def _eligible(self, capability: str, tokens_estimate: int) -> list[ModelConfig]:
        out: list[ModelConfig] = []
        for m in self.registry:
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
            return sorted(eligible, key=lambda m: m.priority)[0]
        if self.strategy == "least_used":
            budgets = []
            for m in eligible:
                rem = await self.rate_limiter.remaining_budget(m.name)
                budgets.append((rem.get("rpd", 0) + rem.get("rpm", 0), m))
            budgets.sort(key=lambda x: x[0], reverse=True)
            return budgets[0][1]
        # round_robin over eligible pool
        idx = self._rr_index % len(eligible)
        self._rr_index += 1
        return eligible[idx]

    async def peek(self, capability: str, tokens_estimate: int = 1) -> ModelConfig | None:
        """Next pick without advancing round-robin state (safe for dashboards)."""
        eligible = await self._eligible(capability, tokens_estimate)
        if not eligible:
            return None
        if self.strategy == "priority_first":
            return sorted(eligible, key=lambda m: m.priority)[0]
        if self.strategy == "least_used":
            budgets = []
            for m in eligible:
                rem = await self.rate_limiter.remaining_budget(m.name)
                budgets.append((rem.get("rpd", 0) + rem.get("rpm", 0), m))
            budgets.sort(key=lambda x: x[0], reverse=True)
            return budgets[0][1]
        return eligible[self._rr_index % len(eligible)]

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
        **_params: Any,
    ) -> LLMResponse:
        tokens_est = estimate_tokens(prompt)
        tried: set[str] = set()
        last_err: Exception | None = None

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
                log.info(
                    "request ok",
                    extra={
                        "model_used": model.name,
                        "latency_ms": resp.latency_ms,
                        "success": True,
                        "tokens_used": used,
                        "provider": model.provider,
                        "capability": capability,
                    },
                )
                events.record(
                    "request ok",
                    level="info",
                    model=model.name,
                    provider=model.provider,
                    success=True,
                    latency_ms=resp.latency_ms,
                    tokens_used=used,
                    capability=capability,
                )
                return resp
            except ProviderError as exc:
                last_err = exc
                metrics.record_failure(model.name)
                log.info(
                    "request fail",
                    extra={
                        "model_used": model.name,
                        "latency_ms": 0,
                        "success": False,
                        "tokens_used": 0,
                        "provider": model.provider,
                        "capability": capability,
                    },
                )
                events.record(
                    "request fail",
                    level="error",
                    model=model.name,
                    provider=model.provider,
                    success=False,
                    error=str(exc),
                    capability=capability,
                )
                continue

        msg = (
            f"no free-tier model succeeded for capability={capability!r}"
            + (f"; last error: {last_err}" if last_err else "")
        )
        events.record(msg, level="error", capability=capability)
        raise AllModelsExhaustedError(msg)
