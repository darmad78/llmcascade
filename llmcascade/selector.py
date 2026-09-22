from __future__ import annotations

import asyncio
import os
import random
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from llmcascade.adapters.base import LLMResponse
from llmcascade.exceptions import AllModelsExhaustedError, ProviderError, safe_error_message
from llmcascade.cascade import classify_failure
from llmcascade.failures import classify_recorded_failure
from llmcascade.event_log import events
from llmcascade.metrics import log, metrics
from llmcascade.rate_limiter import RateLimiter
from llmcascade.registry import ModelConfig
from llmcascade.stats_store import NullStatsStore, StatsStore
from llmcascade.tokens import estimate_tokens

Strategy = Literal["headroom", "round_robin", "least_used", "priority_first", "weighted"]
STRATEGIES: frozenset[str] = frozenset(
    ("headroom", "round_robin", "least_used", "priority_first", "weighted")
)
Executor = Callable[[ModelConfig, str], Awaitable[LLMResponse]]

RETRY_SLEEP_S = 0.25


def strategy_from_env(default: Strategy = "headroom") -> Strategy:
    raw = (os.environ.get("LLMCASCADE_STRATEGY") or default).strip().lower()
    return raw if raw in STRATEGIES else default  # type: ignore[return-value]


def _notes_key(notes: str | None) -> str:
    return (notes or "").strip() or "_"


def allow_paid_models() -> bool:
    """When false (default), models with key_tier=paid are excluded from auto-select."""
    return (os.environ.get("ALLOW_PAID") or "false").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _weighted_pick(eligible: list[ModelConfig]) -> ModelConfig:
    weights = [max(1, int(getattr(m, "weight", 1) or 1)) for m in eligible]
    return random.choices(eligible, weights=weights, k=1)[0]


class ModelSelector:
    def __init__(
        self,
        registry: list[ModelConfig],
        rate_limiter: RateLimiter,
        strategy: Strategy = "headroom",
        stats: StatsStore | NullStatsStore | None = None,
        *,
        cooldowns: Any | None = None,
        quota_learn: Any | None = None,
    ) -> None:
        self.registry = list(registry)
        self.rate_limiter = rate_limiter
        self.strategy = strategy
        self.stats: StatsStore | NullStatsStore = stats or NullStatsStore()
        self.cooldowns = cooldowns
        self.quota_learn = quota_learn
        self._rr_index = 0
        self._last_good: dict[str, str] = {}

    async def _eligible(self, capability: str, tokens_estimate: int) -> list[ModelConfig]:
        paid_ok = allow_paid_models()
        out: list[ModelConfig] = []
        for m in self.registry:
            if not getattr(m, "enabled", True):
                continue
            if capability not in m.capabilities:
                continue
            if getattr(m, "key_tier", "free") == "paid" and not paid_ok:
                continue
            if await self.rate_limiter.can_proceed(m.name, tokens_estimate):
                out.append(m)
        return out

    async def _rpd_score(self, model: ModelConfig) -> tuple[int, int]:
        """Lower source rank wins: live headers, then quota_learn, then YAML."""
        live = self.rate_limiter.live_rpd(model.name)
        if live is not None:
            return (0, live)
        if self.quota_learn is not None:
            learned = self.quota_learn.remaining_rpd(model.name)
            if learned is not None:
                return (1, learned)
        rem = await self.rate_limiter.remaining_budget(model.name)
        return (2, int(rem.get("rpd") or 0))

    async def _headroom_pick(self, eligible: list[ModelConfig]) -> ModelConfig:
        scored: list[tuple[int, int, int, str, ModelConfig]] = []
        for m in eligible:
            src, rpd = await self._rpd_score(m)
            scored.append((src, -rpd, m.priority, m.name, m))
        scored.sort()
        return scored[0][-1]

    def _filter_eligible(
        self,
        eligible: list[ModelConfig],
        exclude: set[str] | None,
    ) -> list[ModelConfig]:
        if not exclude:
            return eligible
        return [m for m in eligible if m.name not in exclude]

    def _sticky(
        self,
        eligible: list[ModelConfig],
        notes: str | None,
    ) -> ModelConfig | None:
        last = self._last_good.get(_notes_key(notes))
        if not last:
            return None
        return next((m for m in eligible if m.name == last), None)

    async def pick(
        self,
        capability: str,
        tokens_estimate: int = 1,
        *,
        notes: str | None = None,
        exclude: set[str] | None = None,
    ) -> ModelConfig | None:
        eligible = self._filter_eligible(
            await self._eligible(capability, tokens_estimate), exclude
        )
        if not eligible:
            return None
        if self.strategy == "headroom":
            sticky = self._sticky(eligible, notes)
            if sticky is not None:
                return sticky
            return await self._headroom_pick(eligible)
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
        return _weighted_pick(eligible)

    async def peek(
        self,
        capability: str,
        tokens_estimate: int = 1,
        *,
        notes: str | None = None,
        exclude: set[str] | None = None,
    ) -> ModelConfig | None:
        """Next pick without advancing round-robin state (safe for dashboards)."""
        eligible = self._filter_eligible(
            await self._eligible(capability, tokens_estimate), exclude
        )
        if not eligible:
            return None
        if self.strategy == "headroom":
            sticky = self._sticky(eligible, notes)
            if sticky is not None:
                return sticky
            return await self._headroom_pick(eligible)
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
            # Retry only timeouts/408. 5xx must fail over — a 60s retry blocks the cascade.
            if exc.retryable and exc.status_code in (None, 408):
                await asyncio.sleep(RETRY_SLEEP_S)
                return await executor(model, prompt)
            raise

    def _with_skipped(self, resp: LLMResponse, skipped: list[dict[str, str]]) -> LLMResponse:
        if not skipped:
            return resp
        return resp.model_copy(update={"skipped_models": skipped})

    async def dispatch_with_fallback(
        self,
        prompt: str,
        capability: str,
        executor: Executor,
        *,
        notes: str | None = None,
        pinned_model: str | None = None,
        fallback: bool | None = None,
        failover_models: list[str] | None = None,
        include_free_cascade: bool | None = None,
        **_params: Any,
    ) -> LLMResponse:
        tokens_est = estimate_tokens(prompt)
        tried: set[str] = set()
        last_err: Exception | None = None
        note = (notes or "").strip() or None
        note_detail = {"notes": note} if note else {}
        allow_fallback = capability != "embed" if fallback is None else fallback
        pin = (pinned_model or "").strip() or None
        budget_blocked = False
        skipped_models: list[dict[str, str]] = []

        preferred: list[str] = []
        if pin:
            seen: set[str] = set()
            extra = failover_models or [] if capability != "embed" else []
            for name in [pin, *extra]:
                n = (name or "").strip()
                if not n or n in seen:
                    continue
                seen.add(n)
                preferred.append(n)

        if pin and capability != "embed":
            include_free = bool(include_free_cascade)
        elif capability == "embed":
            include_free = False
        else:
            include_free = True

        async def succeed(model: ModelConfig, resp: LLMResponse) -> LLMResponse:
            used = resp.tokens_used or tokens_est
            metrics.record_success(model.name, capability)
            await self.stats.record(
                model=model.name,
                provider=model.provider,
                success=True,
                latency_ms=resp.latency_ms,
                tokens_used=used,
                notes=note,
                capability=capability,
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
                    "dimensions": resp.dimensions,
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
            await self.rate_limiter.ingest_headers(
                model.name, getattr(resp, "headers", None) or None, provider=model.provider
            )
            if self.quota_learn is not None:
                self.quota_learn.record_success(
                    (resp.model or model.name), model.provider
                )
            if capability != "embed":
                self._last_good[_notes_key(note)] = model.name
            return self._with_skipped(resp, skipped_models)

        async def fail(model: ModelConfig, exc: ProviderError) -> None:
            nonlocal last_err
            last_err = exc
            metrics.record_failure(model.name, capability)
            await self.stats.record(
                model=model.name,
                provider=model.provider,
                success=False,
                latency_ms=0,
                tokens_used=0,
                notes=note,
                capability=capability,
            )
            self.stats.enqueue_failure(
                model=model.name,
                provider=model.provider,
                capability=capability,
                notes=note,
                exc=exc,
            )
            await self.rate_limiter.ingest_headers(
                model.name, getattr(exc, "headers", None), provider=model.provider
            )
            if self.quota_learn is not None:
                kind = classify_failure(exc.status_code, str(exc))
                self.quota_learn.record_limit(
                    (exc.model or model.name), model.provider, kind
                )
            if self.cooldowns is not None and not (
                model.provider == "gemini" and bool(model.cascade)
            ):
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
                kind=classify_recorded_failure(exc.status_code, str(exc)),
                capability=capability,
                **note_detail,
            )

        for name in preferred:
            model = next((m for m in self.registry if m.name == name), None)
            if model is None:
                skipped_models.append({"model": name, "reason": "unknown"})
                continue
            if capability not in model.capabilities or not getattr(model, "enabled", True):
                skipped_models.append({"model": name, "reason": "unavailable"})
                continue
            if name in tried:
                continue
            if not await self.rate_limiter.try_reserve(model.name, tokens_est):
                budget_blocked = True
                if capability == "embed":
                    break
                skipped_models.append({"model": name, "reason": "budget"})
                continue
            tried.add(model.name)
            try:
                return await succeed(model, await self._try_model(model, prompt, executor))
            except ProviderError as exc:
                skipped_models.append({"model": model.name, "reason": "error"})
                await fail(model, exc)
                if not allow_fallback:
                    break

        while include_free:
            model = await self.pick(
                capability, tokens_est, notes=note, exclude=tried
            )
            if model is None:
                remaining = [
                    m for m in await self._eligible(capability, tokens_est) if m.name not in tried
                ]
                if not remaining:
                    break
                model = remaining[0]
            if not await self.rate_limiter.try_reserve(model.name, tokens_est):
                skipped_models.append({"model": model.name, "reason": "budget"})
                tried.add(model.name)
                continue
            tried.add(model.name)
            try:
                return await succeed(model, await self._try_model(model, prompt, executor))
            except ProviderError as exc:
                skipped_models.append({"model": model.name, "reason": "error"})
                await fail(model, exc)
                if not allow_fallback:
                    break
                continue

        if pin and capability == "embed":
            if budget_blocked:
                rem = await self.rate_limiter.remaining_budget(pin)
                msg = (
                    f"embedding model {pin!r} local budget exhausted "
                    f"(rps {rem.get('rps', 0)} · rpm {rem.get('rpm', 0)} · rpd {rem.get('rpd', 0)} left)"
                )
                status = 429
            else:
                msg = f"embedding model {pin!r} failed or is unavailable"
                status = 503
        else:
            msg = f"no free-tier model succeeded for capability={capability!r}"
            status = 503
        if last_err is not None:
            msg = f"{msg}; last error: {safe_error_message(last_err)}"
        events.record(
            msg.split(";")[0],
            level="error",
            type="request_fail",
            capability=capability,
            model=pin,
            error=safe_error_message(last_err) if last_err else None,
            **note_detail,
        )
        raise AllModelsExhaustedError(msg, http_status=status, skipped_models=skipped_models)
