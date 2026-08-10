from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from llmrouter.adapters.base import LLMResponse
from llmrouter.cascade import (
    GeminiCascadeManager,
    classify_failure,
    cooldown_until,
    needs_thinking_budget_zero,
    resolve_cascade_order,
)
from llmrouter.exceptions import ProviderError
from llmrouter.rate_limiter import RateLimiter
from llmrouter.registry import Limits, ModelConfig
from llmrouter.selector import ModelSelector


def test_classify_daily():
    assert classify_failure(429, "PerDay quota exceeded") == "daily"
    assert classify_failure(429, "daily quota limit") == "daily"
    assert classify_failure(403, "limit:0") == "daily"
    assert classify_failure(429, "Rate limit exceeded: free-models-per-day") == "daily"


def test_classify_credit():
    assert classify_failure(402, "Insufficient Balance") == "credit"
    assert classify_failure(402, '{"error":{"type":"credit_limit"}}') == "credit"
    assert classify_failure(400, "Credit limit exceeded, please add credits") == "credit"


def test_classify_rate():
    assert classify_failure(429, "rate limit") == "rate"
    assert classify_failure(503, "unavailable") == "rate"
    assert classify_failure(None, "timeout") == "rate"


def test_classify_permanent():
    assert classify_failure(404, "model not found") == "permanent"
    assert classify_failure(400, "model is not supported") == "permanent"


def test_classify_transient():
    assert classify_failure(400, "bad request") == "transient"


def test_cooldown_durations():
    now = datetime(2026, 8, 6, 20, 0, tzinfo=timezone.utc)
    assert cooldown_until("transient", now=now) is None
    rate = cooldown_until("rate", now=now)
    assert rate == now + timedelta(seconds=60)
    perm = cooldown_until("permanent", now=now)
    assert perm == now + timedelta(days=365)
    credit = cooldown_until("credit", now=now)
    assert credit == now + timedelta(hours=24)
    daily = cooldown_until("daily", now=now)
    assert daily is not None
    assert daily > now


def test_cooldown_learns_retry_after():
    now = datetime(2026, 8, 6, 20, 0, tzinfo=timezone.utc)
    until = cooldown_until("rate", now=now, headers={"Retry-After": "90"})
    assert until == now + timedelta(seconds=90)


def test_cooldown_learns_ratelimit_reset_ms():
    now = datetime(2026, 8, 6, 20, 0, tzinfo=timezone.utc)
    reset_ms = int((now + timedelta(hours=2)).timestamp() * 1000)
    until = cooldown_until("daily", now=now, headers={"X-RateLimit-Reset": str(reset_ms)})
    assert until is not None
    assert abs((until - now).total_seconds() - 7200) < 2


def test_thinking_budget_flag():
    assert needs_thinking_budget_zero("gemini-2.5-flash")
    assert needs_thinking_budget_zero("gemini-3.6-flash")
    assert not needs_thinking_budget_zero("gemini-2.0-flash")


def test_resolve_cascade_prefers_env(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-flash")
    order = resolve_cascade_order(["gemini-2.5-flash", "gemini-2.0-flash"])
    assert order[0] == "gemini-2.0-flash"
    assert order.count("gemini-2.0-flash") == 1


def test_effective_cascade_hides_and_reorders(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMROUTER_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    from llmrouter.cascade import effective_cascade_members
    from llmrouter.model_store import set_override

    base = ["a", "b", "c"]
    assert effective_cascade_members(base) == base
    set_override("b", enabled=False)
    set_override("c", weight=10)
    assert effective_cascade_members(base) == ["c", "a"]


@pytest.mark.asyncio
async def test_quota_cools_only_failed_model():
    mgr = GeminiCascadeManager(["a", "b", "c"])
    await mgr.apply_cooldown(
        "a",
        "rate",
        body="You exceeded your current quota for this project",
    )
    assert await mgr.is_cooling("a")
    assert not await mgr.is_cooling("b")
    assert not await mgr.is_cooling("c")
    assert await mgr.any_available()


@pytest.mark.asyncio
async def test_skip_cooling_model():
    mgr = GeminiCascadeManager(["a", "b"])
    await mgr.apply_cooldown("a", "rate", body="429")
    calls: list[str] = []

    async def send(model_id: str, prompt: str) -> LLMResponse:
        calls.append(model_id)
        return LLMResponse(text="ok", model=model_id, tokens_used=1)

    resp = await mgr.run(send, "hi")
    assert resp.model == "b"
    assert calls == ["b"]


@pytest.mark.asyncio
async def test_cascade_advances_on_failure():
    mgr = GeminiCascadeManager(["a", "b"])
    calls: list[str] = []

    async def send(model_id: str, prompt: str) -> LLMResponse:
        calls.append(model_id)
        if model_id == "a":
            raise ProviderError("HTTP 503", status_code=503, retryable=True, model="a")
        return LLMResponse(text="ok", model=model_id, tokens_used=1)

    resp = await mgr.run(send, "hi")
    assert resp.model == "b"
    assert calls == ["a", "b"]
    assert await mgr.is_cooling("a")


@pytest.mark.asyncio
async def test_permanent_excludes_model():
    mgr = GeminiCascadeManager(["a", "b"])
    await mgr.apply_cooldown("a", "permanent", body="404 not found")
    status = await mgr.status()
    assert "a" not in status["available"]
    assert "a" in status["available_at"]
    assert "b" in status["available"]


@pytest.mark.asyncio
async def test_default_no_wait_raises_when_all_cooling():
    mgr = GeminiCascadeManager(["a"])
    await mgr.apply_cooldown("a", "rate", body="429")

    async def send(model_id: str, prompt: str) -> LLMResponse:
        raise AssertionError("should not call")

    with pytest.raises(ProviderError, match="cascade exhausted"):
        await mgr.run(send, "hi", wait_for_gemini=False)


@pytest.mark.asyncio
async def test_wait_for_gemini_retries(monkeypatch):
    mgr = GeminiCascadeManager(["a"])
    now = datetime.now(timezone.utc)
    async with mgr._lock:
        mgr._cooldowns["a"] = now + timedelta(seconds=0.05)

    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)
        async with mgr._lock:
            mgr._cooldowns.pop("a", None)

    monkeypatch.setattr("llmrouter.cascade.asyncio.sleep", fake_sleep)

    async def send(model_id: str, prompt: str) -> LLMResponse:
        return LLMResponse(text="ok", model=model_id, tokens_used=1)

    resp = await mgr.run(send, "hi", wait_for_gemini=True)
    assert resp.model == "a"
    assert sleeps


@pytest.mark.asyncio
async def test_rate_limiter_blocks_cooling_family():
    models = [
        ModelConfig(
            name="gemini",
            provider="gemini",
            endpoint="https://example.com/models/{model}:generateContent",
            auth_env_var="GOOGLE_API_KEY",
            limits=Limits(rpd=100, rpm=100, rps=100, tpm=100000, max_context=4096),
            capabilities=["chat"],
            priority=1,
            cascade=["a", "b"],
        ),
        ModelConfig(
            name="other",
            provider="groq",
            endpoint="https://example.com",
            auth_env_var="GROQ_API_KEY",
            limits=Limits(rpd=100, rpm=100, rps=100, tpm=100000, max_context=4096),
            capabilities=["chat"],
            priority=2,
        ),
    ]
    mgr = GeminiCascadeManager(["a", "b"])
    mgr.bind_logical("gemini")
    await mgr.apply_cooldown("a", "rate", body="429")
    await mgr.apply_cooldown("b", "rate", body="429")
    lim = RateLimiter(models, gemini_cascade=mgr)
    assert not await lim.can_proceed("gemini", 1)
    assert await lim.can_proceed("other", 1)
    sel = ModelSelector(models, lim)
    picked = await sel.pick("chat")
    assert picked is not None
    assert picked.name == "other"
