import pytest

from llmrouter.adapters.base import LLMResponse
from llmrouter.exceptions import AllModelsExhaustedError, ProviderError
from llmrouter.rate_limiter import RateLimiter
from llmrouter.registry import Limits, ModelConfig
from llmrouter.selector import ModelSelector
from datetime import datetime, timezone


def _m(name: str, priority: int = 1) -> ModelConfig:
    return ModelConfig(
        name=name,
        provider="groq",
        endpoint="https://example.com",
        auth_env_var="GROQ_API_KEY",
        limits=Limits(rpd=100, rpm=100, rps=100, tpm=100000, max_context=4096),
        capabilities=["chat"],
        priority=priority,
    )


@pytest.mark.asyncio
async def test_round_robin_pick():
    models = [_m("a"), _m("b"), _m("c")]
    sel = ModelSelector(models, RateLimiter(models), strategy="round_robin")
    names = [(await sel.pick("chat")).name for _ in range(3)]
    assert names == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_fallback_on_provider_error():
    models = [_m("a"), _m("b")]
    sel = ModelSelector(models, RateLimiter(models))
    calls: list[str] = []

    async def executor(model, prompt):
        calls.append(model.name)
        if model.name == "a":
            raise ProviderError("fail a", status_code=500, retryable=False, model="a")
        return LLMResponse(text="ok", model=model.name, tokens_used=5)

    resp = await sel.dispatch_with_fallback("hi", "chat", executor)
    assert resp.model == "b"
    assert calls == ["a", "b"]


@pytest.mark.asyncio
async def test_retryable_retries_same_model(monkeypatch):
    models = [_m("a")]
    sel = ModelSelector(models, RateLimiter(models))
    calls = {"n": 0}

    async def no_sleep(_):
        return None

    monkeypatch.setattr("llmrouter.selector.asyncio.sleep", no_sleep)

    async def executor(model, prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ProviderError("temp", status_code=503, retryable=True, model="a")
        return LLMResponse(text="ok", model="a", tokens_used=3)

    resp = await sel.dispatch_with_fallback("hi", "chat", executor)
    assert resp.text == "ok"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_exhaustion():
    models = [_m("a"), _m("b")]
    sel = ModelSelector(models, RateLimiter(models))

    async def executor(model, prompt):
        raise ProviderError("down", status_code=500, retryable=False, model=model.name)

    with pytest.raises(AllModelsExhaustedError):
        await sel.dispatch_with_fallback("hi", "chat", executor)


@pytest.mark.asyncio
async def test_budget_excludes_model():
    models = [
        ModelConfig(
            name="tight",
            provider="groq",
            endpoint="https://example.com",
            auth_env_var="GROQ_API_KEY",
            limits=Limits(rpd=1, rpm=1, rps=1, tpm=10, max_context=4096),
            capabilities=["chat"],
            priority=1,
        ),
        _m("ok"),
    ]
    lim = RateLimiter(models)
    await lim.record_usage("tight", 1)
    sel = ModelSelector(models, lim)
    picked = await sel.pick("chat", tokens_estimate=1)
    assert picked is not None
    assert picked.name == "ok"


@pytest.mark.asyncio
async def test_credit_cooldown_skips_model_on_next_pick():
    from llmrouter.cascade import ModelCooldownTracker

    models = [_m("a"), _m("b")]
    cool = ModelCooldownTracker()
    lim = RateLimiter(models, cooldowns=cool)
    sel = ModelSelector(models, lim, cooldowns=cool)
    calls: list[str] = []

    async def executor(model, prompt):
        calls.append(model.name)
        if model.name == "a":
            raise ProviderError(
                "deepseek HTTP 402: Insufficient Balance",
                status_code=402,
                retryable=False,
                model="a",
            )
        return LLMResponse(text="ok", model=model.name, tokens_used=1)

    resp = await sel.dispatch_with_fallback("hi", "chat", executor)
    assert resp.model == "b"
    assert await cool.is_cooling("a")
    assert not await lim.can_proceed("a", 1)
    picked = await sel.pick("chat")
    assert picked is not None
    assert picked.name == "b"


@pytest.mark.asyncio
async def test_rate_cooldown_learns_retry_after():
    from llmrouter.cascade import ModelCooldownTracker

    models = [_m("a"), _m("b")]
    cool = ModelCooldownTracker()
    lim = RateLimiter(models, cooldowns=cool)
    sel = ModelSelector(models, lim, cooldowns=cool)

    async def executor(model, prompt):
        if model.name == "a":
            raise ProviderError(
                "sambanova HTTP 429: Rate limit exceeded",
                status_code=429,
                retryable=False,
                model="a",
                headers={"Retry-After": "120"},
            )
        return LLMResponse(text="ok", model=model.name, tokens_used=1)

    await sel.dispatch_with_fallback("hi", "chat", executor)
    until = await cool.available_at("a")
    assert until is not None
    remaining = (until - datetime.now(timezone.utc)).total_seconds()
    assert 100 <= remaining <= 120
