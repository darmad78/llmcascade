import pytest

from llmcascade.adapters.base import LLMResponse
from llmcascade.exceptions import AllModelsExhaustedError, ProviderError
from llmcascade.rate_limiter import RateLimiter
from llmcascade.registry import Limits, ModelConfig
from llmcascade.selector import ModelSelector
from datetime import datetime, timezone


def _m(
    name: str,
    priority: int = 1,
    *,
    key_tier: str = "free",
    provider: str = "groq",
    rpd: int = 100,
) -> ModelConfig:
    return ModelConfig(
        name=name,
        provider=provider,
        endpoint="https://example.com",
        auth_env_var="GROQ_API_KEY",
        limits=Limits(rpd=rpd, rpm=100, rps=100, tpm=100000, max_context=4096),
        capabilities=["chat"],
        priority=priority,
        key_tier=key_tier,  # type: ignore[arg-type]
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

    monkeypatch.setattr("llmcascade.selector.asyncio.sleep", no_sleep)

    async def executor(model, prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ProviderError("temp", status_code=None, retryable=True, model="a")
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
    from llmcascade.cascade import ModelCooldownTracker

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
async def test_embed_capability_isolated():
    chat = _m("chat-m")
    embed = ModelConfig(
        name="embed-m",
        provider="groq",
        endpoint="https://example.com",
        auth_env_var="GROQ_API_KEY",
        limits=Limits(rpd=100, rpm=100, rps=100, tpm=100000, max_context=4096),
        capabilities=["embed"],
        priority=1,
    )
    sel = ModelSelector([chat, embed], RateLimiter([chat, embed]))
    assert (await sel.pick("chat")).name == "chat-m"
    assert (await sel.pick("embed")).name == "embed-m"

    async def executor(model, prompt):
        return LLMResponse(
            model=model.name,
            embedding=[1.0],
            dimensions=1,
            tokens_used=2,
        )

    resp = await sel.dispatch_with_fallback("doc", "embed", executor, pinned_model="embed-m")
    assert resp.embedding == [1.0]
    assert resp.model == "embed-m"


@pytest.mark.asyncio
async def test_embed_does_not_fallback_to_another_model():
    a = ModelConfig(
        name="embed-a",
        provider="mistral",
        endpoint="https://example.com",
        auth_env_var="MISTRAL_API_KEY",
        limits=Limits(rpd=100, rpm=100, rps=100, tpm=100000, max_context=4096),
        capabilities=["embed"],
        priority=1,
    )
    b = ModelConfig(
        name="embed-b",
        provider="jina",
        endpoint="https://example.com",
        auth_env_var="JINA_API_KEY",
        limits=Limits(rpd=100, rpm=100, rps=100, tpm=100000, max_context=4096),
        capabilities=["embed"],
        priority=2,
    )
    sel = ModelSelector([a, b], RateLimiter([a, b]))
    calls: list[str] = []

    async def executor(model, prompt):
        calls.append(model.name)
        raise ProviderError("fail", status_code=500, retryable=False, model=model.name)

    with pytest.raises(AllModelsExhaustedError, match="embed-a"):
        await sel.dispatch_with_fallback("doc", "embed", executor, pinned_model="embed-a")
    assert calls == ["embed-a"]


@pytest.mark.asyncio
async def test_embed_budget_exhausted_is_429():
    m = ModelConfig(
        name="embed-a",
        provider="gemini",
        endpoint="https://example.com",
        auth_env_var="GOOGLE_API_KEY",
        limits=Limits(rpd=100, rpm=1, rps=1, tpm=100000, max_context=2048),
        capabilities=["embed"],
        priority=1,
    )
    lim = RateLimiter([m])
    sel = ModelSelector([m], lim)

    async def executor(model, prompt):
        return LLMResponse(text="", model=model.name, embedding=[0.1], dimensions=1)

    await sel.dispatch_with_fallback("a", "embed", executor, pinned_model="embed-a")
    with pytest.raises(AllModelsExhaustedError, match="local budget exhausted") as exc:
        await sel.dispatch_with_fallback("b", "embed", executor, pinned_model="embed-a")
    assert exc.value.http_status == 429


@pytest.mark.asyncio
async def test_rate_cooldown_learns_retry_after():
    from llmcascade.cascade import ModelCooldownTracker

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


@pytest.mark.asyncio
async def test_paid_models_excluded_unless_allow_paid(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ALLOW_PAID", raising=False)
    models = [_m("free-m"), _m("paid-m", key_tier="paid")]
    lim = RateLimiter(models)
    sel = ModelSelector(models, lim, strategy="round_robin")
    eligible = await sel._eligible("chat", 1)
    assert [m.name for m in eligible] == ["free-m"]

    monkeypatch.setenv("ALLOW_PAID", "true")
    eligible = await sel._eligible("chat", 1)
    assert {m.name for m in eligible} == {"free-m", "paid-m"}


@pytest.mark.asyncio
async def test_paid_only_registry_exhausted_when_gated(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ALLOW_PAID", raising=False)
    models = [_m("paid-only", key_tier="paid")]
    sel = ModelSelector(models, RateLimiter(models))
    assert await sel.pick("chat") is None


@pytest.mark.asyncio
async def test_preferred_then_failover_then_free():
    models = [_m("a"), _m("b"), _m("c")]
    sel = ModelSelector(models, RateLimiter(models), strategy="priority_first")
    calls: list[str] = []

    async def executor(model, prompt):
        calls.append(model.name)
        if model.name in {"a", "b"}:
            raise ProviderError("fail", status_code=500, retryable=False, model=model.name)
        return LLMResponse(text="ok", model=model.name, tokens_used=1)

    resp = await sel.dispatch_with_fallback(
        "hi",
        "chat",
        executor,
        pinned_model="a",
        failover_models=["b"],
        include_free_cascade=True,
    )
    assert resp.model == "c"
    assert calls == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_unknown_preferred_skipped_and_reported():
    models = [_m("b")]
    sel = ModelSelector(models, RateLimiter(models))

    async def executor(model, prompt):
        return LLMResponse(text="ok", model=model.name, tokens_used=1)

    resp = await sel.dispatch_with_fallback(
        "hi",
        "chat",
        executor,
        pinned_model="missing",
        failover_models=["b"],
        include_free_cascade=False,
    )
    assert resp.model == "b"
    assert resp.skipped_models == [{"model": "missing", "reason": "unknown"}]


@pytest.mark.asyncio
async def test_no_free_cascade_does_not_use_other_registry_models():
    models = [_m("a"), _m("b")]
    sel = ModelSelector(models, RateLimiter(models))
    calls: list[str] = []

    async def executor(model, prompt):
        calls.append(model.name)
        raise ProviderError("fail", status_code=500, retryable=False, model=model.name)

    with pytest.raises(AllModelsExhaustedError) as exc:
        await sel.dispatch_with_fallback(
            "hi",
            "chat",
            executor,
            pinned_model="a",
            include_free_cascade=False,
        )
    assert calls == ["a"]
    assert exc.value.skipped_models == [{"model": "a", "reason": "error"}]


@pytest.mark.asyncio
async def test_5xx_failsover_without_same_model_retry():
    models = [_m("a"), _m("b")]
    sel = ModelSelector(models, RateLimiter(models), strategy="priority_first")
    calls: list[str] = []

    async def executor(model, prompt):
        calls.append(model.name)
        if model.name == "a":
            raise ProviderError("fail a", status_code=503, retryable=True, model="a")
        return LLMResponse(text="ok", model=model.name, tokens_used=1)

    resp = await sel.dispatch_with_fallback("hi", "chat", executor)
    assert resp.model == "b"
    assert calls == ["a", "b"]


@pytest.mark.asyncio
async def test_headroom_prefers_live_then_learned_then_yaml():
    a = _m("a", provider="groq", rpd=1000)
    b = _m("b", provider="together", rpd=50)

    class Learn:
        def remaining_rpd(self, model_id, **_kw):
            return 12 if model_id == "b" else None

    lim = RateLimiter([a, b])
    sel = ModelSelector([a, b], lim, strategy="headroom", quota_learn=Learn())
    assert (await sel.pick("chat")).name == "b"
    await lim.ingest_headers(
        "a",
        {"x-ratelimit-remaining-requests-day": "3"},
        provider="groq",
    )
    assert (await sel.pick("chat")).name == "a"


@pytest.mark.asyncio
async def test_last_good_is_per_notes():
    a = _m("a", provider="groq", rpd=10)
    b = _m("b", provider="together", rpd=1000)
    lim = RateLimiter([a, b])
    sel = ModelSelector([a, b], lim, strategy="headroom")

    async def executor(model, prompt):
        return LLMResponse(text="ok", model=model.name, tokens_used=1)

    resp = await sel.dispatch_with_fallback("hi", "chat", executor, notes="app1")
    assert resp.model == "b"
    await lim.ingest_headers(
        "a",
        {"x-ratelimit-remaining-requests-day": "9000"},
        provider="groq",
    )
    assert (await sel.pick("chat", notes="app1")).name == "b"
    assert (await sel.pick("chat", notes="app2")).name == "a"
