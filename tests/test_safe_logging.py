"""Ensure provider response bodies never appear in the event ring."""

from __future__ import annotations

import pytest

from llmcascade.adapters.base import LLMResponse
from llmcascade.event_log import EventLog
from llmcascade.exceptions import ProviderError, safe_error_message
from llmcascade.rate_limiter import RateLimiter
from llmcascade.registry import Limits, ModelConfig
from llmcascade.selector import ModelSelector


@pytest.mark.asyncio
async def test_events_omit_provider_body(monkeypatch: pytest.MonkeyPatch):
    from llmcascade import selector as selector_mod

    ring = EventLog(maxlen=50)
    monkeypatch.setattr(selector_mod, "events", ring)

    model = ModelConfig(
        name="m1",
        provider="groq",
        endpoint="https://example.com",
        auth_env_var="GROQ_API_KEY",
        limits=Limits(rpd=10, rpm=10, rps=5, tpm=1000, max_context=1024),
        capabilities=["chat"],
    )
    limiter = RateLimiter([model])
    sel = ModelSelector([model], limiter)

    secret_body = "SUPER_SECRET_PROVIDER_BODY_TOKEN_XYZ"

    async def boom(m, prompt):
        raise ProviderError(
            f"groq HTTP 429: {secret_body}",
            status_code=429,
            provider="groq",
            model="m1",
        )

    with pytest.raises(Exception):
        await sel.dispatch_with_fallback("hi", "chat", boom)

    blob = str(ring.events())
    assert secret_body not in blob
    assert "HTTP 429" in blob or "request fail" in blob


@pytest.mark.asyncio
async def test_request_ok_event_includes_prompt_not_response(monkeypatch: pytest.MonkeyPatch):
    from llmcascade import selector as selector_mod

    ring = EventLog(maxlen=50)
    monkeypatch.setattr(selector_mod, "events", ring)

    model = ModelConfig(
        name="m1",
        provider="groq",
        endpoint="https://example.com",
        auth_env_var="GROQ_API_KEY",
        limits=Limits(rpd=10, rpm=10, rps=5, tpm=1000, max_context=1024),
        capabilities=["chat"],
    )
    limiter = RateLimiter([model])
    sel = ModelSelector([model], limiter)

    user_prompt = "USER_PROMPT_FOR_DASHBOARD_XYZ"
    secret_completion = "SECRET_COMPLETION_TEXT_ABC"

    async def ok_executor(m, prompt):
        assert prompt == user_prompt
        return LLMResponse(text=secret_completion, model=m.name, tokens_used=4)

    resp = await sel.dispatch_with_fallback(user_prompt, "chat", ok_executor)
    assert resp.text == secret_completion

    ev = ring.events()[0]
    assert ev["type"] == "request_ok"
    assert ev["detail"]["prompt"] == user_prompt
    blob = str(ring.events())
    assert secret_completion not in blob


def test_truncate_event_detail_prompt():
    from llmcascade.event_log import EVENT_DETAIL_PROMPT_MAX, truncate_event_detail_prompt

    assert truncate_event_detail_prompt(None) is None
    short = "hello"
    assert truncate_event_detail_prompt(short) == short
    long = "x" * (EVENT_DETAIL_PROMPT_MAX + 10)
    out = truncate_event_detail_prompt(long)
    assert len(out) == EVENT_DETAIL_PROMPT_MAX + 1
    assert out.endswith("…")


def test_safe_error_message_strips_body():
    exc = ProviderError(
        "groq HTTP 500: {\"error\":\"internal leak\"}",
        status_code=500,
        provider="groq",
        model="m1",
    )
    assert safe_error_message(exc) == "groq/m1 HTTP 500"
    assert "leak" not in safe_error_message(exc)
