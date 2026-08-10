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


def test_safe_error_message_strips_body():
    exc = ProviderError(
        "groq HTTP 500: {\"error\":\"internal leak\"}",
        status_code=500,
        provider="groq",
        model="m1",
    )
    assert safe_error_message(exc) == "groq/m1 HTTP 500"
    assert "leak" not in safe_error_message(exc)
