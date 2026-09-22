from __future__ import annotations

import pytest

from llmcascade.exceptions import ProviderError
from llmcascade.rate_limiter import RateLimiter
from llmcascade.registry import Limits, ModelConfig
from llmcascade.selector import ModelSelector
from llmcascade.adapters.base import LLMResponse


def _m(name: str, provider: str = "p") -> ModelConfig:
    return ModelConfig(
        name=name,
        provider=provider,
        endpoint="https://example.com/v1",
        auth_env_var="KEY",
        limits=Limits(rpd=100, rpm=100, rps=10, tpm=100000, max_context=4096),
        capabilities=["chat"],
        priority=1,
    )


@pytest.mark.asyncio
async def test_failed_failover_does_not_burn_rpd_budget():
    models = [_m("a"), _m("b")]
    lim = RateLimiter(models)
    sel = ModelSelector(models, lim)

    async def executor(model, prompt):
        if model.name == "a":
            raise ProviderError("HTTP 403: forbidden", status_code=403, retryable=False, model="a")
        return LLMResponse(text="ok", model=model.name, tokens_used=5)

    await sel.dispatch_with_fallback("hi", "chat", executor)
    rem_a = await lim.remaining_budget("a")
    rem_b = await lim.remaining_budget("b")
    assert rem_a["rpd"] == 100
    assert rem_b["rpd"] == 99
