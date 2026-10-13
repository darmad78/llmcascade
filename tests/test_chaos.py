"""Simulated rate-limit chaos: invariants under concurrent failover."""

from __future__ import annotations

import asyncio
import random

import httpx
import pytest

from llmcascade.exceptions import AllModelsExhaustedError, ProviderError
from llmcascade.queue_worker import RouterClient
from llmcascade.registry import Limits, ModelConfig


def _models(n: int = 3) -> list[ModelConfig]:
    return [
        ModelConfig(
            name=f"m{i}",
            provider="groq",
            endpoint="https://example.com/v1/chat/completions",
            auth_env_var="GROQ_API_KEY",
            limits=Limits(rpd=50, rpm=30, rps=20, tpm=60000, max_context=8192),
            capabilities=["chat"],
            priority=i,
        )
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_chaos_only_expected_exceptions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    rng = random.Random(42)

    async def handler(request: httpx.Request) -> httpx.Response:
        roll = rng.random()
        if roll < 0.3:
            return httpx.Response(429, text="rate limit", headers={"retry-after": "1"})
        if roll < 0.4:
            return httpx.Response(500, text="boom")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 2}},
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    router = RouterClient(registry=_models(3), workers=2, max_queue=50)
    router._client = client
    await router.start()

    from llmcascade.adapters import get_adapter

    async def patched_execute(model, prompt, **kw):
        ad = get_adapter(model, client=client)
        return await ad.send(prompt)

    router._execute = patched_execute  # type: ignore[method-assign]

    async def one() -> None:
        try:
            await router.submit("ping")
        except (AllModelsExhaustedError, ProviderError):
            pass
        except Exception as exc:
            raise AssertionError(f"unexpected {type(exc)}") from exc

    await asyncio.gather(*[one() for _ in range(30)])
    await router.shutdown()
