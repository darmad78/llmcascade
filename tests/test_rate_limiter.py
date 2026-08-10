import asyncio

import pytest

from llmcascade.rate_limiter import RateLimiter
from llmcascade.registry import Limits, ModelConfig


def _model(**overrides) -> ModelConfig:
    base = dict(
        name="m1",
        provider="groq",
        endpoint="https://example.com",
        auth_env_var="GROQ_API_KEY",
        limits=Limits(rpd=100, rpm=10, rps=2, tpm=100, max_context=4096),
        capabilities=["chat"],
        priority=1,
    )
    base.update(overrides)
    return ModelConfig(**base)


@pytest.mark.asyncio
async def test_can_proceed_and_record():
    lim = RateLimiter([_model()])
    assert await lim.can_proceed("m1", 10)
    await lim.record_usage("m1", 10)
    rem = await lim.remaining_budget("m1")
    assert rem["rps"] == 1
    assert rem["rpm"] == 9
    assert rem["tpm"] == 90


@pytest.mark.asyncio
async def test_rps_blocks():
    lim = RateLimiter([_model(limits=Limits(rpd=100, rpm=10, rps=1, tpm=1000, max_context=4096))])
    await lim.record_usage("m1", 1)
    assert not await lim.can_proceed("m1", 1)


@pytest.mark.asyncio
async def test_tpm_blocks_on_estimate():
    lim = RateLimiter([_model(limits=Limits(rpd=100, rpm=10, rps=10, tpm=50, max_context=4096))])
    assert not await lim.can_proceed("m1", 51)
    assert await lim.can_proceed("m1", 50)


@pytest.mark.asyncio
async def test_window_reset(monkeypatch):
    lim = RateLimiter([_model(limits=Limits(rpd=100, rpm=10, rps=1, tpm=1000, max_context=4096))])
    t0 = 1000.0
    monkeypatch.setattr("llmcascade.rate_limiter.time.monotonic", lambda: t0)
    await lim.record_usage("m1", 1)
    assert not await lim.can_proceed("m1", 1)
    monkeypatch.setattr("llmcascade.rate_limiter.time.monotonic", lambda: t0 + 1.1)
    assert await lim.can_proceed("m1", 1)


@pytest.mark.asyncio
async def test_concurrent_record_usage():
    lim = RateLimiter([_model(limits=Limits(rpd=1000, rpm=1000, rps=1000, tpm=100000, max_context=4096))])

    async def one():
        await lim.record_usage("m1", 1)

    await asyncio.gather(*[one() for _ in range(50)])
    rem = await lim.remaining_budget("m1")
    assert rem["rpm"] == 950
