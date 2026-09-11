from llmcascade.provider_quota import parse_openrouter_key, parse_quota_headers
from llmcascade.rate_limiter import RateLimiter
from llmcascade.registry import Limits, ModelConfig

import pytest


def _model(**overrides) -> ModelConfig:
    base = dict(
        name="llama-3.3-70b-versatile",
        provider="groq",
        endpoint="https://api.groq.com/openai/v1/chat/completions",
        auth_env_var="GROQ_API_KEY",
        limits=Limits(rpd=1000, rpm=30, rps=1, tpm=12000, max_context=128000),
        capabilities=["chat"],
        priority=10,
    )
    base.update(overrides)
    return ModelConfig(**base)


def test_parse_groq_headers_are_rpd_and_tpm():
    parsed = parse_quota_headers(
        {
            "x-ratelimit-remaining-requests": "412",
            "x-ratelimit-remaining-tokens": "9000",
        },
        provider="groq",
    )
    assert parsed == {"rpd": 412, "tpm": 9000}


def test_parse_openrouter_error_headers():
    parsed = parse_quota_headers(
        {"X-RateLimit-Limit": "50", "X-RateLimit-Remaining": "0"},
        provider="openrouter",
    )
    assert parsed["rpd"] == 0


def test_parse_openrouter_key_rate_limit():
    parsed = parse_openrouter_key(
        {"data": {"rate_limit": {"requests": 50, "remaining": 12, "interval": "1d"}}}
    )
    assert parsed == {"rpd": 12}


@pytest.mark.asyncio
async def test_live_rpd_overrides_local_and_blocks():
    lim = RateLimiter([_model()])
    await lim.ingest_quota("llama-3.3-70b-versatile", {"rpd": 0}, provider="groq")
    rem = await lim.remaining_budget("llama-3.3-70b-versatile")
    assert rem["rpd"] == 0
    assert rem["rpm"] == 30
    assert not await lim.can_proceed("llama-3.3-70b-versatile", 1)
    assert lim.quota_source("llama-3.3-70b-versatile") == "headers"
