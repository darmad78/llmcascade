import os

import httpx
import pytest
import respx

from llmcascade.adapters.cerebras_adapter import CerebrasAdapter
from llmcascade.adapters.cloudflare_adapter import CloudflareAdapter
from llmcascade.adapters.cohere_adapter import CohereAdapter
from llmcascade.adapters.deepinfra_adapter import DeepInfraAdapter
from llmcascade.adapters.deepseek_adapter import DeepSeekAdapter
from llmcascade.adapters.gemini_adapter import GeminiAdapter
from llmcascade.adapters.groq_adapter import GroqAdapter
from llmcascade.adapters.huggingface_adapter import HuggingFaceAdapter
from llmcascade.adapters.mistral_adapter import MistralAdapter
from llmcascade.adapters.nvidia_adapter import NvidiaAdapter
from llmcascade.adapters.openrouter_adapter import OpenRouterAdapter
from llmcascade.adapters.sambanova_adapter import SambaNovaAdapter
from llmcascade.adapters.together_adapter import TogetherAdapter
from llmcascade.exceptions import ProviderError
from llmcascade.registry import Limits, ModelConfig, resolve_auth_env

OA_BODY = {
    "choices": [{"message": {"content": "hello"}}],
    "usage": {"total_tokens": 12},
}


def _oa_model(provider: str, name: str, endpoint: str) -> ModelConfig:
    return ModelConfig(
        name=name,
        provider=provider,
        endpoint=endpoint,
        auth_env_var="TEST_KEY",
        limits=Limits(rpd=10, rpm=10, rps=10, tpm=1000, max_context=4096),
        capabilities=["chat"],
        priority=1,
    )


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "secret")
    monkeypatch.setenv("GOOGLE_API_KEY", "secret")


@respx.mock
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cls,provider,url",
    [
        (GroqAdapter, "groq", "https://api.groq.com/openai/v1/chat/completions"),
        (OpenRouterAdapter, "openrouter", "https://openrouter.ai/api/v1/chat/completions"),
        (TogetherAdapter, "together", "https://api.together.xyz/v1/chat/completions"),
        (CerebrasAdapter, "cerebras", "https://api.cerebras.ai/v1/chat/completions"),
        (MistralAdapter, "mistral", "https://api.mistral.ai/v1/chat/completions"),
        (SambaNovaAdapter, "sambanova", "https://api.sambanova.ai/v1/chat/completions"),
        (DeepSeekAdapter, "deepseek", "https://api.deepseek.com/chat/completions"),
        (HuggingFaceAdapter, "huggingface", "https://router.huggingface.co/v1/chat/completions"),
        (NvidiaAdapter, "nvidia", "https://integrate.api.nvidia.com/v1/chat/completions"),
        (DeepInfraAdapter, "deepinfra", "https://api.deepinfra.com/v1/openai/chat/completions"),
    ],
)
async def test_openai_compat_success(cls, provider, url):
    respx.post(url).mock(return_value=httpx.Response(200, json=OA_BODY))
    model = _oa_model(provider, "test-model", url)
    async with httpx.AsyncClient() as client:
        adapter = cls(model, os.environ["TEST_KEY"], client=client)
        resp = await adapter.send("hi")
    assert resp.text == "hello"
    assert resp.tokens_used == 12
    assert resp.model == "test-model"


@respx.mock
@pytest.mark.asyncio
async def test_openai_compat_429():
    url = "https://api.groq.com/openai/v1/chat/completions"
    respx.post(url).mock(return_value=httpx.Response(429, text="rate limited"))
    model = _oa_model("groq", "test-model", url)
    async with httpx.AsyncClient() as client:
        adapter = GroqAdapter(model, "secret", client=client)
        with pytest.raises(ProviderError) as ei:
            await adapter.send("hi")
    assert ei.value.status_code == 429
    assert ei.value.retryable is False


@respx.mock
@pytest.mark.asyncio
async def test_gemini_success():
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    respx.post(url__startswith=url).mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "gem"}]}}],
                "usageMetadata": {"totalTokenCount": 7},
            },
        )
    )
    model = ModelConfig(
        name="gemini-2.0-flash",
        provider="gemini",
        endpoint=url,
        auth_env_var="GOOGLE_API_KEY",
        limits=Limits(rpd=10, rpm=10, rps=10, tpm=1000, max_context=4096),
        capabilities=["chat"],
        priority=1,
    )
    async with httpx.AsyncClient() as client:
        adapter = GeminiAdapter(model, "secret", client=client)
        resp = await adapter.send("hi")
    assert resp.text == "gem"
    assert resp.tokens_used == 7


@respx.mock
@pytest.mark.asyncio
async def test_gemini_skips_thought_parts():
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    respx.post(url__startswith=url).mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "thinking...", "thought": True},
                                {"text": "answer"},
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"totalTokenCount": 9},
            },
        )
    )
    model = ModelConfig(
        name="gemini",
        provider="gemini",
        endpoint="https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        auth_env_var="GOOGLE_API_KEY",
        limits=Limits(rpd=10, rpm=10, rps=10, tpm=1000, max_context=4096),
        capabilities=["chat"],
        priority=1,
        cascade=["gemini-2.5-flash"],
    )
    async with httpx.AsyncClient() as client:
        adapter = GeminiAdapter(model, "secret", client=client)
        resp = await adapter.send_model("gemini-2.5-flash", "hi")
    assert resp.text == "answer"


@respx.mock
@pytest.mark.asyncio
async def test_gemini_5xx_retryable():
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    respx.post(url__startswith=url).mock(return_value=httpx.Response(503, text="down"))
    model = ModelConfig(
        name="gemini-2.0-flash",
        provider="gemini",
        endpoint=url,
        auth_env_var="GOOGLE_API_KEY",
        limits=Limits(rpd=10, rpm=10, rps=10, tpm=1000, max_context=4096),
        capabilities=["chat"],
        priority=1,
    )
    async with httpx.AsyncClient() as client:
        adapter = GeminiAdapter(model, "secret", client=client)
        with pytest.raises(ProviderError) as ei:
            await adapter.send("hi")
    assert ei.value.retryable is True


@respx.mock
@pytest.mark.asyncio
async def test_cloudflare_success(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    endpoint = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/meta/llama-3.1-8b-instruct"
    url = endpoint.replace("{account_id}", "acct123")
    respx.post(url).mock(return_value=httpx.Response(200, json={"result": {"response": "cf-ok"}}))
    model = ModelConfig(
        name="@cf/meta/llama-3.1-8b-instruct",
        provider="cloudflare",
        endpoint=endpoint,
        auth_env_var="CLOUDFLARE_API_KEY",
        limits=Limits(rpd=10, rpm=10, rps=10, tpm=1000, max_context=4096),
        capabilities=["chat"],
        priority=1,
    )
    async with httpx.AsyncClient() as client:
        adapter = CloudflareAdapter(model, "secret", client=client)
        resp = await adapter.send("hi")
    assert resp.text == "cf-ok"


@respx.mock
@pytest.mark.asyncio
async def test_cohere_success():
    url = "https://api.cohere.com/v2/chat"
    respx.post(url).mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": [{"type": "text", "text": "co-ok"}]},
                "usage": {"billed_units": {"input_tokens": 2, "output_tokens": 3}},
            },
        )
    )
    model = ModelConfig(
        name="command-r-08-2024",
        provider="cohere",
        endpoint=url,
        auth_env_var="COHERE_API_KEY",
        limits=Limits(rpd=10, rpm=10, rps=10, tpm=1000, max_context=4096),
        capabilities=["chat"],
        priority=1,
    )
    async with httpx.AsyncClient() as client:
        adapter = CohereAdapter(model, "secret", client=client)
        resp = await adapter.send("hi")
    assert resp.text == "co-ok"
    assert resp.tokens_used == 5


def test_resolve_auth_hf_token_first(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "legacy")
    assert resolve_auth_env("HF_TOKEN") == "hf"


def test_resolve_auth_hf_fallback(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "legacy")
    assert resolve_auth_env("HF_TOKEN") == "legacy"
