from __future__ import annotations

import time
from typing import Any

import httpx

from llmcascade.adapters.base import BaseAdapter, LLMResponse, timed_ms
from llmcascade.exceptions import ProviderError


class OpenAICompatibleAdapter(BaseAdapter):
    """Shared chat/completions shape for Groq, OpenRouter, Together, Cerebras, Mistral."""

    extra_headers: dict[str, str] = {}

    async def send(self, prompt: str, **params: Any) -> LLMResponse:
        start = time.perf_counter()
        body = {
            "model": self.model.name,
            "messages": [{"role": "user", "content": prompt}],
            **{k: v for k, v in params.items() if k != "messages"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        client = await self._http()
        try:
            resp = await client.post(self.model.endpoint, json=body, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"{self.model.provider} timeout",
                retryable=True,
                provider=self.model.provider,
                model=self.model.name,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"{self.model.provider} transport error: {exc}",
                retryable=True,
                provider=self.model.provider,
                model=self.model.name,
            ) from exc
        if resp.status_code >= 400:
            self._raise_http(resp)
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"{self.model.provider} unexpected response shape",
                retryable=False,
                provider=self.model.provider,
                model=self.model.name,
            ) from exc
        usage = data.get("usage") or {}
        tokens = int(usage.get("total_tokens") or 0)
        return LLMResponse(
            text=text,
            model=self.model.name,
            tokens_used=tokens,
            latency_ms=timed_ms(start),
            raw=data,
        )


class GroqAdapter(OpenAICompatibleAdapter):
    pass


class OpenRouterAdapter(OpenAICompatibleAdapter):
    extra_headers = {"HTTP-Referer": "https://github.com/darmad78/llmcascade", "X-Title": "llmcascade"}


class TogetherAdapter(OpenAICompatibleAdapter):
    pass


class CerebrasAdapter(OpenAICompatibleAdapter):
    pass


class MistralAdapter(OpenAICompatibleAdapter):
    pass


class SambaNovaAdapter(OpenAICompatibleAdapter):
    pass


class DeepSeekAdapter(OpenAICompatibleAdapter):
    pass


class HuggingFaceAdapter(OpenAICompatibleAdapter):
    pass


class NvidiaAdapter(OpenAICompatibleAdapter):
    pass


class DeepInfraAdapter(OpenAICompatibleAdapter):
    pass
