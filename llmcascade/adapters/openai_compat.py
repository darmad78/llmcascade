from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

import httpx

from llmcascade.adapters.base import BaseAdapter, LLMResponse, StreamChunk, timed_ms
from llmcascade.adapters.sse import iter_sse, wrap_stream_error
from llmcascade.exceptions import ProviderError

_STREAM_DROP = frozenset({"stream", "stream_options", "model"})


class OpenAICompatibleAdapter(BaseAdapter):
    """Shared chat/completions shape for Groq, OpenRouter, Together, Cerebras, Mistral."""

    extra_headers: dict[str, str] = {}
    stream_usage: bool = True

    def _chat_body(self, prompt: str, **params: Any) -> dict[str, Any]:
        return {
            "model": self.model_id(),
            "messages": [{"role": "user", "content": prompt}],
            **{k: v for k, v in params.items() if k not in _STREAM_DROP and k != "messages"},
        }

    async def send(self, prompt: str, **params: Any) -> LLMResponse:
        start = time.perf_counter()
        body = self._chat_body(prompt, **params)
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
        data = self._json(resp)
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
            headers=dict(resp.headers),
        )

    async def stream(self, prompt: str, **params: Any) -> AsyncIterator[StreamChunk]:
        body = self._chat_body(prompt, **params)
        body["stream"] = True
        if self.stream_usage:
            body.setdefault("stream_options", {"include_usage": True})
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        client = await self._http()
        try:
            req = client.build_request("POST", self.model.endpoint, json=body, headers=headers)
            resp = await client.send(req, stream=True)
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
            await resp.aread()
            self._raise_http(resp)
        tokens = 0
        finish: str | None = None
        hdrs = dict(resp.headers)
        try:
            async for _event, data in iter_sse(resp):
                if not data.strip():
                    continue
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = payload.get("usage") or {}
                if usage:
                    tokens = int(usage.get("total_tokens") or tokens)
                for choice in payload.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content") or ""
                    fr = choice.get("finish_reason")
                    if fr:
                        finish = str(fr)
                    if delta:
                        yield StreamChunk(
                            delta=delta,
                            model=self.model.name,
                            finish_reason=finish,
                            tokens_used=tokens,
                            headers=hdrs,
                        )
        except Exception as exc:  # noqa: BLE001
            raise wrap_stream_error(self.model.provider, self.model.name, exc)
        finally:
            await resp.aclose()

    async def embed(self, prompt: str, **params: Any) -> LLMResponse:
        start = time.perf_counter()
        body = {
            "model": self.model.api_model or self.model.name,
            "input": prompt,
            **{k: v for k, v in params.items() if k not in ("input", "messages", *_STREAM_DROP)},
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
        data = self._json(resp)
        try:
            vec = data["data"][0]["embedding"]
            if not isinstance(vec, list) or not vec:
                raise TypeError("empty embedding")
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"{self.model.provider} unexpected embedding shape",
                retryable=False,
                provider=self.model.provider,
                model=self.model.name,
            ) from exc
        usage = data.get("usage") or {}
        tokens = int(usage.get("total_tokens") or usage.get("prompt_tokens") or 0)
        return LLMResponse(
            model=self.model.name,
            tokens_used=tokens,
            latency_ms=timed_ms(start),
            embedding=[float(x) for x in vec],
            dimensions=len(vec),
            raw={"usage": usage, "model": data.get("model") or self.model.name},
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
    stream_usage = False


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


class JinaAdapter(OpenAICompatibleAdapter):
    pass


class VoyageAdapter(OpenAICompatibleAdapter):
    pass


class NomicAdapter(OpenAICompatibleAdapter):
    pass


class MixedbreadAdapter(OpenAICompatibleAdapter):
    pass


class SiliconFlowAdapter(OpenAICompatibleAdapter):
    pass


class FireworksAdapter(OpenAICompatibleAdapter):
    pass


class NovitaAdapter(OpenAICompatibleAdapter):
    pass


class HyperbolicAdapter(OpenAICompatibleAdapter):
    pass
