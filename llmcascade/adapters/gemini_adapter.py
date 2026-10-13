from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

import httpx

from llmcascade.adapters.base import BaseAdapter, LLMResponse, StreamChunk, timed_ms
from llmcascade.adapters.sse import iter_sse, wrap_stream_error
from llmcascade.cascade import gemini_endpoint, needs_thinking_budget_zero
from llmcascade.exceptions import ProviderError

MIN_USEFUL_CHARS = 1
_OPENAI_TO_GEMINI = {
    "max_tokens": "maxOutputTokens",
    "max_completion_tokens": "maxOutputTokens",
    "temperature": "temperature",
    "top_p": "topP",
    "top_k": "topK",
    "stop": "stopSequences",
    "n": "candidateCount",
    "seed": "seed",
    "presence_penalty": "presencePenalty",
    "frequency_penalty": "frequencyPenalty",
}
_DROP = frozenset({"stream", "stream_options", "user", "contents", "model_id", "model", "messages"})


def _generation_config(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in params.items():
        if k in _DROP:
            continue
        key = _OPENAI_TO_GEMINI.get(k, k)
        out[key] = [v] if key == "stopSequences" and isinstance(v, str) else v
    if isinstance(out.get("thinkingConfig"), dict):
        out["thinkingConfig"] = dict(out["thinkingConfig"])
    return out


class GeminiAdapter(BaseAdapter):
    def _request_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "x-goog-api-key": self.api_key}

    def _url(self, model_id: str, *, stream: bool = False) -> str:
        url = gemini_endpoint(self.model.endpoint, model_id)
        if stream and ":generateContent" in url:
            url = url.replace(":generateContent", ":streamGenerateContent") + "?alt=sse"
        return url
    async def send(self, prompt: str, **params: Any) -> LLMResponse:
        model_id = params.pop("model_id", None) or self.model.name
        if self.model.cascade and model_id == self.model.name:
            # Logical family name without cascade runner — use first cascade id.
            model_id = self.model.cascade[0]
        return await self.send_model(model_id, prompt, **params)

    async def send_model(self, model_id: str, prompt: str, **params: Any) -> LLMResponse:
        start = time.perf_counter()
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        }
        generation = _generation_config({k: v for k, v in params.items() if k not in ("contents", "model_id")})
        if needs_thinking_budget_zero(model_id):
            generation.setdefault("thinkingConfig", {})
            if isinstance(generation["thinkingConfig"], dict):
                generation["thinkingConfig"].setdefault("thinkingBudget", 0)
        if generation:
            body["generationConfig"] = generation

        url = self._url(model_id)
        client = await self._http()
        try:
            resp = await client.post(url, json=body, headers=self._request_headers())
        except httpx.TimeoutException as exc:
            raise ProviderError(
                "gemini timeout",
                retryable=True,
                status_code=None,
                provider=self.model.provider,
                model=model_id,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"gemini transport error: {exc}",
                retryable=True,
                provider=self.model.provider,
                model=model_id,
            ) from exc
        if resp.status_code >= 400:
            raise ProviderError(
                f"gemini HTTP {resp.status_code}: {resp.text[:500]}",
                status_code=resp.status_code,
                retryable=resp.status_code >= 500 or resp.status_code == 408,
                provider=self.model.provider,
                model=model_id,
                headers=dict(resp.headers),
            )
        data = self._json(resp)
        try:
            candidate = data["candidates"][0]
            parts = candidate.get("content", {}).get("parts") or []
            text = "".join(
                p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought")
            )
            finish = candidate.get("finishReason")
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"gemini unexpected response shape: {data!r}"[:400],
                retryable=False,
                provider=self.model.provider,
                model=model_id,
            ) from exc

        if not text.strip() or (
            finish == "MAX_TOKENS" and len(text.strip()) < MIN_USEFUL_CHARS
        ):
            raise ProviderError(
                f"gemini empty/short response finishReason={finish!r}",
                retryable=True,
                status_code=resp.status_code,
                provider=self.model.provider,
                model=model_id,
            )

        meta = data.get("usageMetadata") or {}
        tokens = int(meta.get("totalTokenCount") or 0)
        return LLMResponse(
            text=text,
            model=model_id,
            tokens_used=tokens,
            latency_ms=timed_ms(start),
            raw=data,
        )

    async def embed(self, prompt: str, **params: Any) -> LLMResponse:
        model_id = params.pop("model_id", None) or self.model.name
        if self.model.cascade and model_id == self.model.name:
            model_id = self.model.cascade[0]
        return await self.embed_model(model_id, prompt, **params)

    async def embed_model(self, model_id: str, prompt: str, **params: Any) -> LLMResponse:
        start = time.perf_counter()
        body: dict[str, Any] = {
            "content": {"parts": [{"text": prompt}]},
        }
        extra = {k: v for k, v in params.items() if k not in ("content", "contents", "model_id")}
        if "output_dimensionality" in extra:
            extra["outputDimensionality"] = extra.pop("output_dimensionality")
        body.update(extra)
        url = self._url(model_id).replace(":generateContent", ":embedContent")
        client = await self._http()
        try:
            resp = await client.post(url, json=body, headers=self._request_headers())
        except httpx.TimeoutException as exc:
            raise ProviderError(
                "gemini timeout",
                retryable=True,
                provider=self.model.provider,
                model=model_id,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"gemini transport error: {exc}",
                retryable=True,
                provider=self.model.provider,
                model=model_id,
            ) from exc
        if resp.status_code >= 400:
            raise ProviderError(
                f"gemini HTTP {resp.status_code}: {resp.text[:500]}",
                status_code=resp.status_code,
                retryable=resp.status_code >= 500 or resp.status_code == 408,
                provider=self.model.provider,
                model=model_id,
                headers=dict(resp.headers),
            )
        data = self._json(resp)
        try:
            vec = (data.get("embedding") or {}).get("values")
            if not isinstance(vec, list) or not vec:
                raise TypeError("empty embedding")
        except (TypeError, AttributeError) as exc:
            raise ProviderError(
                f"gemini unexpected embedding shape: {data!r}"[:400],
                retryable=False,
                provider=self.model.provider,
                model=model_id,
            ) from exc
        meta = data.get("usageMetadata") or {}
        tokens = int(meta.get("totalTokenCount") or 0)
        return LLMResponse(
            model=model_id,
            tokens_used=tokens,
            latency_ms=timed_ms(start),
            embedding=[float(x) for x in vec],
            dimensions=len(vec),
            raw={"usageMetadata": meta},
        )

    async def stream(self, prompt: str, **params: Any) -> AsyncIterator[StreamChunk]:
        model_id = params.pop("model_id", None) or self.model.name
        if self.model.cascade and model_id == self.model.name:
            model_id = self.model.cascade[0]
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        }
        generation = _generation_config({k: v for k, v in params.items() if k not in ("contents", "model_id")})
        if needs_thinking_budget_zero(model_id):
            generation.setdefault("thinkingConfig", {})
            if isinstance(generation["thinkingConfig"], dict):
                generation["thinkingConfig"].setdefault("thinkingBudget", 0)
        if generation:
            body["generationConfig"] = generation
        url = self._url(model_id, stream=True)
        client = await self._http()
        try:
            req = client.build_request("POST", url, json=body, headers=self._request_headers())
            resp = await client.send(req, stream=True)
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"gemini transport error: {exc}",
                retryable=True,
                provider=self.model.provider,
                model=model_id,
            ) from exc
        if resp.status_code >= 400:
            await resp.aread()
            raise ProviderError(
                f"gemini HTTP {resp.status_code}",
                status_code=resp.status_code,
                retryable=resp.status_code >= 500,
                provider=self.model.provider,
                model=model_id,
                headers=dict(resp.headers),
            )
        tokens = 0
        hdrs = dict(resp.headers)
        try:
            async for _event, data in iter_sse(resp):
                if not data.strip():
                    continue
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
                meta = payload.get("usageMetadata") or {}
                if meta:
                    tokens = int(meta.get("totalTokenCount") or tokens)
                for cand in payload.get("candidates") or []:
                    parts = (cand.get("content") or {}).get("parts") or []
                    text = "".join(
                        p.get("text", "")
                        for p in parts
                        if isinstance(p, dict) and not p.get("thought")
                    )
                    if text:
                        yield StreamChunk(
                            delta=text,
                            model=model_id,
                            tokens_used=tokens,
                            headers=hdrs,
                        )
        except Exception as exc:  # noqa: BLE001
            raise wrap_stream_error(self.model.provider, model_id, exc)
        finally:
            await resp.aclose()
