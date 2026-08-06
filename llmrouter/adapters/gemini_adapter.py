from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlencode

import httpx

from llmrouter.adapters.base import BaseAdapter, LLMResponse, timed_ms
from llmrouter.exceptions import ProviderError


class GeminiAdapter(BaseAdapter):
    async def send(self, prompt: str, **params: Any) -> LLMResponse:
        start = time.perf_counter()
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        }
        if params:
            generation = {k: v for k, v in params.items() if k != "contents"}
            if generation:
                body["generationConfig"] = generation
        qs = urlencode({"key": self.api_key})
        url = f"{self.model.endpoint}?{qs}"
        client = await self._http()
        try:
            resp = await client.post(url, json=body, headers={"Content-Type": "application/json"})
        except httpx.TimeoutException as exc:
            raise ProviderError(
                "gemini timeout",
                retryable=True,
                provider=self.model.provider,
                model=self.model.name,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"gemini transport error: {exc}",
                retryable=True,
                provider=self.model.provider,
                model=self.model.name,
            ) from exc
        if resp.status_code >= 400:
            self._raise_http(resp)
        data = resp.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                "gemini unexpected response shape",
                retryable=False,
                provider=self.model.provider,
                model=self.model.name,
            ) from exc
        meta = data.get("usageMetadata") or {}
        tokens = int(meta.get("totalTokenCount") or 0)
        return LLMResponse(
            text=text,
            model=self.model.name,
            tokens_used=tokens,
            latency_ms=timed_ms(start),
            raw=data,
        )
