from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlencode

import httpx

from llmcascade.adapters.base import BaseAdapter, LLMResponse, timed_ms
from llmcascade.cascade import gemini_endpoint, needs_thinking_budget_zero
from llmcascade.exceptions import ProviderError

MIN_USEFUL_CHARS = 1


class GeminiAdapter(BaseAdapter):
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
        generation: dict[str, Any] = {k: v for k, v in params.items() if k not in ("contents", "model_id")}
        if needs_thinking_budget_zero(model_id):
            generation.setdefault("thinkingConfig", {})
            if isinstance(generation["thinkingConfig"], dict):
                generation["thinkingConfig"].setdefault("thinkingBudget", 0)
        if generation:
            body["generationConfig"] = generation

        qs = urlencode({"key": self.api_key})
        url = f"{gemini_endpoint(self.model.endpoint, model_id)}?{qs}"
        client = await self._http()
        try:
            resp = await client.post(url, json=body, headers={"Content-Type": "application/json"})
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
        data = resp.json()
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
