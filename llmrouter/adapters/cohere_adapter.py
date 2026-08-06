from __future__ import annotations

import time
from typing import Any

import httpx

from llmrouter.adapters.base import BaseAdapter, LLMResponse, timed_ms
from llmrouter.exceptions import ProviderError


class CohereAdapter(BaseAdapter):
    async def send(self, prompt: str, **params: Any) -> LLMResponse:
        start = time.perf_counter()
        body: dict[str, Any] = {
            "model": self.model.name,
            "messages": [{"role": "user", "content": prompt}],
            **{k: v for k, v in params.items() if k not in ("messages", "message")},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        client = await self._http()
        try:
            resp = await client.post(self.model.endpoint, json=body, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderError(
                "cohere timeout",
                retryable=True,
                provider=self.model.provider,
                model=self.model.name,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"cohere transport error: {exc}",
                retryable=True,
                provider=self.model.provider,
                model=self.model.name,
            ) from exc
        if resp.status_code >= 400:
            self._raise_http(resp)
        data = resp.json()
        text = self._extract_text(data)
        if text is None:
            raise ProviderError(
                "cohere unexpected response shape",
                retryable=False,
                provider=self.model.provider,
                model=self.model.name,
            )
        usage = data.get("usage") or {}
        billed = usage.get("billed_units") or {}
        tokens = int(billed.get("input_tokens") or 0) + int(billed.get("output_tokens") or 0)
        if not tokens:
            tokens = int(usage.get("tokens") or 0)
        return LLMResponse(
            text=text,
            model=self.model.name,
            tokens_used=tokens,
            latency_ms=timed_ms(start),
            raw=data,
        )

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str | None:
        msg = data.get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("text"):
                        parts.append(str(item["text"]))
                    elif isinstance(item, str):
                        parts.append(item)
                if parts:
                    return "".join(parts)
        if isinstance(data.get("text"), str):
            return data["text"]
        return None
