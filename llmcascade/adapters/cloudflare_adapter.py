from __future__ import annotations

import os
import time
from typing import Any

import httpx

from llmcascade.adapters.base import BaseAdapter, LLMResponse, timed_ms
from llmcascade.exceptions import ProviderError


class CloudflareAdapter(BaseAdapter):
    async def send(self, prompt: str, **params: Any) -> LLMResponse:
        start = time.perf_counter()
        account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        if not account_id:
            raise ProviderError(
                "missing env CLOUDFLARE_ACCOUNT_ID",
                retryable=False,
                provider=self.model.provider,
                model=self.model.name,
            )
        url = self.model.endpoint.replace("{account_id}", account_id)
        body: dict[str, Any] = {"prompt": prompt, **params}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        client = await self._http()
        try:
            resp = await client.post(url, json=body, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderError(
                "cloudflare timeout",
                retryable=True,
                provider=self.model.provider,
                model=self.model.name,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"cloudflare transport error: {exc}",
                retryable=True,
                provider=self.model.provider,
                model=self.model.name,
            ) from exc
        if resp.status_code >= 400:
            self._raise_http(resp)
        data = resp.json()
        result = data.get("result") if isinstance(data, dict) else None
        text = ""
        if isinstance(result, dict):
            text = str(result.get("response") or result.get("text") or "")
        elif isinstance(result, str):
            text = result
        if not text and isinstance(data, dict):
            text = str(data.get("response") or "")
        if not text:
            raise ProviderError(
                "cloudflare unexpected response shape",
                retryable=False,
                provider=self.model.provider,
                model=self.model.name,
            )
        return LLMResponse(
            text=text,
            model=self.model.name,
            tokens_used=0,
            latency_ms=timed_ms(start),
            raw=data if isinstance(data, dict) else {"raw": data},
        )
