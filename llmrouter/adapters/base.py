from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import httpx
from pydantic import BaseModel, Field

from llmrouter.exceptions import ProviderError
from llmrouter.registry import ModelConfig


class LLMResponse(BaseModel):
    text: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    raw: dict[str, Any] = Field(default_factory=dict)


class BaseAdapter(ABC):
    def __init__(self, model: ModelConfig, api_key: str, client: httpx.AsyncClient | None = None) -> None:
        self.model = model
        self.api_key = api_key
        self._client = client
        self._owns_client = client is None

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _raise_http(self, resp: httpx.Response) -> None:
        code = resp.status_code
        retryable = code >= 500 or code == 408
        raise ProviderError(
            f"{self.model.provider} HTTP {code}: {resp.text[:300]}",
            status_code=code,
            retryable=retryable,
            provider=self.model.provider,
            model=self.model.name,
        )

    @abstractmethod
    async def send(self, prompt: str, **params: Any) -> LLMResponse:
        ...


def timed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
