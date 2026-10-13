"""Server-Sent Events parsing for provider streaming responses."""

from __future__ import annotations

from typing import AsyncIterator

import httpx

from llmcascade.exceptions import ProviderError


async def iter_sse(resp: httpx.Response) -> AsyncIterator[tuple[str, str]]:
    """Yield (event_name, data_payload) from an open SSE response body."""
    event = ""
    data_lines: list[str] = []
    async for line in resp.aiter_lines():
        if line is None:
            continue
        if line.startswith(":"):
            continue
        if not line.strip():
            if data_lines:
                payload = "\n".join(data_lines)
                if payload.strip() == "[DONE]":
                    return
                yield (event or "message", payload)
            event = ""
            data_lines = []
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
    if data_lines:
        payload = "\n".join(data_lines)
        if payload.strip() != "[DONE]":
            yield (event or "message", payload)


def wrap_stream_error(provider: str, model: str, exc: BaseException) -> ProviderError:
    if isinstance(exc, ProviderError):
        return exc
    retryable = isinstance(exc, httpx.TimeoutException)
    return ProviderError(
        f"{provider} stream error: {type(exc).__name__}",
        retryable=retryable,
        provider=provider,
        model=model,
    )
