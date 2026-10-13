"""SSE streaming and failover-before-first-byte tests."""

from __future__ import annotations

import json

import httpx
import pytest

from llmcascade.adapters.openai_compat import GroqAdapter
from llmcascade.adapters.sse import iter_sse
from llmcascade.exceptions import ProviderError
from llmcascade.queue_worker import RouterClient
from llmcascade.registry import Limits, ModelConfig


def _model(name: str, provider: str = "groq") -> ModelConfig:
    return ModelConfig(
        name=name,
        provider=provider,
        endpoint="https://example.com/v1/chat/completions",
        auth_env_var="GROQ_API_KEY",
        limits=Limits(rpd=100, rpm=30, rps=5, tpm=60000, max_context=8192),
        capabilities=["chat"],
        priority=10,
    )


@pytest.mark.asyncio
async def test_openai_stream_parses_deltas(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    lines = [
        'data: {"choices":[{"delta":{"content":"hi"}}]}\n',
        "\n",
        'data: [DONE]\n',
        "\n",
    ]
    body = "".join(lines).encode()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=httpx.ByteStream(body))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    adapter = GroqAdapter(_model("a"), "k", client=client)
    chunks = []
    async for c in adapter.stream("hello"):
        chunks.append(c.delta)
    assert chunks == ["hi"]
    await client.aclose()


@pytest.mark.asyncio
async def test_non_json_response_fails_over(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    models = [_model("a"), _model("b")]

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads((request.content or b"").decode())
        if body.get("model") == "a":
            return httpx.Response(200, text="not json")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 1}},
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    router = RouterClient(registry=models, workers=1, max_queue=10)
    router._client = client
    await router.start()

    async def patched_execute(model, prompt, **kw):
        from llmcascade.adapters import get_adapter

        ad = get_adapter(model, client=client)
        return await ad.send(prompt)

    router._execute = patched_execute  # type: ignore[method-assign]
    resp = await router.submit("hi", model="a", include_free_cascade=True)
    assert resp.text == "ok"
    await router.shutdown()


@pytest.mark.asyncio
async def test_unpinned_stream_cascades_to_next_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("LLMCASCADE_DATA_DIR", str(tmp_path))
    models = [_model("a"), _model("b")]
    models[1] = models[1].model_copy(update={"priority": 20})
    sse = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        name = json.loads((request.content or b"").decode()).get("model")
        seen.append(name)
        if name == "a":
            return httpx.Response(429, text="rate limit", headers={"retry-after": "30"})
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=httpx.ByteStream(sse)
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    router = RouterClient(registry=models, workers=1, max_queue=10, strategy="priority_first")
    router._client = client
    await router.start()
    meta, stream = await router.submit_stream("hello")
    chunks = [c.delta async for c in stream]
    assert meta["model"] == "b"
    assert chunks == ["hi"]
    assert seen == ["a", "b"]
    await router.shutdown()


@pytest.mark.asyncio
async def test_sse_iter_skips_comments():
    payload = ": comment\ndata: {\"x\":1}\n\n"
    resp = httpx.Response(200, stream=httpx.ByteStream(payload.encode()))
    out = []
    async for ev, data in iter_sse(resp):
        out.append((ev, data))
    assert out == [("message", '{"x":1}')]
