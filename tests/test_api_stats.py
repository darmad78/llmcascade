"""HTTP-layer stats: terminal failures once per client call, not at request entry."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llmcascade.exceptions import AllModelsExhaustedError


class _SpyStats:
    def __init__(self) -> None:
        self.terminal_failed: list[str] = []
        self.recv_calls = 0

    def enqueue_recv(self, capability: str = "chat") -> None:
        self.recv_calls += 1

    def enqueue_terminal_failed(self, capability: str = "chat") -> None:
        self.terminal_failed.append(capability)

    async def close(self) -> None:
        return None


@pytest.fixture()
def api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLMCASCADE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SECRET_KEY", "test-secret-stats-xxxxxxxxxxxx")
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    monkeypatch.delenv("MONGODB_URI", raising=False)
    yield


@pytest.fixture()
def client(api_env):
    import llmcascade.api as api_mod

    with TestClient(api_mod.app) as c:
        yield c, api_mod


def test_submit_success_does_not_enqueue_recv(client, monkeypatch: pytest.MonkeyPatch):
    c, api_mod = client
    spy = _SpyStats()
    monkeypatch.setattr(api_mod, "_stats", spy)

    async def ok_submit(prompt, capability="chat", notes=None, **params):
        from llmcascade.adapters.base import LLMResponse

        return LLMResponse(text="ok", model="m", tokens_used=1, latency_ms=1.0)

    monkeypatch.setattr(api_mod._client, "submit", ok_submit)
    r = c.post("/v1/complete", json={"prompt": "hi"})
    assert r.status_code == 200
    assert spy.recv_calls == 0
    assert spy.terminal_failed == []


def test_submit_exhausted_records_one_terminal_failed(client, monkeypatch: pytest.MonkeyPatch):
    c, api_mod = client
    spy = _SpyStats()
    monkeypatch.setattr(api_mod, "_stats", spy)

    async def fail_submit(prompt, capability="chat", notes=None, **params):
        raise AllModelsExhaustedError("nope", http_status=503)

    monkeypatch.setattr(api_mod._client, "submit", fail_submit)
    r = c.post("/v1/complete", json={"prompt": "hi"})
    assert r.status_code == 503
    assert spy.recv_calls == 0
    assert spy.terminal_failed == ["chat"]


def test_retire_watch_terminal_failure_not_counted(client, monkeypatch: pytest.MonkeyPatch):
    c, api_mod = client
    spy = _SpyStats()
    monkeypatch.setattr(api_mod, "_stats", spy)

    async def fail_submit(prompt, capability="chat", notes=None, **params):
        raise ValueError("bad")

    monkeypatch.setattr(api_mod._client, "submit", fail_submit)
    r = c.post("/v1/complete", json={"prompt": "hi", "notes": "retire-watch"})
    assert r.status_code == 400
    assert spy.terminal_failed == []
