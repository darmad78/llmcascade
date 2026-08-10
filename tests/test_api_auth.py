"""Tests for optional /v1/complete API-key auth and per-key RPM."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLMROUTER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-for-jwt-and-fernet")
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    monkeypatch.delenv("REQUIRE_AUTH", raising=False)
    monkeypatch.delenv("LLMROUTER_API_KEYS", raising=False)
    monkeypatch.delenv("LLMROUTER_API_RPM", raising=False)
    monkeypatch.delenv("MONGODB_URI", raising=False)
    yield


@pytest.fixture()
def client(api_env):
    import llmrouter.api as api_mod

    with TestClient(api_mod.app) as c:
        yield c


def test_complete_open_by_default(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    async def fake_submit(prompt, capability="chat", notes=None, **params):
        from llmrouter.adapters.base import LLMResponse

        return LLMResponse(text="hi", model="x", tokens_used=1, latency_ms=1.0)

    import llmrouter.api as api_mod

    monkeypatch.setattr(api_mod._client, "submit", fake_submit)
    r = client.post("/v1/complete", json={"prompt": "hi"})
    assert r.status_code == 200
    assert r.json()["text"] == "hi"


def test_require_auth_401(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REQUIRE_AUTH", "true")
    monkeypatch.setenv("LLMROUTER_API_KEYS", "good-key")
    r = client.post("/v1/complete", json={"prompt": "hi"})
    assert r.status_code == 401


def test_require_auth_ok_and_rpm(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REQUIRE_AUTH", "true")
    monkeypatch.setenv("LLMROUTER_API_KEYS", "good-key")
    monkeypatch.setenv("LLMROUTER_API_RPM", "2")

    async def fake_submit(prompt, capability="chat", notes=None, **params):
        from llmrouter.adapters.base import LLMResponse

        return LLMResponse(text="ok", model="x", tokens_used=1, latency_ms=1.0)

    import llmrouter.api as api_mod

    monkeypatch.setattr(api_mod._client, "submit", fake_submit)
    api_mod._api_limiter = None
    headers = {"Authorization": "Bearer good-key"}
    assert client.post("/v1/complete", json={"prompt": "a"}, headers=headers).status_code == 200
    assert client.post("/v1/complete", json={"prompt": "b"}, headers=headers).status_code == 200
    r = client.post("/v1/complete", json={"prompt": "c"}, headers=headers)
    assert r.status_code == 429
