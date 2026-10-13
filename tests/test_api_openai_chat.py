"""OpenAI-compatible /v1/chat/completions."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLMCASCADE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-for-jwt-and-fernet")
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    monkeypatch.delenv("REQUIRE_AUTH", raising=False)
    monkeypatch.delenv("LLMCASCADE_PROFILE", raising=False)
    monkeypatch.delenv("MONGODB_URI", raising=False)
    yield


@pytest.fixture()
def client(api_env):
    import llmcascade.api as api_mod

    with TestClient(api_mod.app) as c:
        yield c


def test_chat_completions(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    async def fake_submit(prompt, capability="chat", notes=None, **params):
        from llmcascade.adapters.base import LLMResponse

        assert capability == "chat"
        assert "User: Say hi" in prompt
        return LLMResponse(text="hello", model="test-model", tokens_used=4, latency_ms=1.0)

    import llmcascade.api as api_mod

    monkeypatch.setattr(api_mod._client, "submit", fake_submit)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Say hi"}],
            "user": "pytest",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["choices"][0]["message"]["content"] == "hello"
    assert data["model"] == "test-model"


def test_pro_checkout_redirect(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLMCASCADE_PRO_CHECKOUT_URL", "https://example.com/stripe")
    r = client.get("/pro/checkout?utm_source=readme", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "https://example.com/stripe"
