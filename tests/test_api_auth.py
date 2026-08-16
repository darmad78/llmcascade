"""Tests for optional /v1/complete API-key auth and per-key RPM."""

from __future__ import annotations

from pathlib import Path

import bcrypt
import pytest
from fastapi.testclient import TestClient

from llmcascade.api_auth import (
    auth_credentials_configured,
    hash_api_key,
    is_production_profile,
    require_auth_enabled,
    validate_api_key,
)


@pytest.fixture()
def api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLMCASCADE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-for-jwt-and-fernet")
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    monkeypatch.delenv("REQUIRE_AUTH", raising=False)
    monkeypatch.delenv("LLMCASCADE_PROFILE", raising=False)
    monkeypatch.delenv("LLMCASCADE_API_KEYS", raising=False)
    monkeypatch.delenv("LLMCASCADE_API_KEY_HASHES", raising=False)
    monkeypatch.delenv("LLMCASCADE_API_RPM", raising=False)
    monkeypatch.delenv("MONGODB_URI", raising=False)
    yield


@pytest.fixture()
def client(api_env):
    import llmcascade.api as api_mod

    with TestClient(api_mod.app) as c:
        yield c


def test_complete_open_by_default(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    async def fake_submit(prompt, capability="chat", notes=None, **params):
        from llmcascade.adapters.base import LLMResponse

        return LLMResponse(text="hi", model="x", tokens_used=1, latency_ms=1.0)

    import llmcascade.api as api_mod

    monkeypatch.setattr(api_mod._client, "submit", fake_submit)
    r = client.post("/v1/complete", json={"prompt": "hi"})
    assert r.status_code == 200
    assert r.json()["text"] == "hi"


def test_require_auth_401(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REQUIRE_AUTH", "true")
    monkeypatch.setenv("LLMCASCADE_API_KEYS", "good-key")
    r = client.post("/v1/complete", json={"prompt": "hi"})
    assert r.status_code == 401


def test_require_auth_ok_and_rpm(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REQUIRE_AUTH", "true")
    monkeypatch.setenv("LLMCASCADE_API_KEYS", "good-key")
    monkeypatch.setenv("LLMCASCADE_API_RPM", "2")

    async def fake_submit(prompt, capability="chat", notes=None, **params):
        from llmcascade.adapters.base import LLMResponse

        return LLMResponse(text="ok", model="x", tokens_used=1, latency_ms=1.0)

    import llmcascade.api as api_mod

    monkeypatch.setattr(api_mod._client, "submit", fake_submit)
    api_mod._api_limiter = None
    headers = {"Authorization": "Bearer good-key"}
    assert client.post("/v1/complete", json={"prompt": "a"}, headers=headers).status_code == 200
    assert client.post("/v1/complete", json={"prompt": "b"}, headers=headers).status_code == 200
    r = client.post("/v1/complete", json={"prompt": "c"}, headers=headers)
    assert r.status_code == 429


def test_hashed_api_key_ok(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    hashed = hash_api_key("secret-plain")
    monkeypatch.setenv("REQUIRE_AUTH", "true")
    monkeypatch.setenv("LLMCASCADE_API_KEY_HASHES", hashed)

    async def fake_submit(prompt, capability="chat", notes=None, **params):
        from llmcascade.adapters.base import LLMResponse

        return LLMResponse(text="ok", model="x", tokens_used=1, latency_ms=1.0)

    import llmcascade.api as api_mod

    monkeypatch.setattr(api_mod._client, "submit", fake_submit)
    bad = client.post(
        "/v1/complete",
        json={"prompt": "x"},
        headers={"X-API-Key": "wrong"},
    )
    assert bad.status_code == 401
    good = client.post(
        "/v1/complete",
        json={"prompt": "x"},
        headers={"X-API-Key": "secret-plain"},
    )
    assert good.status_code == 200


def test_embed_open_by_default(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    async def fake_submit(prompt, capability="chat", notes=None, **params):
        from llmcascade.adapters.base import LLMResponse

        assert capability == "embed"
        return LLMResponse(model="e", embedding=[0.1], dimensions=1, tokens_used=1, latency_ms=1.0)

    import llmcascade.api as api_mod

    monkeypatch.setattr(api_mod._client, "submit", fake_submit)
    r = client.post("/v1/embed", json={"prompt": "doc", "notes": "app"})
    assert r.status_code == 200
    body = r.json()
    assert body["embedding"] == [0.1]
    assert body["dimensions"] == 1
    assert body["text"] == ""


def test_bcrypt_entry_in_api_keys_env(monkeypatch: pytest.MonkeyPatch):
    hashed = bcrypt.hashpw(b"mixed-key", bcrypt.gensalt()).decode("ascii")
    monkeypatch.setenv("LLMCASCADE_API_KEYS", f"plain-one,{hashed}")
    monkeypatch.delenv("LLMCASCADE_API_KEY_HASHES", raising=False)
    assert validate_api_key("plain-one")
    assert validate_api_key("mixed-key")
    assert not validate_api_key("nope")


def test_dashboard_session_can_complete_with_csrf(
    api_env, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("REQUIRE_AUTH", "true")
    monkeypatch.setenv("LLMCASCADE_API_KEYS", "good-key")

    async def fake_submit(prompt, capability="chat", notes=None, **params):
        from llmcascade.adapters.base import LLMResponse

        return LLMResponse(text="ok", model="x", tokens_used=1, latency_ms=1.0)

    import llmcascade.api as api_mod

    with TestClient(api_mod.app) as client:
        monkeypatch.setattr(api_mod._client, "submit", fake_submit)
        login = client.post(
            "/login",
            data={"username": "admin", "password": "admin"},
            follow_redirects=False,
        )
        assert login.status_code == 303
        csrf = client.cookies.get("llmcascade_csrf")
        assert csrf
        client.post(
            "/admin/change-password",
            data={
                "new_password": "newpassword1",
                "confirm_password": "newpassword1",
                "csrf_token": csrf,
            },
            follow_redirects=False,
        )
        csrf = client.cookies.get("llmcascade_csrf")
        denied = client.post("/v1/complete", json={"prompt": "hi"})
        assert denied.status_code == 403
        ok = client.post(
            "/v1/complete",
            json={"prompt": "hi"},
            headers={"X-CSRF-Token": csrf},
        )
        assert ok.status_code == 200


def test_production_profile_forces_auth(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("LLMCASCADE_PROFILE", "production")
    assert is_production_profile()
    assert require_auth_enabled()


def test_production_profile_refuses_start_without_keys(
    api_env, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("LLMCASCADE_PROFILE", "production")
    monkeypatch.delenv("LLMCASCADE_API_KEYS", raising=False)
    monkeypatch.delenv("LLMCASCADE_API_KEY_HASHES", raising=False)
    assert not auth_credentials_configured()
    import llmcascade.api as api_mod

    with pytest.raises(RuntimeError, match="production requires"):
        with TestClient(api_mod.app):
            pass
