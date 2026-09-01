"""HTTP MCP JSON-RPC at POST /mcp."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llmcascade.api_auth import hash_api_key, mcp_principal, validate_admin_api_key


@pytest.fixture()
def api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLMCASCADE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-for-jwt-and-fernet")
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    monkeypatch.delenv("REQUIRE_AUTH", raising=False)
    monkeypatch.delenv("LLMCASCADE_PROFILE", raising=False)
    monkeypatch.delenv("LLMCASCADE_API_KEYS", raising=False)
    monkeypatch.delenv("LLMCASCADE_API_KEY_HASHES", raising=False)
    monkeypatch.delenv("LLMCASCADE_ADMIN_API_KEYS", raising=False)
    monkeypatch.delenv("LLMCASCADE_ADMIN_API_KEY_HASHES", raising=False)
    monkeypatch.delenv("LLMCASCADE_API_RPM", raising=False)
    monkeypatch.delenv("MONGODB_URI", raising=False)
    yield


@pytest.fixture()
def client(api_env):
    import llmcascade.api as api_mod

    with TestClient(api_mod.app) as c:
        yield c


def _rpc(method: str, params: dict | None = None, req_id: int = 1) -> dict:
    body: dict = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        body["params"] = params
    return body


def test_mcp_principal_admin_not_inference(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLMCASCADE_API_KEYS", "inf-key")
    monkeypatch.setenv("LLMCASCADE_ADMIN_API_KEYS", "adm-key")
    assert mcp_principal("inf-key") == "inference"
    assert mcp_principal("adm-key") == "admin"
    assert mcp_principal("nope") == "none"
    assert not validate_admin_api_key("inf-key")


def test_mcp_hashed_admin_key(monkeypatch: pytest.MonkeyPatch):
    hashed = hash_api_key("adm-plain")
    monkeypatch.setenv("LLMCASCADE_ADMIN_API_KEY_HASHES", hashed)
    assert validate_admin_api_key("adm-plain")
    assert not validate_admin_api_key("wrong")
    assert mcp_principal("adm-plain") == "admin"


def test_mcp_initialize_and_inference_tools_local(client: TestClient):
    r = client.post("/mcp", json=_rpc("initialize", {"protocolVersion": "2025-03-26"}))
    assert r.status_code == 200
    result = r.json()["result"]
    assert result["serverInfo"]["name"] == "llmcascade"
    listed = client.post("/mcp", json=_rpc("tools/list")).json()["result"]["tools"]
    names = {t["name"] for t in listed}
    assert "complete" in names
    assert "save_provider" not in names


def test_mcp_complete(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    async def fake_submit(prompt, capability="chat", notes=None, **params):
        from llmcascade.adapters.base import LLMResponse

        return LLMResponse(text="hi", model="x", tokens_used=1, latency_ms=1.0)

    import llmcascade.api as api_mod

    monkeypatch.setattr(api_mod._client, "submit", fake_submit)
    r = client.post(
        "/mcp",
        json=_rpc("tools/call", {"name": "complete", "arguments": {"prompt": "hey"}}),
    )
    assert r.status_code == 200
    text = r.json()["result"]["content"][0]["text"]
    assert '"text": "hi"' in text
    assert r.json()["result"]["isError"] is False


def test_mcp_require_auth_401(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REQUIRE_AUTH", "true")
    monkeypatch.setenv("LLMCASCADE_API_KEYS", "good-key")
    r = client.post("/mcp", json=_rpc("initialize", {"protocolVersion": "2025-03-26"}))
    assert r.status_code == 401


def test_mcp_inference_key_cannot_admin(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REQUIRE_AUTH", "true")
    monkeypatch.setenv("LLMCASCADE_API_KEYS", "inf-key")
    monkeypatch.setenv("LLMCASCADE_ADMIN_API_KEYS", "adm-key")
    headers = {"Authorization": "Bearer inf-key"}
    listed = client.post("/mcp", json=_rpc("tools/list"), headers=headers).json()["result"]["tools"]
    assert "save_provider" not in {t["name"] for t in listed}
    r = client.post(
        "/mcp",
        json=_rpc("tools/call", {"name": "list_providers", "arguments": {}}),
        headers=headers,
    )
    assert r.json()["result"]["isError"] is True
    assert "inference API keys cannot call admin tools" in r.json()["result"]["content"][0]["text"]


def test_mcp_admin_key_lists_and_calls_admin(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REQUIRE_AUTH", "true")
    monkeypatch.setenv("LLMCASCADE_API_KEYS", "inf-key")
    monkeypatch.setenv("LLMCASCADE_ADMIN_API_KEYS", "adm-key")
    headers = {"Authorization": "Bearer adm-key"}
    listed = client.post("/mcp", json=_rpc("tools/list"), headers=headers).json()["result"]["tools"]
    names = {t["name"] for t in listed}
    assert "complete" in names
    assert "list_providers" in names
    r = client.post(
        "/mcp",
        json=_rpc("tools/call", {"name": "list_providers", "arguments": {}}),
        headers=headers,
    )
    assert r.status_code == 200
    assert r.json()["result"]["isError"] is False
    assert "providers" in r.json()["result"]["content"][0]["text"]


def test_mcp_cookie_session_is_not_admin(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REQUIRE_AUTH", "true")
    monkeypatch.setenv("LLMCASCADE_API_KEYS", "inf-key")
    monkeypatch.setenv("LLMCASCADE_ADMIN_API_KEYS", "adm-key")
    login = client.post(
        "/login",
        data={"username": "admin", "password": "admin"},
        follow_redirects=False,
    )
    assert login.status_code == 303
    r = client.post("/mcp", json=_rpc("initialize", {"protocolVersion": "2025-03-26"}))
    assert r.status_code == 401
