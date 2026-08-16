"""Embedding area nav, capability filters, public help."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llmcascade.ui import filter_dashboard, nav_html


def test_nav_html_embed_prefix():
    html = nav_html("embed", "stats")
    assert 'href="/embed/dashboard"' in html
    assert 'href="/embed/stats"' in html
    assert 'href="/embed/providers"' in html
    assert 'href="/embed/help"' in html
    assert "Embeddings" in html


def test_filter_dashboard_keeps_embed_models_only():
    data = {
        "models": [
            {"name": "a", "capabilities": ["chat"]},
            {"name": "b", "capabilities": ["embed"]},
        ],
        "events": [
            {"type": "request_ok", "capability": "chat"},
            {"type": "request_ok", "capability": "embed"},
            {"type": "system"},
        ],
        "errors": [],
        "next_pick": {"name": "a"},
        "next_embed": {"name": "b"},
        "gemini_cascade": {"x": 1},
    }
    out = filter_dashboard(data, "embed")
    assert [m["name"] for m in out["models"]] == ["b"]
    assert out["next_pick"]["name"] == "b"
    assert out["gemini_cascade"] is None
    types = [(e.get("type"), e.get("capability")) for e in out["events"]]
    assert ("request_ok", "embed") in types
    assert ("system", None) in types
    assert ("request_ok", "chat") not in types


@pytest.fixture(autouse=True)
def _clear_login_lockout():
    from llmcascade.admin_auth import login_lockout

    login_lockout._failures.clear()
    login_lockout._locked_until.clear()
    yield
    login_lockout._failures.clear()
    login_lockout._locked_until.clear()


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLMCASCADE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SECRET_KEY", "test-secret-embed-ui-xxxxxxxxxxxx")
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.delenv("REQUIRE_AUTH", raising=False)
    import importlib
    import llmcascade.admin_auth as admin_auth
    import llmcascade.api as api_mod
    import llmcascade.auth_store as auth_store

    importlib.reload(auth_store)
    importlib.reload(admin_auth)
    admin_auth.login_lockout._failures.clear()
    admin_auth.login_lockout._locked_until.clear()
    importlib.reload(api_mod)
    with TestClient(api_mod.app) as c:
        yield c


def _session(client: TestClient) -> None:
    client.post("/login", data={"username": "admin", "password": "admin"}, follow_redirects=False)
    csrf = client.cookies.get("llmcascade_csrf")
    client.post(
        "/admin/change-password",
        data={
            "new_password": "newpassword1",
            "confirm_password": "newpassword1",
            "csrf_token": csrf,
        },
        follow_redirects=False,
    )


def test_embed_help_is_public(client: TestClient):
    r = client.get("/embed/help")
    assert r.status_code == 200
    assert "area-embed" in r.text
    assert "/v1/embed" in r.text
    assert "Embeddings" in r.text


def test_embed_dashboard_requires_auth(client: TestClient):
    r = client.get("/embed/dashboard", follow_redirects=False)
    assert r.status_code == 303
    assert "/login" in r.headers["location"]


def test_embed_pages_after_login(client: TestClient):
    _session(client)
    dash = client.get("/embed/dashboard")
    assert dash.status_code == 200
    assert "href=\"/embed/stats\"" in dash.text
    assert "Embedding models" in dash.text
    assert "const CAPABILITY = \"embed\"" in dash.text
    assert "Test embed" in dash.text
    assert "/v1/embed" in dash.text
    assert "Embedding models" in dash.text

    stats = client.get("/embed/stats")
    assert stats.status_code == 200
    assert "const CAPABILITY = \"embed\"" in stats.text

    prov = client.get("/embed/providers")
    assert prov.status_code == 200
    assert "const CAPABILITY = \"embed\"" in prov.text
    assert '($("m-priority") && $("m-priority").value)' in prov.text

    llm = client.get("/dashboard")
    assert llm.status_code == 200
    assert "Configured models" in llm.text
    assert "const CAPABILITY = \"chat\"" in llm.text


def test_dashboard_and_stats_capability_query(client: TestClient):
    _session(client)
    r = client.get("/v1/dashboard?capability=embed")
    assert r.status_code == 200
    body = r.json()
    assert body.get("capability") == "embed"
    for m in body.get("models") or []:
        assert "embed" in (m.get("capabilities") or [])

    r = client.get("/v1/stats?capability=embed")
    assert r.status_code == 200
    assert r.json().get("capability") == "embed"
