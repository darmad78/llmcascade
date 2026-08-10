"""Admin auth: first-run, password change, CSRF, lockout, pwd_version."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _clear_login_lockout():
    from llmrouter.admin_auth import login_lockout

    login_lockout._failures.clear()
    login_lockout._locked_until.clear()
    yield
    login_lockout._failures.clear()
    login_lockout._locked_until.clear()


@pytest.fixture()
def admin_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLMROUTER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SECRET_KEY", "test-secret-admin-auth-xxxxxxxx")
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.delenv("REQUIRE_AUTH", raising=False)
    import importlib
    import llmrouter.admin_auth as admin_auth
    import llmrouter.api as api_mod
    import llmrouter.auth_store as auth_store

    importlib.reload(auth_store)
    importlib.reload(admin_auth)
    admin_auth.login_lockout._failures.clear()
    admin_auth.login_lockout._locked_until.clear()
    importlib.reload(api_mod)
    with TestClient(api_mod.app) as c:
        yield c


def _login(client: TestClient, username="admin", password="admin"):
    return client.post(
        "/login",
        data={"username": username, "password": password},
        follow_redirects=False,
    )


def test_first_run_forces_password_change(admin_client: TestClient):
    r = admin_client.get("/dashboard", follow_redirects=False)
    assert r.status_code == 303
    assert "/login" in r.headers["location"]

    r = _login(admin_client)
    assert r.status_code == 303
    assert "/admin/change-password" in r.headers["location"]

    r = admin_client.get("/dashboard", follow_redirects=False)
    assert r.status_code == 303
    assert "/admin/change-password" in r.headers["location"]

    csrf = admin_client.cookies.get("llmrouter_csrf")
    assert csrf
    r = admin_client.post(
        "/admin/change-password",
        data={
            "new_password": "newpassword1",
            "confirm_password": "newpassword1",
            "csrf_token": csrf,
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert "/dashboard" in r.headers["location"]

    r = admin_client.get("/dashboard")
    assert r.status_code == 200


def test_csrf_required_on_password_change(admin_client: TestClient):
    _login(admin_client)
    r = admin_client.post(
        "/admin/change-password",
        data={
            "new_password": "newpassword1",
            "confirm_password": "newpassword1",
            "csrf_token": "wrong",
        },
        follow_redirects=False,
    )
    assert r.status_code == 403


def test_pwd_version_invalidates_old_jwt(admin_client: TestClient, monkeypatch: pytest.MonkeyPatch):
    _login(admin_client)
    csrf = admin_client.cookies.get("llmrouter_csrf")
    old_session = admin_client.cookies.get("llmrouter_session")
    admin_client.post(
        "/admin/change-password",
        data={
            "new_password": "newpassword1",
            "confirm_password": "newpassword1",
            "csrf_token": csrf,
        },
        follow_redirects=False,
    )
    # Attach stale cookie and ensure rejection
    admin_client.cookies.set("llmrouter_session", old_session)
    r = admin_client.get("/v1/events")
    assert r.status_code == 401


def test_login_lockout(admin_client: TestClient):
    for _ in range(5):
        r = _login(admin_client, password="wrong")
        assert r.status_code in (401, 429)
    r = _login(admin_client, password="wrong")
    assert r.status_code == 429
