"""Provider key encrypt/decrypt, env override, reload."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from llmrouter.provider_store import (
    decrypt_key,
    encrypt_key,
    get_decrypted_key,
    save_provider,
)
from llmrouter.queue_worker import RouterClient
from llmrouter.registry import load_registry, resolve_auth_env
from llmrouter.secrets import fernet_key_material


_LIMITS = {"rpd": 10, "rpm": 10, "rps": 1, "tpm": 1000, "max_context": 1024}


@pytest.fixture()
def secret_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLMROUTER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SECRET_KEY", "provider-store-test-secret-key!!")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    yield


def test_fernet_derived_not_raw_secret(secret_env):
    secret = "provider-store-test-secret-key!!"
    material = fernet_key_material(secret)
    assert material != secret.encode()
    token = encrypt_key("sk-test", secret=secret)
    assert decrypt_key(token, secret=secret) == "sk-test"
    assert "sk-test" not in token


def test_store_preferred_env_fallback(secret_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY", "from-env")
    save_provider("groq", api_key="from-store", free_paid="free")
    assert resolve_auth_env("GROQ_API_KEY", provider="groq") == "from-store"
    save_provider("groq", clear_free_keys=True, clear_paid_keys=True, disable_env=False)
    assert resolve_auth_env("GROQ_API_KEY", provider="groq") == "from-env"
    save_provider("groq", clear_free_keys=True, clear_paid_keys=True, disable_env=True)
    assert resolve_auth_env("GROQ_API_KEY", provider="groq") is None


def test_reload_picks_up_stored_key(secret_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    models = tmp_path / "models.yaml"
    models.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "name": "groq-model",
                        "provider": "groq",
                        "endpoint": "https://example.com",
                        "auth_env_var": "GROQ_API_KEY",
                        "limits": _LIMITS,
                        "capabilities": ["chat"],
                    }
                ]
            }
        )
    )
    client = RouterClient(models_path=str(models), allow_empty=True)
    assert client.registry == []
    save_provider("groq", api_key="stored-groq-key", free_paid="paid")
    n = client.reload_registry(allow_empty=True)
    assert n == 1
    assert client.registry[0].name == "groq-model"
    assert get_decrypted_key("groq") == "stored-groq-key"


def test_admin_providers_save_csrf(secret_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY", "bootstrap")
    monkeypatch.delenv("MONGODB_URI", raising=False)
    from fastapi.testclient import TestClient
    import llmrouter.admin_auth as admin_auth
    import llmrouter.api as api_mod

    admin_auth.login_lockout._failures.clear()
    admin_auth.login_lockout._locked_until.clear()

    with TestClient(api_mod.app) as c:
        login = c.post(
            "/login",
            data={"username": "admin", "password": "admin"},
            follow_redirects=False,
        )
        assert login.status_code == 303, login.text
        csrf = c.cookies.get("llmrouter_csrf")
        assert csrf, dict(c.cookies)
        c.post(
            "/admin/change-password",
            data={
                "new_password": "newpassword1",
                "confirm_password": "newpassword1",
                "csrf_token": csrf,
            },
            follow_redirects=False,
        )
        csrf = c.cookies.get("llmrouter_csrf")
        bad = c.post(
            "/admin/providers",
            json={"provider": "groq", "api_key": "ui-key", "free_paid": "paid"},
            headers={"X-CSRF-Token": "nope"},
        )
        assert bad.status_code == 403, bad.text
        ok = c.post(
            "/admin/providers",
            json={
                "provider": "groq",
                "api_key": "ui-key",
                "free_paid": "paid",
                "csrf_token": csrf,
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert ok.status_code == 200
        assert ok.json()["ok"] is True
