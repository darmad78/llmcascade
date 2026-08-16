from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from llmcascade.exceptions import RegistryError
from llmcascade.registry import load_registry


def _write_models(tmp: Path, models: list[dict]) -> Path:
    path = tmp / "models.yaml"
    path.write_text(yaml.safe_dump({"models": models}))
    return path


_LIMITS = {"rpd": 10, "rpm": 10, "rps": 1, "tpm": 1000, "max_context": 1024}


def test_skips_models_without_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
    path = _write_models(
        tmp_path,
        [
            {
                "name": "groq-model",
                "provider": "groq",
                "endpoint": "https://example.com",
                "auth_env_var": "GROQ_API_KEY",
                "limits": _LIMITS,
                "capabilities": ["chat"],
            },
            {
                "name": "gemini",
                "provider": "gemini",
                "endpoint": "https://example.com/{model}",
                "auth_env_var": "GOOGLE_API_KEY",
                "limits": _LIMITS,
                "capabilities": ["chat"],
                "cascade": ["gemini-2.0-flash"],
            },
        ],
    )
    reg = load_registry(path)
    assert [m.name for m in reg] == ["gemini"]


def test_fails_when_no_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    path = _write_models(
        tmp_path,
        [
            {
                "name": "groq-model",
                "provider": "groq",
                "endpoint": "https://example.com",
                "auth_env_var": "GROQ_API_KEY",
                "limits": _LIMITS,
                "capabilities": ["chat"],
            },
        ],
    )
    with pytest.raises(RegistryError, match="no models available"):
        load_registry(path)


def test_cloudflare_needs_account_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cf-key")
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    path = _write_models(
        tmp_path,
        [
            {
                "name": "cf-model",
                "provider": "cloudflare",
                "endpoint": "https://example.com/{account_id}",
                "auth_env_var": "CLOUDFLARE_API_KEY",
                "limits": _LIMITS,
                "capabilities": ["chat"],
            },
        ],
    )
    with pytest.raises(RegistryError, match="no models available"):
        load_registry(path)


def test_yaml_embed_catalog_is_wide_and_wired():
    from llmcascade.adapters import _ADAPTERS
    from llmcascade.registry import _read_yaml_models, default_models_path

    models = _read_yaml_models(default_models_path())
    embeds = [m for m in models if "embed" in m.capabilities]
    names = [m.name for m in embeds]
    assert len(names) >= 25
    assert len(set(names)) == len(names)
    for m in embeds:
        assert m.provider in _ADAPTERS
    providers = {m.provider for m in embeds}
    for extra in ("voyage", "nomic", "mixedbread", "siliconflow"):
        assert extra in providers
