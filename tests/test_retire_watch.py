from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llmcascade.cascade import classify_failure
from llmcascade.registry import Limits, ModelConfig
from llmcascade.retire_watch import (
    RetiredTarget,
    classify_probe,
    parse_catalog_ids,
    parse_suggested_id,
    retired_from_status,
)
from llmcascade.yaml_edit import replace_model_id

_LIMITS = Limits(rpd=10, rpm=10, rps=1, tpm=1000, max_context=1024)


def _model(**kwargs) -> ModelConfig:
    data = {
        "name": "llama-3.3-70b-versatile",
        "provider": "groq",
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "auth_env_var": "GROQ_API_KEY",
        "limits": _LIMITS,
        "capabilities": ["chat"],
    }
    data.update(kwargs)
    return ModelConfig.model_validate(data)


def test_replace_model_id_keeps_comments():
    text = (
        "models:\n"
        "  # Groq free: llama-3.3-70b-versatile\n"
        "  - name: llama-3.3-70b-versatile\n"
        '    free_tier_note: "Groq free · llama-3.3-70b-versatile · 1,000 RPD"\n'
        "    cascade:\n"
        "      - keep-me\n"
    )
    out = replace_model_id(text, "llama-3.3-70b-versatile", "openai/gpt-oss-120b")
    assert "name: openai/gpt-oss-120b" in out
    assert "openai/gpt-oss-120b" in out
    assert "- keep-me" in out
    assert "# Groq free: llama-3.3-70b-versatile" in out


def test_parse_suggested_id_json_and_catalog():
    catalog = ["openai/gpt-oss-120b", "qwen/qwen3.6-27b"]
    assert parse_suggested_id('{"model":"qwen/qwen3.6-27b"}', catalog) == "qwen/qwen3.6-27b"
    assert parse_suggested_id("use openai/gpt-oss-120b please", catalog) == "openai/gpt-oss-120b"
    assert parse_suggested_id("paid-model", catalog) is None


def test_classify_probe():
    assert classify_probe(404, "The model llama-3.3-70b-versatile does not exist") == "gone"
    assert classify_probe(402, "Insufficient Balance") == "paid"
    assert classify_probe(200, "ok") == "free"
    assert classify_probe(429, "rate limit") == "free"
    assert classify_failure(404, "model not found") == "permanent"


def test_retired_from_permanent_cooldown():
    models = [_model()]
    found = retired_from_status(
        models=models,
        cooldowns={"llama-3.3-70b-versatile": {"kind": "permanent", "remaining_s": 1000}},
        gemini=None,
    )
    assert [t.model_id for t in found] == ["llama-3.3-70b-versatile"]
    daily = retired_from_status(
        models=models,
        cooldowns={"llama-3.3-70b-versatile": {"kind": "daily", "remaining_s": 1000}},
        gemini=None,
    )
    assert daily == []


def test_retired_gemini_member():
    gemini = _model(
        name="gemini",
        provider="gemini",
        endpoint="https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        auth_env_var="GOOGLE_API_KEY",
        cascade=["gemini-3.6-flash", "gemini-2.5-flash"],
    )
    found = retired_from_status(
        models=[gemini],
        cooldowns={},
        gemini={"cooldown_kinds": {"gemini-3.6-flash": "permanent"}},
    )
    assert found == [
        RetiredTarget(
            model_id="gemini-3.6-flash",
            provider="gemini",
            endpoint=gemini.endpoint,
            auth_env_var="GOOGLE_API_KEY",
            cascade_parent="gemini",
        )
    ]


def test_parse_catalog_openrouter_free_only():
    payload = {
        "data": [
            {"id": "x/y:free"},
            {"id": "x/y"},
        ]
    }
    assert parse_catalog_ids("openrouter", payload) == ["x/y:free"]


@pytest.mark.asyncio
async def test_replace_one_writes_yaml_and_reloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import llmcascade.retire_watch as rw

    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "name": "old-model",
                        "provider": "groq",
                        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
                        "auth_env_var": "GROQ_API_KEY",
                        "limits": {
                            "rpd": 10,
                            "rpm": 10,
                            "rps": 1,
                            "tpm": 1000,
                            "max_context": 1024,
                        },
                        "capabilities": ["chat"],
                    }
                ]
            }
        )
    )
    monkeypatch.setenv("GROQ_API_KEY", "g-key")

    class FakeHttp:
        pass

    watch = rw.RetireWatch(
        base_url="http://127.0.0.1:9",
        models_path=yaml_path,
        http=FakeHttp(),  # type: ignore[arg-type]
    )

    async def fake_catalog(target):
        return ["paid-id", "free-id"]

    async def fake_suggest(dead_id, provider, catalog):
        return catalog[0]

    probes = {"paid-id": "paid", "free-id": "free"}

    async def fake_probe(target, candidate):
        return probes[candidate]

    reloads = []

    async def fake_reload():
        reloads.append(True)

    monkeypatch.setattr(watch, "list_catalog", fake_catalog)
    monkeypatch.setattr(watch, "suggest_id", fake_suggest)
    monkeypatch.setattr(watch, "probe", fake_probe)
    monkeypatch.setattr(watch, "reload_api", fake_reload)

    target = RetiredTarget(
        model_id="old-model",
        provider="groq",
        endpoint="https://api.groq.com/openai/v1/chat/completions",
        auth_env_var="GROQ_API_KEY",
    )
    result = await watch.replace_one(target)
    assert result["new"] == "free-id"
    assert "name: free-id" in yaml_path.read_text()
    assert reloads == [True]
