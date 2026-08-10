from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from llmrouter.exceptions import RegistryError
from llmrouter.metrics import log

# auth_env_var -> alternate env names accepted at load/runtime
_AUTH_ALIASES: dict[str, tuple[str, ...]] = {
    "HF_TOKEN": ("HF_TOKEN", "HUGGINGFACE_API_KEY"),
}

KeySource = Literal["env", "store", "none"]
KeyTier = Literal["free", "paid"]

_PROVIDER_AUTH: dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "together": "TOGETHER_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "sambanova": "SAMBANOVA_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "huggingface": "HF_TOKEN",
    "cloudflare": "CLOUDFLARE_API_KEY",
    "cohere": "COHERE_API_KEY",
    "nvidia": "NVIDIA_NIM_API_KEY",
    "deepinfra": "DEEPINFRA_API_KEY",
}


class Limits(BaseModel):
    rpd: int = Field(ge=0)
    rpm: int = Field(ge=0)
    rps: int = Field(ge=0)
    tpm: int = Field(ge=0)
    max_context: int = Field(ge=1)


class ModelConfig(BaseModel):
    name: str
    provider: str
    endpoint: str
    auth_env_var: str
    limits: Limits
    capabilities: list[str] = Field(default_factory=list)
    priority: int = 100
    # Relative selection chance among eligible models (higher = more likely).
    weight: int = Field(default=1, ge=1)
    # When false, model is excluded from routing.
    enabled: bool = True
    # Which encrypted key pool to use when env key is absent.
    key_tier: KeyTier = "free"
    custom: bool = False
    # Ordered Gemini model IDs under one logical entry (provider=gemini).
    cascade: list[str] = Field(default_factory=list)
    # When true, limits are treated as researched free-tier caps; dashboard shows remaining.
    free_tier_verified: bool = False
    free_tier_note: str = ""


def default_models_path() -> Path:
    return Path(__file__).resolve().parent / "models.yaml"


def provider_auth_env(provider: str) -> str:
    return _PROVIDER_AUTH.get(provider, f"{provider.upper()}_API_KEY")


def resolve_auth_env(
    auth_env_var: str,
    *,
    provider: str | None = None,
    key_tier: KeyTier = "free",
) -> str | None:
    """Store keys preferred when present; else env (unless disable_env)."""
    if provider:
        try:
            from llmrouter.provider_store import env_disabled, get_decrypted_key

            stored = get_decrypted_key(provider, tier=key_tier)
            if stored:
                return stored
            if env_disabled(provider):
                return None
        except Exception:  # noqa: BLE001
            pass
    for name in _AUTH_ALIASES.get(auth_env_var, (auth_env_var,)):
        val = os.environ.get(name)
        if val:
            return val
    return None


def key_source(auth_env_var: str, *, provider: str, key_tier: KeyTier = "free") -> KeySource:
    try:
        from llmrouter.provider_store import env_disabled, get_decrypted_keys

        if get_decrypted_keys(provider, tier=key_tier) or get_decrypted_keys(
            provider, tier="paid" if key_tier == "free" else "free"
        ):
            return "store"
        if env_disabled(provider):
            return "none"
    except Exception:  # noqa: BLE001
        pass
    for name in _AUTH_ALIASES.get(auth_env_var, (auth_env_var,)):
        if os.environ.get(name):
            return "env"
    return "none"


def _missing_auth_label(auth_env_var: str) -> str:
    alts = _AUTH_ALIASES.get(auth_env_var)
    if alts and len(alts) > 1:
        return " or ".join(alts)
    return auth_env_var


def _model_ready(model: ModelConfig) -> bool:
    if not model.enabled:
        return False
    if not resolve_auth_env(model.auth_env_var, provider=model.provider, key_tier=model.key_tier):
        return False
    if model.provider == "cloudflare" and not os.environ.get("CLOUDFLARE_ACCOUNT_ID"):
        return False
    return True


def _apply_overrides(models: list[ModelConfig]) -> list[ModelConfig]:
    try:
        from llmrouter.model_store import load_overrides
    except Exception:  # noqa: BLE001
        return models
    overrides = load_overrides()
    out: list[ModelConfig] = []
    for m in models:
        ov = overrides.get(m.name) or {}
        data = m.model_dump()
        if "enabled" in ov:
            data["enabled"] = bool(ov["enabled"])
        if "weight" in ov:
            try:
                data["weight"] = max(1, int(ov["weight"]))
            except (TypeError, ValueError):
                pass
        if ov.get("key_tier") in ("free", "paid"):
            data["key_tier"] = ov["key_tier"]
        out.append(ModelConfig.model_validate(data))
    return out


def _read_yaml_models(path: str | Path | None = None) -> list[ModelConfig]:
    cfg_path = Path(path) if path else default_models_path()
    if not cfg_path.is_file():
        raise RegistryError(f"models config not found: {cfg_path}")
    raw: Any = yaml.safe_load(cfg_path.read_text())
    if not isinstance(raw, dict) or "models" not in raw:
        raise RegistryError("models.yaml must contain a top-level 'models' list")
    return [ModelConfig.model_validate(item) for item in raw["models"]]


def _read_custom_models() -> list[ModelConfig]:
    try:
        from llmrouter.model_store import load_custom_models
    except Exception:  # noqa: BLE001
        return []
    out: list[ModelConfig] = []
    for item in load_custom_models():
        try:
            data = dict(item)
            data.setdefault("auth_env_var", provider_auth_env(str(data.get("provider") or "")))
            data["custom"] = True
            out.append(ModelConfig.model_validate(data))
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping invalid custom model %s: %s", item.get("name"), exc)
    return out


def _merged_models(path: str | Path | None = None) -> list[ModelConfig]:
    base = _read_yaml_models(path)
    custom = _read_custom_models()
    by_name = {m.name: m for m in base}
    for m in custom:
        by_name[m.name] = m  # custom overrides same name
    return _apply_overrides(list(by_name.values()))


def list_all_models(path: str | Path | None = None) -> list[ModelConfig]:
    """All models (YAML + custom + overrides) regardless of key readiness."""
    return _merged_models(path)


def list_providers(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Unique providers with representative auth_env_var."""
    seen: dict[str, dict[str, Any]] = {}
    for m in list_all_models(path):
        if m.provider in seen:
            continue
        seen[m.provider] = {
            "provider": m.provider,
            "auth_env_var": m.auth_env_var,
            "needs_account_id": m.provider == "cloudflare",
        }
    # Ensure known providers appear even with zero models somehow
    for provider, env in _PROVIDER_AUTH.items():
        if provider not in seen:
            seen[provider] = {
                "provider": provider,
                "auth_env_var": env,
                "needs_account_id": provider == "cloudflare",
            }
    return list(seen.values())


def load_registry(path: str | Path | None = None, *, allow_empty: bool = False) -> list[ModelConfig]:
    models = _merged_models(path)
    ready: list[ModelConfig] = []
    for m in models:
        if not m.enabled:
            log.warning("skipping model %s — hidden/disabled", m.name)
            continue
        if _model_ready(m):
            ready.append(m)
            continue
        reason = _missing_auth_label(m.auth_env_var)
        if m.provider == "cloudflare" and not os.environ.get("CLOUDFLARE_ACCOUNT_ID"):
            reason = f"{reason}, CLOUDFLARE_ACCOUNT_ID"
        log.warning("skipping model %s — missing %s", m.name, reason)
    if not ready and not allow_empty:
        raise RegistryError("no models available — set at least one provider API key in the environment")
    return ready
