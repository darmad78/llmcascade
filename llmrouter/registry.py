from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from llmrouter.exceptions import RegistryError
from llmrouter.metrics import log

# auth_env_var -> alternate env names accepted at load/runtime
_AUTH_ALIASES: dict[str, tuple[str, ...]] = {
    "HF_TOKEN": ("HF_TOKEN", "HUGGINGFACE_API_KEY"),
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
    # Ordered Gemini model IDs under one logical entry (provider=gemini).
    cascade: list[str] = Field(default_factory=list)
    # When true, limits are treated as researched free-tier caps; dashboard shows remaining.
    free_tier_verified: bool = False
    free_tier_note: str = ""


def default_models_path() -> Path:
    return Path(__file__).resolve().parent / "models.yaml"


def resolve_auth_env(auth_env_var: str) -> str | None:
    for name in _AUTH_ALIASES.get(auth_env_var, (auth_env_var,)):
        val = os.environ.get(name)
        if val:
            return val
    return None


def _missing_auth_label(auth_env_var: str) -> str:
    alts = _AUTH_ALIASES.get(auth_env_var)
    if alts and len(alts) > 1:
        return " or ".join(alts)
    return auth_env_var


def _model_ready(model: ModelConfig) -> bool:
    if not resolve_auth_env(model.auth_env_var):
        return False
    if model.provider == "cloudflare" and not os.environ.get("CLOUDFLARE_ACCOUNT_ID"):
        return False
    return True


def load_registry(path: str | Path | None = None) -> list[ModelConfig]:
    cfg_path = Path(path) if path else default_models_path()
    if not cfg_path.is_file():
        raise RegistryError(f"models config not found: {cfg_path}")
    raw: Any = yaml.safe_load(cfg_path.read_text())
    if not isinstance(raw, dict) or "models" not in raw:
        raise RegistryError("models.yaml must contain a top-level 'models' list")
    models = [ModelConfig.model_validate(item) for item in raw["models"]]
    ready: list[ModelConfig] = []
    for m in models:
        if _model_ready(m):
            ready.append(m)
            continue
        reason = _missing_auth_label(m.auth_env_var)
        if m.provider == "cloudflare" and not os.environ.get("CLOUDFLARE_ACCOUNT_ID"):
            reason = f"{reason}, CLOUDFLARE_ACCOUNT_ID"
        log.warning("skipping model %s — missing %s", m.name, reason)
    if not ready:
        raise RegistryError("no models available — set at least one provider API key in the environment")
    return ready
