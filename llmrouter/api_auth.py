"""Optional API-key auth for POST /v1/complete."""

from __future__ import annotations

import os
from typing import Iterable


def require_auth_enabled() -> bool:
    return (os.environ.get("REQUIRE_AUTH") or "false").strip().lower() in ("1", "true", "yes", "on")


def configured_api_keys() -> set[str]:
    raw = (os.environ.get("LLMROUTER_API_KEYS") or "").strip()
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def api_rpm_limit() -> int | None:
    raw = (os.environ.get("LLMROUTER_API_RPM") or "").strip()
    if not raw:
        return None
    try:
        val = int(raw)
    except ValueError:
        return None
    return val if val >= 1 else None


def extract_api_key(authorization: str | None, x_api_key: str | None) -> str | None:
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()
    if authorization:
        parts = authorization.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip():
            return parts[1].strip()
    return None


def validate_api_key(api_key: str | None, allowed: Iterable[str] | None = None) -> bool:
    keys = set(allowed) if allowed is not None else configured_api_keys()
    if not keys:
        return False
    return bool(api_key) and api_key in keys
