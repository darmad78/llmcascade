"""Optional API-key auth for POST /v1/complete."""

from __future__ import annotations

import os
import secrets
from typing import Iterable

_BCRYPT_PREFIXES = ("$2a$", "$2b$", "$2y$")


def _truthy(raw: str | None, *, default: str = "false") -> bool:
    return (raw if raw is not None else default).strip().lower() in ("1", "true", "yes", "on")


def is_production_profile() -> bool:
    """LLMCASCADE_PROFILE=production|prod forces fail-closed API auth."""
    profile = (os.environ.get("LLMCASCADE_PROFILE") or "").strip().lower()
    return profile in ("production", "prod")


def require_auth_enabled() -> bool:
    if is_production_profile():
        return True
    return _truthy(os.environ.get("REQUIRE_AUTH"), default="false")


def _looks_like_bcrypt(value: str) -> bool:
    return value.startswith(_BCRYPT_PREFIXES)


def _split_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def configured_plaintext_keys() -> set[str]:
    """Plaintext keys from LLMCASCADE_API_KEYS (entries that are not bcrypt hashes)."""
    raw = (os.environ.get("LLMCASCADE_API_KEYS") or "").strip()
    if not raw:
        return set()
    return {p for p in _split_csv(raw) if not _looks_like_bcrypt(p)}


def configured_api_key_hashes() -> list[str]:
    """Bcrypt hashes from LLMCASCADE_API_KEY_HASHES and hash-looking LLMCASCADE_API_KEYS entries."""
    out: list[str] = []
    for env_name in ("LLMCASCADE_API_KEY_HASHES", "LLMCASCADE_API_KEYS"):
        raw = (os.environ.get(env_name) or "").strip()
        if not raw:
            continue
        for part in _split_csv(raw):
            if _looks_like_bcrypt(part):
                out.append(part)
    return out


def configured_api_keys() -> set[str]:
    """Backward-compatible: plaintext keys only (hashes live in configured_api_key_hashes)."""
    return configured_plaintext_keys()


def auth_credentials_configured() -> bool:
    return bool(configured_plaintext_keys() or configured_api_key_hashes())


def api_rpm_limit() -> int | None:
    raw = (os.environ.get("LLMCASCADE_API_RPM") or "").strip()
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


def _match_plaintext(api_key: str, allowed: Iterable[str]) -> bool:
    matched = False
    for candidate in allowed:
        # Constant-time vs each candidate; OR accumulates without short-circuit on match.
        matched = secrets.compare_digest(api_key, candidate) or matched
    return matched


def _match_hashes(api_key: str, hashes: Iterable[str]) -> bool:
    try:
        import bcrypt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "bcrypt is required to validate hashed API keys; install llmcascade[api]"
        ) from exc
    key_bytes = api_key.encode("utf-8")
    matched = False
    for hashed in hashes:
        try:
            ok = bcrypt.checkpw(key_bytes, hashed.encode("ascii"))
        except (ValueError, TypeError):
            ok = False
        matched = ok or matched
    return matched


def validate_api_key(
    api_key: str | None,
    allowed: Iterable[str] | None = None,
    *,
    hashes: Iterable[str] | None = None,
) -> bool:
    if not api_key:
        return False
    plain = set(allowed) if allowed is not None else configured_plaintext_keys()
    hash_list = list(hashes) if hashes is not None else configured_api_key_hashes()
    if not plain and not hash_list:
        return False
    ok_plain = _match_plaintext(api_key, plain) if plain else False
    ok_hash = _match_hashes(api_key, hash_list) if hash_list else False
    return ok_plain or ok_hash


def hash_api_key(plaintext: str) -> str:
    """Helper for operators: bcrypt-hash a plaintext API key."""
    import bcrypt

    return bcrypt.hashpw(plaintext.encode("utf-8"), bcrypt.gensalt()).decode("ascii")
