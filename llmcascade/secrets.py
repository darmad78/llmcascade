"""SECRET_KEY resolution and HKDF derivation for Fernet."""

from __future__ import annotations

import base64
import os
import secrets
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

_FERNET_INFO = b"llmcascade-provider-keys-v1"
_JWT_INFO = b"llmcascade-jwt-v1"


def data_dir() -> Path:
    raw = (os.environ.get("LLMCASCADE_DATA_DIR") or ".llmcascade").strip()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_secret_key() -> str:
    """Env SECRET_KEY, else persist/load from {DATA_DIR}/secret."""
    env = (os.environ.get("SECRET_KEY") or "").strip()
    if env:
        return env
    path = data_dir() / "secret"
    if path.is_file():
        val = path.read_text().strip()
        if val:
            return val
    val = secrets.token_urlsafe(48)
    path.write_text(val + "\n")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return val


def _hkdf(secret: str, *, info: bytes, length: int = 32) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=None,
        info=info,
    ).derive(secret.encode("utf-8"))


def fernet_key_material(secret: str | None = None) -> bytes:
    """urlsafe-base64 32-byte Fernet key derived via HKDF (never raw SECRET_KEY)."""
    raw = _hkdf(secret or resolve_secret_key(), info=_FERNET_INFO, length=32)
    return base64.urlsafe_b64encode(raw)


def jwt_signing_key(secret: str | None = None) -> str:
    """Derived signing material for JWTs (separate HKDF info from Fernet)."""
    raw = _hkdf(secret or resolve_secret_key(), info=_JWT_INFO, length=32)
    return base64.urlsafe_b64encode(raw).decode("ascii")
