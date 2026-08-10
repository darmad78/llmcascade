"""Encrypted-at-rest provider API keys + Free/Paid labels."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Literal

from cryptography.fernet import Fernet, InvalidToken

from llmrouter.secrets import data_dir, fernet_key_material, resolve_secret_key

_LOCK = threading.Lock()
FreePaid = Literal["free", "paid"]


def _path() -> Path:
    return data_dir() / "providers.json"


def _fernet(secret: str | None = None) -> Fernet:
    return Fernet(fernet_key_material(secret or resolve_secret_key()))


def _load_raw() -> dict[str, Any]:
    path = _path()
    if not path.is_file():
        return {}
    with _LOCK:
        data = json.loads(path.read_text())
    return data if isinstance(data, dict) else {}


def _save_raw(data: dict[str, Any]) -> None:
    path = _path()
    with _LOCK:
        path.write_text(json.dumps(data, indent=2) + "\n")
        try:
            path.chmod(0o600)
        except OSError:
            pass


def encrypt_key(plaintext: str, *, secret: str | None = None) -> str:
    return _fernet(secret).encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt_key(token: str, *, secret: str | None = None) -> str:
    return _fernet(secret).decrypt(token.encode("ascii")).decode("utf-8")


def get_provider_entry(provider: str) -> dict[str, Any] | None:
    data = _load_raw()
    entry = data.get(provider)
    return entry if isinstance(entry, dict) else None


def get_decrypted_key(provider: str, *, secret: str | None = None) -> str | None:
    entry = get_provider_entry(provider)
    if not entry:
        return None
    enc = entry.get("key_enc")
    if not enc:
        return None
    try:
        return decrypt_key(str(enc), secret=secret)
    except (InvalidToken, ValueError):
        return None


def get_free_paid(provider: str) -> FreePaid:
    entry = get_provider_entry(provider)
    if not entry:
        return "free"
    val = str(entry.get("free_paid") or "free").lower()
    return "paid" if val == "paid" else "free"


def key_is_set(provider: str) -> bool:
    entry = get_provider_entry(provider)
    return bool(entry and entry.get("key_enc"))


def save_provider(
    provider: str,
    *,
    api_key: str | None = None,
    free_paid: FreePaid | None = None,
    clear_key: bool = False,
    secret: str | None = None,
) -> dict[str, Any]:
    data = _load_raw()
    entry = dict(data.get(provider) or {})
    if clear_key:
        entry.pop("key_enc", None)
    elif api_key is not None and api_key.strip():
        entry["key_enc"] = encrypt_key(api_key.strip(), secret=secret)
    if free_paid is not None:
        entry["free_paid"] = free_paid
    if "free_paid" not in entry:
        entry["free_paid"] = "free"
    data[provider] = entry
    _save_raw(data)
    return entry


def list_stored_providers() -> dict[str, dict[str, Any]]:
    """Public metadata only (no decrypted keys)."""
    out: dict[str, dict[str, Any]] = {}
    for provider, entry in _load_raw().items():
        if not isinstance(entry, dict):
            continue
        out[provider] = {
            "key_set": bool(entry.get("key_enc")),
            "free_paid": "paid" if str(entry.get("free_paid") or "").lower() == "paid" else "free",
        }
    return out
