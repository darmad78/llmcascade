"""Encrypted-at-rest provider API keys (free + paid lists)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Literal

from cryptography.fernet import Fernet, InvalidToken

from llmcascade.secrets import data_dir, fernet_key_material, resolve_secret_key

_LOCK = threading.Lock()
FreePaid = Literal["free", "paid"]
_RR: dict[str, int] = {}


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


def _migrate_entry(entry: dict[str, Any], *, secret: str | None = None) -> dict[str, Any]:
    """Normalize legacy single key_enc/free_paid into free_keys_enc / paid_keys_enc."""
    out = dict(entry)
    free_list = list(out.get("free_keys_enc") or [])
    paid_list = list(out.get("paid_keys_enc") or [])
    legacy = out.get("key_enc")
    if legacy and not free_list and not paid_list:
        tier = str(out.get("free_paid") or "free").lower()
        if tier == "paid":
            paid_list = [legacy]
        else:
            free_list = [legacy]
    out["free_keys_enc"] = [str(x) for x in free_list if x]
    out["paid_keys_enc"] = [str(x) for x in paid_list if x]
    out.pop("key_enc", None)
    # Keep free_paid as a soft default label for UI when lists are empty.
    if "free_paid" not in out:
        out["free_paid"] = "paid" if out["paid_keys_enc"] and not out["free_keys_enc"] else "free"
    return out


def get_provider_entry(provider: str) -> dict[str, Any] | None:
    data = _load_raw()
    entry = data.get(provider)
    if not isinstance(entry, dict):
        return None
    return _migrate_entry(entry)


def _decrypt_list(tokens: list[str], *, secret: str | None = None) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        try:
            out.append(decrypt_key(str(tok), secret=secret))
        except (InvalidToken, ValueError):
            continue
    return out


def get_decrypted_keys(
    provider: str,
    *,
    tier: FreePaid = "free",
    secret: str | None = None,
) -> list[str]:
    entry = get_provider_entry(provider)
    if not entry:
        return []
    key = "paid_keys_enc" if tier == "paid" else "free_keys_enc"
    return _decrypt_list(list(entry.get(key) or []), secret=secret)


def env_disabled(provider: str) -> bool:
    entry = get_provider_entry(provider)
    return bool(entry and entry.get("disable_env"))


def get_decrypted_key(
    provider: str,
    *,
    tier: FreePaid = "free",
    secret: str | None = None,
) -> str | None:
    """Round-robin across keys in the requested tier; fall back to the other tier."""
    primary = get_decrypted_keys(provider, tier=tier, secret=secret)
    secondary = get_decrypted_keys(
        provider, tier="paid" if tier == "free" else "free", secret=secret
    )
    keys = primary or secondary
    if not keys:
        return None
    rr_key = f"{provider}:{tier if primary else ('paid' if tier == 'free' else 'free')}"
    idx = _RR.get(rr_key, 0) % len(keys)
    _RR[rr_key] = idx + 1
    return keys[idx]


def get_free_paid(provider: str) -> FreePaid:
    entry = get_provider_entry(provider)
    if not entry:
        return "free"
    if entry.get("paid_keys_enc") and not entry.get("free_keys_enc"):
        return "paid"
    val = str(entry.get("free_paid") or "free").lower()
    return "paid" if val == "paid" else "free"


def key_is_set(provider: str) -> bool:
    entry = get_provider_entry(provider)
    if not entry:
        return False
    return bool(entry.get("free_keys_enc") or entry.get("paid_keys_enc"))


def save_provider(
    provider: str,
    *,
    api_key: str | None = None,
    free_paid: FreePaid | None = None,
    clear_key: bool = False,
    free_keys: list[str] | None = None,
    paid_keys: list[str] | None = None,
    add_free_key: str | None = None,
    add_paid_key: str | None = None,
    clear_free_keys: bool = False,
    clear_paid_keys: bool = False,
    replace_free_key: str | None = None,
    replace_paid_key: str | None = None,
    disable_env: bool | None = None,
    secret: str | None = None,
) -> dict[str, Any]:
    data = _load_raw()
    entry = _migrate_entry(dict(data.get(provider) or {}), secret=secret)

    if clear_free_keys:
        entry["free_keys_enc"] = []
    if clear_paid_keys:
        entry["paid_keys_enc"] = []

    if free_keys is not None:
        entry["free_keys_enc"] = [
            encrypt_key(k.strip(), secret=secret) for k in free_keys if k and k.strip()
        ]
    if paid_keys is not None:
        entry["paid_keys_enc"] = [
            encrypt_key(k.strip(), secret=secret) for k in paid_keys if k and k.strip()
        ]

    # Legacy single-key helpers
    if clear_key:
        entry["free_keys_enc"] = []
        entry["paid_keys_enc"] = []
    elif api_key is not None and api_key.strip():
        tier = free_paid or "free"
        enc = encrypt_key(api_key.strip(), secret=secret)
        if tier == "paid":
            entry.setdefault("paid_keys_enc", []).append(enc)
        else:
            entry.setdefault("free_keys_enc", []).append(enc)

    if replace_free_key is not None:
        entry["free_keys_enc"] = (
            [encrypt_key(replace_free_key.strip(), secret=secret)]
            if replace_free_key.strip()
            else []
        )
        entry["disable_env"] = True
    if replace_paid_key is not None:
        entry["paid_keys_enc"] = (
            [encrypt_key(replace_paid_key.strip(), secret=secret)]
            if replace_paid_key.strip()
            else []
        )
        entry["disable_env"] = True

    if add_free_key and add_free_key.strip():
        entry.setdefault("free_keys_enc", []).append(
            encrypt_key(add_free_key.strip(), secret=secret)
        )
    if add_paid_key and add_paid_key.strip():
        entry.setdefault("paid_keys_enc", []).append(
            encrypt_key(add_paid_key.strip(), secret=secret)
        )

    if disable_env is not None:
        entry["disable_env"] = bool(disable_env)

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
        migrated = _migrate_entry(entry)
        free_n = len(migrated.get("free_keys_enc") or [])
        paid_n = len(migrated.get("paid_keys_enc") or [])
        out[provider] = {
            "key_set": free_n + paid_n > 0,
            "free_key_count": free_n,
            "paid_key_count": paid_n,
            "free_paid": get_free_paid(provider),
            "disable_env": bool(migrated.get("disable_env")),
        }
    return out
