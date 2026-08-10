"""Local admin credentials store (bcrypt-hashed)."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bcrypt

from llmrouter.secrets import data_dir

_LOCK = threading.Lock()
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "admin"


@dataclass
class AdminUser:
    username: str
    password_hash: str
    must_change_password: bool
    pwd_version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "username": self.username,
            "password_hash": self.password_hash,
            "must_change_password": self.must_change_password,
            "pwd_version": self.pwd_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdminUser:
        return cls(
            username=str(data["username"]),
            password_hash=str(data["password_hash"]),
            must_change_password=bool(data.get("must_change_password", False)),
            pwd_version=int(data.get("pwd_version", 1)),
        )


def _path() -> Path:
    return data_dir() / "admin.json"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except (ValueError, TypeError):
        return False


def load_admin() -> AdminUser | None:
    path = _path()
    if not path.is_file():
        return None
    with _LOCK:
        data = json.loads(path.read_text())
    if not isinstance(data, dict):
        return None
    return AdminUser.from_dict(data)


def save_admin(user: AdminUser) -> None:
    path = _path()
    with _LOCK:
        path.write_text(json.dumps(user.to_dict(), indent=2) + "\n")
        try:
            path.chmod(0o600)
        except OSError:
            pass


def ensure_admin() -> AdminUser:
    """Create default admin/admin on first boot if missing."""
    existing = load_admin()
    if existing is not None:
        return existing
    user = AdminUser(
        username=DEFAULT_USERNAME,
        password_hash=hash_password(DEFAULT_PASSWORD),
        must_change_password=True,
        pwd_version=1,
    )
    save_admin(user)
    return user


def change_password(user: AdminUser, new_password: str) -> AdminUser:
    updated = AdminUser(
        username=user.username,
        password_hash=hash_password(new_password),
        must_change_password=False,
        pwd_version=user.pwd_version + 1,
    )
    save_admin(updated)
    return updated
