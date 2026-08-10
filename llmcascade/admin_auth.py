"""Admin JWT cookie auth, CSRF, and login lockout."""

from __future__ import annotations

import os
import secrets
import time
from dataclasses import dataclass
from typing import Any

import jwt

from llmcascade.auth_store import AdminUser, ensure_admin, load_admin, verify_password
from llmcascade.secrets import jwt_signing_key, resolve_secret_key

SESSION_COOKIE = "llmcascade_session"
CSRF_COOKIE = "llmcascade_csrf"
JWT_TTL_S = 12 * 60 * 60
LOCKOUT_MAX_FAILURES = 5
LOCKOUT_BASE_S = 30.0


@dataclass
class SessionClaims:
    username: str
    must_change_password: bool
    pwd_version: int


class LoginLockout:
    """Process-local brute-force protection for /login."""

    def __init__(self) -> None:
        self._failures: dict[str, list[float]] = {}
        self._locked_until: dict[str, float] = {}

    def _bucket(self, ip: str, username: str) -> str:
        return f"{ip}|{username.lower()}"

    def is_locked(self, ip: str, username: str) -> tuple[bool, float]:
        key = self._bucket(ip, username)
        until = self._locked_until.get(key, 0.0)
        now = time.monotonic()
        if until > now:
            return True, until - now
        return False, 0.0

    def record_failure(self, ip: str, username: str) -> float:
        key = self._bucket(ip, username)
        now = time.monotonic()
        recent = [t for t in self._failures.get(key, []) if now - t < 900]
        recent.append(now)
        self._failures[key] = recent
        if len(recent) >= LOCKOUT_MAX_FAILURES:
            # Escalating: 30s, 60s, 120s…
            strikes = len(recent) // LOCKOUT_MAX_FAILURES
            delay = LOCKOUT_BASE_S * (2 ** max(0, strikes - 1))
            self._locked_until[key] = now + delay
            return delay
        return 0.0

    def record_success(self, ip: str, username: str) -> None:
        key = self._bucket(ip, username)
        self._failures.pop(key, None)
        self._locked_until.pop(key, None)


login_lockout = LoginLockout()


def cookie_secure(request_https: bool = False) -> bool:
    env = (os.environ.get("LLMCASCADE_COOKIE_SECURE") or "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    return request_https


def create_session_token(user: AdminUser) -> str:
    now = int(time.time())
    payload = {
        "sub": user.username,
        "must_change_password": user.must_change_password,
        "pwd_version": user.pwd_version,
        "iat": now,
        "exp": now + JWT_TTL_S,
    }
    return jwt.encode(payload, jwt_signing_key(resolve_secret_key()), algorithm="HS256")


def decode_session_token(token: str) -> SessionClaims | None:
    try:
        payload = jwt.decode(token, jwt_signing_key(resolve_secret_key()), algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
    username = str(payload.get("sub") or "")
    if not username:
        return None
    return SessionClaims(
        username=username,
        must_change_password=bool(payload.get("must_change_password")),
        pwd_version=int(payload.get("pwd_version") or 0),
    )


def validate_session(token: str | None) -> tuple[AdminUser, SessionClaims] | None:
    if not token:
        return None
    claims = decode_session_token(token)
    if claims is None:
        return None
    ensure_admin()
    user = load_admin()
    if user is None or user.username != claims.username:
        return None
    if claims.pwd_version != user.pwd_version:
        return None
    # Prefer live must_change flag from store
    claims.must_change_password = user.must_change_password
    return user, claims


def new_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def csrf_ok(cookie_token: str | None, submitted: str | None) -> bool:
    if not cookie_token or not submitted:
        return False
    if len(cookie_token) != len(submitted):
        return False
    return secrets.compare_digest(cookie_token, submitted)


def authenticate(username: str, password: str) -> AdminUser | None:
    user = ensure_admin()
    if username != user.username:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def protected_paths() -> tuple[str, ...]:
    return (
        "/dashboard",
        "/stats",
        "/admin",
        "/v1/dashboard",
        "/v1/stats",
        "/v1/events",
        "/v1/errors",
        "/v1/metrics",
    )


def path_requires_auth(path: str) -> bool:
    for prefix in protected_paths():
        if path == prefix or path.startswith(prefix + "/"):
            return True
    return False


def path_allowed_during_password_change(path: str) -> bool:
    allowed = {
        "/login",
        "/logout",
        "/admin/change-password",
        "/favicon.ico",
    }
    return path in allowed or path.startswith("/static/")


def wants_html(accept: str | None) -> bool:
    if not accept:
        return False
    return "text/html" in accept.lower()


def session_cookie_kwargs(*, secure: bool) -> dict[str, Any]:
    return {
        "httponly": True,
        "samesite": "lax",
        "secure": secure,
        "path": "/",
        "max_age": JWT_TTL_S,
    }


def csrf_cookie_kwargs(*, secure: bool) -> dict[str, Any]:
    return {
        "httponly": False,
        "samesite": "lax",
        "secure": secure,
        "path": "/",
        "max_age": JWT_TTL_S,
    }
