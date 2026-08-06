from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlencode

import httpx

from llmrouter.event_log import events
from llmrouter.registry import ModelConfig, resolve_auth_env


@dataclass
class HealthStatus:
    state: str  # ok | auth_error | down | unknown
    latency_ms: float = 0.0
    message: str = ""
    checked_at: float = 0.0
    http_status: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _auth_headers(model: ModelConfig, api_key: str) -> tuple[str, dict[str, str]]:
    """Return (url, headers) for a non-completion probe."""
    url = model.endpoint
    headers: dict[str, str] = {}

    if model.provider == "gemini":
        qs = urlencode({"key": api_key})
        return f"{url}?{qs}", headers

    if model.provider == "cloudflare":
        account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        url = url.replace("{account_id}", account_id)

    headers["Authorization"] = f"Bearer {api_key}"
    return url, headers


def _classify(status_code: int | None, exc: Exception | None) -> tuple[str, str]:
    if exc is not None:
        if isinstance(exc, httpx.TimeoutException):
            return "down", "timeout"
        return "down", f"transport: {exc}"
    assert status_code is not None
    if status_code in (401, 403):
        return "auth_error", f"HTTP {status_code}"
    # Reachable: 2xx/3xx/4xx (incl. 405 Method Not Allowed on GET)
    if status_code < 500:
        return "ok", f"HTTP {status_code}"
    return "down", f"HTTP {status_code}"


async def probe_model(client: httpx.AsyncClient, model: ModelConfig) -> HealthStatus:
    api_key = resolve_auth_env(model.auth_env_var)
    if not api_key:
        return HealthStatus(
            state="auth_error",
            message=f"missing env {model.auth_env_var}",
            checked_at=time.time(),
        )

    url, headers = _auth_headers(model, api_key)
    start = time.perf_counter()
    status_code: int | None = None
    err: Exception | None = None
    try:
        resp = await client.get(url, headers=headers, timeout=5.0)
        status_code = resp.status_code
    except Exception as exc:  # noqa: BLE001 — classify any transport failure
        err = exc
    latency = (time.perf_counter() - start) * 1000.0
    state, message = _classify(status_code, err)
    return HealthStatus(
        state=state,
        latency_ms=round(latency, 2),
        message=message,
        checked_at=time.time(),
        http_status=status_code,
    )


class HealthCache:
    """Throttle live probes so dashboard polling does not hammer providers."""

    def __init__(self, ttl_s: float = 30.0) -> None:
        self.ttl_s = ttl_s
        self._cache: dict[str, HealthStatus] = {}

    async def statuses(
        self,
        client: httpx.AsyncClient,
        models: list[ModelConfig],
        *,
        force: bool = False,
    ) -> dict[str, dict[str, Any]]:
        now = time.time()
        out: dict[str, dict[str, Any]] = {}
        for model in models:
            cached = self._cache.get(model.name)
            if not force and cached is not None and (now - cached.checked_at) < self.ttl_s:
                out[model.name] = cached.to_dict()
                continue
            status = await probe_model(client, model)
            self._cache[model.name] = status
            events.record(
                f"health {status.state}",
                level="error" if status.state in ("down", "auth_error") else "info",
                model=model.name,
                provider=model.provider,
                health=status.state,
                latency_ms=status.latency_ms,
                message=status.message,
            )
            out[model.name] = status.to_dict()
        return out


health_cache = HealthCache()
