"""Gemini cascade: failure classification, per-model cooldowns, ordered fallback.

Default when every cascade member is cooling: do NOT wait — raise so the outer
selector can fall through to other free providers. Pass wait_for_gemini=True to
block in ~15s chunks until the earliest finite cooldown ends (optional path).
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Literal
from zoneinfo import ZoneInfo

from llmrouter.adapters.base import LLMResponse
from llmrouter.exceptions import ProviderError
from llmrouter.registry import ModelConfig

FailureKind = Literal["daily", "credit", "rate", "permanent", "transient"]

PACIFIC = ZoneInfo("America/Los_Angeles")
# Gemini free RPM/TPM is a rolling ~60s window; retry after that, not 10m.
RATE_COOLDOWN = timedelta(seconds=60)
CREDIT_COOLDOWN = timedelta(hours=24)
PERMANENT_COOLDOWN = timedelta(days=365)
WAIT_CHUNK_S = 15.0
MIN_TEXT_LEN = 1

# gemini-2.0-* shut down Jun 2026. Unknown IDs 404 → permanent cooldown.
DEFAULT_GEMINI_CASCADE: tuple[str, ...] = (
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
)

SendFn = Callable[[str, str], Awaitable[LLMResponse]]  # (model_id, prompt) -> response


def classify_failure(status_code: int | None, body: str = "") -> FailureKind:
    text = (body or "").lower()
    if (
        status_code == 402
        or "insufficient balance" in text
        or "credit limit" in text
        or "credit_limit" in text
        or "out of credits" in text
    ):
        return "credit"
    if (
        "perday" in text
        or "per day" in text
        or "per-day" in text
        or "free-models-per-day" in text
        or ("daily" in text and ("quota" in text or "limit" in text))
        or "limit: 0" in text
        or "limit:0" in text
    ):
        return "daily"
    if status_code == 429:
        return "rate"
    if status_code == 404 or "not supported" in text or "is not found" in text:
        return "permanent"
    if status_code is None or (status_code is not None and status_code >= 500):
        return "rate"
    return "transient"


def parse_reset_at(
    headers: dict[str, str] | None = None,
    *,
    now: datetime | None = None,
) -> datetime | None:
    """Learn cooldown end from Retry-After / X-RateLimit-Reset when present."""
    if not headers:
        return None
    now = now or datetime.now(timezone.utc)
    lower = {k.lower(): v for k, v in headers.items()}

    retry_after = lower.get("retry-after")
    if retry_after:
        try:
            return now + timedelta(seconds=max(0.0, float(retry_after)))
        except ValueError:
            try:
                # HTTP-date
                from email.utils import parsedate_to_datetime

                dt = parsedate_to_datetime(retry_after)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except (TypeError, ValueError, IndexError):
                pass

    for key in ("x-ratelimit-reset", "ratelimit-reset"):
        raw = lower.get(key)
        if not raw:
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        # ms epoch vs seconds epoch vs delta-seconds
        if val > 1e12:  # epoch ms
            return datetime.fromtimestamp(val / 1000.0, tz=timezone.utc)
        if val > 1e9:  # epoch seconds
            return datetime.fromtimestamp(val, tz=timezone.utc)
        return now + timedelta(seconds=max(0.0, val))
    return None


def cooldown_until(
    kind: FailureKind,
    *,
    now: datetime | None = None,
    headers: dict[str, str] | None = None,
) -> datetime | None:
    """Return UTC available_at, or None for transient (no cooldown)."""
    now = now or datetime.now(timezone.utc)
    if kind == "transient":
        return None
    learned = parse_reset_at(headers, now=now)
    if kind == "rate":
        if learned is not None and learned > now:
            return learned
        return now + RATE_COOLDOWN
    if kind == "credit":
        return now + CREDIT_COOLDOWN
    if kind == "permanent":
        return now + PERMANENT_COOLDOWN
    # daily → prefer provider reset header, else next America/Los_Angeles midnight
    if learned is not None and learned > now:
        return learned
    local = now.astimezone(PACIFIC)
    next_midnight = (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return next_midnight.astimezone(timezone.utc)


def needs_thinking_budget_zero(model_id: str) -> bool:
    mid = model_id.lower()
    return "2.5" in mid or mid.startswith("gemini-3")


def resolve_cascade_order(configured: list[str] | None = None) -> list[str]:
    """Ordered cascade; GEMINI_MODEL env becomes preferred head when set."""
    base = list(configured) if configured else list(DEFAULT_GEMINI_CASCADE)
    preferred = (os.environ.get("GEMINI_MODEL") or "").strip()
    if preferred:
        base = [m for m in base if m != preferred]
        base.insert(0, preferred)
    # de-dupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for m in base:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def effective_cascade_members(configured: list[str] | None = None) -> list[str]:
    """Cascade IDs after admin Hide + weight reorder (higher weight tried first)."""
    base = resolve_cascade_order(configured)
    try:
        from llmrouter.model_store import load_overrides

        overrides = load_overrides()
    except Exception:  # noqa: BLE001
        overrides = {}
    scored: list[tuple[int, int, str]] = []
    for i, mid in enumerate(base):
        ov = overrides.get(mid) or {}
        if ov.get("enabled") is False:
            continue
        try:
            weight = max(1, int(ov.get("weight", 1)))
        except (TypeError, ValueError):
            weight = 1
        scored.append((-weight, i, mid))
    scored.sort()
    return [mid for _, _, mid in scored]


def gemini_endpoint(base_endpoint: str, model_id: str) -> str:
    if "{model}" in base_endpoint:
        return base_endpoint.replace("{model}", model_id)
    # legacy full URL ending with models/<name>:generateContent
    marker = "/models/"
    if marker in base_endpoint and ":generateContent" in base_endpoint:
        prefix, _, rest = base_endpoint.partition(marker)
        return f"{prefix}{marker}{model_id}:generateContent"
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"


class GeminiCascadeManager:
    """Process-local per-model cooldown map for the Gemini cascade family."""

    def __init__(self, models: list[str] | None = None) -> None:
        self.models = resolve_cascade_order(models)
        self._cooldowns: dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    def bind_logical(self, logical_name: str) -> None:
        self.logical_name = logical_name

    @property
    def logical_name(self) -> str:
        return getattr(self, "_logical_name", "gemini")

    @logical_name.setter
    def logical_name(self, value: str) -> None:
        self._logical_name = value

    async def available_at(self, model_id: str) -> datetime | None:
        async with self._lock:
            until = self._cooldowns.get(model_id)
            if until is None:
                return None
            now = datetime.now(timezone.utc)
            if until <= now:
                self._cooldowns.pop(model_id, None)
                return None
            return until

    async def is_cooling(self, model_id: str) -> bool:
        return (await self.available_at(model_id)) is not None

    async def any_available(self) -> bool:
        for m in self.models:
            if not await self.is_cooling(m):
                return True
        return False

    async def earliest_available_at(self) -> datetime | None:
        times: list[datetime] = []
        for m in self.models:
            until = await self.available_at(m)
            if until is not None:
                times.append(until)
            else:
                return None  # at least one ready now
        return min(times) if times else None

    async def apply_cooldown(
        self,
        model_id: str,
        kind: FailureKind,
        *,
        body: str = "",
        now: datetime | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        # Quotas are per model ID — never fan out to siblings. `body` kept for call-site compat.
        until = cooldown_until(kind, now=now, headers=headers)
        if until is None:
            return
        async with self._lock:
            prev = self._cooldowns.get(model_id)
            if prev is None or until > prev:
                self._cooldowns[model_id] = until

    async def status(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        available: list[str] = []
        cooling: dict[str, str] = {}
        remaining_s: dict[str, int] = {}
        async with self._lock:
            expired = [m for m, t in self._cooldowns.items() if t <= now]
            for m in expired:
                self._cooldowns.pop(m, None)
            for m in self.models:
                until = self._cooldowns.get(m)
                if until is None:
                    available.append(m)
                else:
                    cooling[m] = until.isoformat()
                    remaining_s[m] = max(0, int((until - now).total_seconds()))
        next_ready_in_s: int | None
        if available:
            next_ready_in_s = 0
        elif remaining_s:
            next_ready_in_s = min(remaining_s.values())
        else:
            next_ready_in_s = None
        return {
            "logical": self.logical_name,
            "models": list(self.models),
            "available": available,
            "available_at": cooling,
            "cooldown_remaining_s": remaining_s,
            "next_ready_in_s": next_ready_in_s,
            "family_ready": bool(available),
        }

    async def run(
        self,
        send: SendFn,
        prompt: str,
        *,
        wait_for_gemini: bool = False,
    ) -> LLMResponse:
        """Try cascade in order. Default: if all cooling, raise (outer fallthrough)."""
        while True:
            last_err: ProviderError | None = None
            tried = 0
            for model_id in self.models:
                if await self.is_cooling(model_id):
                    continue
                tried += 1
                try:
                    return await send(model_id, prompt)
                except ProviderError as exc:
                    last_err = exc
                    body = str(exc)
                    kind = classify_failure(exc.status_code, body)
                    await self.apply_cooldown(
                        model_id, kind, body=body, headers=getattr(exc, "headers", None)
                    )
                    continue

            if tried == 0 and wait_for_gemini:
                earliest = await self.earliest_available_at()
                if earliest is not None:
                    now = datetime.now(timezone.utc)
                    remaining = (earliest - now).total_seconds()
                    if remaining > 0:
                        await asyncio.sleep(min(WAIT_CHUNK_S, remaining))
                        continue

            msg = "gemini cascade exhausted"
            if last_err:
                msg = f"{msg}: {last_err.safe_message}"
            raise ProviderError(
                msg,
                status_code=last_err.status_code if last_err else None,
                retryable=False,
                provider="gemini",
                model=self.logical_name,
                headers=getattr(last_err, "headers", None) if last_err else None,
            )


class ModelCooldownTracker:
    """Process-local per registry-model cooldowns (keyed by model name)."""

    def __init__(self) -> None:
        self._cooldowns: dict[str, datetime] = {}
        self._kinds: dict[str, FailureKind] = {}
        self._lock = asyncio.Lock()

    async def available_at(self, model_name: str) -> datetime | None:
        async with self._lock:
            until = self._cooldowns.get(model_name)
            if until is None:
                return None
            now = datetime.now(timezone.utc)
            if until <= now:
                self._cooldowns.pop(model_name, None)
                self._kinds.pop(model_name, None)
                return None
            return until

    async def is_cooling(self, model_name: str) -> bool:
        return (await self.available_at(model_name)) is not None

    async def apply_from_error(
        self,
        model_name: str,
        *,
        status_code: int | None,
        body: str = "",
        headers: dict[str, str] | None = None,
        now: datetime | None = None,
    ) -> FailureKind | None:
        kind = classify_failure(status_code, body)
        until = cooldown_until(kind, now=now, headers=headers)
        if until is None:
            return None
        async with self._lock:
            prev = self._cooldowns.get(model_name)
            if prev is None or until > prev:
                self._cooldowns[model_name] = until
                self._kinds[model_name] = kind
        return kind

    async def status(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        cooling: dict[str, dict[str, Any]] = {}
        async with self._lock:
            expired = [m for m, t in self._cooldowns.items() if t <= now]
            for m in expired:
                self._cooldowns.pop(m, None)
                self._kinds.pop(m, None)
            for name, until in self._cooldowns.items():
                cooling[name] = {
                    "kind": self._kinds.get(name),
                    "available_at": until.isoformat(),
                    "remaining_s": max(0, int((until - now).total_seconds())),
                }
        return cooling


def cascade_manager_from_registry(registry: list[ModelConfig]) -> GeminiCascadeManager | None:
    for m in registry:
        if m.provider == "gemini" and m.cascade:
            members = effective_cascade_members(m.cascade)
            if not members:
                continue
            mgr = GeminiCascadeManager(members)
            mgr.bind_logical(m.name)
            return mgr
        if m.provider == "gemini" and not m.cascade:
            members = effective_cascade_members([m.name])
            if not members:
                continue
            mgr = GeminiCascadeManager(members)
            mgr.bind_logical(m.name)
            return mgr
    return None
