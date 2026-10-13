"""Gemini cascade: failure classification, per-model cooldowns, ordered fallback.

Default when every cascade member is cooling: do NOT wait — raise so the outer
selector can fall through to other free providers. Pass wait_for_gemini=True to
block in ~15s chunks until the earliest finite cooldown ends (optional path).
"""

from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Iterable, Literal
from zoneinfo import ZoneInfo

from llmcascade.persist import JsonState
from llmcascade.secrets import data_dir

from llmcascade.adapters.base import LLMResponse
from llmcascade.exceptions import ProviderError
from llmcascade.model_pool import ModelPool
from llmcascade.registry import ModelConfig

FailureKind = Literal["daily", "credit", "rate", "permanent", "missing", "auth", "transient"]

PACIFIC = ZoneInfo("America/Los_Angeles")
DAILY_RESET_TZ: dict[str, ZoneInfo | timezone] = {"gemini": PACIFIC}
_GO_DUR = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_GO_UNITS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


def _go_duration_s(raw: str) -> float | None:
    parts = _GO_DUR.findall(raw.strip())
    if not parts:
        return None
    return sum(float(n) * _GO_UNITS[u] for n, u in parts)


def _gemini_cooldown_state() -> JsonState:
    return JsonState(data_dir() / "gemini_cooldowns.json")
# Gemini free RPM/TPM is a rolling ~60s window; retry after that, not 10m.
RATE_COOLDOWN = timedelta(seconds=60)
CREDIT_COOLDOWN = timedelta(hours=24)
PERMANENT_COOLDOWN = timedelta(days=365)
# Bare 404s (e.g. OpenRouter "No endpoints found") are often temporary.
MISSING_COOLDOWN = timedelta(hours=1)
GONE_SIGNALS: tuple[str, ...] = (
    "does not exist",
    "is not found",
    "model not found",
    "model_not_found",
    "decommissioned",
    "end of life",
    "end-of-life",
    "no longer available",
    "no longer supported",
)
AUTH_COOLDOWN = timedelta(seconds=30)
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


def is_gone_message(body: str) -> bool:
    text = (body or "").lower()
    return any(sig in text for sig in GONE_SIGNALS)


def classify_failure(status_code: int | None, body: str = "") -> FailureKind:
    text = (body or "").lower()
    if status_code == 402:
        return "credit"
    if status_code == 429 and (
        "insufficient balance" in text
        or "out of credits" in text
        or "credit_limit" in text
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
        or "exceeded your current quota" in text
        or "check your plan and billing" in text
    ):
        return "daily"
    if status_code == 429:
        return "rate"
    if status_code is not None and status_code >= 500:
        return "transient"
    if status_code == 410 or "not supported" in text or is_gone_message(text):
        return "permanent"
    if status_code == 404:
        return "missing"
    if status_code == 401:
        return "auth"
    if status_code == 403:
        if any(
            token in text
            for token in (
                "unauthorized",
                "invalid api",
                "invalid key",
                "api key",
                "authentication",
                "permission denied",
                "access denied",
                "does not have access",
                "free tier",
                "free plan",
                "free models",
                "not available on the free",
                "forbidden",
            )
        ):
            return "auth"
        return "auth"
    if status_code is None:
        return "transient"
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

    for key in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        raw = lower.get(key)
        if not raw:
            continue
        secs = _go_duration_s(str(raw))
        if secs is not None:
            return now + timedelta(seconds=max(0.0, secs))
    return None


def cooldown_until(
    kind: FailureKind,
    *,
    now: datetime | None = None,
    headers: dict[str, str] | None = None,
    provider: str = "",
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
    if kind == "auth":
        if learned is not None and learned > now:
            return learned
        return now + AUTH_COOLDOWN
    if kind == "permanent":
        return now + PERMANENT_COOLDOWN
    if kind == "missing":
        return now + MISSING_COOLDOWN
    # daily → Pacific midnight. Ignore short Retry-After (Google often sends 60s on quota 429).
    if learned is not None and learned - now >= timedelta(minutes=30):
        return learned
    tz = DAILY_RESET_TZ.get(provider, timezone.utc)
    local = now.astimezone(tz)
    next_midnight = (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return next_midnight.astimezone(timezone.utc)


def needs_thinking_budget_zero(model_id: str) -> bool:
    mid = model_id.lower()
    return "2.5" in mid or mid.startswith("gemini-3")


def resolve_cascade_order(
    configured: list[str] | None = None,
    *,
    preferred_env: str | None = "GEMINI_MODEL",
) -> list[str]:
    """Ordered cascade; optional env var becomes preferred head when set."""
    base = list(configured) if configured else list(DEFAULT_GEMINI_CASCADE)
    preferred = (os.environ.get(preferred_env) or "").strip() if preferred_env else ""
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


def effective_cascade_members(
    configured: list[str] | None = None,
    *,
    preferred_env: str | None = "GEMINI_MODEL",
) -> list[str]:
    """Cascade IDs after admin Hide + weight reorder (higher weight tried first)."""
    base = resolve_cascade_order(configured, preferred_env=preferred_env)
    try:
        from llmcascade.model_store import load_overrides

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

    def __init__(
        self,
        models: list[str] | None = None,
        *,
        preferred_env: str | None = "GEMINI_MODEL",
        state: JsonState | None = None,
    ) -> None:
        self.models = resolve_cascade_order(models, preferred_env=preferred_env)
        self._cooldowns: dict[str, datetime] = {}
        self._kinds: dict[str, FailureKind] = {}
        self._lock = asyncio.Lock()
        self._state = state or _gemini_cooldown_state()
        self._on_limit: Callable[[str, FailureKind], None] | None = None
        self._load_persisted_cooldowns()

    def set_on_limit(self, cb: Callable[[str, FailureKind], None] | None) -> None:
        self._on_limit = cb

    def adopt_state(self, other: "GeminiCascadeManager") -> None:
        keep = set(self.models)
        self._cooldowns = {m: t for m, t in other._cooldowns.items() if m in keep}
        self._kinds = {m: k for m, k in other._kinds.items() if m in keep}

    def flush_state(self) -> None:
        self._persist_cooldowns()

    def _load_persisted_cooldowns(self) -> None:
        now = datetime.now(timezone.utc)
        with self._state.lock:
            raw = self._state.data.get("cooldowns") if isinstance(self._state.data.get("cooldowns"), dict) else {}
            kinds = self._state.data.get("kinds") if isinstance(self._state.data.get("kinds"), dict) else {}
        for mid, iso in raw.items():
            if mid not in self.models:
                continue
            try:
                until = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
            except ValueError:
                continue
            if until.tzinfo is None:
                until = until.replace(tzinfo=timezone.utc)
            if until > now:
                self._cooldowns[mid] = until
                k = kinds.get(mid)
                if k in ("daily", "credit", "rate", "permanent", "missing", "auth", "transient"):
                    self._kinds[mid] = k  # type: ignore[assignment]

    def _persist_cooldowns(self) -> None:
        with self._state.lock:
            self._state.data["cooldowns"] = {
                m: t.isoformat() for m, t in self._cooldowns.items()
            }
            self._state.data["kinds"] = dict(self._kinds)
        self._state.mark_dirty()

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
                self._kinds.pop(model_id, None)
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
        until = cooldown_until(kind, now=now, headers=headers, provider="gemini")
        if until is None:
            return
        targets = [model_id]
        async with self._lock:
            for mid in targets:
                prev = self._cooldowns.get(mid)
                if prev is None or until > prev:
                    self._cooldowns[mid] = until
                    self._kinds[mid] = kind
            self._persist_cooldowns()
        if self._on_limit is not None:
            self._on_limit(model_id, kind)

    async def status(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        available: list[str] = []
        cooling: dict[str, str] = {}
        remaining_s: dict[str, int] = {}
        async with self._lock:
            expired = [m for m, t in self._cooldowns.items() if t <= now]
            for m in expired:
                self._cooldowns.pop(m, None)
                self._kinds.pop(m, None)
            kinds: dict[str, str] = {}
            for m in self.models:
                until = self._cooldowns.get(m)
                if until is None:
                    available.append(m)
                else:
                    cooling[m] = until.isoformat()
                    remaining_s[m] = max(0, int((until - now).total_seconds()))
                    if self._kinds.get(m):
                        kinds[m] = self._kinds[m]
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
            "cooldown_kinds": kinds,
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
            last_kind: FailureKind | None = None
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
                    last_kind = kind
                    await self.apply_cooldown(
                        model_id, kind, body=body, headers=getattr(exc, "headers", None)
                    )
                    continue

            if tried == 0 and wait_for_gemini:
                earliest = await self.earliest_available_at()
                if earliest is None:
                    continue
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
                kind=last_kind,
            )

    async def run_stream(
        self,
        open_stream: Callable[[str, str], Any],
        prompt: str,
        *,
        wait_for_gemini: bool = False,
    ) -> tuple[str, Any, Any]:
        """Return (model_id, first_chunk, async_iter) after priming first chunk."""

        while True:
            last_err: ProviderError | None = None
            last_kind: FailureKind | None = None
            tried = 0
            for model_id in self.models:
                if await self.is_cooling(model_id):
                    continue
                tried += 1
                try:
                    gen = open_stream(model_id, prompt)
                    if asyncio.iscoroutine(gen):
                        gen = await gen
                    first = await gen.__anext__()
                    return model_id, first, gen
                except StopAsyncIteration:
                    raise ProviderError(
                        "gemini empty stream",
                        retryable=True,
                        provider="gemini",
                        model=model_id,
                    )
                except ProviderError as exc:
                    last_err = exc
                    kind = classify_failure(exc.status_code, str(exc))
                    last_kind = kind
                    await self.apply_cooldown(
                        model_id, kind, body=str(exc), headers=getattr(exc, "headers", None)
                    )
                    continue

            if tried == 0 and wait_for_gemini:
                earliest = await self.earliest_available_at()
                if earliest is None:
                    continue
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
                kind=last_kind,
            )


class ModelCooldownTracker:
    """Unavailable pool + cooldowns (persisted). Dispatch only uses available models."""

    def __init__(self, pool: ModelPool | None = None) -> None:
        self.pool = pool or ModelPool()

    async def available_at(self, model_name: str) -> datetime | None:
        return self.pool.available_at(model_name)

    async def is_cooling(self, model_name: str) -> bool:
        return self.pool.is_unavailable(model_name)

    def due_for_health(self, *, now: datetime | None = None) -> list[str]:
        return self.pool.due_for_health(now=now)

    def apply_health(
        self,
        model_name: str,
        *,
        state: str,
        http_status: int | None = None,
        message: str = "",
        now: datetime | None = None,
    ) -> str:
        return self.pool.apply_health(
            model_name,
            state=state,
            http_status=http_status,
            message=message,
            now=now,
        )

    def sync_registry(self, names: Iterable[str]) -> None:
        self.pool.sync_registry(names)

    def snapshot_pools(self, names: Iterable[str]) -> dict[str, Any]:
        return self.pool.snapshot(names)

    async def apply_from_error(
        self,
        model_name: str,
        *,
        status_code: int | None,
        body: str = "",
        headers: dict[str, str] | None = None,
        now: datetime | None = None,
        failover_probe: bool = False,
        provider: str = "",
        kind: FailureKind | None = None,
    ) -> FailureKind | None:
        classified = kind or classify_failure(status_code, body)
        pool_kind: FailureKind = classified
        if classified == "credit":
            pool_kind = "daily"
        if failover_probe and classified == "daily":
            pool_kind = "rate"
        until = cooldown_until(pool_kind, now=now, headers=headers, provider=provider)
        if until is None:
            return None
        self.pool.mark_unavailable(model_name, pool_kind, until)
        return pool_kind

    async def status(self) -> dict[str, Any]:
        return self.pool.cooldown_status()


def cascade_manager_from_registry(
    registry: list[ModelConfig],
    *,
    capability: str = "chat",
) -> GeminiCascadeManager | None:
    preferred_env = "GEMINI_EMBED_MODEL" if capability == "embed" else "GEMINI_MODEL"
    for m in registry:
        if m.provider != "gemini":
            continue
        if capability not in (m.capabilities or []):
            continue
        ids = list(m.cascade) if m.cascade else [m.name]
        members = effective_cascade_members(ids, preferred_env=preferred_env)
        if not members:
            continue
        mgr = GeminiCascadeManager(members, preferred_env=None)
        mgr.bind_logical(m.name)
        return mgr
    return None
