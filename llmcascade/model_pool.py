"""Available vs unavailable model pools.

Unavailable models are not dispatched. Rate/daily/credit/auth return only after
the cooldown elapses *and* a health probe succeeds. Permanent (404/410) stays
out; a replacement ID is a new name and starts available.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from llmcascade.secrets import data_dir

RATE_RETRY = timedelta(seconds=60)
AUTH_RETRY = timedelta(seconds=30)
GONE_RETRY = timedelta(days=365)
GONE_KINDS = frozenset({"permanent"})
HEALTH_KINDS = frozenset({"rate", "daily", "credit", "auth"})
HealthAction = Literal["promoted", "held", "gone"]


def _path() -> Path:
    return data_dir() / "model_pools.json"


def _now(now: datetime | None = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _parse_dt(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class ModelPool:
    """Persisted unavailable set; everything else in the registry is available."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _path()
        self._lock = threading.Lock()
        self._unavail: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.is_file():
            self._unavail = {}
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._unavail = {}
            return
        raw = data.get("unavailable") if isinstance(data, dict) else None
        out: dict[str, dict[str, Any]] = {}
        if isinstance(raw, dict):
            for name, row in raw.items():
                if isinstance(name, str) and isinstance(row, dict) and row.get("kind"):
                    out[name] = dict(row)
        self._unavail = out

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"unavailable": self._unavail}
        self._path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        try:
            self._path.chmod(0o600)
        except OSError:
            pass

    def sync_registry(self, names: Iterable[str]) -> None:
        keep = {n for n in names if n}
        with self._lock:
            dropped = [k for k in self._unavail if k not in keep]
            if not dropped:
                return
            for k in dropped:
                self._unavail.pop(k, None)
            self._save()

    def mark_unavailable(
        self,
        name: str,
        kind: str,
        until: datetime,
    ) -> None:
        key = (name or "").strip()
        if not key or not kind:
            return
        until = _now(until)
        with self._lock:
            prev = self._unavail.get(key) or {}
            prev_until = _parse_dt(prev.get("available_at"))
            if prev_until is not None and prev_until > until and prev.get("kind") == "permanent":
                return
            if prev.get("kind") == "permanent" and kind != "permanent":
                return
            self._unavail[key] = {
                "kind": kind,
                "available_at": until.isoformat(),
            }
            self._save()

    def mark_available(self, name: str) -> None:
        key = (name or "").strip()
        if not key:
            return
        with self._lock:
            if key not in self._unavail:
                return
            self._unavail.pop(key, None)
            self._save()

    def is_unavailable(self, name: str) -> bool:
        key = (name or "").strip()
        with self._lock:
            return key in self._unavail

    def available_at(self, name: str) -> datetime | None:
        key = (name or "").strip()
        with self._lock:
            row = self._unavail.get(key)
            if not row:
                return None
            return _parse_dt(row.get("available_at"))

    def kind(self, name: str) -> str | None:
        key = (name or "").strip()
        with self._lock:
            row = self._unavail.get(key)
            if not row:
                return None
            return str(row.get("kind") or "") or None

    def waiting_health(self, name: str, *, now: datetime | None = None) -> bool:
        now = _now(now)
        key = (name or "").strip()
        with self._lock:
            row = self._unavail.get(key)
            if not row:
                return False
            if str(row.get("kind") or "") in GONE_KINDS:
                return False
            until = _parse_dt(row.get("available_at"))
            return until is not None and until <= now

    def due_for_health(self, *, now: datetime | None = None) -> list[str]:
        now = _now(now)
        due: list[str] = []
        with self._lock:
            for name, row in self._unavail.items():
                kind = str(row.get("kind") or "")
                if kind in GONE_KINDS or kind not in HEALTH_KINDS:
                    continue
                until = _parse_dt(row.get("available_at"))
                if until is None or until <= now:
                    due.append(name)
        return due

    def apply_health(
        self,
        name: str,
        *,
        state: str,
        http_status: int | None = None,
        message: str = "",
        now: datetime | None = None,
    ) -> HealthAction:
        now = _now(now)
        key = (name or "").strip()
        text = (message or "").lower()
        with self._lock:
            row = self._unavail.get(key)
            if not row:
                return "held"
            kind = str(row.get("kind") or "")
            if kind in GONE_KINDS:
                return "gone"
            until = _parse_dt(row.get("available_at"))
            if until is not None and until > now:
                return "held"

            if http_status in (404, 410) or "does not exist" in text or "is not found" in text:
                self._unavail[key] = {
                    "kind": "permanent",
                    "available_at": (now + GONE_RETRY).isoformat(),
                }
                self._save()
                return "gone"
            if http_status in (401, 403) or state == "auth_error":
                self._unavail[key] = {
                    "kind": "auth",
                    "available_at": (now + AUTH_RETRY).isoformat(),
                }
                self._save()
                return "held"
            if http_status == 429 or state == "warn":
                self._unavail[key] = {
                    "kind": "rate",
                    "available_at": (now + RATE_RETRY).isoformat(),
                }
                self._save()
                return "held"
            if state in ("down", "unknown"):
                self._unavail[key] = {
                    "kind": kind or "rate",
                    "available_at": (now + RATE_RETRY).isoformat(),
                }
                self._save()
                return "held"

            self._unavail.pop(key, None)
            self._save()
            return "promoted"

    def snapshot(self, registry_names: Iterable[str], *, now: datetime | None = None) -> dict[str, Any]:
        now = _now(now)
        names = [n for n in registry_names if n]
        with self._lock:
            unavailable: list[dict[str, Any]] = []
            un_set = set()
            for name in names:
                row = self._unavail.get(name)
                if not row:
                    continue
                un_set.add(name)
                until = _parse_dt(row.get("available_at"))
                kind = str(row.get("kind") or "cooldown")
                waiting = kind not in GONE_KINDS and until is not None and until <= now
                unavailable.append(
                    {
                        "name": name,
                        "kind": kind,
                        "available_at": until.isoformat() if until else None,
                        "remaining_s": max(0, int((until - now).total_seconds())) if until else 0,
                        "waiting_health": waiting,
                    }
                )
            available = [n for n in names if n not in un_set]
        return {"available": available, "unavailable": unavailable}

    def cooldown_status(self, *, now: datetime | None = None) -> dict[str, dict[str, Any]]:
        now = _now(now)
        out: dict[str, dict[str, Any]] = {}
        with self._lock:
            for name, row in self._unavail.items():
                until = _parse_dt(row.get("available_at"))
                kind = str(row.get("kind") or "cooldown")
                waiting = kind not in GONE_KINDS and until is not None and until <= now
                out[name] = {
                    "kind": kind,
                    "available_at": until.isoformat() if until else None,
                    "remaining_s": max(0, int((until - now).total_seconds())) if until else 0,
                    "waiting_health": waiting,
                }
        return out
