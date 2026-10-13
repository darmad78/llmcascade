"""Learn per-model daily quota by counting successes until the provider 429s.

YAML rpd/rpm are guesses. This store is the observed cap for each model ID.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from pathlib import Path

from llmcascade.persist import JsonState
from llmcascade.secrets import data_dir

PACIFIC = ZoneInfo("America/Los_Angeles")
_LOCK = threading.Lock()
_PRUNE_DAYS = 30


def _path() -> Path:
    return data_dir() / "quota_learned.json"


def _path_state() -> JsonState:
    return JsonState(_path())


def _day_key(provider: str, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if provider == "gemini":
        return now.astimezone(PACIFIC).date().isoformat()
    return now.astimezone(timezone.utc).date().isoformat()


def _prune_old(data: dict[str, Any], now: datetime) -> None:
    cutoff = (now - timedelta(days=_PRUNE_DAYS)).date().isoformat()
    for key in list(data.keys()):
        raw = data.get(key)
        if not isinstance(raw, dict):
            data.pop(key, None)
            continue
        day = str(raw.get("day") or "")
        if day and day < cutoff:
            data.pop(key, None)


def _row(data: dict[str, Any], key: str, provider: str, now: datetime) -> dict[str, Any]:
    day = _day_key(provider, now)
    raw = data.get(key) if isinstance(data.get(key), dict) else {}
    row = dict(raw)
    if row.get("day") != day:
        row["ok"] = 0
        row["rpd_remaining"] = row.get("learned_rpd")
        row["day"] = day
    row["provider"] = provider
    row.setdefault("ok", 0)
    row.setdefault("learned_rpd", None)
    row.setdefault("rpd_remaining", row.get("learned_rpd"))
    data[key] = row
    return row


class QuotaLearner:
    """Process + disk map of observed RPD per provider model id."""

    def __init__(self, state: JsonState | None = None) -> None:
        self._state = state or _path_state()
        with _LOCK:
            with self._state.lock:
                _prune_old(self._state.data, datetime.now(timezone.utc))

    def flush(self) -> None:
        self._state.flush()

    def record_success(
        self,
        model_id: str,
        provider: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        key = (model_id or "").strip()
        if not key:
            return {}
        now = now or datetime.now(timezone.utc)
        with _LOCK:
            with self._state.lock:
                data = self._state.data
                row = _row(data, key, provider, now)
                row["ok"] = int(row.get("ok") or 0) + 1
                learned = row.get("learned_rpd")
                if isinstance(learned, int) and learned >= 0:
                    row["rpd_remaining"] = max(0, learned - row["ok"])
            self._state.mark_dirty()
            return dict(row)

    def record_limit(
        self,
        model_id: str,
        provider: str,
        kind: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        key = (model_id or "").strip()
        if not key:
            return {}
        now = now or datetime.now(timezone.utc)
        with _LOCK:
            with self._state.lock:
                data = self._state.data
                row = _row(data, key, provider, now)
                ok = int(row.get("ok") or 0)
                if kind in ("daily", "credit"):
                    if ok > 0:
                        row["learned_rpd"] = max(int(row.get("learned_rpd") or 0), ok)
                    row["rpd_remaining"] = 0
                    row["learned_kind"] = kind
                row["last_limit"] = kind
            self._state.mark_dirty()
            return dict(row)

    def remaining_rpd(
        self,
        model_id: str,
        *,
        now: datetime | None = None,
    ) -> int | None:
        key = (model_id or "").strip()
        if not key:
            return None
        now = now or datetime.now(timezone.utc)
        with _LOCK:
            with self._state.lock:
                data = self._state.data
                raw = data.get(key)
                if not isinstance(raw, dict) or raw.get("learned_rpd") is None:
                    return None
                row = _row(data, key, str(raw.get("provider") or ""), now)
                rem = row.get("rpd_remaining")
            return None if rem is None else max(0, int(rem))

    def snapshot(self, now: datetime | None = None) -> dict[str, dict[str, Any]]:
        now = now or datetime.now(timezone.utc)
        with _LOCK:
            with self._state.lock:
                data = self._state.data
                out: dict[str, dict[str, Any]] = {}
                for key, raw in list(data.items()):
                    if not isinstance(raw, dict):
                        continue
                    provider = str(raw.get("provider") or "")
                    row = _row(data, key, provider, now)
                    out[key] = {
                        "provider": provider,
                        "day": row.get("day"),
                        "ok": int(row.get("ok") or 0),
                        "learned_rpd": row.get("learned_rpd"),
                        "rpd_remaining": row.get("rpd_remaining"),
                        "last_limit": row.get("last_limit"),
                    }
            return out
