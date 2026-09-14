"""Learn per-model daily quota by counting successes until the provider 429s.

YAML rpd/rpm are guesses. This store is the observed cap for each model ID.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from llmcascade.secrets import data_dir

PACIFIC = ZoneInfo("America/Los_Angeles")
_LOCK = threading.Lock()


def _path() -> Path:
    return data_dir() / "quota_learned.json"


def _day_key(provider: str, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if provider == "gemini":
        return now.astimezone(PACIFIC).date().isoformat()
    return now.astimezone(timezone.utc).date().isoformat()


def _load() -> dict[str, Any]:
    path = _path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save(data: dict[str, Any]) -> None:
    path = _path()
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


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
            data = _load()
            row = _row(data, key, provider, now)
            row["ok"] = int(row.get("ok") or 0) + 1
            learned = row.get("learned_rpd")
            if isinstance(learned, int) and learned >= 0:
                row["rpd_remaining"] = max(0, learned - row["ok"])
            _save(data)
            return dict(row)

    def record_limit(
        self,
        model_id: str,
        provider: str,
        kind: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Call when the provider refuses (daily/credit 429). Cap = successes today."""
        key = (model_id or "").strip()
        if not key:
            return {}
        now = now or datetime.now(timezone.utc)
        with _LOCK:
            data = _load()
            row = _row(data, key, provider, now)
            ok = int(row.get("ok") or 0)
            if kind in ("daily", "credit"):
                row["learned_rpd"] = max(ok, int(row.get("learned_rpd") or 0) or ok)
                row["rpd_remaining"] = 0
                row["learned_kind"] = kind
            row["last_limit"] = kind
            _save(data)
            return dict(row)

    def remaining_rpd(
        self,
        model_id: str,
        *,
        now: datetime | None = None,
    ) -> int | None:
        """Observed remaining daily calls, or None if this ID has no learned cap."""
        key = (model_id or "").strip()
        if not key:
            return None
        now = now or datetime.now(timezone.utc)
        with _LOCK:
            data = _load()
            raw = data.get(key)
            if not isinstance(raw, dict) or raw.get("learned_rpd") is None:
                return None
            row = _row(data, key, str(raw.get("provider") or ""), now)
            rem = row.get("rpd_remaining")
            if rem is None:
                return None
            return max(0, int(rem))

    def snapshot(self, now: datetime | None = None) -> dict[str, dict[str, Any]]:
        now = now or datetime.now(timezone.utc)
        with _LOCK:
            data = _load()
            out: dict[str, dict[str, Any]] = {}
            for key, raw in data.items():
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
