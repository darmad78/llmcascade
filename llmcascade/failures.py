"""Classify provider failures for Mongo stats. Never store raw bodies for known kinds."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal

from llmcascade.cascade import classify_failure
from llmcascade.exceptions import ProviderError, safe_error_message

RecordedKind = Literal["daily", "credit", "rate", "auth", "timeout", "permanent", "unknown"]
RECORDED_KINDS: tuple[RecordedKind, ...] = (
    "daily",
    "credit",
    "rate",
    "auth",
    "timeout",
    "permanent",
    "unknown",
)

_MAX_UNKNOWN = 2000
_SECRET_RES = (
    re.compile(r"(?i)(bearer\s+)\S+"),
    re.compile(r"(?i)((?:api[_-]?key|token|secret|password)[\"'\s:=]+)[^\s\"'&,]+"),
    re.compile(r"(?i)([?&]key=)[^&\s]+"),
    re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
)


def redact_failure_text(text: str) -> str:
    out = text or ""
    for pat in _SECRET_RES:
        out = pat.sub(r"\1[redacted]" if pat.groups else "[redacted]", out)
    return out[:_MAX_UNKNOWN]


def classify_recorded_failure(status_code: int | None, body: str = "") -> RecordedKind:
    """Map a provider error onto the stats taxonomy. Leftovers are `unknown`."""
    text = (body or "").lower()
    kind = classify_failure(status_code, body)
    if kind in ("daily", "credit", "permanent", "auth"):
        return kind
    if kind == "rate":
        if status_code is None or "timeout" in text or "timed out" in text:
            return "timeout"
        if status_code is not None and status_code >= 500:
            return "unknown"
        return "rate"
    if status_code in (401, 403):
        return "auth"
    if status_code == 408 or "timeout" in text or "timed out" in text:
        return "timeout"
    return "unknown"


def build_failure_record(
    *,
    model: str,
    provider: str,
    capability: str,
    notes: str | None,
    exc: BaseException,
    at: datetime | None = None,
) -> dict[str, Any]:
    status = getattr(exc, "status_code", None)
    body = str(exc)
    kind = classify_recorded_failure(status if isinstance(status, int) else None, body)
    safe = safe_error_message(exc)
    doc: dict[str, Any] = {
        "ts": at or datetime.now(timezone.utc),
        "model": model,
        "provider": provider,
        "capability": (capability or "chat").strip() or "chat",
        "notes": (notes or "").strip() or None,
        "kind": kind,
        "status_code": status if isinstance(status, int) else None,
        "message": safe,
    }
    if kind == "unknown":
        doc["detail"] = redact_failure_text(body)
    else:
        doc["detail"] = None
    if isinstance(exc, ProviderError):
        doc["retryable"] = bool(exc.retryable)
    return doc


def summarize_failures(docs: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, int] = {k: 0 for k in RECORDED_KINDS}
    by_model: dict[str, dict[str, Any]] = {}
    by_provider: dict[str, dict[str, Any]] = {}
    daily: dict[str, dict[str, int]] = {}
    unknowns: list[dict[str, Any]] = []

    def _bump(bucket: dict[str, dict[str, Any]], name: str, kind: str) -> None:
        row = bucket.setdefault(name, {"count": 0, "by_kind": {k: 0 for k in RECORDED_KINDS}})
        row["count"] += 1
        row["by_kind"][kind] = int(row["by_kind"].get(kind) or 0) + 1

    for doc in docs:
        kind = str(doc.get("kind") or "unknown")
        if kind not in by_kind:
            kind = "unknown"
        by_kind[kind] += 1
        model = str(doc.get("model") or "unknown")
        provider = str(doc.get("provider") or "unknown")
        _bump(by_model, model, kind)
        by_model[model]["provider"] = provider
        _bump(by_provider, provider, kind)
        ts = doc.get("ts")
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            day = ts.astimezone(timezone.utc).date().isoformat()
        else:
            day = str(ts)[:10] if ts else "unknown"
        day_row = daily.setdefault(day, {k: 0 for k in RECORDED_KINDS})
        day_row[kind] = int(day_row.get(kind) or 0) + 1
        day_row["total"] = int(day_row.get("total") or 0) + 1
        if kind == "unknown":
            unknowns.append(
                {
                    "ts": ts.isoformat().replace("+00:00", "Z") if isinstance(ts, datetime) else str(ts or ""),
                    "model": model,
                    "provider": provider,
                    "status_code": doc.get("status_code"),
                    "message": doc.get("message"),
                    "detail": doc.get("detail"),
                    "notes": doc.get("notes"),
                    "capability": doc.get("capability") or "chat",
                }
            )

    unknowns.sort(key=lambda r: r.get("ts") or "", reverse=True)
    models = [
        {"name": name, "provider": row.get("provider") or "", "count": row["count"], "by_kind": row["by_kind"]}
        for name, row in by_model.items()
    ]
    models.sort(key=lambda r: r["count"], reverse=True)
    providers = [
        {"name": name, "count": row["count"], "by_kind": row["by_kind"]}
        for name, row in by_provider.items()
    ]
    providers.sort(key=lambda r: r["count"], reverse=True)
    series = [{"bucket": day, **counts} for day, counts in sorted(daily.items())]
    return {
        "range": "30d",
        "total": sum(by_kind.values()),
        "by_kind": by_kind,
        "models": models,
        "providers": providers,
        "daily": series,
        "unknowns": unknowns[:50],
    }
