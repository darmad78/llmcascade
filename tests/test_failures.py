from __future__ import annotations

from datetime import datetime, timezone

from llmcascade.exceptions import ProviderError
from llmcascade.failures import (
    build_failure_record,
    classify_recorded_failure,
    redact_failure_text,
    summarize_failures,
)
from llmcascade.stats_store import NullStatsStore


def test_classify_recorded_known_kinds():
    assert classify_recorded_failure(429, "rate limit") == "rate"
    assert classify_recorded_failure(429, "PerDay quota exceeded") == "daily"
    assert classify_recorded_failure(402, "Insufficient Balance") == "credit"
    assert classify_recorded_failure(401, "unauthorized") == "auth"
    assert classify_recorded_failure(403, "forbidden") == "auth"
    assert classify_recorded_failure(None, "timeout") == "timeout"
    assert classify_recorded_failure(408, "request timeout") == "timeout"
    assert classify_recorded_failure(404, "model not found") == "permanent"
    assert classify_recorded_failure(410, "Gone") == "permanent"


def test_classify_recorded_unknown_keeps_5xx_and_400():
    assert classify_recorded_failure(503, "unavailable") == "unknown"
    assert classify_recorded_failure(400, "bad request") == "unknown"
    assert classify_recorded_failure(403, "limit:0") == "daily"


def test_unknown_record_stores_redacted_detail_only():
    exc = ProviderError(
        "groq HTTP 400: bad schema Bearer sk-SECRETBODY AIzaSyNotARealKey1234567890",
        status_code=400,
        provider="groq",
        model="m1",
    )
    doc = build_failure_record(
        model="m1",
        provider="groq",
        capability="chat",
        notes="app",
        exc=exc,
        at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    assert doc["kind"] == "unknown"
    assert doc["message"] == "groq/m1 HTTP 400"
    assert doc["detail"]
    assert "sk-SECRETBODY" not in doc["detail"]
    assert "AIzaSyNotARealKey1234567890" not in doc["detail"]
    assert "bad schema" in doc["detail"]


def test_known_record_omits_provider_body():
    exc = ProviderError(
        "groq HTTP 429: SUPER_SECRET_PROVIDER_BODY_TOKEN_XYZ",
        status_code=429,
        provider="groq",
        model="m1",
    )
    doc = build_failure_record(
        model="m1", provider="groq", capability="chat", notes=None, exc=exc
    )
    assert doc["kind"] == "rate"
    assert doc["detail"] is None
    assert "SUPER_SECRET" not in str(doc)


def test_redact_query_api_key():
    text = redact_failure_text("https://generativelanguage.googleapis.com/v1?key=REALSECRET")
    assert "REALSECRET" not in text
    assert "[redacted]" in text


def test_summarize_failures_30d_shape():
    ts = datetime(2026, 8, 6, 10, tzinfo=timezone.utc)
    docs = [
        {"ts": ts, "model": "a", "provider": "p1", "kind": "rate", "status_code": 429, "capability": "chat"},
        {"ts": ts, "model": "a", "provider": "p1", "kind": "unknown", "status_code": 400,
         "detail": "bad schema", "message": "p1/a HTTP 400", "capability": "chat"},
        {"ts": ts, "model": "b", "provider": "p2", "kind": "timeout", "status_code": None, "capability": "chat"},
    ]
    out = summarize_failures(docs)
    assert out["total"] == 3
    assert out["by_kind"]["rate"] == 1
    assert out["by_kind"]["unknown"] == 1
    assert out["by_kind"]["timeout"] == 1
    assert out["models"][0]["name"] == "a"
    assert out["models"][0]["count"] == 2
    assert len(out["unknowns"]) == 1
    assert out["unknowns"][0]["detail"] == "bad schema"


async def test_null_failure_snapshot():
    store = NullStatsStore(detail="no uri")
    store.enqueue_failure(model="m", provider="p", capability="chat", notes=None, exc=Exception("x"))
    snap = await store.failure_snapshot("chat")
    assert snap["configured"] is False
    assert snap["range"] == "30d"
    assert snap["total"] == 0
    assert snap["unknowns"] == []
