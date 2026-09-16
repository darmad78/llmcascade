from __future__ import annotations

from datetime import datetime, timezone

import pytest

from llmcascade.stats_store import NullStatsStore, StatsStore, floor_day, floor_hour


def test_floor_hour_day_utc():
    dt = datetime(2026, 8, 6, 23, 45, 12, tzinfo=timezone.utc)
    assert floor_hour(dt) == datetime(2026, 8, 6, 23, 0, 0, tzinfo=timezone.utc)
    assert floor_day(dt) == datetime(2026, 8, 6, 0, 0, 0, tzinfo=timezone.utc)


def test_pivot_series_rolls_provider():
    store = StatsStore.__new__(StatsStore)
    docs = [
        {
            "grain": "hour",
            "bucket": datetime(2026, 8, 6, 10, 0, tzinfo=timezone.utc),
            "model": "a",
            "provider": "p1",
            "requests": 3,
            "failures": 1,
            "latency_sum_ms": 300.0,
            "latency_max_ms": 150.0,
            "tokens_sum": 30,
        },
        {
            "grain": "hour",
            "bucket": datetime(2026, 8, 6, 10, 0, tzinfo=timezone.utc),
            "model": "b",
            "provider": "p1",
            "requests": 2,
            "failures": 0,
            "latency_sum_ms": 100.0,
            "latency_max_ms": 60.0,
            "tokens_sum": 20,
        },
        {
            "grain": "hour",
            "bucket": datetime(2026, 8, 6, 10, 0, tzinfo=timezone.utc),
            "model": "__note__:app",
            "kind": "notes",
            "notes": "app",
            "provider": "",
            "requests": 4,
            "failures": 1,
            "latency_sum_ms": 200.0,
            "latency_max_ms": 80.0,
            "tokens_sum": 40,
        },
    ]
    series = store._pivot_series(docs)
    assert len(series) == 1
    row = series[0]
    assert row["requests"] == 5
    assert row["failures"] == 1
    assert row["by_model"]["a"]["requests"] == 3
    assert "__note__:app" not in row["by_model"]
    assert row["by_provider"]["p1"]["requests"] == 5
    assert row["by_provider"]["p1"]["avg_latency_ms"] == 80.0
    assert row["by_provider"]["p1"]["success_rate"] == 0.8
    assert row["by_notes"]["app"]["requests"] == 4
    assert row["by_notes"]["app"]["success_rate"] == 0.75


def test_pivot_series_by_capability():
    store = StatsStore.__new__(StatsStore)
    docs = [
        {
            "grain": "hour",
            "bucket": datetime(2026, 8, 16, 10, 0, tzinfo=timezone.utc),
            "model": "chat-m",
            "provider": "p1",
            "capability": "chat",
            "requests": 4,
            "failures": 0,
            "latency_sum_ms": 400.0,
            "latency_max_ms": 120.0,
            "tokens_sum": 40,
        },
        {
            "grain": "hour",
            "bucket": datetime(2026, 8, 16, 10, 0, tzinfo=timezone.utc),
            "model": "embed-m",
            "provider": "p2",
            "capability": "embed",
            "requests": 2,
            "failures": 1,
            "latency_sum_ms": 80.0,
            "latency_max_ms": 50.0,
            "tokens_sum": 10,
        },
    ]
    series = store._pivot_series(docs)
    row = series[0]
    assert row["by_capability"]["chat"]["requests"] == 4
    assert row["by_capability"]["embed"]["requests"] == 2
    assert row["by_capability"]["embed"]["success_rate"] == 0.5
    assert row["by_model"]["embed-m"]["capability"] == "embed"


async def test_null_stats_store():
    store = NullStatsStore(detail="no uri")
    store.enqueue(model="m", provider="p", success=True)
    await store.record(model="m", provider="p", success=True)
    snap = await store.snapshot("7d")
    assert snap["configured"] is False
    assert snap["totals"]["models"] == []
    assert snap["totals"]["notes"] == []
    assert snap["detail"] == "no uri"
    fail = await store.failure_snapshot()
    assert fail["configured"] is False
    assert fail["unknowns"] == []


def test_note_model_key_roundtrip():
    from llmcascade.stats_store import note_model_key, parse_note_model_key

    key = note_model_key("billing|eu", "openai/gpt-4o")
    assert parse_note_model_key(key) == ("billing|eu", "openai/gpt-4o")
    assert parse_note_model_key("__note__:x") is None


def test_pivot_series_note_model_cross_dim():
    store = StatsStore.__new__(StatsStore)
    docs = [
        {
            "grain": "day",
            "bucket": datetime(2026, 8, 6, 0, 0, tzinfo=timezone.utc),
            "model": "a",
            "provider": "p1",
            "requests": 3,
            "failures": 1,
            "latency_sum_ms": 300.0,
            "latency_max_ms": 150.0,
            "tokens_sum": 30,
        },
        {
            "grain": "day",
            "bucket": datetime(2026, 8, 6, 0, 0, tzinfo=timezone.utc),
            "model": "__nm__:app\x1fa",
            "kind": "note_model",
            "notes": "app",
            "base_model": "a",
            "provider": "p1",
            "requests": 3,
            "failures": 1,
            "latency_sum_ms": 300.0,
            "latency_max_ms": 150.0,
            "tokens_sum": 30,
        },
        {
            "grain": "day",
            "bucket": datetime(2026, 8, 6, 0, 0, tzinfo=timezone.utc),
            "model": "__nm__:jobs\x1fa",
            "kind": "note_model",
            "notes": "jobs",
            "base_model": "a",
            "provider": "p1",
            "requests": 2,
            "failures": 0,
            "latency_sum_ms": 100.0,
            "latency_max_ms": 60.0,
            "tokens_sum": 20,
        },
    ]
    series = store._pivot_series(docs)
    row = series[0]
    assert row["by_model"]["a"]["requests"] == 3
    assert "__nm__:app" not in str(row["by_model"])
    assert row["by_note_model"]["app"]["a"]["failures"] == 1
    assert row["by_note_model"]["jobs"]["a"]["requests"] == 2
    assert row["by_note_model"]["jobs"]["a"]["avg_latency_ms"] == 50.0


class _Hits:
    def __init__(self) -> None:
        self.docs: list[dict] = []

    async def insert_one(self, doc):
        self.docs.append(dict(doc))

    async def count_documents(self, q):
        cap = q.get("capability")
        gte = (q.get("ts") or {}).get("$gte")
        n = 0
        for d in self.docs:
            if cap is not None and d.get("capability") != cap:
                continue
            if gte is not None and d.get("ts") < gte:
                continue
            n += 1
        return n


class _Peaks:
    def __init__(self) -> None:
        self.docs: dict = {}

    async def find_one(self, q):
        return self.docs.get(q.get("_id"))

    async def update_one(self, q, upd, upsert=False):
        _id = q["_id"]
        row = dict(self.docs.get(_id) or {"_id": _id, "peak": 0})
        mx = (upd.get("$max") or {}).get("peak")
        if mx is not None:
            row["peak"] = max(int(row.get("peak") or 0), int(mx))
        self.docs[_id] = row


@pytest.mark.asyncio
async def test_mongo_24h_counter_keeps_max():
    from datetime import timedelta

    store = StatsStore.__new__(StatsStore)
    store._req_24h = _Hits()
    store._req_24h_peak = _Peaks()
    store._recv_24h = _Hits()
    t0 = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)
    await store._bump_peak_24h("chat", t0)
    await store._bump_peak_24h("chat", t0 + timedelta(hours=1))
    snap = await store.snapshot_peak_24h(now=t0 + timedelta(hours=2))
    assert snap["chat"] == {"window": 2, "peak": 2, "recv": 0}
    await store._bump_peak_24h("chat", t0 + timedelta(hours=25))
    later = await store.snapshot_peak_24h(now=t0 + timedelta(hours=26))
    assert later["chat"]["window"] == 1
    assert later["chat"]["peak"] == 2
    assert later["chat"]["recv"] == 0


@pytest.mark.asyncio
async def test_recv_24h_counts_incoming_not_handled():
    store = StatsStore.__new__(StatsStore)
    store._req_24h = _Hits()
    store._req_24h_peak = _Peaks()
    store._recv_24h = _Hits()
    t0 = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)
    await store.record_recv("chat", now=t0)
    await store.record_recv("chat", now=t0)
    await store._bump_peak_24h("chat", t0)
    snap = await store.snapshot_peak_24h(now=t0)
    assert snap["chat"]["recv"] == 2
    assert snap["chat"]["window"] == 1
    assert snap["embed"]["recv"] == 0
