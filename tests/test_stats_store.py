from __future__ import annotations

from datetime import datetime, timezone

from llmrouter.stats_store import NullStatsStore, StatsStore, floor_day, floor_hour


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


async def test_null_stats_store():
    store = NullStatsStore(detail="no uri")
    store.enqueue(model="m", provider="p", success=True)
    await store.record(model="m", provider="p", success=True)
    snap = await store.snapshot("7d")
    assert snap["configured"] is False
    assert snap["totals"]["models"] == []
    assert snap["totals"]["notes"] == []
    assert snap["detail"] == "no uri"
