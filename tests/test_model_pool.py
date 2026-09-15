from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from llmcascade.cascade import ModelCooldownTracker, classify_failure
from llmcascade.model_pool import ModelPool


def test_410_is_permanent():
    assert classify_failure(410, "Gone") == "permanent"


def test_new_id_starts_available(tmp_path: Path):
    pool = ModelPool(path=tmp_path / "pools.json")
    now = datetime(2026, 9, 15, tzinfo=timezone.utc)
    pool.mark_unavailable("old-id", "permanent", now + timedelta(days=365))
    snap = pool.snapshot(["old-id", "new-id"], now=now)
    assert "new-id" in snap["available"]
    assert snap["unavailable"][0]["name"] == "old-id"


def test_rate_stays_out_until_health(tmp_path: Path):
    pool = ModelPool(path=tmp_path / "pools.json")
    now = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)
    pool.mark_unavailable("m1", "rate", now + timedelta(seconds=60))
    assert pool.is_unavailable("m1")
    assert pool.due_for_health(now=now) == []
    later = now + timedelta(seconds=61)
    assert pool.due_for_health(now=later) == ["m1"]
    assert pool.apply_health("m1", state="ok", http_status=405, now=now) == "held"
    assert pool.apply_health("m1", state="ok", http_status=405, now=later) == "promoted"
    assert not pool.is_unavailable("m1")


def test_permanent_ignores_health(tmp_path: Path):
    pool = ModelPool(path=tmp_path / "pools.json")
    now = datetime(2026, 9, 15, tzinfo=timezone.utc)
    pool.mark_unavailable("dead", "permanent", now + timedelta(days=365))
    assert pool.apply_health("dead", state="ok", http_status=200, now=now) == "gone"
    assert pool.is_unavailable("dead")


def test_health_410_marks_gone(tmp_path: Path):
    pool = ModelPool(path=tmp_path / "pools.json")
    now = datetime(2026, 9, 15, tzinfo=timezone.utc)
    pool.mark_unavailable("m1", "rate", now)
    assert pool.apply_health("m1", state="down", http_status=410, now=now) == "gone"
    assert pool.kind("m1") == "permanent"


def test_persist_reload(tmp_path: Path):
    path = tmp_path / "pools.json"
    pool = ModelPool(path=path)
    now = datetime(2026, 9, 15, tzinfo=timezone.utc)
    pool.mark_unavailable("m1", "auth", now + timedelta(seconds=30))
    again = ModelPool(path=path)
    assert again.is_unavailable("m1")
    assert again.kind("m1") == "auth"


def test_sync_drops_removed_ids(tmp_path: Path):
    pool = ModelPool(path=tmp_path / "pools.json")
    now = datetime(2026, 9, 15, tzinfo=timezone.utc)
    pool.mark_unavailable("gone", "permanent", now + timedelta(days=1))
    pool.sync_registry(["kept"])
    assert not pool.is_unavailable("gone")


@pytest.mark.asyncio
async def test_tracker_410_blocks(tmp_path: Path):
    cool = ModelCooldownTracker(pool=ModelPool(path=tmp_path / "p.json"))
    kind = await cool.apply_from_error("nvidia-8b", status_code=410, body="Gone")
    assert kind == "permanent"
    assert await cool.is_cooling("nvidia-8b")
    assert "nvidia-8b" not in cool.snapshot_pools(["nvidia-8b", "ok"]).get("available")
