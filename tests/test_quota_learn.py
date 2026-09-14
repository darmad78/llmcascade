from datetime import datetime, timezone

from llmcascade.quota_learn import QuotaLearner


def test_learns_rpd_on_daily_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMCASCADE_DATA_DIR", str(tmp_path))
    now = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
    q = QuotaLearner()
    q.record_success("gemini-3.6-flash", "gemini", now=now)
    q.record_success("gemini-3.6-flash", "gemini", now=now)
    row = q.record_limit("gemini-3.6-flash", "gemini", "daily", now=now)
    assert row["ok"] == 2
    assert row["learned_rpd"] == 2
    assert row["rpd_remaining"] == 0
    snap = q.snapshot(now=now)
    assert snap["gemini-3.6-flash"]["learned_rpd"] == 2
    assert q.remaining_rpd("gemini-3.6-flash", now=now) == 0


def test_new_pacific_day_resets_ok(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMCASCADE_DATA_DIR", str(tmp_path))
    q = QuotaLearner()
    day1 = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
    q.record_success("m", "gemini", now=day1)
    q.record_limit("m", "gemini", "daily", now=day1)
    day2 = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)
    row = q.record_success("m", "gemini", now=day2)
    assert row["ok"] == 1
    assert row["learned_rpd"] == 1
    assert row["day"] == "2026-09-15"
