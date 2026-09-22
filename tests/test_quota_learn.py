from __future__ import annotations

from llmcascade.quota_learn import QuotaLearner


def test_record_limit_without_success_does_not_learn_zero_cap(tmp_path, monkeypatch):
    monkeypatch.setattr("llmcascade.quota_learn._path", lambda: tmp_path / "quota_learned.json")
    learn = QuotaLearner()
    learn.record_limit("m1", "together", "credit")
    snap = learn.snapshot()
    assert snap["m1"]["rpd_remaining"] == 0
    assert snap["m1"].get("learned_rpd") is None
    assert learn.remaining_rpd("m1") is None
