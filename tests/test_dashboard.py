from __future__ import annotations

from llmrouter.event_log import EventLog
from llmrouter.health import _classify


def test_event_log_splits_errors():
    log = EventLog(maxlen=10)
    log.record("ok", level="info", success=True)
    log.record("boom", level="error", model="m1")
    log.record("fail", level="info", success=False, model="m2")

    assert len(log.events()) == 3
    errs = log.errors()
    assert len(errs) == 2
    assert errs[0]["message"] == "fail"
    assert errs[1]["message"] == "boom"


def test_event_log_ring_buffer():
    log = EventLog(maxlen=3)
    for i in range(5):
        log.record(f"e{i}")
    msgs = [e["message"] for e in log.events()]
    assert msgs == ["e4", "e3", "e2"]


def test_classify_health():
    assert _classify(200, None)[0] == "ok"
    assert _classify(405, None)[0] == "ok"
    assert _classify(401, None)[0] == "auth_error"
    assert _classify(503, None)[0] == "down"
    assert _classify(None, TimeoutError())[0] == "down"
