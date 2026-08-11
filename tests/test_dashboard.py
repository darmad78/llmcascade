from __future__ import annotations

from llmcascade.event_log import EventLog
from llmcascade.health import _classify


def test_event_log_splits_errors():
    log = EventLog(maxlen=10)
    log.record("ok", level="info", type="request_ok", success=True)
    log.record("boom", level="error", type="system", model="m1")
    log.record("fail", level="info", type="request_fail", success=False, model="m2")

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


def test_event_log_records_type():
    log = EventLog(maxlen=10)
    log.record("health ok", type="health", model="m1")
    log.record("request ok", type="request_ok", success=True)
    log.record("request fail", type="request_fail", success=False)
    types = [e["type"] for e in log.events()]
    assert types == ["request_fail", "request_ok", "health"]
    assert log.events()[0]["type"] == "request_fail"


def test_classify_health():
    assert _classify(200, None)[0] == "ok"
    assert _classify(405, None)[0] == "ok"
    assert _classify(401, None)[0] == "auth_error"
    assert _classify(503, None)[0] == "down"
    assert _classify(None, TimeoutError())[0] == "down"
