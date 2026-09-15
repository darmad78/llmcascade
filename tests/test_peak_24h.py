from llmcascade.peak_24h import Peak24h, WINDOW_S


def test_peak_only_rises_when_24h_count_is_greater(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMCASCADE_DATA_DIR", str(tmp_path))
    p = Peak24h()
    t0 = 1_000_000.0
    assert p.record_success("chat", now=t0) == {"window": 1, "peak": 1}
    assert p.record_success("chat", now=t0 + 1) == {"window": 2, "peak": 2}
    later = p.record_success("chat", now=t0 + WINDOW_S + 2)
    assert later["window"] == 1
    assert later["peak"] == 2


def test_peak_persists_and_does_not_drop(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMCASCADE_DATA_DIR", str(tmp_path))
    a = Peak24h()
    a.record_success("chat", now=10.0)
    a.record_success("chat", now=11.0)
    b = Peak24h()
    snap = b.snapshot(now=12.0)
    assert snap["chat"]["peak"] == 2
    assert snap["chat"]["window"] == 0
