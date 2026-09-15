"""High-water mark of successful requests in a rolling 24h window."""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from llmcascade.secrets import data_dir

WINDOW_S = 86400.0
_LOCK = threading.Lock()


def _path() -> Path:
    return data_dir() / "peak_24h.json"


def _load_peaks() -> dict[str, int]:
    path = _path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, int] = {}
    for key, val in data.items():
        try:
            out[str(key)] = max(0, int(val))
        except (TypeError, ValueError):
            continue
    return out


def _save_peaks(peaks: dict[str, int]) -> None:
    path = _path()
    path.write_text(json.dumps(peaks, indent=2) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


class Peak24h:
    def __init__(self) -> None:
        self._times: dict[str, deque[float]] = defaultdict(deque)
        self._peak = _load_peaks()

    def record_success(self, capability: str, *, now: float | None = None) -> dict[str, int]:
        cap = (capability or "chat").strip() or "chat"
        ts = time.time() if now is None else float(now)
        with _LOCK:
            q = self._times[cap]
            q.append(ts)
            cutoff = ts - WINDOW_S
            while q and q[0] < cutoff:
                q.popleft()
            window = len(q)
            peak = int(self._peak.get(cap) or 0)
            if window > peak:
                self._peak[cap] = window
                _save_peaks(dict(self._peak))
                peak = window
            return {"window": window, "peak": peak}

    def reset_for_tests(self) -> None:
        with _LOCK:
            self._times.clear()
            self._peak = _load_peaks()

    def snapshot(self, *, now: float | None = None) -> dict[str, dict[str, int]]:
        ts = time.time() if now is None else float(now)
        cutoff = ts - WINDOW_S
        with _LOCK:
            caps = set(self._times) | set(self._peak)
            out: dict[str, dict[str, int]] = {}
            for cap in caps:
                q = self._times.get(cap)
                if q:
                    while q and q[0] < cutoff:
                        q.popleft()
                    window = len(q)
                else:
                    window = 0
                out[cap] = {"window": window, "peak": int(self._peak.get(cap) or 0)}
            return out


peak_24h = Peak24h()
