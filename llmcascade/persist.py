"""Debounced atomic JSON persistence for process-local state."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class JsonState:
    """In-memory JSON dict with debounced atomic flush (off the hot path)."""

    def __init__(self, path: Path, *, delay_s: float = 2.0) -> None:
        self.path = path
        self.delay_s = delay_s
        self.lock = threading.Lock()
        self.data: dict[str, Any] = self._read()
        self._pending: asyncio.TimerHandle | None = None

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def mark_dirty(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.flush()
            return
        with self.lock:
            if self._pending is not None:
                self._pending.cancel()
            self._pending = loop.call_later(self.delay_s, self._flush_async, loop)

    def _flush_async(self, loop: asyncio.AbstractEventLoop) -> None:
        self._pending = None
        with self.lock:
            payload = json.dumps(self.data, indent=2) + "\n"
        loop.run_in_executor(None, atomic_write, self.path, payload)

    def flush(self) -> None:
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None
        with self.lock:
            payload = json.dumps(self.data, indent=2) + "\n"
        atomic_write(self.path, payload)
