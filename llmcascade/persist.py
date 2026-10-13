"""Debounced atomic JSON persistence for process-local state."""

from __future__ import annotations

import asyncio
import copy
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

    async def aflush(self) -> None:
        self.flush()


class MongoState:
    """Same contract as JsonState, persisted as one Mongo document per top-level key.

    Reads stay in memory; dirty keys are upserted after a debounce delay.
    """

    def __init__(
        self,
        collection: Any,
        data: dict[str, Any],
        *,
        saved: dict[str, Any] | None = None,
        delay_s: float = 2.0,
    ) -> None:
        self.collection = collection
        self.delay_s = delay_s
        self.lock = threading.Lock()
        self.data: dict[str, Any] = data
        self._saved: dict[str, Any] = copy.deepcopy(saved or {})
        self._pending: asyncio.TimerHandle | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    def mark_dirty(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        with self.lock:
            if self._pending is not None:
                self._pending.cancel()
            self._pending = loop.call_later(self.delay_s, self._kick)

    def _kick(self) -> None:
        self._pending = None
        task = asyncio.ensure_future(self.aflush())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def flush(self) -> None:
        """Sync callers can only schedule; use ``aflush`` to wait for the write."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        self._kick()

    async def aflush(self) -> None:
        from pymongo import DeleteOne, ReplaceOne

        if self._pending is not None:
            self._pending.cancel()
            self._pending = None
        with self.lock:
            snapshot = copy.deepcopy(self.data)
        ops: list[Any] = [
            ReplaceOne({"_id": key}, dict(row), upsert=True)
            for key, row in snapshot.items()
            if isinstance(row, dict) and self._saved.get(key) != row
        ]
        ops += [DeleteOne({"_id": key}) for key in self._saved if key not in snapshot]
        if ops:
            try:
                await self.collection.bulk_write(ops, ordered=False)
            except Exception as exc:  # noqa: BLE001 — learning must never break routing
                from llmcascade.metrics import log

                log.warning("quota learn mongo write failed: %s", type(exc).__name__)
                return
        self._saved = snapshot
