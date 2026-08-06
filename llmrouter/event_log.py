from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Event:
    ts: float
    level: str
    message: str
    model: str | None = None
    provider: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventLog:
    """Process-local ring buffers for all events and errors-only."""

    def __init__(self, maxlen: int = 500) -> None:
        self._lock = threading.Lock()
        self._events: deque[Event] = deque(maxlen=maxlen)
        self._errors: deque[Event] = deque(maxlen=maxlen)

    def record(
        self,
        message: str,
        *,
        level: str = "info",
        model: str | None = None,
        provider: str | None = None,
        **detail: Any,
    ) -> Event:
        event = Event(
            ts=time.time(),
            level=level,
            message=message,
            model=model,
            provider=provider,
            detail=detail,
        )
        with self._lock:
            self._events.append(event)
            if level in ("error", "critical") or detail.get("success") is False:
                self._errors.append(event)
        return event

    def events(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._events)
        # Newest first for dashboard / API consumers.
        items.reverse()
        if limit is not None:
            items = items[:limit]
        return [e.to_dict() for e in items]

    def errors(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._errors)
        items.reverse()
        if limit is not None:
            items = items[:limit]
        return [e.to_dict() for e in items]


events = EventLog()
