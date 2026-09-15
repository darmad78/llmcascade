from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        for key in (
            "model_used",
            "latency_ms",
            "success",
            "tokens_used",
            "provider",
            "capability",
            "dimensions",
        ):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload)


def get_logger(name: str = "llmcascade") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class MetricsCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests_total: dict[str, int] = defaultdict(int)
        self.failures_total: dict[str, int] = defaultdict(int)
        self.requests_by_capability: dict[str, int] = defaultdict(int)
        self.failures_by_capability: dict[str, int] = defaultdict(int)

    def record_success(self, model: str, capability: str = "chat") -> None:
        with self._lock:
            self.requests_total[model] += 1
            self.requests_by_capability[capability] += 1

    def record_failure(self, model: str, capability: str = "chat") -> None:
        with self._lock:
            self.requests_total[model] += 1
            self.failures_total[model] += 1
            self.requests_by_capability[capability] += 1
            self.failures_by_capability[capability] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests_total": dict(self.requests_total),
                "failures_total": dict(self.failures_total),
                "requests_by_capability": dict(self.requests_by_capability),
                "failures_by_capability": dict(self.failures_by_capability),
            }


metrics = MetricsCollector()
log = get_logger()
