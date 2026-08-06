from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from llmrouter.metrics import log

# Long-running single-process API: modest pool (free-tier router is low QPS).
_DEFAULT_MAX_POOL = 10
_DEFAULT_MIN_POOL = 0
_DEFAULT_MAX_IDLE_MS = 60_000
_DEFAULT_CONNECT_MS = 5_000
_DEFAULT_SERVER_SELECTION_MS = 5_000


def floor_hour(dt: datetime) -> datetime:
    dt = dt.astimezone(timezone.utc)
    return dt.replace(minute=0, second=0, microsecond=0)


def floor_day(dt: datetime) -> datetime:
    dt = dt.astimezone(timezone.utc)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _parse_range(range_key: str) -> tuple[timedelta, timedelta]:
    """Return (hourly_lookback, daily_lookback)."""
    key = (range_key or "7d").strip().lower()
    if key in ("24h", "1d"):
        return timedelta(hours=24), timedelta(days=7)
    if key in ("30d",):
        return timedelta(days=30), timedelta(days=90)
    # default 7d
    return timedelta(days=7), timedelta(days=30)


class StatsStore:
    """Persists per-model / per-provider counters into MongoDB (hourly + daily + totals)."""

    def __init__(self, client: Any, db_name: str) -> None:
        self._client = client
        self._db = client[db_name]
        self._buckets = self._db["stats_buckets"]
        self._totals = self._db["stats_totals"]
        self.configured = True

    @classmethod
    async def connect(cls, uri: str | None = None, db_name: str | None = None) -> StatsStore | None:
        uri = (uri if uri is not None else os.environ.get("MONGODB_URI", "")).strip()
        if not uri:
            return None
        db_name = (db_name or os.environ.get("LLMROUTER_MONGO_DB") or "llmrouter").strip()
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install llmrouter[api] (motor) for MongoDB stats") from exc

        client = AsyncIOMotorClient(
            uri,
            maxPoolSize=_DEFAULT_MAX_POOL,
            minPoolSize=_DEFAULT_MIN_POOL,
            maxIdleTimeMS=_DEFAULT_MAX_IDLE_MS,
            connectTimeoutMS=_DEFAULT_CONNECT_MS,
            serverSelectionTimeoutMS=_DEFAULT_SERVER_SELECTION_MS,
        )
        store = cls(client, db_name)
        await store.ensure_indexes()
        # Fail fast if URI is wrong
        await client.admin.command("ping")
        log.info("mongodb stats connected", extra={"provider": "mongodb", "capability": "stats"})
        return store

    async def ensure_indexes(self) -> None:
        await self._buckets.create_index(
            [("grain", 1), ("bucket", 1), ("model", 1)],
            unique=True,
            name="grain_bucket_model",
        )
        await self._buckets.create_index([("grain", 1), ("bucket", 1), ("provider", 1)])
        await self._totals.create_index([("scope", 1), ("name", 1)], unique=True, name="scope_name")

    async def close(self) -> None:
        self._client.close()

    async def record(
        self,
        *,
        model: str,
        provider: str,
        success: bool,
        latency_ms: float = 0.0,
        tokens_used: int = 0,
        at: datetime | None = None,
    ) -> None:
        now = at or datetime.now(timezone.utc)
        hour = floor_hour(now)
        day = floor_day(now)
        inc = {
            "requests": 1,
            "failures": 0 if success else 1,
            "latency_sum_ms": float(latency_ms or 0),
            "tokens_sum": int(tokens_used or 0),
        }
        max_lat = float(latency_ms or 0)
        try:
            for grain, bucket in (("hour", hour), ("day", day)):
                await self._buckets.update_one(
                    {"grain": grain, "bucket": bucket, "model": model},
                    {
                        "$inc": inc,
                        "$max": {"latency_max_ms": max_lat},
                        "$setOnInsert": {"provider": provider},
                        "$set": {"provider": provider},
                    },
                    upsert=True,
                )
            await self._totals.update_one(
                {"scope": "model", "name": model},
                {
                    "$inc": inc,
                    "$max": {"latency_max_ms": max_lat},
                    "$set": {"provider": provider},
                },
                upsert=True,
            )
            await self._totals.update_one(
                {"scope": "provider", "name": provider},
                {
                    "$inc": inc,
                    "$max": {"latency_max_ms": max_lat},
                },
                upsert=True,
            )
        except Exception as exc:  # noqa: BLE001 — never fail request path on stats
            log.info(
                f"stats persist failed: {exc}",
                extra={"model_used": model, "provider": provider, "success": success},
            )

    @staticmethod
    def _row_metrics(doc: dict[str, Any]) -> dict[str, Any]:
        requests = int(doc.get("requests") or 0)
        failures = int(doc.get("failures") or 0)
        latency_sum = float(doc.get("latency_sum_ms") or 0)
        tokens = int(doc.get("tokens_sum") or 0)
        avg_latency = (latency_sum / requests) if requests else 0.0
        success_rate = ((requests - failures) / requests) if requests else 0.0
        return {
            "requests": requests,
            "failures": failures,
            "success_rate": round(success_rate, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "max_latency_ms": round(float(doc.get("latency_max_ms") or 0), 2),
            "tokens_sum": tokens,
        }

    async def snapshot(self, range_key: str = "7d") -> dict[str, Any]:
        hourly_delta, daily_delta = _parse_range(range_key)
        now = datetime.now(timezone.utc)
        hour_from = floor_hour(now - hourly_delta)
        day_from = floor_day(now - daily_delta)

        totals_raw = await self._totals.find({}).to_list(length=500)
        models: list[dict[str, Any]] = []
        providers: list[dict[str, Any]] = []
        for doc in totals_raw:
            row = {
                "name": doc["name"],
                **self._row_metrics(doc),
            }
            if doc.get("scope") == "model":
                row["provider"] = doc.get("provider") or ""
                models.append(row)
            elif doc.get("scope") == "provider":
                providers.append(row)
        models.sort(key=lambda r: r["requests"], reverse=True)
        providers.sort(key=lambda r: r["requests"], reverse=True)

        hourly_docs = await self._buckets.find(
            {"grain": "hour", "bucket": {"$gte": hour_from}}
        ).to_list(length=50_000)
        daily_docs = await self._buckets.find(
            {"grain": "day", "bucket": {"$gte": day_from}}
        ).to_list(length=20_000)

        return {
            "configured": True,
            "range": range_key,
            "totals": {"models": models, "providers": providers},
            "series": {
                "hourly": self._pivot_series(hourly_docs),
                "daily": self._pivot_series(daily_docs),
            },
            "performance": {
                "models": [
                    {
                        "name": m["name"],
                        "provider": m.get("provider", ""),
                        "requests": m["requests"],
                        "failures": m["failures"],
                        "success_rate": m["success_rate"],
                        "avg_latency_ms": m["avg_latency_ms"],
                        "max_latency_ms": m["max_latency_ms"],
                    }
                    for m in models
                ],
                "providers": [
                    {
                        "name": p["name"],
                        "requests": p["requests"],
                        "failures": p["failures"],
                        "success_rate": p["success_rate"],
                        "avg_latency_ms": p["avg_latency_ms"],
                        "max_latency_ms": p["max_latency_ms"],
                    }
                    for p in providers
                ],
            },
        }

    def _pivot_series(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_bucket: dict[datetime, dict[str, Any]] = {}
        for doc in docs:
            bucket: datetime = doc["bucket"]
            if bucket.tzinfo is None:
                bucket = bucket.replace(tzinfo=timezone.utc)
            entry = by_bucket.setdefault(
                bucket,
                {
                    "bucket": bucket.isoformat().replace("+00:00", "Z"),
                    "by_model": {},
                    "by_provider": {},
                    "requests": 0,
                    "failures": 0,
                },
            )
            metrics = self._row_metrics(doc)
            model = doc.get("model") or "unknown"
            provider = doc.get("provider") or "unknown"
            entry["by_model"][model] = {
                "provider": provider,
                **metrics,
            }
            # roll provider within bucket
            prev = entry["by_provider"].get(provider)
            if prev is None:
                entry["by_provider"][provider] = {
                    "requests": metrics["requests"],
                    "failures": metrics["failures"],
                    "latency_sum_ms": float(doc.get("latency_sum_ms") or 0),
                    "tokens_sum": metrics["tokens_sum"],
                    "max_latency_ms": metrics["max_latency_ms"],
                }
            else:
                prev["requests"] += metrics["requests"]
                prev["failures"] += metrics["failures"]
                prev["latency_sum_ms"] += float(doc.get("latency_sum_ms") or 0)
                prev["tokens_sum"] += metrics["tokens_sum"]
                prev["max_latency_ms"] = max(prev["max_latency_ms"], metrics["max_latency_ms"])
            entry["requests"] += metrics["requests"]
            entry["failures"] += metrics["failures"]

        out: list[dict[str, Any]] = []
        for bucket in sorted(by_bucket):
            entry = by_bucket[bucket]
            # finalize provider avg/success
            finalized: dict[str, Any] = {}
            for name, raw in entry["by_provider"].items():
                req = raw["requests"]
                fail = raw["failures"]
                lat_sum = raw.pop("latency_sum_ms", 0.0)
                finalized[name] = {
                    "requests": req,
                    "failures": fail,
                    "success_rate": round(((req - fail) / req) if req else 0.0, 4),
                    "avg_latency_ms": round((lat_sum / req) if req else 0.0, 2),
                    "max_latency_ms": raw["max_latency_ms"],
                    "tokens_sum": raw["tokens_sum"],
                }
            entry["by_provider"] = finalized
            out.append(entry)
        return out


class NullStatsStore:
    """No-op when MONGODB_URI is unset."""

    configured = False

    async def record(self, **_kwargs: Any) -> None:
        return None

    async def snapshot(self, range_key: str = "7d") -> dict[str, Any]:
        return {
            "configured": False,
            "range": range_key,
            "totals": {"models": [], "providers": []},
            "series": {"hourly": [], "daily": []},
            "performance": {"models": [], "providers": []},
            "detail": "Set MONGODB_URI to enable historical stats",
        }

    async def close(self) -> None:
        return None

    async def ensure_indexes(self) -> None:
        return None
