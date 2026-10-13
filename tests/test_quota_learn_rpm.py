"""Learned RPM: AIMD estimate, gating, per-attempt timeout, Mongo persistence."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from llmcascade.persist import MongoState
from llmcascade.quota_learn import QuotaLearner
from llmcascade.queue_worker import RouterClient
from llmcascade.rate_limiter import RateLimiter
from llmcascade.registry import Limits, ModelConfig


@pytest.fixture
def learn(tmp_path, monkeypatch) -> QuotaLearner:
    monkeypatch.setattr("llmcascade.quota_learn._path", lambda: tmp_path / "quota_learned.json")
    return QuotaLearner()


def _model(name: str, priority: int = 10, rps: int = 50) -> ModelConfig:
    return ModelConfig(
        name=name,
        provider="groq",
        endpoint="https://example.com/v1/chat/completions",
        auth_env_var="GROQ_API_KEY",
        limits=Limits(rpd=1000, rpm=1000, rps=rps, tpm=10**7, max_context=8192),
        capabilities=["chat"],
        priority=priority,
    )


def test_rate_limit_learns_window_and_only_lowers(learn: QuotaLearner):
    assert learn.record_rate_limit("m", "groq", 12) == 12
    assert learn.record_rate_limit("m", "groq", 20) == 12
    assert learn.record_rate_limit("m", "groq", 7) == 7
    assert learn.learned_rpm("m") == 7
    assert learn.snapshot()["m"]["learned_rpm"] == 7


def test_rate_limit_ignores_token_limits_and_empty_windows(learn: QuotaLearner):
    body = "Rate limit reached for model on tokens per minute (TPM): Limit 8000"
    assert learn.record_rate_limit("m", "groq", 3, body=body) is None
    assert learn.record_rate_limit("m", "groq", 0) is None
    assert learn.learned_rpm("m") is None


def test_success_at_cap_probes_upward(learn: QuotaLearner):
    learn.record_rate_limit("m", "groq", 5)
    assert learn.observe_rpm_success("m", "groq", 4) == 5
    assert learn.observe_rpm_success("m", "groq", 5) == 6
    assert learn.learned_rpm("m") == 6


@pytest.mark.asyncio
async def test_rate_limiter_gates_on_learned_rpm(learn: QuotaLearner):
    lim = RateLimiter([_model("m")])
    lim.learned_rpm = learn.learned_rpm
    learn.record_rate_limit("m", "groq", 2)
    assert await lim.try_reserve("m", 1)
    assert await lim.try_reserve("m", 1)
    assert not await lim.can_proceed("m", 1)
    assert not await lim.try_reserve("m", 1)
    budget = await lim.remaining_budget("m")
    assert budget["rpm"] == 0 and budget["limit_rpm"] == 2


@pytest.mark.asyncio
async def test_router_learns_rpm_then_skips_capped_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("LLMCASCADE_DATA_DIR", str(tmp_path))
    calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        name = json.loads(request.content.decode())["model"]
        calls.append(name)
        if name == "a" and calls.count("a") > 2:
            return httpx.Response(429, text="Rate limit reached: requests per minute")
        return httpx.Response(
            200, json={"choices": [{"message": {"content": name}}], "usage": {"total_tokens": 1}}
        )

    router = RouterClient(
        registry=[_model("a", 10), _model("b", 20)], workers=1, max_queue=10, strategy="priority_first"
    )
    router._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await router.start()
    for _ in range(3):
        await router.submit("hi")
    assert router.quota_learn.learned_rpm("a") == 2
    router.cooldowns.pool.mark_available("a")
    calls.clear()
    resp = await router.submit("hi")
    assert resp.model == "b" or resp.text == "b"
    assert calls == ["b"]
    await router.shutdown()


@pytest.mark.asyncio
async def test_slow_attempt_times_out_and_fails_over(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("LLMCASCADE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LLMCASCADE_ATTEMPT_TIMEOUT_S", "1")

    async def handler(request: httpx.Request) -> httpx.Response:
        name = json.loads(request.content.decode())["model"]
        if name == "slow":
            await asyncio.sleep(10)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 1}}
        )

    router = RouterClient(
        registry=[_model("slow", 10), _model("fast", 20)], workers=1, max_queue=10, strategy="priority_first"
    )
    router._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await router.start()
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    resp = await router.submit("hi")
    assert loop.time() - t0 < 5
    assert resp.text == "ok"
    assert [s["model"] for s in resp.skipped_models or []] == ["slow"]
    assert router.cooldowns.pool.kind("slow") == "rate"
    await router.shutdown()


class _FakeCursor:
    def __init__(self, docs: list[dict]) -> None:
        self._docs = docs

    async def to_list(self, length=None):
        return [dict(d) for d in self._docs]


class _FakeCollection:
    def __init__(self, docs: list[dict] | None = None) -> None:
        self.docs = {d["_id"]: dict(d) for d in (docs or [])}

    def find(self, _query):
        return _FakeCursor(list(self.docs.values()))

    async def bulk_write(self, ops, ordered=False):
        for op in ops:
            doc = op._doc if hasattr(op, "_doc") else None
            key = op._filter["_id"]
            if doc is None:
                self.docs.pop(key, None)
            else:
                self.docs[key] = {"_id": key, **doc}


@pytest.mark.asyncio
async def test_attach_mongo_seeds_from_json_then_persists(learn: QuotaLearner):
    learn.record_rate_limit("m", "groq", 4)
    coll = _FakeCollection()
    await learn.attach_mongo(coll)
    assert learn.backend == "mongo"
    assert coll.docs["m"]["learned_rpm"] == 4
    learn.record_rate_limit("m", "groq", 3)
    await learn.aflush()
    assert coll.docs["m"]["learned_rpm"] == 3


@pytest.mark.asyncio
async def test_attach_mongo_prefers_remote_state(learn: QuotaLearner):
    learn.record_rate_limit("m", "groq", 9)
    coll = _FakeCollection([{"_id": "m", "provider": "groq", "learned_rpm": 2, "day": "2099-01-01"}])
    await learn.attach_mongo(coll)
    assert learn.learned_rpm("m") == 2
    assert isinstance(learn._state, MongoState)
