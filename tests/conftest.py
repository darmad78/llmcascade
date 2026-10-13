"""Shared pytest fixtures: reset process-global state between tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_process_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent order-dependent leaks (api limiter, login lockout, health cache)."""
    from llmcascade.admin_auth import login_lockout
    from llmcascade.health import health_cache

    login_lockout._failures.clear()
    login_lockout._locked_until.clear()
    health_cache._cache.clear()

    try:
        import llmcascade.api as api_mod

        api_mod._api_limiter = None
    except ImportError:
        pass

    yield

    login_lockout._failures.clear()
    login_lockout._locked_until.clear()
    health_cache._cache.clear()
    try:
        import llmcascade.api as api_mod

        api_mod._api_limiter = None
    except ImportError:
        pass
