"""
Optional FastAPI HTTP layer for llmrouter.

v1 budgets are process-local — run a single uvicorn worker only:
  uvicorn llmrouter.api:app --host 0.0.0.0 --port 12000
Do not use --workers > 1 until a shared BudgetStore (e.g. Redis) is implemented.

Historical stats require MONGODB_URI (optional LLMROUTER_MONGO_DB, default llmrouter).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from llmrouter.adapters.base import LLMResponse
from llmrouter.admin_auth import (
    CSRF_COOKIE,
    SESSION_COOKIE,
    authenticate,
    cookie_secure,
    create_session_token,
    csrf_cookie_kwargs,
    csrf_ok,
    login_lockout,
    new_csrf_token,
    path_allowed_during_password_change,
    path_requires_auth,
    session_cookie_kwargs,
    validate_session,
    wants_html,
)
from llmrouter.api_auth import (
    api_rpm_limit,
    configured_api_keys,
    extract_api_key,
    require_auth_enabled,
    validate_api_key,
)
from llmrouter.auth_store import change_password, ensure_admin
from llmrouter.event_log import events
from llmrouter.exceptions import safe_error_message
from llmrouter.metrics import log
from llmrouter.provider_store import list_stored_providers, save_provider
from llmrouter.queue_worker import RouterClient
from llmrouter.rate_limiter import ApiKeyRateLimiter
from llmrouter.registry import key_source, list_providers
from llmrouter.secrets import resolve_secret_key
from llmrouter.stats_store import NullStatsStore, StatsStore

try:
    from fastapi import FastAPI, Form, HTTPException, Query, Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install llmrouter[api] to use the HTTP layer") from exc


_client: RouterClient | None = None
_stats: StatsStore | NullStatsStore | None = None
_api_limiter: ApiKeyRateLimiter | None = None
_STATIC = Path(__file__).resolve().parent / "static"
_DASHBOARD_HTML = _STATIC / "dashboard.html"
_STATS_HTML = _STATIC / "stats.html"
_LOGIN_HTML = _STATIC / "login.html"
_CHANGE_PASSWORD_HTML = _STATIC / "change_password.html"
_ADMIN_PROVIDERS_HTML = _STATIC / "admin_providers.html"


class CompleteRequest(BaseModel):
    prompt: str
    capability: str = "chat"
    params: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class ProviderSaveBody(BaseModel):
    provider: str
    api_key: str | None = None
    free_paid: Literal["free", "paid"] = "free"
    clear_key: bool = False
    csrf_token: str | None = None


def _is_https(request: Request) -> bool:
    if request.url.scheme == "https":
        return True
    return (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower() == "https"


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _set_auth_cookies(response: Response, token: str, *, secure: bool) -> str:
    csrf = new_csrf_token()
    response.set_cookie(SESSION_COOKIE, token, **session_cookie_kwargs(secure=secure))
    response.set_cookie(CSRF_COOKIE, csrf, **csrf_cookie_kwargs(secure=secure))
    return csrf


def _clear_auth_cookies(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")
    response.delete_cookie(CSRF_COOKIE, path="/")


def _load_env_files() -> None:
    """Load .env files; non-empty process env wins, empty placeholders are filled from file."""
    try:
        from dotenv import dotenv_values
    except ImportError:
        return
    for path in (Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"):
        if not path.is_file():
            continue
        for key, val in dotenv_values(path).items():
            if val is None:
                continue
            cur = os.environ.get(key)
            if cur is None or str(cur).strip() == "":
                os.environ[key] = val


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client, _stats, _api_limiter
    _load_env_files()

    secret = resolve_secret_key()
    if not (os.environ.get("SECRET_KEY") or "").strip():
        log.warning("SECRET_KEY unset — using auto-generated key in LLMROUTER_DATA_DIR; set SECRET_KEY in production")
    ensure_admin()

    rpm = api_rpm_limit()
    _api_limiter = ApiKeyRateLimiter(rpm) if rpm is not None else None

    models_path = Path(__file__).resolve().parent / "models.yaml"
    try:
        _stats = await StatsStore.connect()
    except Exception as exc:  # noqa: BLE001
        detail = f"MongoDB connect failed: {exc}"
        events.record(detail, level="error")
        _stats = NullStatsStore(detail=detail)
    if _stats is None:
        _stats = NullStatsStore(
            detail="MONGODB_URI is empty — add it to .env and restart"
        )
    _client = RouterClient(models_path=str(models_path), stats=_stats, allow_empty=True)
    await _client.start()
    events.record(
        "router started",
        level="info",
        models=len(_client.registry),
        stats_configured=_stats.configured,
    )
    app.state.secret_key = secret
    yield
    await _client.shutdown(graceful=True)
    await _stats.close()
    events.record("router stopped", level="info")
    _client = None
    _stats = None
    _api_limiter = None


app = FastAPI(title="llmrouter", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def admin_auth_middleware(request: Request, call_next):
    path = request.url.path
    if not path_requires_auth(path):
        return await call_next(request)

    session = validate_session(request.cookies.get(SESSION_COOKIE))
    if session is None:
        if wants_html(request.headers.get("accept")) or path in ("/dashboard", "/stats") or path.startswith("/admin"):
            return RedirectResponse(url="/login", status_code=303)
        return JSONResponse({"detail": "authentication required"}, status_code=401)

    _user, claims = session
    if claims.must_change_password and not path_allowed_during_password_change(path):
        if wants_html(request.headers.get("accept")) or path.startswith("/admin") or path in ("/dashboard", "/stats"):
            return RedirectResponse(url="/admin/change-password", status_code=303)
        return JSONResponse({"detail": "password change required"}, status_code=403)

    request.state.admin_user = _user
    request.state.admin_claims = claims
    return await call_next(request)


def _require_client() -> RouterClient:
    if _client is None:
        raise HTTPException(status_code=503, detail="router not ready")
    return _client


def _require_csrf(request: Request, submitted: str | None) -> None:
    if not csrf_ok(request.cookies.get(CSRF_COOKIE), submitted):
        raise HTTPException(status_code=403, detail="invalid CSRF token")


@app.post("/v1/complete", response_model=LLMResponse)
async def complete(request: Request, body: CompleteRequest) -> LLMResponse:
    global _api_limiter
    if require_auth_enabled():
        keys = configured_api_keys()
        if not keys:
            raise HTTPException(
                status_code=503,
                detail="REQUIRE_AUTH=true but LLMROUTER_API_KEYS is empty",
            )
        api_key = extract_api_key(
            request.headers.get("authorization"),
            request.headers.get("x-api-key"),
        )
        if not validate_api_key(api_key, keys):
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        rpm = api_rpm_limit()
        if rpm is not None and api_key is not None:
            if _api_limiter is None or _api_limiter.rpm != rpm:
                _api_limiter = ApiKeyRateLimiter(rpm)
            if not await _api_limiter.check_and_record(api_key):
                raise HTTPException(status_code=429, detail="API key rate limit exceeded")

    client = _require_client()
    try:
        return await client.submit(
            body.prompt, body.capability, notes=body.notes, **body.params
        )
    except Exception as exc:
        detail: dict[str, Any] = {"capability": body.capability}
        if body.notes and str(body.notes).strip():
            detail["notes"] = str(body.notes).strip()
        safe = safe_error_message(exc)
        events.record(safe, level="error", **detail)
        raise HTTPException(status_code=502, detail=safe) from exc


@app.get("/v1/status")
async def status() -> dict[str, Any]:
    return await _require_client().status()


@app.get("/v1/status/gemini")
async def gemini_status() -> dict[str, Any]:
    return await _require_client().gemini_status()


@app.get("/v1/metrics")
async def metrics_endpoint() -> dict[str, Any]:
    return await _require_client().metrics_snapshot()


@app.get("/v1/stats")
async def stats_endpoint(range: str = Query(default="7d", pattern="^(24h|1d|7d|30d)$")) -> dict[str, Any]:
    return await _require_client().stats_snapshot(range)


@app.get("/v1/health")
async def health_endpoint(force: bool = False) -> dict[str, dict[str, Any]]:
    return await _require_client().health_snapshot(force=force)


@app.get("/v1/events")
async def events_endpoint(limit: int | None = None) -> list[dict[str, Any]]:
    return events.events(limit=limit)


@app.get("/v1/errors")
async def errors_endpoint(limit: int | None = None) -> list[dict[str, Any]]:
    return events.errors(limit=limit)


@app.get("/v1/dashboard")
async def dashboard_data(force_health: bool = False) -> dict[str, Any]:
    return await _require_client().dashboard_snapshot(force_health=force_health)


@app.get("/dashboard")
async def dashboard_page() -> FileResponse:
    if not _DASHBOARD_HTML.is_file():
        raise HTTPException(status_code=404, detail="dashboard.html missing")
    return FileResponse(_DASHBOARD_HTML, media_type="text/html")


@app.get("/stats")
async def stats_page() -> FileResponse:
    if not _STATS_HTML.is_file():
        raise HTTPException(status_code=404, detail="stats.html missing")
    return FileResponse(_STATS_HTML, media_type="text/html")


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> Response:
    if validate_session(request.cookies.get(SESSION_COOKIE)):
        session = validate_session(request.cookies.get(SESSION_COOKIE))
        assert session is not None
        _, claims = session
        if claims.must_change_password:
            return RedirectResponse(url="/admin/change-password", status_code=303)
        return RedirectResponse(url="/dashboard", status_code=303)
    if not _LOGIN_HTML.is_file():
        raise HTTPException(status_code=404, detail="login.html missing")
    return FileResponse(_LOGIN_HTML, media_type="text/html")


@app.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
) -> Response:
    ip = _client_ip(request)
    locked, remaining = login_lockout.is_locked(ip, username)
    if locked:
        raise HTTPException(
            status_code=429,
            detail=f"too many failed attempts; try again in {int(remaining) + 1}s",
        )
    user = authenticate(username.strip(), password)
    if user is None:
        delay = login_lockout.record_failure(ip, username)
        if delay:
            raise HTTPException(
                status_code=429,
                detail=f"too many failed attempts; try again in {int(delay)}s",
            )
        raise HTTPException(status_code=401, detail="invalid credentials")
    login_lockout.record_success(ip, username)
    token = create_session_token(user)
    secure = cookie_secure(_is_https(request))
    dest = "/admin/change-password" if user.must_change_password else "/dashboard"
    resp = RedirectResponse(url=dest, status_code=303)
    _set_auth_cookies(resp, token, secure=secure)
    return resp


@app.get("/logout")
@app.post("/logout")
async def logout(request: Request) -> Response:
    if request.method == "POST":
        _require_csrf(
            request,
            request.headers.get("x-csrf-token")
            or (await request.form()).get("csrf_token"),  # type: ignore[arg-type]
        )
    resp = RedirectResponse(url="/login", status_code=303)
    _clear_auth_cookies(resp)
    return resp


@app.get("/admin/change-password", response_class=HTMLResponse)
async def change_password_page() -> FileResponse:
    if not _CHANGE_PASSWORD_HTML.is_file():
        raise HTTPException(status_code=404, detail="change_password.html missing")
    return FileResponse(_CHANGE_PASSWORD_HTML, media_type="text/html")


@app.post("/admin/change-password")
async def change_password_submit(
    request: Request,
    new_password: str = Form(...),
    confirm_password: str = Form(...),
    csrf_token: str = Form(""),
) -> Response:
    _require_csrf(request, csrf_token or request.headers.get("x-csrf-token"))
    session = validate_session(request.cookies.get(SESSION_COOKIE))
    if session is None:
        return RedirectResponse(url="/login", status_code=303)
    user, _claims = session
    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="password must be at least 8 characters")
    if new_password != confirm_password:
        raise HTTPException(status_code=400, detail="passwords do not match")
    updated = change_password(user, new_password)
    token = create_session_token(updated)
    secure = cookie_secure(_is_https(request))
    resp = RedirectResponse(url="/dashboard", status_code=303)
    _set_auth_cookies(resp, token, secure=secure)
    return resp


@app.get("/admin/providers", response_class=HTMLResponse)
async def admin_providers_page() -> FileResponse:
    if not _ADMIN_PROVIDERS_HTML.is_file():
        raise HTTPException(status_code=404, detail="admin_providers.html missing")
    return FileResponse(_ADMIN_PROVIDERS_HTML, media_type="text/html")


@app.get("/admin/providers/data")
async def admin_providers_data() -> dict[str, Any]:
    stored = list_stored_providers()
    rows = []
    for info in list_providers():
        provider = info["provider"]
        meta = stored.get(provider, {})
        src = key_source(info["auth_env_var"], provider=provider)
        rows.append(
            {
                "provider": provider,
                "auth_env_var": info["auth_env_var"],
                "needs_account_id": info["needs_account_id"],
                "key_set": src != "none",
                "key_source": src,
                "free_paid": meta.get("free_paid", "free"),
                "masked": "••••••••" if src != "none" else "",
            }
        )
    return {"providers": rows, "csrf_cookie": CSRF_COOKIE}


@app.post("/admin/providers")
async def admin_providers_save(request: Request, body: ProviderSaveBody) -> dict[str, Any]:
    submitted = body.csrf_token or request.headers.get("x-csrf-token")
    _require_csrf(request, submitted)
    provider = body.provider.strip()
    known = {p["provider"] for p in list_providers()}
    if provider not in known:
        raise HTTPException(status_code=400, detail="unknown provider")
    save_provider(
        provider,
        api_key=body.api_key,
        free_paid=body.free_paid,
        clear_key=body.clear_key,
    )
    n = _require_client().reload_registry(allow_empty=True)
    events.record("provider keys reloaded", level="info", models=n, provider=provider)
    return {"ok": True, "models": n, "provider": provider}
