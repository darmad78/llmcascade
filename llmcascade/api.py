"""
Optional FastAPI HTTP layer for llmcascade.

v1 budgets are process-local — run a single uvicorn worker only:
  uvicorn llmcascade.api:app --host 0.0.0.0 --port 12000
Do not use --workers > 1 until a shared BudgetStore (e.g. Redis) is implemented.

Historical stats require MONGODB_URI (optional LLMCASCADE_MONGO_DB, default llmcascade).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from llmcascade.adapters.base import LLMResponse
from llmcascade.admin_auth import (
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
from llmcascade.api_auth import (
    api_rpm_limit,
    auth_credentials_configured,
    extract_api_key,
    is_production_profile,
    require_auth_enabled,
    validate_api_key,
)
from llmcascade.auth_store import change_password, ensure_admin
from llmcascade.event_log import events
from llmcascade.exceptions import safe_error_message
from llmcascade.metrics import log
from llmcascade.provider_store import list_stored_providers, save_provider
from llmcascade.queue_worker import RouterClient
from llmcascade.rate_limiter import ApiKeyRateLimiter
from llmcascade.registry import key_source, list_all_models, list_providers, provider_auth_env
from llmcascade.secrets import resolve_secret_key
from llmcascade.stats_store import NullStatsStore, StatsStore
from llmcascade.model_store import (
    delete_custom_model,
    set_override,
    upsert_custom_model,
)
from llmcascade.ui import filter_dashboard, filter_stats_snapshot, nav_html
from llmcascade.health import probe_model
from llmcascade.registry import ModelConfig, Limits

try:
    from fastapi import FastAPI, Form, HTTPException, Query, Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install llmcascade[api] to use the HTTP layer") from exc


_client: RouterClient | None = None
_stats: StatsStore | NullStatsStore | None = None
_api_limiter: ApiKeyRateLimiter | None = None
_STATIC = Path(__file__).resolve().parent / "static"
_DASHBOARD_HTML = _STATIC / "dashboard.html"
_STATS_HTML = _STATIC / "stats.html"
_LOGIN_HTML = _STATIC / "login.html"
_CHANGE_PASSWORD_HTML = _STATIC / "change_password.html"
_ADMIN_PROVIDERS_HTML = _STATIC / "admin_providers.html"
_HELP_HTML = _STATIC / "help.html"


class CompleteRequest(BaseModel):
    prompt: str
    capability: str = "chat"
    params: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class EmbedRequest(BaseModel):
    prompt: str
    model: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class ProviderSaveBody(BaseModel):
    provider: str
    api_key: str | None = None
    free_paid: Literal["free", "paid"] = "free"
    clear_key: bool = False
    add_free_key: str | None = None
    add_paid_key: str | None = None
    clear_free_keys: bool = False
    clear_paid_keys: bool = False
    replace_free_key: str | None = None
    replace_paid_key: str | None = None
    disable_env: bool | None = None
    csrf_token: str | None = None


class ModelSaveBody(BaseModel):
    name: str
    provider: str
    endpoint: str
    auth_env_var: str | None = None
    capabilities: list[str] = Field(default_factory=lambda: ["chat"])
    priority: int = 100
    weight: int = 1
    enabled: bool = True
    key_tier: Literal["free", "paid"] = "free"
    limits: dict[str, int] = Field(
        default_factory=lambda: {
            "rpd": 100,
            "rpm": 30,
            "rps": 2,
            "tpm": 60000,
            "max_context": 8192,
        }
    )
    free_tier_verified: bool = False
    free_tier_note: str = ""
    cascade: list[str] = Field(default_factory=list)
    csrf_token: str | None = None


class ModelOverrideBody(BaseModel):
    name: str
    enabled: bool | None = None
    weight: int | None = None
    key_tier: Literal["free", "paid"] | None = None
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
        log.warning(
            "SECRET_KEY unset — using auto-generated key in LLMCASCADE_DATA_DIR; set SECRET_KEY in production"
        )
    if is_production_profile():
        if not auth_credentials_configured():
            raise RuntimeError(
                "LLMCASCADE_PROFILE=production requires REQUIRE_AUTH credentials "
                "(LLMCASCADE_API_KEYS and/or LLMCASCADE_API_KEY_HASHES)"
            )
        if not (os.environ.get("SECRET_KEY") or "").strip():
            log.warning(
                "LLMCASCADE_PROFILE=production but SECRET_KEY unset — set an explicit SECRET_KEY"
            )
    elif not require_auth_enabled():
        log.warning(
            "POST /v1/complete is open (REQUIRE_AUTH unset). "
            "Set REQUIRE_AUTH=true or LLMCASCADE_PROFILE=production before exposing beyond localhost"
        )
    ensure_admin()

    rpm = api_rpm_limit()
    _api_limiter = ApiKeyRateLimiter(rpm) if rpm is not None else None

    models_path = Path(__file__).resolve().parent / "models.yaml"
    try:
        _stats = await StatsStore.connect()
    except Exception as exc:  # noqa: BLE001
        detail = f"MongoDB connect failed: {exc}"
        events.record(detail, level="error", type="system")
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
        type="lifecycle",
        models=len(_client.registry),
        stats_configured=_stats.configured,
    )
    app.state.secret_key = secret
    yield
    await _client.shutdown(graceful=True)
    await _stats.close()
    events.record("router stopped", level="info", type="lifecycle")
    _client = None
    _stats = None
    _api_limiter = None


app = FastAPI(title="llmcascade", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def admin_auth_middleware(request: Request, call_next):
    path = request.url.path
    if not path_requires_auth(path):
        return await call_next(request)

    session = validate_session(request.cookies.get(SESSION_COOKIE))
    if session is None:
        if wants_html(request.headers.get("accept")) or path in (
            "/dashboard",
            "/stats",
        ) or path.startswith("/admin") or path.startswith("/embed/"):
            return RedirectResponse(url="/login", status_code=303)
        return JSONResponse({"detail": "authentication required"}, status_code=401)

    _user, claims = session
    if claims.must_change_password and not path_allowed_during_password_change(path):
        if wants_html(request.headers.get("accept")) or path.startswith("/admin") or path.startswith("/embed/") or path in ("/dashboard", "/stats"):
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


def _inject_page(html: str, *, area: str, active: str) -> str:
    prefix = "" if area == "llm" else "/embed"
    cap = "chat" if area == "llm" else "embed"
    heading = "Configured models" if area == "llm" else "Embedding models"
    html = html.replace("__NAV__", nav_html(area, active))
    html = html.replace("__AREA__", area)
    html = html.replace("__PREFIX__", prefix)
    html = html.replace("__CAPABILITY__", cap)
    html = html.replace("__MODELS_HEADING__", heading)
    return html


def _html_file(
    path: Path, *, area: str, active: str, request: Request | None = None
) -> HTMLResponse:
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{path.name} missing")
    resp = HTMLResponse(_inject_page(path.read_text(encoding="utf-8"), area=area, active=active))
    if request is not None and not request.cookies.get(CSRF_COOKIE):
        token = new_csrf_token()
        resp.set_cookie(
            CSRF_COOKIE,
            token,
            **csrf_cookie_kwargs(secure=cookie_secure(_is_https(request))),
        )
    return resp


def _dashboard_session_ok(request: Request) -> bool:
    session = validate_session(request.cookies.get(SESSION_COOKIE))
    if session is None:
        return False
    _, claims = session
    return not claims.must_change_password


async def _authorize_inference(request: Request) -> None:
    global _api_limiter
    if not require_auth_enabled():
        return
    if not auth_credentials_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "API auth required but no credentials configured "
                "(set LLMCASCADE_API_KEYS and/or LLMCASCADE_API_KEY_HASHES)"
            ),
        )
    api_key = extract_api_key(
        request.headers.get("authorization"),
        request.headers.get("x-api-key"),
    )
    if validate_api_key(api_key):
        rpm = api_rpm_limit()
        if rpm is not None and api_key is not None:
            if _api_limiter is None or _api_limiter.rpm != rpm:
                _api_limiter = ApiKeyRateLimiter(rpm)
            if not await _api_limiter.check_and_record(api_key):
                raise HTTPException(status_code=429, detail="API key rate limit exceeded")
        return
    if _dashboard_session_ok(request):
        _require_csrf(request, request.headers.get("x-csrf-token"))
        return
    raise HTTPException(status_code=401, detail="invalid or missing API key")


async def _submit_inference(
    prompt: str,
    capability: str,
    notes: str | None,
    params: dict[str, Any],
    *,
    model: str | None = None,
) -> LLMResponse:
    client = _require_client()
    try:
        return await client.submit(prompt, capability, notes=notes, model=model, **params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        detail: dict[str, Any] = {"capability": capability}
        if notes and str(notes).strip():
            detail["notes"] = str(notes).strip()
        if model:
            detail["model"] = model
        safe = safe_error_message(exc)
        events.record(safe, level="error", type="request_fail", **detail)
        raise HTTPException(status_code=502, detail=safe) from exc


@app.post("/v1/complete", response_model=LLMResponse)
async def complete(request: Request, body: CompleteRequest) -> LLMResponse:
    await _authorize_inference(request)
    return await _submit_inference(body.prompt, body.capability, body.notes, body.params)


@app.post("/v1/embed", response_model=LLMResponse)
async def embed(request: Request, body: EmbedRequest) -> LLMResponse:
    await _authorize_inference(request)
    return await _submit_inference(
        body.prompt, "embed", body.notes, body.params, model=body.model
    )


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
async def stats_endpoint(
    range: str = Query(default="7d", pattern="^(24h|1d|7d|30d)$"),
    capability: str | None = Query(default=None, pattern="^(chat|embed)$"),
) -> dict[str, Any]:
    snap = await _require_client().stats_snapshot(range)
    if capability:
        snap = filter_stats_snapshot(snap, capability)
    return snap


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
async def dashboard_data(
    force_health: bool = False,
    capability: str | None = Query(default=None, pattern="^(chat|embed)$"),
) -> dict[str, Any]:
    snap = await _require_client().dashboard_snapshot(force_health=force_health)
    if capability:
        snap = filter_dashboard(snap, capability)
    return snap


@app.get("/dashboard")
async def dashboard_page(request: Request) -> HTMLResponse:
    return _html_file(_DASHBOARD_HTML, area="llm", active="status", request=request)


@app.get("/embed/dashboard")
async def embed_dashboard_page(request: Request) -> HTMLResponse:
    return _html_file(_DASHBOARD_HTML, area="embed", active="status", request=request)


@app.get("/stats")
async def stats_page() -> HTMLResponse:
    return _html_file(_STATS_HTML, area="llm", active="stats")


@app.get("/embed/stats")
async def embed_stats_page() -> HTMLResponse:
    return _html_file(_STATS_HTML, area="embed", active="stats")


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


@app.get("/help", response_class=HTMLResponse)
async def help_page() -> HTMLResponse:
    return _html_file(_HELP_HTML, area="llm", active="help")


@app.get("/embed/help", response_class=HTMLResponse)
async def embed_help_page() -> HTMLResponse:
    return _html_file(_HELP_HTML, area="embed", active="help")


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
async def change_password_page(request: Request) -> Response:
    if not _CHANGE_PASSWORD_HTML.is_file():
        raise HTTPException(status_code=404, detail="change_password.html missing")
    html = _CHANGE_PASSWORD_HTML.read_text(encoding="utf-8")
    token = new_csrf_token()
    html = html.replace("__CSRF_TOKEN__", token)
    secure = cookie_secure(_is_https(request))
    resp = HTMLResponse(html)
    resp.set_cookie(CSRF_COOKIE, token, **csrf_cookie_kwargs(secure=secure))
    return resp


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


def _providers_page(request: Request, area: str) -> HTMLResponse:
    if not _ADMIN_PROVIDERS_HTML.is_file():
        raise HTTPException(status_code=404, detail="admin_providers.html missing")
    html = _ADMIN_PROVIDERS_HTML.read_text(encoding="utf-8")
    token = new_csrf_token()
    html = html.replace("__CSRF_TOKEN__", token)
    html = _inject_page(html, area=area, active="providers")
    secure = cookie_secure(_is_https(request))
    resp = HTMLResponse(html)
    resp.set_cookie(CSRF_COOKIE, token, **csrf_cookie_kwargs(secure=secure))
    return resp


@app.get("/admin/providers", response_class=HTMLResponse)
async def admin_providers_page(request: Request) -> Response:
    return _providers_page(request, "llm")


@app.get("/embed/providers", response_class=HTMLResponse)
async def embed_providers_page(request: Request) -> Response:
    return _providers_page(request, "embed")


@app.get("/admin/providers/data")
async def admin_providers_data(
    capability: str | None = Query(default=None, pattern="^(chat|embed)$"),
) -> dict[str, Any]:
    import os

    stored = list_stored_providers()
    rows = []
    for info in list_providers():
        provider = info["provider"]
        meta = stored.get(provider, {})
        src = key_source(info["auth_env_var"], provider=provider)
        env_set = bool(os.environ.get(info["auth_env_var"]))
        if info["auth_env_var"] == "HF_TOKEN":
            env_set = env_set or bool(os.environ.get("HUGGINGFACE_API_KEY"))
        free_n = int(meta.get("free_key_count") or 0)
        paid_n = int(meta.get("paid_key_count") or 0)
        free_set = free_n > 0 or (src == "env" and env_set)
        paid_set = paid_n > 0
        rows.append(
            {
                "provider": provider,
                "auth_env_var": info["auth_env_var"],
                "needs_account_id": info["needs_account_id"],
                "key_set": src != "none",
                "key_source": src,
                "free_key_count": free_n,
                "paid_key_count": paid_n,
                "free_set": free_set,
                "paid_set": paid_set,
                "env_set": env_set,
                "disable_env": bool(meta.get("disable_env")),
                "free_paid": meta.get("free_paid", "free"),
            }
        )
    active = {m.name for m in (_client.registry if _client else [])}
    try:
        from llmcascade.model_store import get_override
    except Exception:  # noqa: BLE001
        get_override = lambda _n: {}  # noqa: E731
    models = []
    for m in list_all_models():
        parent_active = m.name in active
        if m.cascade:
            for mid in m.cascade:
                ov = get_override(mid)
                enabled = True if "enabled" not in ov else bool(ov["enabled"])
                try:
                    weight = max(1, int(ov.get("weight", 1)))
                except (TypeError, ValueError):
                    weight = 1
                key_tier = ov.get("key_tier") if ov.get("key_tier") in ("free", "paid") else m.key_tier
                models.append(
                    {
                        "name": mid,
                        "provider": m.provider,
                        "endpoint": m.endpoint,
                        "auth_env_var": m.auth_env_var,
                        "priority": m.priority,
                        "weight": weight,
                        "enabled": enabled,
                        "key_tier": key_tier,
                        "custom": False,
                        "active": parent_active and enabled,
                        "capabilities": m.capabilities,
                        "limits": m.limits.model_dump(),
                        "cascade_of": m.name,
                        "is_cascade_member": True,
                    }
                )
            continue
        models.append(
            {
                "name": m.name,
                "provider": m.provider,
                "endpoint": m.endpoint,
                "auth_env_var": m.auth_env_var,
                "priority": m.priority,
                "weight": m.weight,
                "enabled": m.enabled,
                "key_tier": m.key_tier,
                "custom": m.custom,
                "active": m.name in active,
                "capabilities": m.capabilities,
                "limits": m.limits.model_dump(),
                "cascade_of": None,
                "is_cascade_member": False,
            }
        )
    models.sort(key=lambda e: (e["provider"], e["priority"], e["name"]))
    if capability:
        models = [m for m in models if capability in (m.get("capabilities") or [])]
    return {
        "providers": rows,
        "models": models,
        "provider_options": sorted({p["provider"] for p in rows}),
        "csrf_cookie": CSRF_COOKIE,
    }


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
        add_free_key=body.add_free_key,
        add_paid_key=body.add_paid_key,
        clear_free_keys=body.clear_free_keys,
        clear_paid_keys=body.clear_paid_keys,
        replace_free_key=body.replace_free_key,
        replace_paid_key=body.replace_paid_key,
        disable_env=body.disable_env,
    )
    n = _require_client().reload_registry(allow_empty=True)
    events.record(
        "provider keys reloaded",
        level="info",
        type="admin",
        models=n,
        provider=provider,
    )
    return {"ok": True, "models": n, "provider": provider}


@app.post("/admin/models")
async def admin_models_save(request: Request, body: ModelSaveBody) -> dict[str, Any]:
    submitted = body.csrf_token or request.headers.get("x-csrf-token")
    _require_csrf(request, submitted)
    name = body.name.strip()
    provider = body.provider.strip()
    if not name or not provider or not body.endpoint.strip():
        raise HTTPException(status_code=400, detail="name, provider, and endpoint are required")
    auth_env = (body.auth_env_var or provider_auth_env(provider)).strip()
    try:
        limits = Limits.model_validate(body.limits)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid limits: {exc}") from exc
    payload = {
        "name": name,
        "provider": provider,
        "endpoint": body.endpoint.strip(),
        "auth_env_var": auth_env,
        "capabilities": body.capabilities or ["chat"],
        "priority": body.priority,
        "weight": max(1, body.weight),
        "enabled": body.enabled,
        "key_tier": body.key_tier,
        "limits": limits.model_dump(),
        "free_tier_verified": body.free_tier_verified,
        "free_tier_note": body.free_tier_note,
        "cascade": body.cascade,
        "custom": True,
    }
    upsert_custom_model(payload)
    set_override(name, enabled=body.enabled, weight=max(1, body.weight), key_tier=body.key_tier)

    # Probe before accepting into live registry
    model = ModelConfig.model_validate(payload)
    client = _require_client()
    probe = await probe_model(client._client, model)
    if probe.state != "ok":
        # Keep stored but report failure — operator can hide or fix keys
        client.reload_registry(allow_empty=True)
        return {
            "ok": False,
            "saved": True,
            "test": probe.to_dict(),
            "detail": f"model saved but probe failed: {probe.state} ({probe.message})",
            "models": len(client.registry),
        }
    n = client.reload_registry(allow_empty=True)
    events.record(
        "custom model saved",
        level="info",
        type="admin",
        model=name,
        provider=provider,
        models=n,
    )
    return {"ok": True, "saved": True, "test": probe.to_dict(), "models": n, "name": name}


@app.post("/admin/models/override")
async def admin_models_override(request: Request, body: ModelOverrideBody) -> dict[str, Any]:
    submitted = body.csrf_token or request.headers.get("x-csrf-token")
    _require_csrf(request, submitted)
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name required")
    set_override(name, enabled=body.enabled, weight=body.weight, key_tier=body.key_tier)
    n = _require_client().reload_registry(allow_empty=True)
    events.record(
        "model override",
        level="info",
        type="admin",
        model=name,
        enabled=body.enabled,
        weight=body.weight,
        models=n,
    )
    return {"ok": True, "models": n, "name": name}


@app.post("/admin/models/delete")
async def admin_models_delete(request: Request, body: ModelOverrideBody) -> dict[str, Any]:
    submitted = body.csrf_token or request.headers.get("x-csrf-token")
    _require_csrf(request, submitted)
    name = body.name.strip()
    if not delete_custom_model(name):
        raise HTTPException(status_code=404, detail="custom model not found")
    n = _require_client().reload_registry(allow_empty=True)
    return {"ok": True, "models": n, "deleted": name}


@app.post("/admin/models/test")
async def admin_models_test(request: Request, body: ModelOverrideBody) -> dict[str, Any]:
    submitted = body.csrf_token or request.headers.get("x-csrf-token")
    _require_csrf(request, submitted)
    name = body.name.strip()
    model = next((m for m in list_all_models() if m.name == name), None)
    if model is None:
        parent = next((m for m in list_all_models() if name in (m.cascade or [])), None)
        if parent is None:
            raise HTTPException(status_code=404, detail="model not found")
        # Probe a specific cascade member id using the parent endpoint/auth.
        model = parent.model_copy(update={"name": name, "cascade": []})
    client = _require_client()
    probe = await probe_model(client._client, model)
    return {"ok": probe.state == "ok", "test": probe.to_dict(), "name": name}
