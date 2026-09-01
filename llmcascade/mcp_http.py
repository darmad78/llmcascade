"""Streamable HTTP MCP (JSON-RPC) mounted on the FastAPI app.

Auth: Bearer / X-API-Key. Inference keys cannot call admin tools.
Admin tools require LLMCASCADE_ADMIN_API_KEYS (or hashes), never cookies.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Literal

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from llmcascade.api_auth import (
    admin_credentials_configured,
    api_rpm_limit,
    extract_api_key,
    mcp_principal,
    require_auth_enabled,
)
from llmcascade.exceptions import AllModelsExhaustedError, safe_error_message

PROTOCOL_VERSION = "2025-03-26"
SUPPORTED_PROTOCOL_VERSIONS = frozenset(
    {"2024-11-05", "2025-03-26", "2025-06-18"}
)

ToolScope = Literal["inference", "admin"]
JsonDict = dict[str, Any]
ToolHandler = Callable[[JsonDict], Awaitable[Any]]


def _text_result(payload: Any, *, is_error: bool = False) -> JsonDict:
    if isinstance(payload, (dict, list)):
        text = json.dumps(payload, default=str)
    else:
        text = str(payload)
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def _rpc_result(req_id: Any, result: Any) -> JsonDict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _rpc_error(req_id: Any, code: int, message: str) -> JsonDict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _tool_schema(properties: JsonDict, required: list[str] | None = None) -> JsonDict:
    schema: JsonDict = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        schema["required"] = required
    return schema


INFERENCE_TOOLS: dict[str, JsonDict] = {
    "complete": {
        "description": "Run a chat completion through the llmcascade dispatcher.",
        "inputSchema": _tool_schema(
            {
                "prompt": {"type": "string"},
                "capability": {"type": "string", "default": "chat"},
                "params": {"type": "object"},
                "notes": {"type": "string"},
                "model": {"type": "string"},
                "failover_models": {"type": "array", "items": {"type": "string"}},
                "include_free_cascade": {"type": "boolean"},
            },
            ["prompt"],
        ),
    },
    "embed": {
        "description": "Embed text with a pinned embedding model (no cascade).",
        "inputSchema": _tool_schema(
            {
                "prompt": {"type": "string"},
                "model": {"type": "string"},
                "params": {"type": "object"},
                "notes": {"type": "string"},
            },
            ["prompt", "model"],
        ),
    },
    "status": {
        "description": "Live rate-limit budgets and dispatcher status.",
        "inputSchema": _tool_schema({}),
    },
    "health": {
        "description": "Provider reachability snapshot.",
        "inputSchema": _tool_schema({"force": {"type": "boolean", "default": False}}),
    },
}

ADMIN_TOOLS: dict[str, JsonDict] = {
    "list_providers": {
        "description": "List providers and models (no secret values).",
        "inputSchema": _tool_schema(
            {"capability": {"type": "string", "enum": ["chat", "embed"]}}
        ),
    },
    "save_provider": {
        "description": "Store or rotate provider API keys (encrypted at rest). Do not echo keys.",
        "inputSchema": _tool_schema(
            {
                "provider": {"type": "string"},
                "api_key": {"type": "string"},
                "free_paid": {"type": "string", "enum": ["free", "paid"]},
                "clear_key": {"type": "boolean"},
                "add_free_key": {"type": "string"},
                "add_paid_key": {"type": "string"},
                "clear_free_keys": {"type": "boolean"},
                "clear_paid_keys": {"type": "boolean"},
                "replace_free_key": {"type": "string"},
                "replace_paid_key": {"type": "string"},
                "disable_env": {"type": "boolean"},
            },
            ["provider"],
        ),
    },
    "save_model": {
        "description": "Create or update a custom model, then probe it.",
        "inputSchema": _tool_schema(
            {
                "name": {"type": "string"},
                "provider": {"type": "string"},
                "endpoint": {"type": "string"},
                "auth_env_var": {"type": "string"},
                "capabilities": {"type": "array", "items": {"type": "string"}},
                "priority": {"type": "integer"},
                "weight": {"type": "integer"},
                "enabled": {"type": "boolean"},
                "key_tier": {"type": "string", "enum": ["free", "paid"]},
                "limits": {"type": "object"},
                "free_tier_verified": {"type": "boolean"},
                "free_tier_note": {"type": "string"},
                "cascade": {"type": "array", "items": {"type": "string"}},
            },
            ["name", "provider", "endpoint"],
        ),
    },
    "override_model": {
        "description": "Enable/disable a model or change weight / key tier.",
        "inputSchema": _tool_schema(
            {
                "name": {"type": "string"},
                "enabled": {"type": "boolean"},
                "weight": {"type": "integer"},
                "key_tier": {"type": "string", "enum": ["free", "paid"]},
            },
            ["name"],
        ),
    },
    "delete_custom_model": {
        "description": "Delete a custom (UI-added) model.",
        "inputSchema": _tool_schema({"name": {"type": "string"}}, ["name"]),
    },
    "test_model": {
        "description": "Probe a model without changing config.",
        "inputSchema": _tool_schema({"name": {"type": "string"}}, ["name"]),
    },
    "metrics": {
        "description": "In-process metrics snapshot.",
        "inputSchema": _tool_schema({}),
    },
    "stats": {
        "description": "Historical stats (Mongo when configured).",
        "inputSchema": _tool_schema(
            {
                "range": {"type": "string", "enum": ["24h", "1d", "7d", "30d"]},
                "capability": {"type": "string", "enum": ["chat", "embed"]},
            }
        ),
    },
    "failures": {
        "description": "Classified provider failures for the last 30 days (Mongo).",
        "inputSchema": _tool_schema(
            {"capability": {"type": "string", "enum": ["chat", "embed"]}}
        ),
    },
    "events": {
        "description": "Recent event log (metadata only).",
        "inputSchema": _tool_schema({"limit": {"type": "integer"}}),
    },
    "errors": {
        "description": "Recent error events.",
        "inputSchema": _tool_schema({"limit": {"type": "integer"}}),
    },
    "dashboard": {
        "description": "Dashboard snapshot (budgets, health, models).",
        "inputSchema": _tool_schema(
            {
                "force_health": {"type": "boolean"},
                "capability": {"type": "string", "enum": ["chat", "embed"]},
            }
        ),
    },
}


def tool_scope(name: str) -> ToolScope | None:
    if name in INFERENCE_TOOLS:
        return "inference"
    if name in ADMIN_TOOLS:
        return "admin"
    return None


def listed_tools(principal: str) -> list[JsonDict]:
    names = list(INFERENCE_TOOLS.items())
    if principal == "admin":
        names.extend(ADMIN_TOOLS.items())
    return [
        {"name": name, "description": spec["description"], "inputSchema": spec["inputSchema"]}
        for name, spec in names
    ]


def _exc_payload(exc: HTTPException) -> Any:
    return exc.detail


async def dispatch_tool(name: str, arguments: JsonDict) -> Any:
    import llmcascade.api as api

    args = arguments or {}
    if name == "complete":
        body = api.CompleteRequest.model_validate(args)
        return (
            await api._submit_inference(
                body.prompt,
                body.capability,
                body.notes,
                body.params,
                model=body.model,
                failover_models=body.failover_models,
                include_free_cascade=body.include_free_cascade,
            )
        ).model_dump()
    if name == "embed":
        body = api.EmbedRequest.model_validate(args)
        return (
            await api._submit_inference(
                body.prompt, "embed", body.notes, body.params, model=body.model
            )
        ).model_dump()
    if name == "status":
        return await api._require_client().status()
    if name == "health":
        return await api._require_client().health_snapshot(force=bool(args.get("force")))
    if name == "list_providers":
        cap = args.get("capability")
        return api.providers_catalog(capability=cap)
    if name == "save_provider":
        body = api.ProviderSaveBody.model_validate(args)
        return api.apply_provider_save(body)
    if name == "save_model":
        body = api.ModelSaveBody.model_validate(args)
        return await api.apply_model_save(body)
    if name == "override_model":
        body = api.ModelOverrideBody.model_validate(args)
        return api.apply_model_override(body)
    if name == "delete_custom_model":
        return api.apply_model_delete(str(args.get("name") or ""))
    if name == "test_model":
        return await api.apply_model_test(str(args.get("name") or ""))
    if name == "metrics":
        return await api._require_client().metrics_snapshot()
    if name == "stats":
        rng = str(args.get("range") or "7d")
        snap = await api._require_client().stats_snapshot(rng)
        cap = args.get("capability")
        if cap:
            from llmcascade.ui import filter_stats_snapshot

            snap = filter_stats_snapshot(snap, cap)
        return snap
    if name == "failures":
        cap = args.get("capability")
        return await api._require_client().failure_snapshot(cap)
    if name == "events":
        from llmcascade.event_log import events

        limit = args.get("limit")
        return events.events(limit=int(limit) if limit is not None else None)
    if name == "errors":
        from llmcascade.event_log import events

        limit = args.get("limit")
        return events.errors(limit=int(limit) if limit is not None else None)
    if name == "dashboard":
        snap = await api._require_client().dashboard_snapshot(
            force_health=bool(args.get("force_health"))
        )
        cap = args.get("capability")
        if cap:
            from llmcascade.ui import filter_dashboard

            snap = filter_dashboard(snap, cap)
        return snap
    raise HTTPException(status_code=404, detail=f"unknown tool {name}")


async def handle_rpc(message: JsonDict, *, principal: str) -> JsonDict | None:
    if message.get("jsonrpc") != "2.0":
        return _rpc_error(message.get("id"), -32600, "invalid JSON-RPC version")
    method = message.get("method")
    req_id = message.get("id")
    params = message.get("params") if isinstance(message.get("params"), dict) else {}
    if req_id is None:
        return None
    if not isinstance(method, str):
        return _rpc_error(req_id, -32600, "method required")
    if method == "initialize":
        requested = str((params or {}).get("protocolVersion") or PROTOCOL_VERSION)
        version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else PROTOCOL_VERSION
        return _rpc_result(
            req_id,
            {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "llmcascade", "version": "0.1.0"},
                "instructions": (
                    "Inference tools use LLMCASCADE_API_KEYS. "
                    "Admin tools require a separate LLMCASCADE_ADMIN_API_KEYS Bearer token."
                ),
            },
        )
    if method == "ping":
        return _rpc_result(req_id, {})
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return _rpc_result(req_id, {"tools": listed_tools(principal)})
    if method == "tools/call":
        name = str((params or {}).get("name") or "")
        arguments = (params or {}).get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        scope = tool_scope(name)
        if scope is None:
            return _rpc_result(req_id, _text_result(f"unknown tool {name}", is_error=True))
        if scope == "admin" and principal != "admin":
            msg = (
                "admin API key required (LLMCASCADE_ADMIN_API_KEYS / "
                "LLMCASCADE_ADMIN_API_KEY_HASHES)"
                if not admin_credentials_configured()
                else "inference API keys cannot call admin tools"
            )
            return _rpc_result(req_id, _text_result(msg, is_error=True))
        try:
            result = await dispatch_tool(name, arguments)
        except HTTPException as exc:
            return _rpc_result(req_id, _text_result(_exc_payload(exc), is_error=True))
        except ValidationError as exc:
            return _rpc_result(req_id, _text_result(exc.errors(), is_error=True))
        except AllModelsExhaustedError as exc:
            detail: Any = str(exc)
            if exc.skipped_models:
                detail = {"message": str(exc), "skipped_models": exc.skipped_models}
            return _rpc_result(req_id, _text_result(detail, is_error=True))
        except Exception as exc:  # noqa: BLE001
            return _rpc_result(req_id, _text_result(safe_error_message(exc), is_error=True))
        return _rpc_result(req_id, _text_result(result))
    return _rpc_error(req_id, -32601, f"method not found: {method}")


async def _maybe_rate_limit(api_key: str | None) -> JSONResponse | None:
    import llmcascade.api as api

    rpm = api_rpm_limit()
    if rpm is None or not api_key:
        return None
    if api._api_limiter is None or api._api_limiter.rpm != rpm:
        from llmcascade.rate_limiter import ApiKeyRateLimiter

        api._api_limiter = ApiKeyRateLimiter(rpm)
    if not await api._api_limiter.check_and_record(api_key):
        return JSONResponse({"detail": "API key rate limit exceeded"}, status_code=429)
    return None


async def mcp_endpoint(request: Request) -> Response:
    if request.method == "DELETE":
        return Response(status_code=204)
    if request.method != "POST":
        return JSONResponse({"detail": "MCP is POST JSON-RPC at /mcp"}, status_code=405)

    api_key = extract_api_key(
        request.headers.get("authorization"),
        request.headers.get("x-api-key"),
    )
    principal = mcp_principal(api_key)
    if require_auth_enabled() and principal == "none":
        return JSONResponse({"detail": "invalid or missing API key"}, status_code=401)

    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"detail": "invalid JSON"}, status_code=400)

    rate_tools = False
    if isinstance(payload, dict) and payload.get("method") == "tools/call":
        params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
        rate_tools = str(params.get("name") or "") in {"complete", "embed"}
    if rate_tools and require_auth_enabled() and api_key:
        limited = await _maybe_rate_limit(api_key)
        if limited is not None:
            return limited

    if isinstance(payload, list):
        out = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            msg = await handle_rpc(item, principal=principal)
            if msg is not None:
                out.append(msg)
        return JSONResponse(out)

    if not isinstance(payload, dict):
        return JSONResponse({"detail": "JSON-RPC object required"}, status_code=400)

    msg = await handle_rpc(payload, principal=principal)
    if msg is None:
        return Response(status_code=202)
    return JSONResponse(msg)
