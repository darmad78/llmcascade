"""Shared admin UI nav and capability-scoped dashboard/stats views."""

from __future__ import annotations

from typing import Any


def event_capability(event: dict[str, Any]) -> str:
    detail = event.get("detail") if isinstance(event.get("detail"), dict) else {}
    cap = event.get("capability") or detail.get("capability")
    return str(cap or "chat")


def keep_event_for_area(event: dict[str, Any], capability: str) -> bool:
    etype = event.get("type") or "system"
    if etype in ("request_ok", "request_fail", "cooldown", "queue"):
        return event_capability(event) == capability
    return True


def nav_html(area: str, active: str) -> str:
    llm = area != "embed"
    prefix = "" if llm else "/embed"
    help_href = "/help" if llm else "/embed/help"
    providers = "/admin/providers" if llm else "/embed/providers"
    items = [
        ("status", f"{prefix}/dashboard" if prefix else "/dashboard", "Status"),
        ("stats", f"{prefix}/stats" if prefix else "/stats", "Stats"),
        ("providers", providers, "Providers"),
        ("help", help_href, "Help"),
        ("logout", "/logout", "Logout"),
    ]
    llm_cls = "nav-btn is-active" if llm else "nav-btn"
    embed_cls = "nav-btn is-active" if not llm else "nav-btn"
    sub = []
    for key, href, label in items:
        cls = "nav-btn is-active" if key == active else "nav-btn"
        sub.append(f'<a class="{cls}" href="{href}">{label}</a>')
    return f"""<div class="nav-stack">
      <nav class="area-nav" aria-label="Area">
        <a class="{llm_cls}" href="/dashboard">LLM</a>
        <a class="{embed_cls}" href="/embed/dashboard">Embeddings</a>
      </nav>
      <nav class="page-nav" aria-label="Pages">
        {"".join(sub)}
      </nav>
    </div>"""


def filter_dashboard(data: dict[str, Any], capability: str) -> dict[str, Any]:
    models = [
        m
        for m in (data.get("models") or [])
        if capability in (m.get("capabilities") or [])
    ]
    events = [e for e in (data.get("events") or []) if keep_event_for_area(e, capability)]
    errors = [e for e in (data.get("errors") or []) if keep_event_for_area(e, capability)]
    out = dict(data)
    out["models"] = models
    out["events"] = events
    out["errors"] = errors
    out["capability"] = capability
    if capability == "embed":
        out["gemini_cascade"] = None
        out["next_pick"] = data.get("next_embed")
    else:
        out["next_embed"] = None
    return out


def _row_from_parts(
    *,
    requests: int,
    failures: int,
    latency_sum: float,
    max_latency_ms: float,
    tokens_sum: int,
) -> dict[str, Any]:
    req = int(requests or 0)
    fail = int(failures or 0)
    return {
        "requests": req,
        "failures": fail,
        "success_rate": round(((req - fail) / req) if req else 0.0, 4),
        "avg_latency_ms": round((latency_sum / req) if req else 0.0, 2),
        "max_latency_ms": round(float(max_latency_ms or 0), 2),
        "tokens_sum": int(tokens_sum or 0),
    }


def _caps_by_model_name() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        from llmcascade.registry import list_all_models

        for m in list_all_models():
            caps = m.capabilities or []
            if "embed" in caps and "chat" not in caps:
                out[m.name] = "embed"
            elif "chat" in caps:
                out[m.name] = "chat"
            elif "embed" in caps:
                out[m.name] = "embed"
    except Exception:  # noqa: BLE001
        pass
    return out


def _row_cap(row: dict[str, Any], name: str | None = None, caps: dict[str, str] | None = None) -> str:
    cap = row.get("capability")
    if cap:
        return str(cap)
    if name and caps and name in caps:
        return caps[name]
    if name and ("embed" in name.lower() or "embedding" in name.lower()):
        return "embed"
    return "chat"


def _filter_series(series: list[dict[str, Any]], capability: str, caps: dict[str, str] | None = None) -> list[dict[str, Any]]:
    caps = caps or {}
    out: list[dict[str, Any]] = []
    for raw in series:
        bucket = dict(raw)
        by_model = {
            name: row
            for name, row in (bucket.get("by_model") or {}).items()
            if _row_cap(row, name, caps) == capability
        }
        by_provider: dict[str, dict[str, Any]] = {}
        for row in by_model.values():
            prov = row.get("provider") or "unknown"
            prev = by_provider.get(prov)
            lat_sum = float(row.get("avg_latency_ms") or 0) * int(row.get("requests") or 0)
            if prev is None:
                by_provider[prov] = {
                    "requests": int(row.get("requests") or 0),
                    "failures": int(row.get("failures") or 0),
                    "latency_sum_ms": lat_sum,
                    "tokens_sum": int(row.get("tokens_sum") or 0),
                    "max_latency_ms": float(row.get("max_latency_ms") or 0),
                }
            else:
                prev["requests"] += int(row.get("requests") or 0)
                prev["failures"] += int(row.get("failures") or 0)
                prev["latency_sum_ms"] += lat_sum
                prev["tokens_sum"] += int(row.get("tokens_sum") or 0)
                prev["max_latency_ms"] = max(prev["max_latency_ms"], float(row.get("max_latency_ms") or 0))
        finalized = {}
        for name, rawp in by_provider.items():
            req = rawp["requests"]
            fail = rawp["failures"]
            finalized[name] = {
                "requests": req,
                "failures": fail,
                "success_rate": round(((req - fail) / req) if req else 0.0, 4),
                "avg_latency_ms": round((rawp["latency_sum_ms"] / req) if req else 0.0, 2),
                "max_latency_ms": rawp["max_latency_ms"],
                "tokens_sum": rawp["tokens_sum"],
            }
        nm: dict[str, Any] = {}
        for note, models in (bucket.get("by_note_model") or {}).items():
            kept = {n: r for n, r in (models or {}).items() if _row_cap(r, n, caps) == capability}
            if kept:
                nm[note] = kept
        by_notes: dict[str, Any] = {}
        for note, models in nm.items():
            req = sum(int(r.get("requests") or 0) for r in models.values())
            fail = sum(int(r.get("failures") or 0) for r in models.values())
            lat = sum(float(r.get("avg_latency_ms") or 0) * int(r.get("requests") or 0) for r in models.values())
            tok = sum(int(r.get("tokens_sum") or 0) for r in models.values())
            mx = max((float(r.get("max_latency_ms") or 0) for r in models.values()), default=0.0)
            by_notes[note] = _row_from_parts(
                requests=req, failures=fail, latency_sum=lat, max_latency_ms=mx, tokens_sum=tok
            )
        cap_row = (bucket.get("by_capability") or {}).get(capability)
        if not cap_row and by_model:
            req = sum(int(r.get("requests") or 0) for r in by_model.values())
            fail = sum(int(r.get("failures") or 0) for r in by_model.values())
            lat = sum(float(r.get("avg_latency_ms") or 0) * int(r.get("requests") or 0) for r in by_model.values())
            tok = sum(int(r.get("tokens_sum") or 0) for r in by_model.values())
            mx = max((float(r.get("max_latency_ms") or 0) for r in by_model.values()), default=0.0)
            cap_row = _row_from_parts(
                requests=req, failures=fail, latency_sum=lat, max_latency_ms=mx, tokens_sum=tok
            )
        bucket["by_model"] = by_model
        bucket["by_provider"] = finalized
        bucket["by_note_model"] = nm
        bucket["by_notes"] = by_notes
        bucket["by_capability"] = {capability: cap_row} if cap_row else {}
        bucket["requests"] = sum(int(r.get("requests") or 0) for r in by_model.values())
        bucket["failures"] = sum(int(r.get("failures") or 0) for r in by_model.values())
        out.append(bucket)
    return out


def filter_stats_snapshot(snap: dict[str, Any], capability: str) -> dict[str, Any]:
    caps_map = _caps_by_model_name()
    totals = dict(snap.get("totals") or {})
    models = [
        m
        for m in (totals.get("models") or [])
        if _row_cap(m, m.get("name"), caps_map) == capability
    ]
    providers_map: dict[str, dict[str, float]] = {}
    for m in models:
        name = m.get("provider") or "unknown"
        prev = providers_map.setdefault(
            name,
            {"requests": 0, "failures": 0, "latency_sum": 0.0, "max_latency_ms": 0.0, "tokens_sum": 0},
        )
        req = int(m.get("requests") or 0)
        prev["requests"] += req
        prev["failures"] += int(m.get("failures") or 0)
        prev["latency_sum"] += float(m.get("avg_latency_ms") or 0) * req
        prev["max_latency_ms"] = max(prev["max_latency_ms"], float(m.get("max_latency_ms") or 0))
        prev["tokens_sum"] += int(m.get("tokens_sum") or 0)
    providers = [
        {"name": name, **_row_from_parts(
            requests=int(p["requests"]),
            failures=int(p["failures"]),
            latency_sum=float(p["latency_sum"]),
            max_latency_ms=float(p["max_latency_ms"]),
            tokens_sum=int(p["tokens_sum"]),
        )}
        for name, p in providers_map.items()
    ]
    providers.sort(key=lambda r: r["requests"], reverse=True)
    caps = [c for c in (totals.get("capabilities") or []) if c.get("name") == capability]
    hourly = _filter_series((snap.get("series") or {}).get("hourly") or [], capability, caps_map)
    daily = _filter_series((snap.get("series") or {}).get("daily") or [], capability, caps_map)
    notes_map: dict[str, dict[str, float]] = {}
    for bucket in hourly + daily:
        for note, row in (bucket.get("by_notes") or {}).items():
            prev = notes_map.setdefault(
                note,
                {"requests": 0, "failures": 0, "latency_sum": 0.0, "max_latency_ms": 0.0, "tokens_sum": 0},
            )
            req = int(row.get("requests") or 0)
            prev["requests"] += req
            prev["failures"] += int(row.get("failures") or 0)
            prev["latency_sum"] += float(row.get("avg_latency_ms") or 0) * req
            prev["max_latency_ms"] = max(prev["max_latency_ms"], float(row.get("max_latency_ms") or 0))
            prev["tokens_sum"] += int(row.get("tokens_sum") or 0)
    notes = [
        {"name": name, **_row_from_parts(
            requests=int(p["requests"]),
            failures=int(p["failures"]),
            latency_sum=float(p["latency_sum"]),
            max_latency_ms=float(p["max_latency_ms"]),
            tokens_sum=int(p["tokens_sum"]),
        )}
        for name, p in notes_map.items()
    ]
    notes.sort(key=lambda r: r["requests"], reverse=True)
    perf_models = [
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
    ]
    out = dict(snap)
    out["capability"] = capability
    out["totals"] = {"models": models, "providers": providers, "notes": notes, "capabilities": caps}
    out["series"] = {"hourly": hourly, "daily": daily}
    out["performance"] = {
        "models": perf_models,
        "providers": [
            {k: p[k] for k in ("name", "requests", "failures", "success_rate", "avg_latency_ms", "max_latency_ms")}
            for p in providers
        ],
        "notes": [
            {k: n[k] for k in ("name", "requests", "failures", "success_rate", "avg_latency_ms", "max_latency_ms")}
            for n in notes
        ],
        "capabilities": caps,
    }
    return out
