"""Separate process: replace permanently dead model IDs and email the admin.

Polls the running API for `permanent` cooldowns, asks `/v1/complete` for the next
free ID from the provider catalog, probes until one works, writes models.yaml,
then hot-reloads the API.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import smtplib
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
import yaml

from llmcascade.cascade import classify_failure
from llmcascade.registry import ModelConfig, default_models_path, resolve_auth_env
from llmcascade.secrets import data_dir
from llmcascade.yaml_edit import replace_model_id

ProbeKind = Literal["free", "paid", "gone", "error"]
MAX_RUNS = 40
MAX_CATALOG_IDS = 80
_STATE_LOCK = threading.Lock()


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key:
            out[key] = val
    return out


def load_env_files() -> None:
    here = Path(__file__).resolve().parents[1]
    for path in (Path.cwd() / ".env", here / ".env"):
        if not path.is_file():
            continue
        for key, val in _parse_env_file(path).items():
            cur = os.environ.get(key)
            if cur is None or str(cur).strip() == "":
                os.environ[key] = val


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _state_path() -> Path:
    return data_dir() / "retire_watch.json"


def _empty_state() -> dict[str, Any]:
    return {
        "replacements": {},
        "abandoned": {},
        "runs": [],
        "interval_s": _env_int("RETIRE_WATCH_INTERVAL_S", 120),
        "pid": None,
        "last_poll_at": None,
        "last_error": None,
        "started_at": None,
    }


def load_state() -> dict[str, Any]:
    path = _state_path()
    empty = _empty_state()
    if not path.is_file():
        return empty
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return empty
    if not isinstance(raw, dict):
        return empty
    empty.update(raw)
    empty.setdefault("replacements", {})
    empty.setdefault("abandoned", {})
    empty.setdefault("runs", [])
    if not isinstance(empty.get("runs"), list):
        empty["runs"] = []
    if not isinstance(empty.get("replacements"), dict):
        empty["replacements"] = {}
    if not isinstance(empty.get("abandoned"), dict):
        empty["abandoned"] = {}
    return empty


def save_state(state: dict[str, Any]) -> None:
    path = _state_path()
    path.write_text(json.dumps(state, indent=2) + "\n")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(raw: str) -> datetime | None:
    try:
        return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def mark_watcher_alive(*, pid: int, interval_s: int) -> None:
    with _STATE_LOCK:
        state = load_state()
        state["pid"] = pid
        state["interval_s"] = interval_s
        state["started_at"] = state.get("started_at") or _iso_now()
        save_state(state)


def append_run(run: dict[str, Any]) -> None:
    with _STATE_LOCK:
        state = load_state()
        runs = list(state.get("runs") or [])
        runs.insert(0, run)
        state["runs"] = runs[:MAX_RUNS]
        state["last_poll_at"] = str(run.get("at") or _iso_now())
        err = run.get("error")
        state["last_error"] = str(err) if err else None
        save_state(state)


def snapshot(*, now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    with _STATE_LOCK:
        state = load_state()
    interval = int(state.get("interval_s") or _env_int("RETIRE_WATCH_INTERVAL_S", 120))
    last = str(state.get("last_poll_at") or "")
    age_s: int | None = None
    parsed = _parse_iso(last) if last else None
    if parsed is not None:
        age_s = max(0, int((datetime.fromtimestamp(now, tz=timezone.utc) - parsed).total_seconds()))
    stale_after = max(interval * 3, interval + 30)
    running = age_s is not None and age_s <= stale_after
    replacements = []
    for old, row in (state.get("replacements") or {}).items():
        if isinstance(row, dict):
            replacements.append({"old": old, **row})
        else:
            replacements.append({"old": old, "new": str(row)})
    abandoned = []
    for mid, reason in (state.get("abandoned") or {}).items():
        abandoned.append({"id": mid, "reason": str(reason)})
    return {
        "running": running,
        "stale": bool(last) and not running,
        "interval_s": interval,
        "pid": state.get("pid"),
        "started_at": state.get("started_at"),
        "last_poll_at": last or None,
        "last_error": state.get("last_error"),
        "age_s": age_s,
        "replacements": replacements,
        "abandoned": abandoned,
        "runs": list(state.get("runs") or []),
    }


@dataclass(frozen=True)
class RetiredTarget:
    model_id: str
    provider: str
    endpoint: str
    auth_env_var: str
    cascade_parent: str | None = None


def retired_from_status(
    *,
    models: list[ModelConfig],
    cooldowns: dict[str, Any],
    gemini: dict[str, Any] | None,
) -> list[RetiredTarget]:
    by_name = {m.name: m for m in models}
    found: dict[str, RetiredTarget] = {}
    for name, cd in (cooldowns or {}).items():
        if name in ("gemini_cascade",) or not isinstance(cd, dict):
            continue
        if cd.get("kind") != "permanent":
            continue
        row = by_name.get(str(name))
        if row is None or "chat" not in (row.capabilities or ["chat"]):
            continue
        if row.cascade:
            continue
        found[row.name] = RetiredTarget(
            model_id=row.name,
            provider=row.provider,
            endpoint=row.endpoint,
            auth_env_var=row.auth_env_var,
        )
    kinds = (gemini or {}).get("cooldown_kinds") or {}
    parent = next((m for m in models if m.provider == "gemini" and m.cascade), None)
    if parent is not None and isinstance(kinds, dict):
        for mid, kind in kinds.items():
            if kind != "permanent":
                continue
            if mid not in parent.cascade:
                continue
            found[str(mid)] = RetiredTarget(
                model_id=str(mid),
                provider="gemini",
                endpoint=parent.endpoint,
                auth_env_var=parent.auth_env_var,
                cascade_parent=parent.name,
            )
    return list(found.values())


def models_list_url(endpoint: str, provider: str) -> str | None:
    if provider == "gemini":
        return "https://generativelanguage.googleapis.com/v1beta/models"
    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        return None
    path = parsed.path or ""
    if path.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")] + "/models"
    elif path.endswith("/completions"):
        path = path[: -len("/completions")] + "/models"
    else:
        path = path.rstrip("/") + "/models"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def parse_catalog_ids(provider: str, payload: Any) -> list[str]:
    ids: list[str] = []
    if provider == "gemini":
        rows = payload.get("models") if isinstance(payload, dict) else None
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "")
            if name.startswith("models/"):
                name = name[len("models/") :]
            methods = row.get("supportedGenerationMethods") or []
            if methods and "generateContent" not in methods:
                continue
            low = name.lower()
            if any(x in low for x in ("embed", "imagen", "aqa", "tts", "pro", "ultra")):
                continue
            if name:
                ids.append(name)
        return ids
    data = payload.get("data") if isinstance(payload, dict) else None
    for row in data or []:
        if not isinstance(row, dict):
            continue
        mid = str(row.get("id") or "").strip()
        if mid:
            ids.append(mid)
    if provider == "openrouter":
        ids = [i for i in ids if i.endswith(":free")]
    return ids


def parse_suggested_id(text: str, catalog: list[str]) -> str | None:
    if not text or not catalog:
        return None
    catalog_set = {c: c for c in catalog}
    lowered = {c.lower(): c for c in catalog}
    stripped = text.strip().strip("`").strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            cand = str(obj.get("model") or obj.get("id") or "").strip()
            if cand in catalog_set:
                return cand
            if cand.lower() in lowered:
                return lowered[cand.lower()]
    except json.JSONDecodeError:
        pass
    token = re.search(r"[A-Za-z0-9][A-Za-z0-9_./:-]{2,}", stripped)
    if token:
        cand = token.group(0).strip(".,;\"'")
        if cand in catalog_set:
            return cand
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    for cid in sorted(catalog, key=len, reverse=True):
        if cid in text:
            return cid
    return None


def classify_probe(status_code: int | None, body: str) -> ProbeKind:
    kind = classify_failure(status_code, body)
    if kind == "permanent":
        return "gone"
    if kind == "credit":
        return "paid"
    text = (body or "").lower()
    if any(
        s in text
        for s in (
            "requires a paid",
            "paid plan",
            "upgrade your plan",
            "not available on the free",
            "free tier is not",
        )
    ):
        return "paid"
    if status_code is not None and 200 <= status_code < 300:
        return "free"
    if status_code == 429:
        return "free"
    if kind == "daily":
        return "free"
    return "error"


def advisor_prompt(dead_id: str, provider: str, catalog: list[str]) -> str:
    listed = "\n".join(f"- {i}" for i in catalog[:80])
    return (
        f"The {provider} model id `{dead_id}` is retired (HTTP 404 / not supported).\n"
        "Pick the best remaining FREE-TIER chat replacement from this catalog only.\n"
        "Do not pick paid, preview-paid, embedding, or image models.\n"
        "Reply with JSON only: {\"model\":\"exact-id-from-list\"}\n\n"
        f"Catalog:\n{listed}\n"
    )


def send_admin_email(subject: str, body: str) -> bool:
    to_addr = (os.environ.get("ADMIN_EMAIL") or "").strip()
    host = (os.environ.get("SMTP_HOST") or "").strip()
    if not to_addr or not host:
        return False
    port = int((os.environ.get("SMTP_PORT") or "587").strip() or "587")
    user = (os.environ.get("SMTP_USER") or "").strip()
    password = os.environ.get("SMTP_PASSWORD") or ""
    from_addr = (os.environ.get("SMTP_FROM") or user or to_addr).strip()
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(body)
    with smtplib.SMTP(host, port, timeout=30) as smtp:
        if _env_bool("SMTP_STARTTLS", True):
            smtp.starttls()
        if user:
            smtp.login(user, password)
        smtp.send_message(msg)
    return True


class RetireWatch:
    def __init__(
        self,
        *,
        base_url: str,
        models_path: Path,
        http: httpx.AsyncClient,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.models_path = models_path
        self.http = http

    def _headers(self, *, admin: bool = False) -> dict[str, str]:
        names = (
            ("LLMCASCADE_RETIRE_ADMIN_KEY", "LLMCASCADE_ADMIN_API_KEYS")
            if admin
            else ("LLMCASCADE_RETIRE_API_KEY", "LLMCASCADE_API_KEYS")
        )
        for env_name in names:
            raw = (os.environ.get(env_name) or "").strip()
            for part in raw.split(","):
                key = part.strip()
                if key and not key.startswith("$2"):
                    return {"Authorization": f"Bearer {key}"}
        return {}

    def _load_models(self) -> list[ModelConfig]:
        raw = yaml.safe_load(self.models_path.read_text())
        return [ModelConfig.model_validate(item) for item in raw["models"]]

    async def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        resp = await self.http.request(method, f"{self.base_url}{path}", **kwargs)
        resp.raise_for_status()
        return resp.json()

    async def suggest_id(self, dead_id: str, provider: str, catalog: list[str]) -> str | None:
        payload = {
            "prompt": advisor_prompt(dead_id, provider, catalog),
            "capability": "chat",
            "notes": "retire-watch",
        }
        resp = await self.http.post(
            f"{self.base_url}/v1/complete",
            json=payload,
            headers=self._headers(),
            timeout=120.0,
        )
        resp.raise_for_status()
        text = str((resp.json() or {}).get("text") or "")
        return parse_suggested_id(text, catalog)

    async def list_catalog(self, target: RetiredTarget) -> list[str]:
        url = models_list_url(target.endpoint, target.provider)
        if not url:
            return []
        api_key = resolve_auth_env(
            target.auth_env_var, provider=target.provider, key_tier="free"
        )
        if not api_key:
            return []
        headers: dict[str, str] = {}
        params: dict[str, str] = {}
        if target.provider == "gemini":
            params["key"] = api_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = await self.http.get(url, headers=headers, params=params, timeout=30.0)
        resp.raise_for_status()
        return parse_catalog_ids(target.provider, resp.json())

    async def probe(self, target: RetiredTarget, candidate: str) -> ProbeKind:
        api_key = resolve_auth_env(
            target.auth_env_var, provider=target.provider, key_tier="free"
        )
        if not api_key:
            return "error"
        if target.provider == "gemini":
            from llmcascade.cascade import gemini_endpoint

            url = gemini_endpoint(target.endpoint, candidate)
            resp = await self.http.post(
                url,
                params={"key": api_key},
                json={"contents": [{"role": "user", "parts": [{"text": "ping"}]}]},
                timeout=60.0,
            )
            return classify_probe(resp.status_code, resp.text)
        resp = await self.http.post(
            target.endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": candidate,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 8,
            },
            timeout=60.0,
        )
        return classify_probe(resp.status_code, resp.text)

    def apply_yaml(self, old: str, new: str) -> None:
        text = self.models_path.read_text(encoding="utf-8")
        updated = replace_model_id(text, old, new)
        if updated == text:
            raise RuntimeError(f"models.yaml did not contain {old}")
        self.models_path.write_text(updated, encoding="utf-8")

    async def reload_api(self) -> None:
        headers = self._headers(admin=True)
        if not headers:
            headers = self._headers()
        resp = await self.http.post(
            f"{self.base_url}/v1/admin/reload-registry",
            headers=headers,
            timeout=30.0,
        )
        resp.raise_for_status()

    async def replace_one(
        self, target: RetiredTarget, job: dict[str, Any] | None = None
    ) -> dict[str, str] | None:
        job = job if job is not None else {}
        try:
            catalog = [i for i in await self.list_catalog(target) if i != target.model_id]
        except Exception as exc:
            job["catalog_error"] = str(exc)
            job["catalog"] = []
            job["catalog_n"] = 0
            raise RuntimeError(f"catalog fetch failed for {target.model_id}: {exc}") from exc
        existing = {m.name for m in self._load_models()}
        if target.cascade_parent:
            parent = next(m for m in self._load_models() if m.name == target.cascade_parent)
            existing.update(parent.cascade)
        catalog = [i for i in catalog if i not in existing or i == target.model_id]
        job["catalog_n"] = len(catalog)
        job["catalog"] = catalog[:MAX_CATALOG_IDS]
        job["probes"] = []
        tried: list[str] = []
        last_reason = "no catalog"
        max_tries = min(8, max(1, len(catalog)))
        for _ in range(max_tries):
            remaining = [i for i in catalog if i not in tried]
            if not remaining:
                break
            suggested = await self.suggest_id(target.model_id, target.provider, remaining)
            if not suggested:
                suggested = remaining[0]
            tried.append(suggested)
            kind = await self.probe(target, suggested)
            job["probes"].append({"id": suggested, "kind": kind})
            if kind == "free":
                self.apply_yaml(target.model_id, suggested)
                await self.reload_api()
                job["chosen"] = suggested
                return {"old": target.model_id, "new": suggested, "provider": target.provider}
            last_reason = kind
            if kind == "error":
                continue
        raise RuntimeError(
            f"no free replacement for {target.model_id} ({last_reason}); tried {tried}"
        )

    async def run_once(self) -> list[dict[str, str]]:
        run: dict[str, Any] = {
            "at": _iso_now(),
            "ok": True,
            "error": None,
            "targets": [],
            "skipped": [],
            "jobs": [],
        }
        try:
            status = await self._json("GET", "/v1/status")
            gemini = None
            try:
                gemini = await self._json("GET", "/v1/status/gemini")
            except httpx.HTTPError:
                gemini = status.get("gemini_cascade") if isinstance(status, dict) else None
            cooldowns = status.get("model_cooldowns") if isinstance(status, dict) else {}
            models = self._load_models()
            targets = retired_from_status(
                models=models,
                cooldowns=cooldowns or {},
                gemini=gemini if isinstance(gemini, dict) else None,
            )
            run["targets"] = [t.model_id for t in targets]
            with _STATE_LOCK:
                state = load_state()
            done: list[dict[str, str]] = []
            for target in targets:
                if target.model_id in (state.get("replacements") or {}):
                    run["skipped"].append({"id": target.model_id, "reason": "already replaced"})
                    continue
                if target.model_id in (state.get("abandoned") or {}):
                    run["skipped"].append({"id": target.model_id, "reason": "abandoned"})
                    continue
                job: dict[str, Any] = {
                    "id": target.model_id,
                    "provider": target.provider,
                }
                try:
                    result = await self.replace_one(target, job=job)
                except RuntimeError as exc:
                    job["error"] = str(exc)
                    run["jobs"].append(job)
                    with _STATE_LOCK:
                        state = load_state()
                        state.setdefault("abandoned", {})[target.model_id] = str(exc)
                        save_state(state)
                    try:
                        send_admin_email(
                            f"llmcascade: could not replace {target.model_id}",
                            f"Provider {target.provider} model `{target.model_id}` is retired.\n\n{exc}\n",
                        )
                    except Exception:
                        pass
                    continue
                except Exception as exc:  # noqa: BLE001
                    job["error"] = str(exc)
                    run["jobs"].append(job)
                    continue
                run["jobs"].append(job)
                if not result:
                    continue
                with _STATE_LOCK:
                    state = load_state()
                    state.setdefault("replacements", {})[target.model_id] = {
                        **result,
                        "at": _iso_now(),
                    }
                    save_state(state)
                try:
                    send_admin_email(
                        f"llmcascade: replaced {result['old']}",
                        (
                            f"Retired `{result['old']}` on {result['provider']}.\n"
                            f"Now using `{result['new']}`.\n"
                            "models.yaml was updated and the API registry was reloaded.\n"
                        ),
                    )
                except Exception:
                    pass
                done.append(result)
            append_run(run)
            return done
        except Exception as exc:  # noqa: BLE001
            run["ok"] = False
            run["error"] = str(exc)
            append_run(run)
            raise


async def async_main(interval_s: int) -> None:
    load_env_files()
    mark_watcher_alive(pid=os.getpid(), interval_s=interval_s)
    base = (os.environ.get("LLMCASCADE_BASE_URL") or "http://127.0.0.1:12000").rstrip("/")
    models_path = Path(os.environ.get("LLMCASCADE_MODELS_YAML") or default_models_path())
    async with httpx.AsyncClient() as http:
        watch = RetireWatch(base_url=base, models_path=models_path, http=http)
        while True:
            try:
                await watch.run_once()
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(interval_s)


def main() -> None:
    load_env_files()
    parser = argparse.ArgumentParser(description="Replace retired free-tier model IDs")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=0)
    args = parser.parse_args()
    interval = args.interval or _env_int("RETIRE_WATCH_INTERVAL_S", 120)
    mark_watcher_alive(pid=os.getpid(), interval_s=interval)
    if args.once:
        async def _once() -> None:
            load_env_files()
            base = (os.environ.get("LLMCASCADE_BASE_URL") or "http://127.0.0.1:12000").rstrip("/")
            models_path = Path(os.environ.get("LLMCASCADE_MODELS_YAML") or default_models_path())
            async with httpx.AsyncClient() as http:
                watch = RetireWatch(base_url=base, models_path=models_path, http=http)
                await watch.run_once()

        asyncio.run(_once())
        return
    asyncio.run(async_main(interval))


if __name__ == "__main__":
    main()
