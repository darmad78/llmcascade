# llmrouter

Standalone **free-tier LLM dispatcher**. At request time it picks an eligible free model from a live rate-limit budget, calls that provider’s native HTTP API, and falls back to the next model on failure.

Framework-independent library (`RouterClient`) plus an optional FastAPI HTTP layer. Python **3.11+**. Self-hosted only (not a multi-tenant hosted service).

## Features

- YAML model registry with Pydantic validation
- Per-model sliding-window limits: `rpd` / `rpm` / `rps` / `tpm`
- Round-robin selection over the eligible pool (also `least_used`, `priority_first`)
- Same-model retry once on timeout/5xx, then fallback; `AllModelsExhaustedError` if none succeed
- Async queue + worker pool with backpressure (`QueueFullError`)
- Structured JSON request logs (metadata only) + in-memory metrics
- Status dashboard (`/dashboard`) with per-model health, budgets, event/error logs
- Admin UI (`/admin/providers`) for encrypted provider keys + Free/Paid labels
- Optional free-text `notes` on completions (event log + stats grouping by source)
- Historical stats UI (`/stats`) with MongoDB persistence when `MONGODB_URI` is set
- Optional `POST /v1/complete` HTTP API with opt-in API-key auth + per-key RPM

## Supported providers

| Provider | Auth env | Notes |
|----------|----------|--------|
| Groq | `GROQ_API_KEY` | OpenAI-compatible |
| Google Gemini | `GOOGLE_API_KEY` | Cascade family (`generateContent`) — see below |
| OpenRouter | `OPENROUTER_API_KEY` | Free models (`:free`) |
| Together | `TOGETHER_API_KEY` | OpenAI-compatible |
| Cerebras | `CEREBRAS_API_KEY` | OpenAI-compatible |
| Mistral | `MISTRAL_API_KEY` | OpenAI-compatible |
| SambaNova | `SAMBANOVA_API_KEY` | OpenAI-compatible |
| DeepSeek | `DEEPSEEK_API_KEY` | OpenAI-compatible |
| Hugging Face | `HF_TOKEN` or `HUGGINGFACE_API_KEY` | Router OpenAI API |
| Cloudflare Workers AI | `CLOUDFLARE_API_KEY` + `CLOUDFLARE_ACCOUNT_ID` | Edge inference |
| Cohere | `COHERE_API_KEY` | Chat v2 |
| NVIDIA NIM | `NVIDIA_NIM_API_KEY` | OpenAI-compatible |
| DeepInfra | `DEEPINFRA_API_KEY` | OpenAI-compatible |

Model IDs, endpoints, and free-tier limit estimates live in [`llmrouter/models.yaml`](llmrouter/models.yaml).

## Install

```bash
git clone git@github.com:darmad78/llmrouter.git
cd llmrouter
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -e ".[api]"   # optional FastAPI + uvicorn + admin auth crypto
pip install -e ".[dev]"   # pytest
```

Or from another project:

```bash
pip install "git+https://github.com/darmad78/llmrouter.git"
# with HTTP layer:
pip install "llmrouter[api] @ git+https://github.com/darmad78/llmrouter.git"
```

## Configuration

```bash
cp .env.example .env
# fill the keys you have — models without a key are skipped at startup
# or leave keys empty and configure them later in /admin/providers
```

Hugging Face: `HF_TOKEN` is preferred; if unset, `HUGGINGFACE_API_KEY` is used.  
Cloudflare: both `CLOUDFLARE_API_KEY` and `CLOUDFLARE_ACCOUNT_ID` are required for the Cloudflare model to load (`ACCOUNT_ID` stays env-only).  
Keys may also be saved encrypted via the admin UI; **environment variables always override** UI-stored keys.

Token estimates for TPM gating use `len(text) // 4` (no tiktoken). After a successful call, provider-reported `tokens_used` is preferred for `record_usage`.

## Programmatic usage (Python)

### A. HTTP client (self-hosted instance)

No llmrouter install required — call `POST /v1/complete` on your instance.

```python
import httpx

BASE = "http://localhost:12000"

def complete(prompt: str, capability: str = "chat", notes: str | None = None, **params) -> dict:
    body = {"prompt": prompt, "capability": capability, "params": params}
    if notes:
        body["notes"] = notes  # e.g. "app" / "system" — shown in event log + stats
    headers = {}
    # If REQUIRE_AUTH=true on the server:
    # headers["Authorization"] = "Bearer YOUR_LLMROUTER_API_KEY"
    r = httpx.post(f"{BASE}/v1/complete", json=body, headers=headers, timeout=120.0)
    r.raise_for_status()
    return r.json()  # text, model, tokens_used, latency_ms, raw

data = complete("Say hi in one sentence.", notes="app")
print(data["model"], data["latency_ms"], data["text"])
```

Async:

```python
import httpx
import asyncio

BASE = "http://localhost:12000"

async def complete(prompt: str, capability: str = "chat", notes: str | None = None, **params) -> dict:
    body = {"prompt": prompt, "capability": capability, "params": params}
    if notes:
        body["notes"] = notes
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{BASE}/v1/complete", json=body)
        r.raise_for_status()
        return r.json()

asyncio.run(complete("Ping", notes="system"))
```

Useful GETs: `/v1/status`, `/v1/metrics`, `/v1/health`, `/v1/status/gemini`, `/v1/stats`, `/v1/events`.  
`502` means the router exhausted eligible models (or a provider error bubbled up).

### B. In-process library (`RouterClient`)

Runs the dispatcher inside your app (needs provider keys in the environment / `.env` / admin store).

```python
import asyncio
from llmrouter import RouterClient

async def main():
    client = RouterClient()  # loads packaged models.yaml
    await client.start()
    try:
        resp = await client.submit(
            "Summarize free-tier LLM routing.",
            capability="chat",
            notes="app",
        )
        print(resp.model, resp.latency_ms, resp.text)
    finally:
        await client.shutdown()

asyncio.run(main())
```

Custom registry / tuning:

```python
client = RouterClient(
    models_path="/path/to/models.yaml",
    strategy="round_robin",  # or least_used, priority_first
    workers=4,
    max_queue=100,
)
```

Response shape (both paths): `text`, `model`, `tokens_used`, `latency_ms`, `raw`.

## Run (HTTP + dashboard)

Budgets, metrics, health cache, event logs, API-key RPM, and login lockout are **in-memory and process-local**. Always run **one** process (one uvicorn worker / one PM2 instance). Do **not** use `--workers 2+` or PM2 cluster mode until a shared `BudgetStore` exists.

### 1. Env + install

```bash
cd /path/to/llmrouter
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[api]"
cp .env.example .env   # fill provider keys and optional security settings
set -a && source .env && set +a
```

### 2. Foreground (dev)

```bash
uvicorn llmrouter.api:app --host 0.0.0.0 --port 12000
```

Open **http://localhost:12000/login** (default first-run: `admin` / `admin` — you must change the password before the dashboard unlocks).

### 3. Process manager (example)

Single instance only (`instances: 1`, no cluster). Load `.env` before start so keys are inherited.

```bash
cd /path/to/llmrouter
set -a && source .env && set +a
pm2 delete llmrouter 2>/dev/null || true
pm2 start .venv/bin/uvicorn --name llmrouter --interpreter none --cwd /path/to/llmrouter -- \
  llmrouter.api:app --host 127.0.0.1 --port 12000
pm2 save
```

Prefer binding uvicorn to `127.0.0.1` and terminating TLS at a reverse proxy when exposing beyond localhost.

### 4. Nginx reverse proxy (example)

```nginx
server {
    listen 443 ssl;
    server_name llmrouter.example.com;
    # ssl_certificate /path/to/fullchain.pem;
    # ssl_certificate_key /path/to/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:12000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Set `LLMROUTER_COOKIE_SECURE=true` (or terminate HTTPS so `X-Forwarded-Proto: https` is set) so admin cookies get the `Secure` flag.

### API / UI routes

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/complete` | Body: `{ "prompt", "capability"?, "params"?, "notes"? }` → `LLMResponse` |
| `GET` | `/v1/status` | Remaining budget per model + `gemini_cascade` snapshot |
| `GET` | `/v1/status/gemini` | Per-cascade-model `available` / `available_at` |
| `GET` | `/v1/metrics` | Auth-protected: counters + budgets |
| `GET` | `/v1/stats` | Auth-protected historical counters |
| `GET` | `/v1/health` | Live provider reachability |
| `GET` | `/v1/events` | Auth-protected event ring |
| `GET` | `/v1/errors` | Auth-protected errors ring |
| `GET` | `/v1/dashboard` | Auth-protected dashboard JSON |
| `GET` | `/help` | Public error / scenario catalog (no login) |
| `GET` | `/login` | Admin login form |
| `GET` | `/dashboard` | Status UI (login required) |
| `GET` | `/stats` | Stats UI (login required) |
| `GET` | `/admin/providers` | Provider key management (login required) |
| `GET` | `/admin/change-password` | Forced on first login |

Optional `notes` is free text tagging the **calling service** (e.g. `"finwin:tr"`, `"app"`). It is logged on request events, never forwarded to providers, and rolled into Mongo service stats under `(none)` when omitted. The `/stats` page shows hourly/daily charts by service with interactive toggles.

Example:

```bash
curl -s http://localhost:12000/v1/complete \
  -H 'content-type: application/json' \
  -d '{"prompt":"Hello","capability":"chat","notes":"app"}'
```

### Optional API-key auth on `/v1/complete`

Disabled by default (`REQUIRE_AUTH=false`). To require a key:

```bash
REQUIRE_AUTH=true
LLMROUTER_API_KEYS=change-me,another-key
# optional per-key RPM (process-local):
LLMROUTER_API_RPM=60
```

Clients send `Authorization: Bearer <key>` or `X-API-Key: <key>`.

### Admin auth + provider keys

- First boot creates local admin `admin` / `admin` under `LLMROUTER_DATA_DIR` (default `.llmrouter/`) and forces a password change before dashboard access.
- Admin session is a JWT in an HttpOnly `SameSite=Lax` cookie (`Secure` when HTTPS / `LLMROUTER_COOKIE_SECURE=true`). Password changes bump `pwd_version` and invalidate older JWTs.
- Admin POSTs use double-submit CSRF (`llmrouter_csrf` cookie + form/header token). Login has basic lockout after 5 failures (process-local).
- Provider keys saved in the UI are encrypted at rest with Fernet using an **HKDF-derived** key from `SECRET_KEY` (never the raw secret as the Fernet key).

## Failure / fallback behavior

1. Pick next eligible model (capability match + under budget).
2. On retryable error (timeout / 5xx): wait 250ms, retry **same** model once.
3. On 429 / non-retryable / second failure: try the **next** eligible model.
4. If every eligible model fails → `AllModelsExhaustedError`.

Streaming is **not** supported in v1 (`LLMResponse.text` is a complete string).

## Gemini cascade

Gemini is a **provider family**, not a single model row. `models.yaml` has one logical entry (`name: gemini`) with an ordered `cascade:` list. When the selector picks Gemini, llmrouter walks that list internally before raising to the outer free-provider fallback.

**Order (default):** `gemini-3.6-flash` → `gemini-3-flash-preview` → `gemini-2.5-flash` → `gemini-2.5-flash-lite` → `gemini-2.0-flash` → `gemini-2.0-flash-lite`. Set `GEMINI_MODEL` to prepend a preferred head. `gemini-3.*` IDs may be speculative/preview — a 404 permanently cools that ID only.

**Failure classification → cooldown (process-local, UTC):**

| Kind | Trigger | Cooldown |
|------|---------|----------|
| `daily` | PerDay / daily quota / `limit:0` | Until next `America/Los_Angeles` midnight |
| `rate` | 429, 5xx, timeout | +60 seconds (rolling RPM/TPM window) |
| `permanent` | 404 / not supported | +365 days |
| `transient` | other 4xx | No cooldown; try next cascade member |

Cooldowns are **per cascade model ID only** — exhausting one Gemini model does not cool siblings.

**Default when every member is cooling:** do **not** wait — mark the Gemini family ineligible and fall through to other free providers. Optional `wait_for_gemini=True` on `submit` / complete params waits in ~15s chunks until the earliest cooldown ends.

Thinking models (`*2.5*` or `gemini-3*`) send `thinkingConfig.thinkingBudget: 0`. Thought parts are skipped when parsing candidates; empty / short `MAX_TOKENS` responses count as failure and advance the cascade.

## Development

```bash
pip install -e ".[api,dev]"
pytest -q
```

## Layout

```
llmrouter/
  pyproject.toml
  LICENSE
  .env.example
  README.md
  llmrouter/
    registry.py      # YAML + env / encrypted-store auth resolution
    rate_limiter.py  # BudgetStore + in-memory + API-key limiter
    tokens.py
    selector.py
    cascade.py       # Gemini classify + cooldown + cascade runner
    queue_worker.py  # RouterClient (+ hot reload)
    metrics.py
    event_log.py     # in-memory event / error rings
    health.py        # provider reachability probes
    auth_store.py    # local admin credentials
    provider_store.py# encrypted provider keys
    admin_auth.py    # JWT cookie + CSRF + lockout
    api_auth.py      # optional /v1/complete API keys
    api.py           # optional FastAPI
    static/          # dashboard / stats / admin UI
    adapters/        # one adapter per provider
  tests/
```

## Security

**Self-hosted only.** This is not a multi-tenant hosted service.

- Uvicorn examples often bind `0.0.0.0`. That exposes the process on all interfaces. Prefer `127.0.0.1` behind a reverse proxy, or firewall the port.
- **`POST /v1/complete` has no auth unless you set `REQUIRE_AUTH=true` and `LLMROUTER_API_KEYS`.** Do not expose it to the public internet without that (plus TLS).
- The admin UI (`/login`, `/dashboard`, `/stats`, `/admin/*`, and related `/v1/*` data routes) uses a **separate** JWT cookie auth layer. Enabling API-key auth does not protect the dashboard by itself, and logging into the dashboard does not authorize `/v1/complete`.
- Do **not** put an unauthenticated llmrouter on the public internet. Minimum for any internet exposure: `REQUIRE_AUTH=true`, strong `LLMROUTER_API_KEYS`, changed admin password, reverse proxy + TLS, and `LLMROUTER_COOKIE_SECURE=true`.
- **Free/Paid** on `/admin/providers` is a **display-only** label. Paid providers with a configured key are still selected automatically by routing — the label does not block spend or exclude models.
- Provider keys: `.env` (gitignored) and/or encrypted UI store under `LLMROUTER_DATA_DIR`. Never commit secrets.
- Logging / `/v1/events`: metadata only (model, provider, latency, tokens, capability, safe error status). No prompts, completions, API keys, or raw provider response bodies.
- Single-worker limitation remains: budgets and lockouts are process-local.

## Disclaimer

- llmrouter is **not affiliated with, endorsed by, or sponsored by** any listed LLM provider (Google, Groq, OpenRouter, Together, Cerebras, Mistral, SambaNova, DeepSeek, Hugging Face, Cloudflare, Cohere, NVIDIA, DeepInfra, or others).
- **You** are responsible for complying with each provider’s Terms of Service, rate limits, and acceptable-use policies when you configure and use API keys with this software.
- **BYOK:** This project does not provide or resell API access. You must supply your own authorized credentials.
- Software is provided **“as is”** with **no warranty**, as stated in the [MIT License](LICENSE). Use at your own risk — including account suspensions, rate-limit bans, service disruptions, or unexpected billing.

## License

Distributed under the [MIT License](LICENSE).
