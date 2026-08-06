# llmrouter

Standalone **free-tier LLM dispatcher**. At request time it picks an eligible free model from a live rate-limit budget, calls that provider’s native HTTP API, and falls back to the next model on failure.

Framework-independent library (`RouterClient`) plus an optional FastAPI HTTP layer. Python **3.11+**.

## Features

- YAML model registry with Pydantic validation
- Per-model sliding-window limits: `rpd` / `rpm` / `rps` / `tpm`
- Round-robin selection over the eligible pool (also `least_used`, `priority_first`)
- Same-model retry once on timeout/5xx, then fallback; `AllModelsExhaustedError` if none succeed
- Async queue + worker pool with backpressure (`QueueFullError`)
- Structured JSON request logs + in-memory metrics
- Status dashboard (`/dashboard`) with per-model health, budgets, event/error logs
- Optional `POST /v1/complete` HTTP API

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
pip install -e ".[api]"   # optional FastAPI + uvicorn
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
```

Hugging Face: `HF_TOKEN` is preferred; if unset, `HUGGINGFACE_API_KEY` is used.  
Cloudflare: both `CLOUDFLARE_API_KEY` and `CLOUDFLARE_ACCOUNT_ID` are required for the Cloudflare model to load.  
Startup fails only if **no** model has a usable key.

Token estimates for TPM gating use `len(text) // 4` (no tiktoken). After a successful call, provider-reported `tokens_used` is preferred for `record_usage`.

## Library usage

```python
import asyncio
from llmrouter import RouterClient

async def main():
    client = RouterClient()  # loads packaged models.yaml
    await client.start()
    try:
        resp = await client.submit("Summarize free-tier LLM routing.", capability="chat")
        print(resp.model, resp.latency_ms, resp.text)
    finally:
        await client.shutdown()

asyncio.run(main())
```

Custom registry path:

```python
client = RouterClient(models_path="/path/to/models.yaml", strategy="round_robin", workers=4, max_queue=100)
```

## Run (HTTP + dashboard)

Budgets, metrics, health cache, and event logs are **in-memory and process-local**. Always run **one** process (one uvicorn worker / one PM2 instance). Do **not** use `--workers 2+` or PM2 cluster mode until a shared `BudgetStore` exists.

### 1. Env + install

```bash
cd /path/to/llmrouter
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[api]"
cp .env.example .env   # fill provider keys
set -a && source .env && set +a
```

### 2. Foreground (dev)

```bash
uvicorn llmrouter.api:app --host 0.0.0.0 --port 12000
```

Open **http://localhost:12000/dashboard** for model status, live health, event log, and error log.

### 3. PM2 (production)

Single instance only (`instances: 1`, no cluster):

```bash
cd /path/to/llmrouter
source .venv/bin/activate
set -a && source .env && set +a

# first start
pm2 start "uvicorn llmrouter.api:app --host 0.0.0.0 --port 12000" \
  --name llmrouter --interpreter bash

# after git pull / code changes
git pull
pip install -e ".[api]"
pm2 restart llmrouter
```

Persist across reboot:

```bash
pm2 save
pm2 startup
```

### API routes

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/complete` | Body: `{ "prompt", "capability"?, "params"? }` → `LLMResponse` |
| `GET` | `/v1/status` | Remaining budget per model + `gemini_cascade` snapshot |
| `GET` | `/v1/status/gemini` | Per-cascade-model `available` / `available_at` |
| `GET` | `/v1/metrics` | `requests_total`, `failures_total`, `current_budget` |
| `GET` | `/v1/health` | Live provider reachability (cached ~30s; `?force=true` to refresh) |
| `GET` | `/v1/events` | In-memory event ring buffer |
| `GET` | `/v1/errors` | Errors-only ring buffer |
| `GET` | `/v1/dashboard` | Combined JSON for the UI |
| `GET` | `/dashboard` | Status UI |

Example:

```bash
curl -s localhost:12000/v1/complete \
  -H 'content-type: application/json' \
  -d '{"prompt":"Hello","capability":"chat"}'
```

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
| `rate` | 429, 5xx, timeout | +10 minutes |
| `permanent` | 404 / not supported | +365 days |
| `transient` | other 4xx | No cooldown; try next cascade member |

Shared quota text (`exceeded your current quota`, plan+billing) cools **all** cascade members together.

**Default when every member is cooling:** do **not** wait — mark the Gemini family ineligible and fall through to other free providers. Optional `wait_for_gemini=True` on `submit` / complete params waits in ~15s chunks until the earliest cooldown ends.

Thinking models (`*2.5*` or `gemini-3*`) send `thinkingConfig.thinkingBudget: 0`. Thought parts are skipped when parsing candidates; empty / short `MAX_TOKENS` responses count as failure and advance the cascade.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

## Layout

```
llmrouter/
  pyproject.toml
  .env.example
  README.md
  llmrouter/
    registry.py      # YAML + env validation
    rate_limiter.py  # BudgetStore + in-memory limiter
    tokens.py
    selector.py
    cascade.py       # Gemini classify + cooldown + cascade runner
    queue_worker.py  # RouterClient
    metrics.py
    event_log.py     # in-memory event / error rings
    health.py        # provider reachability probes
    api.py           # optional FastAPI
    static/          # /dashboard UI
    adapters/        # one adapter per provider
  tests/
```

## License

Private repository — all rights reserved unless a license file is added later.
