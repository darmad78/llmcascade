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
- Optional `POST /v1/complete` HTTP API

## Supported providers

| Provider | Auth env | Notes |
|----------|----------|--------|
| Groq | `GROQ_API_KEY` | OpenAI-compatible |
| Google Gemini | `GOOGLE_API_KEY` | `generateContent` |
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
# fill provider keys — load_registry fails if any configured model's auth env is missing
```

Hugging Face: `HF_TOKEN` is preferred; if unset, `HUGGINGFACE_API_KEY` is used.  
Cloudflare: both `CLOUDFLARE_API_KEY` and `CLOUDFLARE_ACCOUNT_ID` are required when a Cloudflare model is in the registry.

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

## HTTP API (single process only)

Budgets and metrics are **in-memory and process-local**. Run **one** uvicorn worker:

```bash
uvicorn llmrouter.api:app --host 0.0.0.0 --port 8080
```

Do **not** use `--workers 2+` until a shared `BudgetStore` (e.g. Redis) exists — each process would under-count usage and overshoot provider free-tier limits.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/complete` | Body: `{ "prompt", "capability"?, "params"? }` → `LLMResponse` |
| `GET` | `/v1/status` | Remaining budget per model |
| `GET` | `/v1/metrics` | `requests_total`, `failures_total`, `current_budget` |

Example:

```bash
curl -s localhost:8080/v1/complete \
  -H 'content-type: application/json' \
  -d '{"prompt":"Hello","capability":"chat"}'
```

## Failure / fallback behavior

1. Pick next eligible model (capability match + under budget).
2. On retryable error (timeout / 5xx): wait 250ms, retry **same** model once.
3. On 429 / non-retryable / second failure: try the **next** eligible model.
4. If every eligible model fails → `AllModelsExhaustedError`.

Streaming is **not** supported in v1 (`LLMResponse.text` is a complete string).

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
    queue_worker.py  # RouterClient
    metrics.py
    api.py           # optional FastAPI
    adapters/        # one adapter per provider
  tests/
```

## License

Private repository — all rights reserved unless a license file is added later.
