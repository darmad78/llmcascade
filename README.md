# llmcascade

Self-hosted **free-tier LLM dispatcher**. At request time it picks an eligible free model from a live rate-limit budget, calls that provider’s native HTTP API, and falls back on failure.

Python **3.11+**. Optional in-process library (`RouterClient`). Not a multi-tenant hosted service. Streaming is not supported in v1.

## Quick start

```bash
git clone git@github.com:darmad78/llmcascade.git
cd llmcascade
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[api]"
cp .env.example .env          # at least one provider key, or add keys later in /admin/providers
set -a && source .env && set +a
uvicorn llmcascade.api:app --host 127.0.0.1 --port 12000
```

1. Open **http://127.0.0.1:12000/login** — first run is `admin` / `admin`; change the password before the dashboard unlocks.
2. Confirm it works:

```bash
curl -s http://127.0.0.1:12000/v1/complete \
  -H 'content-type: application/json' \
  -d '{"prompt":"Say hi in one sentence.","capability":"chat","notes":"app"}'
```

```json
{
  "text": "Hello.",
  "model": "llama-3.1-8b-instant",
  "tokens_used": 12,
  "latency_ms": 184.2,
  "raw": {}
}
```

`notes` tags the calling service for the event log and `/stats`. It is never sent to providers.

**Local:** leave auth off. **Internet:** `LLMCASCADE_PROFILE=production`, hashed API keys, changed admin password, reverse proxy + TLS. See [`.env.example`](.env.example) and [docs/ops.md](docs/ops.md).

Install as a dependency: `pip install "llmcascade[api] @ git+https://github.com/darmad78/llmcascade.git"`.

## Call it

```python
import httpx

r = httpx.post(
    "http://127.0.0.1:12000/v1/complete",
    json={"prompt": "Say hi in one sentence.", "capability": "chat", "notes": "app"},
    timeout=120.0,
)
r.raise_for_status()
print(r.json()["text"])
```

If the server has `REQUIRE_AUTH=true` (or production profile), send `Authorization: Bearer <key>` or `X-API-Key: <key>`. `502` means every eligible model failed.

| Method | Path | Notes |
|--------|------|--------|
| `POST` | `/v1/complete` | `{ "prompt", "capability"?, "params"?, "notes"? }` |
| `GET` | `/v1/status` | Budgets + Gemini snapshot |
| `GET` | `/v1/health` | Provider reachability |
| `GET` | `/help` | Public error catalog (no login) |
| `GET` | `/dashboard` | Status UI (login) |
| `GET` | `/stats` | Charts by `notes` (login; Mongo when `MONGODB_URI` is set) |
| `GET` | `/admin/providers` | Encrypted keys (login) |

Also: `/v1/status/gemini`, `/v1/metrics`, `/v1/stats`, `/v1/events`, `/v1/errors`, `/v1/dashboard` (those `/v1/*` data routes need the **admin** cookie, not the complete API key).

## Providers

| Provider | Auth env |
|----------|----------|
| Groq | `GROQ_API_KEY` |
| Google Gemini | `GOOGLE_API_KEY` — family cascade, see [docs/gemini.md](docs/gemini.md) |
| OpenRouter | `OPENROUTER_API_KEY` |
| Together | `TOGETHER_API_KEY` |
| Cerebras | `CEREBRAS_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| SambaNova | `SAMBANOVA_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Hugging Face | `HF_TOKEN` (or `HUGGINGFACE_API_KEY`) |
| Cloudflare Workers AI | `CLOUDFLARE_API_KEY` + `CLOUDFLARE_ACCOUNT_ID` |
| Cohere | `COHERE_API_KEY` |
| NVIDIA NIM | `NVIDIA_NIM_API_KEY` |
| DeepInfra | `DEEPINFRA_API_KEY` |

IDs and free-tier limits: [`llmcascade/models.yaml`](llmcascade/models.yaml). Missing keys skip that provider. Env vars **override** UI-stored keys.

Fallback: pick eligible model → retry same model once on timeout/5xx → next model on 429 / hard fail → `AllModelsExhaustedError`. Round-robin by default (`least_used`, `priority_first` available). Paid-tier models stay out of auto-select unless `ALLOW_PAID=true`.

## Library (`RouterClient`)

```python
import asyncio
from llmcascade import RouterClient

async def main():
    client = RouterClient()
    await client.start()
    try:
        resp = await client.submit("Summarize free-tier LLM routing.", notes="app")
        print(resp.model, resp.text)
    finally:
        await client.shutdown()

asyncio.run(main())
```

`RouterClient(models_path=..., strategy="round_robin", workers=4, max_queue=100)`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| No models / empty status | Set at least one provider key in `.env` or `/admin/providers`, then restart. |
| `502` on `/v1/complete` | All eligible models failed or were over budget. Check `/v1/errors` (after login) and `/help`. |
| Gemini 429 / daily quota | Expected; cascade cools that ID and tries siblings, then other providers. [docs/gemini.md](docs/gemini.md) |
| Dashboard locked after login | Change the first-run admin password at `/admin/change-password`. |
| Auth errors on complete | Production profile requires an API key header. |
| Stats empty after restart | In-memory unless `MONGODB_URI` is set. |

`pytest -q` after `pip install -e ".[api,dev]"`. Production bind/proxy: [docs/ops.md](docs/ops.md).

## Disclaimer

Not affiliated with listed providers. **BYOK** — you supply credentials and must follow each provider’s terms. **“As is”**, no warranty — see [LICENSE](LICENSE).

## License

**Source-available** (v1.1) — [LICENSE](LICENSE) (draft; not legal advice) and [LICENSING-FAQ.md](LICENSING-FAQ.md). Free for Non-Commercial Use. Commercial Use needs a written license: **support@conceptgame.co.uk**. Apache 2.0 from **10 August 2030**. England and Wales.
