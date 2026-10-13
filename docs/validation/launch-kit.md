# Launch kit (single day)

Prerequisites: Apache 2.0 license live, `/v1/chat/completions` documented, PyPI optional.

## Checklist (launch morning)

1. Tag `v0.1.1` (or current) on GitHub with release notes (OpenAI endpoint + Apache 2.0).
2. Set `LLMCASCADE_PRO_CHECKOUT_URL` on any public demo, or leave mailto waitlist.
3. Post **same day** (stagger 2–4 hours): Show HN → r/LocalLLaMA → r/selfhosted.
4. Drop link to `docs/validation/` only in your own notes — public posts link to repo README.
5. Start [metrics-log.md](metrics-log.md) row **Launch D0**.

## Show HN title options

- `llmcascade – self-hosted dispatcher that routes free-tier LLM calls by live rate-limit budget`
- `llmcascade – BYOK router that won't send a call your remaining RPM/RPD can't take`

## Show HN comment (first post)

Self-hosted Python dispatcher for people who already have Groq/Gemini/OpenRouter keys and hit **429s**. It picks the next eligible **free** model using provider headers + learned daily caps, then fails over. Embeddings stay pinned (no cross-model vector spaces). OpenAI-compatible `POST /v1/chat/completions` (non-streaming). Apache 2.0.

Not LiteLLM (heavier gateway) and not OpenRouter (keys leave your host). Single process; HA needs Pro waitlist / shared budget store.

Repo: `https://github.com/darmad78/llmcascade`

## r/LocalLLaMA

Title: `Self-hosted free-tier LLM router with live quota headroom (Apache 2.0)`

Body: Short problem (429s across free providers), what you built, link, ask for feedback on whether `quota_aware` routing beats their current setup.

## r/selfhosted

Title: `llmcascade – BYOK FastAPI dispatcher for free-tier LLM APIs + dashboard`

Body: Docker/uvicorn quick start from README, MCP endpoint for Cursor, link to ops.md.

## After launch

- Reply to every comment for 48h.
- Begin [interview-script.md](interview-script.md) outreach (20 DMs, 10 calls).
- D+14: score distribution kill line.
