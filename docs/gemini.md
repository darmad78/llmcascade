# Gemini cascade

Gemini is a **provider family**, not a single model row. [`llmcascade/models.yaml`](../llmcascade/models.yaml) has one logical entry (`name: gemini`) with an ordered `cascade:` list. When the selector picks Gemini, llmcascade walks that list internally before raising to the outer free-provider fallback.

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

Status: `GET /v1/status/gemini` (`available` / `available_at` per cascade ID).
