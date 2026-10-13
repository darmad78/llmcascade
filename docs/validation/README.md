# 30-day validation runbook

Start date: **fill on launch day**. Do not spend on legal, BudgetStore, streaming, or ads until day-30 decision.

## Metrics tracker

Copy [metrics-log.md](metrics-log.md) weekly. Pull Pro clicks from `GET /v1/events` (filter `type=validation`) or server logs.

## Kill lines (60 days from launch)

| Test | Kill if |
|------|---------|
| Distribution | &lt; 300 unique GitHub visitors **or** &lt; 25 stars in first 14 days |
| Willingness to pay | &lt; 3 `/pro/checkout` clicks per 100 README visitors **or** zero waitlist emails |
| Interviews | &lt; 5 of 10 interviewees have **ever paid** to solve 429 / multi-provider routing |

**Stop monetization** if any **two** kill lines hit.

## Files

- [launch-kit.md](launch-kit.md) — Show HN / Reddit posts + launch checklist
- [interview-script.md](interview-script.md) — outreach + 15-min script
- [metrics-log.md](metrics-log.md) — stars, traffic, Pro clicks, emails
- [day-30-decision.md](day-30-decision.md) — Pursue / Pivot / Stop worksheet
