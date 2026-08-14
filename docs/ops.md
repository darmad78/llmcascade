# Ops (single process)

Budgets, metrics, health cache, event logs, API-key RPM, and login lockout are **in-memory and process-local**. Always run **one** process (one uvicorn worker / one PM2 instance). Do **not** use `--workers 2+` or PM2 cluster mode until a shared `BudgetStore` exists.

Bind uvicorn to `127.0.0.1`. Terminate TLS at a reverse proxy.

## PM2

```bash
cd /path/to/llmcascade
set -a && source .env && set +a
pm2 delete llmcascade 2>/dev/null || true
pm2 start .venv/bin/uvicorn --name llmcascade --interpreter none --cwd /path/to/llmcascade -- \
  llmcascade.api:app --host 127.0.0.1 --port 12000
pm2 save
```

## Nginx

```nginx
server {
    listen 443 ssl;
    server_name llmcascade.example.com;
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

Set `LLMCASCADE_COOKIE_SECURE=true` (or terminate HTTPS so `X-Forwarded-Proto: https` is set) so admin cookies get the `Secure` flag.

## Auth details

Two independent layers (API key vs admin cookie). Env vars, bcrypt hashes, `ALLOW_PAID`, `SECRET_KEY`, and cookie flags are documented in [`.env.example`](../.env.example).

- First boot creates local admin `admin` / `admin` under `LLMCASCADE_DATA_DIR` (default `.llmcascade/`) and forces a password change.
- Admin session is a JWT in an HttpOnly `SameSite=Lax` cookie. Password changes bump `pwd_version` and invalidate older JWTs.
- Admin POSTs use double-submit CSRF. Login lockout after 5 failures is process-local.
- Provider keys saved in the UI are encrypted at rest with Fernet using an **HKDF-derived** key from `SECRET_KEY`.
- Models with `key_tier=paid` are skipped unless `ALLOW_PAID=true`.
- Logging / `/v1/events`: metadata only. No prompts, completions, API keys, or raw provider bodies.
