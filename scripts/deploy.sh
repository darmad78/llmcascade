#!/usr/bin/env bash
# Full production deploy for the PM2 llmrouter process.
#
# - git pull (unless DEPLOY_SKIP_PULL=1)
# - pip install -e '.[api]' into .venv
# - load repo .env (shell-exported vars win)
# - recreate PM2 app so env is fully applied
# - pm2 save
#
# Usage (on the server):
#   ./scripts/deploy.sh
#   MONGODB_URI='mongodb+srv://...' ./scripts/deploy.sh
#
# Optional:
#   DEPLOY_SKIP_PULL=1     skip git pull
#   PM2_APP_NAME=llmrouter
#   UVICORN_HOST=0.0.0.0 UVICORN_PORT=12000
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

APP_NAME="${PM2_APP_NAME:-llmrouter}"
HOST="${UVICORN_HOST:-0.0.0.0}"
PORT="${UVICORN_PORT:-12000}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
UVICORN_BIN="${UVICORN_BIN:-$VENV_DIR/bin/uvicorn}"
PIP_BIN="${PIP_BIN:-$VENV_DIR/bin/pip}"

if ! command -v pm2 >/dev/null 2>&1; then
  echo "error: pm2 not found on PATH" >&2
  exit 1
fi

if [[ "${DEPLOY_SKIP_PULL:-0}" != "1" ]]; then
  if [[ -d "$ROOT/.git" ]]; then
    echo "info: git pull"
    git pull --ff-only
  else
    echo "warn: not a git checkout — skipping pull" >&2
  fi
else
  echo "info: skipping git pull (DEPLOY_SKIP_PULL=1)"
fi

if [[ ! -x "$PIP_BIN" ]]; then
  echo "info: creating venv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "info: pip install -e '.[api]'"
"$PIP_BIN" install -e '.[api]'

if [[ ! -x "$UVICORN_BIN" ]]; then
  echo "error: uvicorn missing after install ($UVICORN_BIN)" >&2
  exit 1
fi

# Merge .env under the current environment.
# Shell/non-empty exports win. Uses python-dotenv so values with & ? # work
# (bash `source`/eval breaks on unquoted mongodb+srv URIs).
load_dotenv() {
  local file="$1"
  local py="${VENV_DIR}/bin/python"
  [[ -f "$file" ]] || return 0
  [[ -x "$py" ]] || py="$PYTHON_BIN"
  # shellcheck disable=SC1090
  eval "$(
    "$py" - "$file" <<'PY'
import os, shlex, sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    from dotenv import dotenv_values
except ImportError:
    # Minimal fallback: KEY=VALUE, strip optional quotes; skip if already set.
    vals = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
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
        vals[key] = val
else:
    vals = {k: v for k, v in dotenv_values(path).items() if k and v is not None}

for key, val in vals.items():
    if not key:
        continue
    # Prefer non-empty shell exports; allow .env to fill empty placeholders.
    if os.environ.get(key):
        continue
    print(f"export {key}={shlex.quote(val)}")
PY
  )"
}

if [[ ! -f "$ROOT/.env" ]]; then
  echo "warn: $ROOT/.env missing — copy from .env.example and fill keys" >&2
else
  load_dotenv "$ROOT/.env"
  echo "info: loaded .env from $ROOT/.env"
fi

if [[ -z "${MONGODB_URI:-}" ]]; then
  echo "warn: MONGODB_URI is not set after loading .env (stats will stay disabled)" >&2
  echo "warn: put it in $ROOT/.env as MONGODB_URI='mongodb+srv://...'" >&2
else
  echo "info: MONGODB_URI is set (${#MONGODB_URI} chars)"
fi

if pm2 describe "$APP_NAME" >/dev/null 2>&1; then
  echo "info: deleting existing PM2 app '$APP_NAME'"
  pm2 delete "$APP_NAME"
fi

echo "info: starting PM2 app '$APP_NAME' (host=$HOST port=$PORT)"
pm2 start "$UVICORN_BIN" \
  --name "$APP_NAME" \
  --interpreter none \
  --cwd "$ROOT" \
  -- \
  llmrouter.api:app --host "$HOST" --port "$PORT"

pm2 save
pm2 show "$APP_NAME" | sed -n '1,40p'
echo "ok: deploy complete — $APP_NAME is up"
