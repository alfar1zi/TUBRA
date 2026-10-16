#!/usr/bin/env bash
# Serve the API from this machine and expose it publicly through a quick
# Cloudflare Tunnel. Quick tunnels need no account or card, and the hostname
# changes every run, so the script prints the URL to paste into VITE_API_BASE.
#
# Usage:
#   ./scripts/serve_public.sh          # API only
#   ./scripts/serve_public.sh web      # also serve the built web app
#
# Requires: cloudflared (https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_PORT="${API_PORT:-8080}"
WEB_PORT="${WEB_PORT:-5173}"
WITH_WEB=0
[[ "${1:-}" == "web" ]] && WITH_WEB=1

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared not found; install it first" >&2
  exit 1
fi

export TBCARE_ARTIFACTS="$REPO_ROOT/ml/artifacts"
export DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-staging}"
export ALLOW_BLOCKED_CANDIDATE="${ALLOW_BLOCKED_CANDIDATE:-true}"
# Quick tunnels hand out a fresh hostname each run, so the allowed origin is
# widened here; pin a real domain in ALLOWED_ORIGINS if you have one.
export ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-*}"

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  kill 0 2>/dev/null || true
  exit "$code"
}
trap cleanup EXIT INT TERM

cd "$REPO_ROOT/apps/api"
PYTHONPATH=. gunicorn \
  --bind "127.0.0.1:${API_PORT}" \
  --workers 1 --threads 4 --preload --timeout 120 \
  "tbcare_api.app:create_app()" &
API_PID=$!

# Wait for the API to answer before opening the tunnel, otherwise the first
# requests through the public URL hit a closed port.
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${API_PORT}/api/v1/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "api exited before it became healthy" >&2
    exit 1
  fi
  sleep 1
done

if [[ "$WITH_WEB" -eq 1 ]]; then
  cd "$REPO_ROOT/apps/web"
  npx vite --port "$WEB_PORT" --host 127.0.0.1 &
fi

echo "api listening on http://127.0.0.1:${API_PORT}"
echo "opening a quick tunnel (no account needed) ..."
exec cloudflared tunnel --url "http://127.0.0.1:${API_PORT}" --no-autoupdate
